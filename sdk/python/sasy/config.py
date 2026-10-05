"""
Shared configuration and gRPC channel management for SASY.

All services (policy engine, reference monitor, credential server,
observability) run on a single SASY binary endpoint. This module
provides the shared URL, TLS, and auth configuration plus a cached
gRPC channel reused across all subsystems.

There is no built-in endpoint. Choose one with ``SASY_URL``,
``configure(url=...)``, or an explicitly provisioned local engine profile
(``sasy engine start``). Without a choice, opening a connection raises
:class:`SasyEndpointNotConfigured`.

Environment variables:
    SASY_URL          SASY endpoint, host:port (required)
    TLS_CA_PATH       CA certificate for server verification
    TLS_CERT_PATH     Client certificate for mTLS
    TLS_KEY_PATH      Client private key for mTLS

Set ``SASY_URL`` to the endpoint configured by your deployment. A SASY
binary started locally listens on ``localhost:10089`` unless it was given
another address. A selected profile in ``~/.sasy`` is a fallback for the whole
connection only when no deployment URL is configured.
"""

import itertools
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast
from urllib.parse import urlsplit

import grpc
from pydantic import Field, PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict

from sasy.auth.hooks import AuthHook, NoAuthHook
from sasy.capture import capture_logger

logger = capture_logger(__name__)

#: Value suggested in the error when an endpoint variable is unset — the
#: address a SASY binary listens on when it is started without one.
LOCAL_DEV_URL = "localhost:10089"

#: The file pydantic reads settings from, alongside the environment. Named
#: once so every setting in this module comes from the same place.
DOTENV_FILE = ".env"


class SasyEndpointNotConfigured(RuntimeError):
    """Raised when a call needs an endpoint the caller never chose.

    The SDK ships no default endpoint: it never picks a host on the
    caller's behalf, so nothing leaves the machine until an endpoint is
    named explicitly.
    """


def _require_endpoint(value: str, env_var: str, local_default: str) -> str:
    """Return *value*, or raise naming the variable that must be set."""
    if value:
        return value
    raise SasyEndpointNotConfigured(
        f"{env_var} is not set, and the SASY SDK has no default endpoint. "
        f"Set {env_var} in the environment or in .env (for a SASY binary "
        f"running locally: {env_var}={local_default}), or pass it to "
        f"sasy.configure()."
    )

# Round-robin synchronous calls across channels to reduce contention. Tune the
# connection count with SASY_CHANNEL_POOL_SIZE; size 1 uses a single channel.
_pool_lock = threading.Lock()
_channel_pool: list[grpc.Channel] = []
_channel_rr = itertools.count()

# asyncio gRPC channels are bound to the event loop they were
# created on — sharing one across loops/threads triggers runtime
# errors. Key the async pool on the running loop so each loop gets
# its own pool. Sync channels don't have this constraint, so they
# stay process-wide.
_async_channel_pools: dict[int, list[grpc.aio.Channel]] = {}
_async_channel_rrs: dict[int, "itertools.count[int]"] = {}

# Per-(stub_class, channel) stub cache. Stubs are stateless gRPC
# handles, but constructing one per call still allocates Python
# objects + acquires the GIL. Caching keys off ``id(channel)`` so the
# round-robin pool gets one stub per channel; total stubs in the
# cache after warmup is bounded by ``pool_size × n_stub_classes``.
_Stub = TypeVar("_Stub")
_stub_cache: dict[tuple[Callable[[Any], object], int], object] = {}
_async_stub_cache: dict[tuple[Callable[[Any], object], int], object] = {}
_stub_cache_lock = threading.Lock()


class SasyConfig(BaseSettings):
    """Global SASY configuration.

    Single endpoint for all services, with TLS and auth settings.
    """

    model_config = SettingsConfigDict(
        env_file=DOTENV_FILE,
        extra="ignore",
        arbitrary_types_allowed=True,
    )

    # No default: the SDK never picks an endpoint for the caller.
    url: str = Field(
        default="",
        description="SASY binary URL (host:port); required",
        alias="SASY_URL",
    )

    engine_profile: str | None = Field(default=None, alias="SASY_ENGINE_PROFILE")

    # TLS
    ca_path: str | None = Field(default=None, alias="TLS_CA_PATH")
    cert_path: str | None = Field(default=None, alias="TLS_CERT_PATH")
    key_path: str | None = Field(default=None, alias="TLS_KEY_PATH")

    # Auth — either pass an explicit auth_hook via configure(), or set
    # SASY_API_KEY in the env and we'll wire an APIKeyAuthHook for you.
    # The complete key is opaque and is sent unchanged.
    api_key: str = Field(default="", alias="SASY_API_KEY")
    auth_hook: AuthHook = Field(
        default_factory=NoAuthHook,
        description="Auth hook for gRPC metadata (API key, JWT, etc.)",
    )

    # More channels trade additional connections for less contention between
    # concurrent synchronous callers. Async pools are separate per event loop.
    channel_pool_size: int = Field(
        default=64,
        ge=1,
        description="Number of gRPC channels to round-robin across",
        alias="SASY_CHANNEL_POOL_SIZE",
    )

    _profile_fields: set[str] = PrivateAttr(default_factory=set)
    _profile_auth: bool = PrivateAttr(default=False)

    def model_post_init(self, __context: object) -> None:
        """Use a full SASY_API_KEY when no auth hook was supplied.

        Missing or empty keys leave authentication unconfigured. The SDK does
        not derive credentials from entity names or other environment settings.
        """
        from sasy.local_profiles import load_profile

        # A deployment URL must never inherit a different engine's key or CA.
        # Checking fields_set also respects an explicitly empty URL.
        profile = None
        if "url" not in self.model_fields_set:
            profile = load_profile(self.engine_profile)
        if profile is not None:
            for field, value in {"url": profile["url"], "api_key": profile["api_key"],
                                 "ca_path": profile["ca_path"]}.items():
                if field not in self.model_fields_set:
                    object.__setattr__(self, field, value)
                    self._profile_fields.add(field)
        if isinstance(self.auth_hook, NoAuthHook) and self.api_key:
            from sasy.auth.hooks import APIKeyAuthHook
            object.__setattr__(self, "auth_hook", APIKeyAuthHook(api_key=self.api_key))
            self._profile_auth = "api_key" in self._profile_fields

    def require_url(self) -> str:
        """Return the gRPC endpoint, or raise if none was configured."""
        return _require_endpoint(self.url, "SASY_URL", LOCAL_DEV_URL)

    def load_ca_cert(self) -> bytes | None:
        if self.ca_path and Path(self.ca_path).exists():
            return Path(self.ca_path).read_bytes()
        return None

    def load_cert(self) -> bytes | None:
        if self.cert_path and Path(self.cert_path).exists():
            return Path(self.cert_path).read_bytes()
        return None

    def load_key(self) -> bytes | None:
        if self.key_path and Path(self.key_path).exists():
            return Path(self.key_path).read_bytes()
        return None

    def build_credentials(self) -> grpc.ChannelCredentials:
        """Build channel credentials from the optional CA / client cert+key.

        mTLS if all three are present, custom-CA TLS if only a CA is set,
        otherwise system-root TLS (works for cloud endpoints). Shared by the
        sync and async channel builders.
        """
        ca_cert = self.load_ca_cert()
        client_cert = self.load_cert()
        client_key = self.load_key()

        if client_cert and client_key and ca_cert:
            logger.debug("Creating mTLS channel to %s", self.url)
            return grpc.ssl_channel_credentials(
                root_certificates=ca_cert,
                private_key=client_key,
                certificate_chain=client_cert,
            )
        if ca_cert:
            logger.debug("Creating TLS channel to %s (custom CA)", self.url)
            return grpc.ssl_channel_credentials(root_certificates=ca_cert)
        logger.debug("Creating TLS channel to %s (system CAs)", self.url)
        return grpc.ssl_channel_credentials()

    def create_channel(self) -> grpc.Channel:
        """Create a gRPC channel. Always uses TLS (system CAs if none configured)."""
        return grpc.secure_channel(self.require_url(), self.build_credentials())

    def get_metadata(self, endpoint: str | None = None) -> list[tuple[str, str]]:
        """Get authentication for the engine, or an explicitly named endpoint.

        Automatically loaded local-profile credentials are scoped to the
        engine's TLS origin. Separately configured HTTP services must not
        receive them. Explicit hooks and environment/project credentials
        retain their existing cross-service behavior.
        """
        if self._profile_auth and endpoint is not None:
            try:
                destination = urlsplit(endpoint if "://" in endpoint else f"https://{endpoint}")
                engine = urlsplit(self.url if "://" in self.url else f"https://{self.url}")
                if (destination.scheme, destination.hostname, destination.port if destination.port is not None else 443) != (
                    engine.scheme, engine.hostname, engine.port if engine.port is not None else 443
                ):
                    return []
            except ValueError:
                return []
        return self.auth_hook.get_metadata()


# ── Singleton config and channel ──────────────────────────

_config: SasyConfig | None = None


def get_config() -> SasyConfig:
    """Get the global SASY configuration (creates on first call)."""
    global _config
    if _config is None:
        _config = SasyConfig()
    return _config


def get_config_for_http() -> SasyConfig:
    """Resolve independently configured authoring services without an engine.

    Reuse explicit process configuration, including its auth hook, if present.
    Otherwise build temporary settings without a local-engine connection;
    missing local profiles must not prevent calls to independent HTTP services.
    Do not cache this temporary config: a later engine call must still resolve
    its selected profile, or report that the selected engine is unavailable.
    """
    return _config if _config is not None else SasyConfig(SASY_URL="")


def _build_async_channel(cfg: SasyConfig) -> grpc.aio.Channel:
    """Construct one async gRPC channel from a config."""
    return grpc.aio.secure_channel(cfg.require_url(), cfg.build_credentials())


def get_channel() -> grpc.Channel:
    """Pick the next sync gRPC channel from the round-robin pool.

    Channels are created lazily on first call and reused for the
    process lifetime. Pool size is ``cfg.channel_pool_size`` (default
    64, override via ``SASY_CHANNEL_POOL_SIZE``); bumping it spreads
    concurrent RPCs across N independent gRPC I/O threads so they
    don't serialize behind a single channel under high concurrency.
    """
    cfg = get_config()
    n = max(1, cfg.channel_pool_size)
    if len(_channel_pool) < n:
        with _pool_lock:
            while len(_channel_pool) < n:
                _channel_pool.append(cfg.create_channel())
    return _channel_pool[next(_channel_rr) % n]


def get_async_channel() -> grpc.aio.Channel:
    """Pick the next async gRPC channel from the round-robin pool.

    Same shape as :func:`get_channel`, but the pool is *per event
    loop*: asyncio gRPC channel handles bind to the loop they're
    constructed on, so a channel created on loop A must not be
    used from loop B. Multi-loop callers (each thread typically
    runs its own loop) get an independent pool keyed on
    ``id(asyncio.get_running_loop())``.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError as e:
        raise RuntimeError(
            "get_async_channel() must be called from inside a running "
            "asyncio event loop"
        ) from e
    loop_key = id(loop)
    cfg = get_config()
    n = max(1, cfg.channel_pool_size)
    pool = _async_channel_pools.get(loop_key)
    if pool is None or len(pool) < n:
        with _pool_lock:
            pool = _async_channel_pools.setdefault(loop_key, [])
            while len(pool) < n:
                pool.append(_build_async_channel(cfg))
            _async_channel_rrs.setdefault(loop_key, itertools.count())
    rr = _async_channel_rrs[loop_key]
    return pool[next(rr) % n]


def get_stub(stub_cls: Callable[[Any], _Stub]) -> _Stub:
    """Return a sync stub of ``stub_cls`` bound to the next pooled channel.

    Stubs are cached per ``(stub_cls, channel)`` so subsequent calls
    on the same channel reuse the same handle. The round-robin still
    picks the next channel on every call, so call patterns spread
    evenly across the pool.
    """
    ch = get_channel()
    key = (stub_cls, id(ch))
    stub = _stub_cache.get(key)
    if stub is not None:
        return cast(_Stub, stub)
    with _stub_cache_lock:
        stub = _stub_cache.get(key)
        if stub is None:
            stub = stub_cls(ch)
            _stub_cache[key] = stub
        return cast(_Stub, stub)


def get_async_stub(stub_cls: Callable[[Any], _Stub]) -> _Stub:
    """Async counterpart to :func:`get_stub`."""
    ch = get_async_channel()
    key = (stub_cls, id(ch))
    stub = _async_stub_cache.get(key)
    if stub is not None:
        return cast(_Stub, stub)
    with _stub_cache_lock:
        stub = _async_stub_cache.get(key)
        if stub is None:
            stub = stub_cls(ch)
            _async_stub_cache[key] = stub
        return cast(_Stub, stub)


def reset_channel() -> None:
    """Reset cached channel pools (e.g. after config change).

    Closes each pooled channel before dropping the reference.
    Clearing the list alone would leave the TCP connection and the
    gRPC I/O thread open until garbage collection runs the channel's
    ``__del__``. Async channels' ``close()`` returns a coroutine, which
    is run on the channel's bound event loop when one is available.
    """
    with _pool_lock:
        for ch in _channel_pool:
            try:
                ch.close()
            except Exception:
                # Best-effort — a half-broken channel shouldn't
                # block the reset.
                pass
        _channel_pool.clear()

        for loop_key, pool in list(_async_channel_pools.items()):
            for async_ch in pool:
                _close_async_channel_best_effort(async_ch)
            pool.clear()
        _async_channel_pools.clear()
        _async_channel_rrs.clear()
    with _stub_cache_lock:
        _stub_cache.clear()
        _async_stub_cache.clear()


def _close_async_channel_best_effort(ch: grpc.aio.Channel) -> None:
    """Run an async channel's ``close()`` from a sync context.

    If we're in a running loop, schedule the close as a task. If
    no loop is running, build a one-shot loop just to drain the
    coroutine. Failures are swallowed — the channel is being
    discarded anyway and the caller is in the middle of a
    teardown path.
    """
    import asyncio

    try:
        coro = ch.close(grace=None)
    except Exception:
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
        return
    except RuntimeError:
        pass
    try:
        asyncio.run(coro)
    except Exception:
        pass


def configure(
    url: str | None = None,
    ca_path: str | None = None,
    cert_path: str | None = None,
    key_path: str | None = None,
    auth_hook: AuthHook | None = None,
    entity: str | None = None,
    process_global_session: bool | None = None,
) -> SasyConfig:
    """Configure the shared SASY connection.

    Args:
        url: SASY binary URL (host:port)
        ca_path: Path to CA certificate for TLS
        cert_path: Path to client certificate for mTLS
        key_path: Path to client private key for mTLS
        auth_hook: Auth hook for gRPC metadata
        process_global_session: Opt into a generated process-default session.
            False disables it; None preserves the current choice.
        entity: Process-wide default user-supplied actor stamped onto
            events and edges. Distinct from the auth-derived
            ``principal`` (immutable, set server-side from auth) and
            from the conversation-level ``agent`` field. Most apps
            leave this unset and override per-block via
            ``with sasy.session(entity=...)``.
    """
    global _config
    if _config is None and url is not None:
        _config = SasyConfig(SASY_URL=url)
    cfg = get_config()
    if url is not None:
        # Drop credentials inherited from a local profile before changing host.
        if "api_key" in cfg._profile_fields:
            cfg.api_key = ""
            if cfg._profile_auth:
                cfg.auth_hook = NoAuthHook()
                cfg._profile_auth = False
        if "ca_path" in cfg._profile_fields:
            cfg.ca_path = None
        cfg._profile_fields.clear()
        cfg.url = url
    if ca_path is not None:
        cfg.ca_path = ca_path
        cfg._profile_fields.discard("ca_path")
    if cert_path is not None:
        cfg.cert_path = cert_path
    if key_path is not None:
        cfg.key_path = key_path
    if auth_hook is not None:
        cfg.auth_hook = auth_hook
        cfg._profile_auth = False
    if entity is not None:
        from sasy.instrumentation.session import configure_default_entity
        configure_default_entity(entity)
    if process_global_session is not None:
        from sasy.instrumentation.session import configure_process_global_session
        configure_process_global_session(process_global_session)
    reset_channel()
    return cfg


def _print_url(argv: list[str]) -> int:
    """Print the endpoint this SDK would dial, and nothing else.

    One resolver, one answer: a caller that has to decide whether a run is
    local or remote must classify the address the SDK will actually use before
    selecting credentials. Reading ``SASY_URL`` for itself is not the same
    question: this reads the process environment and ``.env`` through
    pydantic-settings, which matches variable names case-insensitively, so a
    lower-case ``sasy_url`` is an endpoint here and invisible to a reader that
    only knows the upper-case spelling.

    Prints an empty line when no endpoint is configured, which is a resolved
    answer ("none"), not a failure; a failure is a non-zero exit.
    """
    if argv != ["--print-url"]:
        print("usage: python -m sasy --print-url", file=sys.stderr)
        return 2
    print(SasyConfig().url)
    return 0

