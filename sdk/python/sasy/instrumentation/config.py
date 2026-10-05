"""
Instrumentation configuration.

Connection settings (URL, TLS, auth) are managed by the shared
``sasy.config`` module. This module adds instrumentation-specific
options (logging, OTel, feedback callbacks).
"""

from collections.abc import Callable
from typing import Any

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from sasy.auth.hooks import AuthHook

# Signature: callback(accumulator: FeedbackAccumulator, output: ChatDocument | None, agent: Agent)
FeedbackCallback = Callable[[Any, Any, Any], None]

# Settings that only the program itself may set. An agent with a file-writing
# tool can drop a ``.env`` next to the application, and a generic variable name
# collides with other software, so the one switch that lets unauthorized work
# through is not readable from either place.
_CODE_ONLY_SETTINGS = frozenset({"tool_policy_fail_closed"})


class _CodeOnly(PydanticBaseSettingsSource):
    """One of pydantic-settings' sources with the code-only fields removed."""

    def __init__(self, source: PydanticBaseSettingsSource) -> None:
        super().__init__(source.settings_cls)
        self._source = source

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self._source.get_field_value(field, field_name)

    def __call__(self) -> dict[str, Any]:
        return {name: value for name, value in self._source().items()
                if name not in _CODE_ONLY_SETTINGS}


class InstrumentationConfig(BaseSettings):
    """Instrumentation-specific settings.

    Connection settings (URL, TLS, auth) come from ``sasy.config``.
    This class holds only the instrumentation-layer options.

    Every setting can be passed to :func:`configure`. Those that can also be
    read from the environment, or from a ``.env`` file in the working
    directory, carry the ``SASY_`` prefix:

        SASY_LOG_DENIALS=true
        SASY_LOG_POLICY_DECISIONS=false
        SASY_LOG_POLICY_DECISIONS_TRANSFORMS_ONLY=false
        SASY_LOG_TRANSFORMS=false
        OTEL_ENABLED=true
        OTEL_SERVICE_NAME=instrumentation

    The two OpenTelemetry settings keep the names the OpenTelemetry ecosystem
    uses, which the SDK's own ``OTelConfig`` also reads. ``tool_policy_fail_closed``
    is deliberately absent from this list: see :data:`_CODE_ONLY_SETTINGS`.
    """

    model_config = SettingsConfigDict(
        env_prefix="SASY_",
        env_file=".env",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Keep the code-only settings out of everything but the constructor."""
        return (init_settings, _CodeOnly(env_settings), _CodeOnly(dotenv_settings),
                _CodeOnly(file_secret_settings))

    log_denials: bool = Field(
        default=True,
        description="Log authorization denials at responder boundaries",
    )
    log_policy_decisions: bool = Field(
        default=False,
        description="Print colored [POLICY] lines for HTTP authorization decisions",
    )
    log_policy_decisions_transforms_only: bool = Field(
        default=False,
        description="Only print AUTHORIZED when transforms are present",
    )
    log_transforms: bool = Field(
        default=False,
        description="Print colored [TRANSFORM] lines for applied transforms",
    )
    feedback_callback: FeedbackCallback | None = Field(
        default=None,
        description="Callback for handling authorization feedback",
    )

    otel_enabled: bool = Field(
        default=True,
        description="Enable OpenTelemetry tracing",
        alias="OTEL_ENABLED",
    )
    otel_service_name: str = Field(
        default="instrumentation",
        description="Service name for OTel traces",
        alias="OTEL_SERVICE_NAME",
    )

    tool_policy_fail_closed: bool = Field(
        default=True,
        description=("Langroid adapter only: if True, deny a tool call when no "
                     "decision could be obtained. The ADK and LangChain "
                     "adapters always fail closed. Settable in code only; no "
                     "environment variable and no .env line changes it."),
    )



_config: InstrumentationConfig | None = None


def get_config() -> InstrumentationConfig:
    """Get the global instrumentation configuration."""
    global _config
    if _config is None:
        _config = InstrumentationConfig()
    return _config


def configure(
    sasy_url: str | None = None,
    ca_path: str | None = None,
    cert_path: str | None = None,
    key_path: str | None = None,
    auth_hook: AuthHook | None = None,
    log_denials: bool | None = None,
    log_policy_decisions: bool | None = None,
    log_policy_decisions_transforms_only: bool | None = None,
    log_transforms: bool | None = None,
    feedback_callback: FeedbackCallback | None = None,
    otel_enabled: bool | None = None,
    otel_service_name: str | None = None,
    tool_policy_fail_closed: bool | None = None,
    process_global_session: bool | None = None,
) -> InstrumentationConfig:
    """Set how this process reaches the SASY engine and how the adapters behave.

    Calling it is optional: with no arguments the connection settings are read
    from the environment and ``.env`` (``SASY_URL``, ``SASY_API_KEY``,
    ``TLS_CA_PATH``, ``TLS_CERT_PATH``, ``TLS_KEY_PATH``) and the
    instrumentation settings from their ``SASY_``-prefixed names (see
    :class:`InstrumentationConfig`). Arguments left as ``None`` keep their
    current value, so it can be called again to change one setting. Every call
    closes the pooled gRPC channels and drops them; the next call that needs a
    channel opens a new one against the settings in force then.

    Args:
        sasy_url: Engine address as ``host:port`` (a locally started binary
            listens on ``localhost:10089``). There is no default; without one,
            the first call that would open a connection raises
            :class:`sasy.config.SasyEndpointNotConfigured`.
        ca_path: PEM CA bundle used to verify the engine's certificate. Needed
            for the self-signed certificate that ``make certs`` creates.
        cert_path: Client certificate for mutual TLS.
        key_path: Client private key for mutual TLS.
        auth_hook: Supplies per-call credentials, for example
            ``APIKeyAuthHook(api_key=...)``. If omitted and ``SASY_API_KEY`` is
            set, that key is sent unchanged.
        log_denials: Log each denial at WARNING (default ``True``). Read by the
            Langroid adapter's denial feedback; the ADK and LangChain adapters
            report denials to the model and do not log them.
        log_policy_decisions: Print a coloured ``[POLICY]`` line per decision
            made by the HTTP hooks and by the Langroid adapter's tool checks.
        log_policy_decisions_transforms_only: With the above, print AUTHORIZED
            lines only when a transform was applied.
        log_transforms: Print a ``[TRANSFORM]`` line per transform applied by
            the HTTP hooks.
        feedback_callback: ``callback(accumulator, output, agent)``, called by
            the Langroid adapter after each responder that saw a denial.
        otel_enabled: Export OpenTelemetry spans.
        otel_service_name: Service name those spans carry.
        process_global_session: Opt into a generated process-default session.
            False disables it; None preserves the current choice.
        tool_policy_fail_closed: Langroid adapter only. ``True`` (the default)
            blocks a tool when no decision could be obtained. The ADK and
            LangChain adapters always fail closed. This one is settable here
            only: no environment variable and no ``.env`` line turns it off,
            and ``sasy.instrument()`` logs a warning when it is off.

    Returns:
        The process-wide :class:`InstrumentationConfig`.
    """
    global _config
    if _config is None:
        _config = InstrumentationConfig()

    # Forward connection settings to shared config
    from sasy.config import configure as _configure_shared
    _configure_shared(
        url=sasy_url,
        ca_path=ca_path,
        cert_path=cert_path,
        key_path=key_path,
        auth_hook=auth_hook,
        process_global_session=process_global_session,
    )

    # Apply instrumentation-specific overrides
    if log_denials is not None:
        _config.log_denials = log_denials
    if log_policy_decisions is not None:
        _config.log_policy_decisions = log_policy_decisions
    if log_policy_decisions_transforms_only is not None:
        _config.log_policy_decisions_transforms_only = log_policy_decisions_transforms_only
    if log_transforms is not None:
        _config.log_transforms = log_transforms
    if feedback_callback is not None:
        _config.feedback_callback = feedback_callback
    if otel_enabled is not None:
        _config.otel_enabled = otel_enabled
    if otel_service_name is not None:
        _config.otel_service_name = otel_service_name
    if tool_policy_fail_closed is not None:
        _config.tool_policy_fail_closed = tool_policy_fail_closed

    return _config
