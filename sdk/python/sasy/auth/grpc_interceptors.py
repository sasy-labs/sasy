"""
gRPC server interceptors for authentication.

Provides interceptors that can be added to any gRPC server to enforce
authentication at the application level, complementing transport-level mTLS.

The interceptors store authentication results in a context variable that can
be accessed by service methods using get_auth_context().
"""

import contextvars
from collections.abc import Awaitable, Callable

import grpc
from grpc import aio

from sasy.capture import capture_logger

from .providers import AuthProvider, AuthResult, MTLSAuthConfig, MTLSAuthProvider

logger = capture_logger(__name__)

# Context variable to store auth result for the current request
_auth_context: contextvars.ContextVar[AuthResult | None] = contextvars.ContextVar(
    "auth_context", default=None
)

# Global flag to disable RBAC enforcement (for development only)
_rbac_disabled: bool = False


def disable_rbac() -> None:
    """
    Disable RBAC enforcement globally. FOR DEVELOPMENT USE ONLY.

    When disabled, @require_role decorators will skip authorization checks.
    This allows running services without mTLS/auth configured.

    WARNING: Never call this in production.
    """
    global _rbac_disabled
    _rbac_disabled = True
    logger.warning("RBAC disabled - authorization checks will be skipped")


def enable_rbac() -> None:
    """Re-enable RBAC enforcement."""
    global _rbac_disabled
    _rbac_disabled = False
    logger.info("RBAC enabled")


def get_auth_context() -> AuthResult | None:
    """
    Get the authentication result for the current request.

    Returns the AuthResult set by the interceptor, or None if not authenticated.

    Example:
        def MyServiceMethod(self, request, context):
            auth = get_auth_context()
            if auth and "admin" in auth.roles:
                # Allow admin operation
                ...
            else:
                context.abort(grpc.StatusCode.PERMISSION_DENIED, "Admin required")
    """
    return _auth_context.get()


def require_role(role: str) -> Callable:
    """
    Decorator to require a specific role for a service method.

    Works with both sync and async methods. If RBAC is disabled via disable_rbac(),
    the check is skipped.

    Args:
        role: The role required to access this method

    Example:
        @require_role("observability-writer")
        def RecordEvents(self, request, context):
            ...

        @require_role("credential-reader")
        async def GetCredential(self, request, context):
            ...
    """
    import asyncio
    import functools
    import inspect

    def decorator(func: Callable) -> Callable:
        if inspect.isasyncgenfunction(func):
            @functools.wraps(func)
            async def async_gen_wrapper(self, request, context: aio.ServicerContext, *args, **kwargs):
                if not _rbac_disabled:
                    auth = get_auth_context()
                    if not auth or not auth.authenticated:
                        await context.abort(
                            grpc.StatusCode.UNAUTHENTICATED,
                            "Not authenticated",
                        )
                    if role not in auth.roles:
                        await context.abort(
                            grpc.StatusCode.PERMISSION_DENIED,
                            f"Role '{role}' required, "
                            f"have: {auth.roles}",
                        )
                async for item in func(
                    self, request, context,
                    *args, **kwargs
                ):
                    yield item
            return async_gen_wrapper
        elif asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(self, request, context: aio.ServicerContext, *args, **kwargs):
                if _rbac_disabled:
                    return await func(self, request, context, *args, **kwargs)
                auth = get_auth_context()
                if not auth or not auth.authenticated:
                    await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Not authenticated")
                if role not in auth.roles:
                    await context.abort(
                        grpc.StatusCode.PERMISSION_DENIED,
                        f"Role '{role}' required, have: {auth.roles}"
                    )
                return await func(self, request, context, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(self, request, context: grpc.ServicerContext, *args, **kwargs):
                if _rbac_disabled:
                    return func(self, request, context, *args, **kwargs)
                auth = get_auth_context()
                if not auth or not auth.authenticated:
                    context.abort(grpc.StatusCode.UNAUTHENTICATED, "Not authenticated")
                if role not in auth.roles:
                    context.abort(
                        grpc.StatusCode.PERMISSION_DENIED,
                        f"Role '{role}' required, have: {auth.roles}"
                    )
                return func(self, request, context, *args, **kwargs)
            return sync_wrapper
    return decorator


def require_any_role(*roles: str) -> Callable:
    """
    Decorator to require any of the specified roles for a service method.

    Works with both sync and async methods. If RBAC is disabled via disable_rbac(),
    the check is skipped.

    Args:
        roles: The roles, any of which grants access

    Example:
        @require_any_role("observability-writer", "observability-admin")
        def RecordEvents(self, request, context):
            ...

        @require_any_role("credential-reader", "admin")
        async def GetCredential(self, request, context):
            ...
    """
    import asyncio
    import functools
    import inspect

    def decorator(func: Callable) -> Callable:
        if inspect.isasyncgenfunction(func):
            @functools.wraps(func)
            async def asyncgen_wrapper(self, request, context: aio.ServicerContext, *args, **kwargs):
                if not _rbac_disabled:
                    auth = get_auth_context()
                    if not auth or not auth.authenticated:
                        await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Not authenticated")
                    if not any(r in auth.roles for r in roles):
                        await context.abort(
                            grpc.StatusCode.PERMISSION_DENIED,
                            f"One of {roles} required, have: {auth.roles}"
                        )
                async for item in func(self, request, context, *args, **kwargs):
                    yield item
            return asyncgen_wrapper
        elif asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(self, request, context: aio.ServicerContext, *args, **kwargs):
                if _rbac_disabled:
                    return await func(self, request, context, *args, **kwargs)
                auth = get_auth_context()
                if not auth or not auth.authenticated:
                    await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Not authenticated")
                if not any(r in auth.roles for r in roles):
                    await context.abort(
                        grpc.StatusCode.PERMISSION_DENIED,
                        f"One of {roles} required, have: {auth.roles}"
                    )
                return await func(self, request, context, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(self, request, context: grpc.ServicerContext, *args, **kwargs):
                if _rbac_disabled:
                    return func(self, request, context, *args, **kwargs)
                auth = get_auth_context()
                if not auth or not auth.authenticated:
                    context.abort(grpc.StatusCode.UNAUTHENTICATED, "Not authenticated")
                if not any(r in auth.roles for r in roles):
                    context.abort(
                        grpc.StatusCode.PERMISSION_DENIED,
                        f"One of {roles} required, have: {auth.roles}"
                    )
                return func(self, request, context, *args, **kwargs)
            return sync_wrapper
    return decorator


class AuthInterceptor(grpc.ServerInterceptor):
    """
    Synchronous gRPC server interceptor for authentication.

    Uses an AuthProvider to authenticate requests and reject unauthorized ones.
    """

    def __init__(
        self,
        auth_provider: AuthProvider,
        allow_reflection: bool = True,
        allow_health: bool = True,
    ):
        """
        Initialize the interceptor.

        Args:
            auth_provider: The authentication provider to use
            allow_reflection: Whether to allow unauthenticated access to gRPC reflection
            allow_health: Whether to allow unauthenticated access to health check methods
        """
        self.auth_provider = auth_provider
        self.allow_reflection = allow_reflection
        self.allow_health = allow_health

    def _should_skip_auth(self, method: str) -> bool:
        """Check if authentication should be skipped for this method."""
        if self.allow_reflection and "grpc.reflection" in method:
            return True
        if self.allow_health and ("Health" in method or "health" in method):
            return True
        return False

    def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], grpc.RpcMethodHandler | None],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler | None:
        """Intercept and authenticate the request."""
        method = handler_call_details.method

        # Get the original handler
        handler = continuation(handler_call_details)
        if handler is None:
            return handler

        # Skip auth for allowed methods
        if self._should_skip_auth(method):
            return handler

        # Wrap the handler to add authentication
        return self._wrap_handler(handler)

    def _wrap_handler(
        self, handler: grpc.RpcMethodHandler
    ) -> grpc.RpcMethodHandler:
        """Wrap a handler to add authentication checking."""

        def _auth_wrapper(
            behavior: Callable,
            request_streaming: bool,
            response_streaming: bool,
        ) -> Callable:
            if request_streaming and response_streaming:
                # Bidirectional streaming
                def wrapped(request_iterator, context):
                    result = self.auth_provider.authenticate(context)
                    if not result.authenticated:
                        context.abort(grpc.StatusCode.UNAUTHENTICATED, result.error or "Authentication failed")
                    _auth_context.set(result)
                    return behavior(request_iterator, context)
                return wrapped
            elif request_streaming:
                # Client streaming
                def wrapped(request_iterator, context):
                    result = self.auth_provider.authenticate(context)
                    if not result.authenticated:
                        context.abort(grpc.StatusCode.UNAUTHENTICATED, result.error or "Authentication failed")
                    _auth_context.set(result)
                    return behavior(request_iterator, context)
                return wrapped
            elif response_streaming:
                # Server streaming
                def wrapped(request, context):
                    result = self.auth_provider.authenticate(context)
                    if not result.authenticated:
                        context.abort(grpc.StatusCode.UNAUTHENTICATED, result.error or "Authentication failed")
                    _auth_context.set(result)
                    return behavior(request, context)
                return wrapped
            else:
                # Unary
                def wrapped(request, context):
                    result = self.auth_provider.authenticate(context)
                    if not result.authenticated:
                        context.abort(grpc.StatusCode.UNAUTHENTICATED, result.error or "Authentication failed")
                    _auth_context.set(result)
                    return behavior(request, context)
                return wrapped

        # Create new handler with wrapped behaviors
        if handler.unary_unary:
            return grpc.unary_unary_rpc_method_handler(
                _auth_wrapper(handler.unary_unary, False, False),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        elif handler.unary_stream:
            return grpc.unary_stream_rpc_method_handler(
                _auth_wrapper(handler.unary_stream, False, True),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        elif handler.stream_unary:
            return grpc.stream_unary_rpc_method_handler(
                _auth_wrapper(handler.stream_unary, True, False),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        elif handler.stream_stream:
            return grpc.stream_stream_rpc_method_handler(
                _auth_wrapper(handler.stream_stream, True, True),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        return handler


class AsyncAuthInterceptor(aio.ServerInterceptor):
    """
    Async gRPC server interceptor for authentication.

    Uses an AuthProvider to authenticate requests and reject unauthorized ones.
    Compatible with grpc.aio servers.
    """

    def __init__(
        self,
        auth_provider: AuthProvider,
        allow_reflection: bool = True,
        allow_health: bool = True,
    ):
        """
        Initialize the interceptor.

        Args:
            auth_provider: The authentication provider to use
            allow_reflection: Whether to allow unauthenticated access to gRPC reflection
            allow_health: Whether to allow unauthenticated access to health check methods
        """
        self.auth_provider = auth_provider
        self.allow_reflection = allow_reflection
        self.allow_health = allow_health

    def _should_skip_auth(self, method: str) -> bool:
        """Check if authentication should be skipped for this method."""
        if self.allow_reflection and "grpc.reflection" in method:
            return True
        if self.allow_health and ("Health" in method or "health" in method):
            return True
        return False

    async def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Intercept and authenticate the request."""
        method = handler_call_details.method

        # Get the original handler (must await for async interceptor)
        handler = await continuation(handler_call_details)

        # Skip auth for allowed methods
        if self._should_skip_auth(method):
            return handler

        # Wrap the handler to add authentication
        return self._wrap_handler(handler)

    def _wrap_handler(
        self, handler: grpc.RpcMethodHandler
    ) -> grpc.RpcMethodHandler:
        """Wrap a handler to add authentication checking.

        Handles both sync and async handlers transparently by checking
        the return value type at runtime.
        """
        import inspect

        auth_provider = self.auth_provider

        def _auth_wrapper(
            behavior: Callable,
            request_streaming: bool,
            response_streaming: bool,
        ) -> Callable:
            # Streaming handlers can be sync or async generators depending on server type.
            # Check at runtime and handle both cases.

            if request_streaming and response_streaming:
                # Bidirectional streaming
                async def wrapped_stream_stream(request_iterator, context):
                    result = auth_provider.authenticate(context)
                    if not result.authenticated:
                        await context.abort(grpc.StatusCode.UNAUTHENTICATED, result.error or "Authentication failed")
                    _auth_context.set(result)
                    gen = behavior(request_iterator, context)
                    if inspect.isasyncgen(gen):
                        async for response in gen:
                            yield response
                    else:
                        for response in gen:
                            yield response
                return wrapped_stream_stream
            elif request_streaming:
                # Client streaming (returns single response)
                async def wrapped_stream_unary(request_iterator, context):
                    result = auth_provider.authenticate(context)
                    if not result.authenticated:
                        await context.abort(grpc.StatusCode.UNAUTHENTICATED, result.error or "Authentication failed")
                    _auth_context.set(result)
                    response = behavior(request_iterator, context)
                    if inspect.iscoroutine(response):
                        return await response
                    return response
                return wrapped_stream_unary
            elif response_streaming:
                # Server streaming
                async def wrapped_unary_stream(request, context):
                    result = auth_provider.authenticate(context)
                    if not result.authenticated:
                        await context.abort(grpc.StatusCode.UNAUTHENTICATED, result.error or "Authentication failed")
                    _auth_context.set(result)
                    gen = behavior(request, context)
                    if inspect.isasyncgen(gen):
                        async for response in gen:
                            yield response
                    else:
                        for response in gen:
                            yield response
                return wrapped_unary_stream
            else:
                # Unary - check return value at runtime
                async def wrapped_unary_unary(request, context):
                    result = auth_provider.authenticate(context)
                    if not result.authenticated:
                        await context.abort(grpc.StatusCode.UNAUTHENTICATED, result.error or "Authentication failed")
                    _auth_context.set(result)
                    response = behavior(request, context)
                    if inspect.iscoroutine(response):
                        return await response
                    return response
                return wrapped_unary_unary

        # Create new handler with wrapped behaviors
        if handler.unary_unary:
            return grpc.unary_unary_rpc_method_handler(
                _auth_wrapper(handler.unary_unary, False, False),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        elif handler.unary_stream:
            return grpc.unary_stream_rpc_method_handler(
                _auth_wrapper(handler.unary_stream, False, True),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        elif handler.stream_unary:
            return grpc.stream_unary_rpc_method_handler(
                _auth_wrapper(handler.stream_unary, True, False),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        elif handler.stream_stream:
            return grpc.stream_stream_rpc_method_handler(
                _auth_wrapper(handler.stream_stream, True, True),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        return handler


def create_mtls_interceptor(
    allowed_cns: list[str] | None = None,
    allowed_pattern: str | None = None,
    entity_source: str = "cn",
    role_mappings: dict[str, list[str]] | None = None,
    async_mode: bool = True,
) -> AuthInterceptor | AsyncAuthInterceptor:
    """
    Create an mTLS authentication interceptor.

    This interceptor verifies client certificates at the application level,
    checking that the CN or SAN matches expected values.

    Args:
        allowed_cns: List of allowed Common Names (exact match)
        allowed_pattern: Regex pattern for allowed entities
        entity_source: Where to extract entity from cert ("cn", "san_dns", "san_uri")
        role_mappings: Map of entity -> roles (e.g., {"policy-engine": ["updates-subscriber"]})
        async_mode: Whether to return an async interceptor (for grpc.aio)

    Returns:
        An interceptor that can be added to the gRPC server

    Example:
        # Allow specific CNs with roles
        interceptor = create_mtls_interceptor(
            allowed_cns=["policy-engine", "reference-monitor"],
            role_mappings={
                "policy-engine": ["updates-subscriber", "observability-reader"],
                "reference-monitor": ["credential-reader"],
            }
        )

        # Add to server
        server = grpc.aio.server(interceptors=[interceptor])
    """
    # Build allowed pattern from list of CNs or use provided pattern
    if allowed_cns and not allowed_pattern:
        # Escape special regex chars and join with |
        import re
        escaped = [re.escape(cn) for cn in allowed_cns]
        allowed_pattern = f"^({'|'.join(escaped)})$"

    config = MTLSAuthConfig(
        entity_source=entity_source,  # type: ignore
        allowed_entities_pattern=allowed_pattern,
        role_mappings=role_mappings or {},
    )

    provider = MTLSAuthProvider(config)

    if async_mode:
        return AsyncAuthInterceptor(provider)
    else:
        return AuthInterceptor(provider)


def create_spiffe_interceptor(
    trust_domain: str,
    allowed_workloads: list[str] | None = None,
    allowed_pattern: str | None = None,
    role_mappings: dict[str, list[str]] | None = None,
    async_mode: bool = True,
) -> AuthInterceptor | AsyncAuthInterceptor:
    """
    Create a SPIFFE-based authentication interceptor.

    This interceptor extracts SPIFFE IDs from X.509-SVID certificates
    (URI SANs in the form spiffe://trust-domain/workload-path) and
    validates them against allowed workloads.

    Args:
        trust_domain: The SPIFFE trust domain (e.g., "example.org")
        allowed_workloads: List of allowed workload paths (e.g., ["policy-engine", "ns/prod/sa/myapp"])
        allowed_pattern: Regex pattern for allowed workload paths
        role_mappings: Map of workload path -> roles
        async_mode: Whether to return an async interceptor (for grpc.aio)

    Returns:
        An interceptor that can be added to the gRPC server

    Example:
        # Allow specific SPIFFE workloads
        interceptor = create_spiffe_interceptor(
            trust_domain="observability.local",
            allowed_workloads=["policy-engine", "reference-monitor"],
            role_mappings={
                "policy-engine": ["updates-subscriber"],
                "reference-monitor": ["credential-reader"],
            }
        )

        # SPIFFE IDs will be:
        #   spiffe://observability.local/policy-engine
        #   spiffe://observability.local/reference-monitor
    """
    import re

    # Build pattern to match SPIFFE URIs and extract workload path
    # Pattern extracts the path after spiffe://trust-domain/
    trust_domain_escaped = re.escape(trust_domain)
    entity_pattern = f"^spiffe://{trust_domain_escaped}/(.+)$"

    # Build allowed pattern from workload list
    if allowed_workloads and not allowed_pattern:
        escaped = [re.escape(w) for w in allowed_workloads]
        allowed_pattern = f"^({'|'.join(escaped)})$"

    config = MTLSAuthConfig(
        entity_source="san_uri",
        entity_pattern=entity_pattern,
        allowed_entities_pattern=allowed_pattern,
        role_mappings=role_mappings or {},
    )

    provider = MTLSAuthProvider(config)

    if async_mode:
        return AsyncAuthInterceptor(provider)
    else:
        return AuthInterceptor(provider)


def create_service_interceptor(
    service_name: str,
    auth_config_path: str | None = None,
    use_spiffe: bool = False,
    async_mode: bool = True,
) -> AuthInterceptor | AsyncAuthInterceptor:
    """
    Create an interceptor for a service using centralized auth config.

    Loads role mappings from the global AuthorizationConfig. The interceptor
    authenticates requests and looks up roles - authorization is enforced
    at the RPC level using @require_role decorators.

    Args:
        service_name: Name of this service (for logging)
        auth_config_path: Path to auth config YAML/JSON (loads if not already loaded)
        use_spiffe: Use SPIFFE/SVID identity (URI SANs) instead of CN-based identity
        async_mode: Whether to return an async interceptor

    Returns:
        An interceptor configured for this service

    Example:
        # CN-based (default)
        interceptor = create_service_interceptor("credential-server", "config/auth_config.yaml")

        # SPIFFE-based (uses trust_domain from config)
        interceptor = create_service_interceptor(
            "credential-server",
            "config/auth_config.yaml",
            use_spiffe=True,
        )
    """
    from .config import DEFAULT_CONFIG, get_auth_config, load_auth_config

    # Load config if path provided and not already loaded
    config = get_auth_config()
    if config is None:
        if auth_config_path:
            config = load_auth_config(auth_config_path)
        else:
            logger.warning("No auth config loaded, using defaults")
            config = DEFAULT_CONFIG

    # Get role mappings for all entities
    role_mappings = config.get_role_mappings()

    if use_spiffe:
        logger.info(
            f"Service '{service_name}' using SPIFFE auth (trust_domain={config.trust_domain})"
        )
        return create_spiffe_interceptor(
            trust_domain=config.trust_domain,
            role_mappings=role_mappings,
            async_mode=async_mode,
        )
    else:
        logger.info(f"Service '{service_name}' using mTLS CN auth")
        return create_mtls_interceptor(
            role_mappings=role_mappings,
            async_mode=async_mode,
        )
