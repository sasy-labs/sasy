"""
Authentication providers for the reference monitor.

Supports multiple authentication mechanisms:
- JWT/OAuth2: Token validation against JWKS endpoints
- API Keys: Simple key-based authentication via credential server
- mTLS: Entity extraction from client certificates

Authentication is performed per-request and returns an authenticated entity
identifier that is used for policy evaluation and credential lookup.

Roles are looked up from the centralized auth_config (config/auth_config.yaml)
after authentication extracts the entity name.

Configuration is loaded from a JSON/YAML file using Pydantic models with
discriminated unions for type-safe provider configuration.
"""


import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import grpc
from pydantic import BaseModel, Field

from sasy._headers import API_KEY, AUTHORIZATION, ENTITY
from sasy.capture import capture_logger

logger = capture_logger(__name__)


def _lookup_roles(entity: str, fallback_mappings: dict[str, list[str]] | None = None) -> list[str]:
    """
    Look up roles for an entity from the central auth config.

    Args:
        entity: The authenticated entity name
        fallback_mappings: Optional fallback role mappings (for backwards compatibility)

    Returns:
        List of roles for the entity
    """
    # Import here to avoid circular imports
    from sasy.auth.config import get_auth_config

    auth_config = get_auth_config()
    if auth_config:
        roles = auth_config.get_roles_for_entity(entity)
        if roles:
            return roles

    # Fall back to inline role_mappings if central config doesn't have this entity
    if fallback_mappings and entity in fallback_mappings:
        return fallback_mappings[entity]

    return []


@dataclass
class AuthResult:
    """Result of an authentication attempt."""

    authenticated: bool
    entity: str | None = None
    roles: list[str] = None  # type: ignore
    error: str | None = None

    def __post_init__(self):
        if self.roles is None:
            self.roles = []


class AuthProvider(ABC):
    """Base class for authentication providers."""

    @abstractmethod
    def authenticate(self, context: grpc.ServicerContext) -> AuthResult:
        """
        Authenticate a request from the gRPC context.

        Args:
            context: gRPC servicer context containing metadata and peer info

        Returns:
            AuthResult indicating success/failure and extracted entity
        """
        pass


# =============================================================================
# Pydantic Configuration Models
# =============================================================================


class MTLSAuthConfig(BaseModel):
    """Configuration for mTLS authentication."""

    type: Literal["mtls"] = "mtls"

    # How to extract entity from certificate
    # Options: "cn" (Common Name), "san_dns" (DNS SAN), "san_uri" (URI SAN)
    entity_source: Literal["cn", "san_dns", "san_uri"] = "cn"

    # Regex pattern to extract entity from the certificate field
    # If None, the entire field value is used
    # Example: r"spiffe://[^/]+/(.+)" to extract path from SPIFFE URI
    entity_pattern: str | None = None

    # Prefix to strip from entity (e.g., "client-" from "client-myapp")
    strip_prefix: str | None = None

    # Only allow certs with entities matching this pattern
    allowed_entities_pattern: str | None = None

    # Map of certificate entity -> roles
    # The key is matched against the extracted entity (after pattern/prefix processing)
    # Example: {"fda-agent": ["user", "fda-access"], "admin-service": ["admin", "user"]}
    role_mappings: dict[str, list[str]] = Field(default_factory=dict)


class APIKeyEntityConfig(BaseModel):
    """Configuration for an API key's entity and roles."""

    entity: str
    roles: list[str] = Field(default_factory=list)


class APIKeyAuthConfig(BaseModel):
    """Configuration for API key authentication."""

    type: Literal["api_key"] = "api_key"

    # Metadata key containing the API key
    metadata_key: str = API_KEY

    # Map of API key -> entity (string) or entity config (object with entity and roles)
    # If empty, the API key itself is used as the entity identifier
    # Examples:
    #   "my-key": "my-entity"  # Simple string entity
    #   "my-key": {"entity": "my-entity", "roles": ["admin", "user"]}  # With roles
    static_keys: dict[str, str | APIKeyEntityConfig] = Field(default_factory=dict)


class JWTAuthConfig(BaseModel):
    """Configuration for JWT authentication."""

    type: Literal["jwt"] = "jwt"

    # JWKS endpoint URL for key validation (preferred)
    jwks_url: str | None = None

    # Static public key (PEM format) - alternative to JWKS
    public_key: str | None = None

    # Path to public key file - alternative to inline public_key
    public_key_path: str | None = None

    # Expected issuer (iss claim)
    issuer: str | None = None

    # Expected audience (aud claim)
    audience: str | None = None

    # Claim to use as entity identifier
    entity_claim: str = "sub"

    # Claim containing roles (can be nested with dot notation, e.g., "realm_access.roles")
    roles_claim: str = "roles"

    # Metadata key containing the JWT
    metadata_key: str = AUTHORIZATION

    # Token prefix (e.g., "Bearer ")
    token_prefix: str = "Bearer "

    # Cache JWKS for this many seconds
    jwks_cache_seconds: int = 300

    # If True, use roles from central auth_config.yaml instead of token roles.
    # Default is False: token roles are used exclusively (respects scope/consent).
    # WARNING: Only enable for service accounts or when the IdP doesn't provide roles.
    # Enabling this for user-facing flows may bypass consent restrictions.
    use_central_roles: bool = False


class PassthroughAuthConfig(BaseModel):
    """Configuration for passthrough (no authentication)."""

    type: Literal["none"] = "none"

    # Metadata key to extract entity from (optional)
    entity_metadata_key: str = ENTITY


class ChainAuthConfig(BaseModel):
    """Configuration for chained authentication (try multiple providers)."""

    type: Literal["chain"] = "chain"

    # List of provider configs to try in order
    providers: list[
        Annotated[
            MTLSAuthConfig | APIKeyAuthConfig | JWTAuthConfig | PassthroughAuthConfig,
            Field(discriminator="type"),
        ]
    ]


# Discriminated union of all auth config types
AuthConfigUnion = Annotated[
    MTLSAuthConfig | APIKeyAuthConfig | JWTAuthConfig | PassthroughAuthConfig | ChainAuthConfig,
    Field(discriminator="type"),
]


class AuthConfigFile(BaseModel):
    """Root configuration file model."""

    auth: AuthConfigUnion = Field(default_factory=lambda: PassthroughAuthConfig())


class MTLSAuthProvider(AuthProvider):
    """
    Authenticates clients via mTLS certificates.

    Extracts the entity identifier from the client certificate's
    Common Name (CN) or Subject Alternative Names (SANs).
    """

    def __init__(self, config: MTLSAuthConfig):
        self.config = config
        self._entity_regex = (
            re.compile(config.entity_pattern) if config.entity_pattern else None
        )
        self._allowed_regex = (
            re.compile(config.allowed_entities_pattern)
            if config.allowed_entities_pattern
            else None
        )

    def authenticate(self, context: grpc.ServicerContext) -> AuthResult:
        """Extract entity from client certificate."""
        # Get peer identity from gRPC context
        # In mTLS, this contains certificate info
        auth_context = context.auth_context()

        if not auth_context:
            return AuthResult(
                authenticated=False,
                error="No authentication context (mTLS not configured or no client cert)",
            )

        # Extract entity based on configured source
        entity = self._extract_entity(auth_context)

        if entity is None:
            return AuthResult(
                authenticated=False,
                error=f"Could not extract entity from certificate ({self.config.entity_source})",
            )

        # Apply pattern extraction if configured
        if self._entity_regex:
            match = self._entity_regex.search(entity)
            if match:
                entity = match.group(1) if match.lastindex else match.group(0)
            else:
                return AuthResult(
                    authenticated=False,
                    error=f"Entity '{entity}' does not match pattern",
                )

        # Strip prefix if configured
        if self.config.strip_prefix and entity.startswith(self.config.strip_prefix):
            entity = entity[len(self.config.strip_prefix) :]

        # Check against allowed entities pattern
        if self._allowed_regex and not self._allowed_regex.match(entity):
            return AuthResult(
                authenticated=False,
                error=f"Entity '{entity}' not in allowed list",
            )

        # Look up roles from central config (with fallback to inline role_mappings)
        roles = _lookup_roles(entity, self.config.role_mappings)
        logger.debug(f"mTLS auth: entity={entity}, roles={roles}")

        return AuthResult(authenticated=True, entity=entity, roles=roles)

    def _extract_entity(self, auth_context: Mapping[str, Any]) -> str | None:
        """Extract entity from auth context based on configuration."""
        if self.config.entity_source == "cn":
            # Common Name is in x509_common_name
            cn_list = auth_context.get("x509_common_name", [])
            if cn_list:
                return cn_list[0].decode() if isinstance(cn_list[0], bytes) else cn_list[0]

        elif self.config.entity_source == "san_dns":
            # DNS SANs
            dns_list = auth_context.get("x509_dns_name", [])
            if dns_list:
                return dns_list[0].decode() if isinstance(dns_list[0], bytes) else dns_list[0]

        elif self.config.entity_source == "san_uri":
            # URI SANs (e.g., SPIFFE IDs)
            uri_list = auth_context.get("x509_uri_name", [])
            if uri_list:
                return uri_list[0].decode() if isinstance(uri_list[0], bytes) else uri_list[0]

        return None


# =============================================================================
# Provider Implementations
# =============================================================================


class APIKeyAuthProvider(AuthProvider):
    """
    Authenticates clients via API keys in request metadata.

    API keys can be validated against:
    - A static key->entity map (for simple setups)
    - The credential server (for dynamic key management)
    """

    def __init__(self, config: APIKeyAuthConfig):
        self.config = config

    def authenticate(self, context: grpc.ServicerContext) -> AuthResult:
        """Validate API key from request metadata."""
        metadata = dict(context.invocation_metadata())
        api_key = metadata.get(self.config.metadata_key)
        logger.debug(
            f"API key auth: looking for '{self.config.metadata_key}' "
            f"in {list(metadata.keys())}, found={api_key is not None}"
        )

        if not api_key:
            return AuthResult(
                authenticated=False,
                error=f"Missing API key in metadata key '{self.config.metadata_key}'",
            )

        # Ensure api_key is a string (gRPC metadata values can be str or bytes)
        if isinstance(api_key, bytes):
            api_key = api_key.decode("utf-8")

        # Check static keys first
        if api_key in self.config.static_keys:
            key_config = self.config.static_keys[api_key]

            # Handle both string and object formats
            if isinstance(key_config, str):
                # Simple string format: "api-key": "entity-name"
                entity = key_config
                roles = _lookup_roles(entity)
                return AuthResult(
                    authenticated=True,
                    entity=entity,
                    roles=roles,
                )
            else:
                # Object format: "api-key": {"entity": "...", "roles": [...]}
                # Lookup from central config, fall back to inline roles
                entity = key_config.entity
                roles = _lookup_roles(entity, {entity: key_config.roles} if key_config.roles else None)
                return AuthResult(
                    authenticated=True,
                    entity=entity,
                    roles=roles,
                )

        # When a static map is configured it is the whole list of accepted
        # keys, so a key that is not in it is unauthenticated. With no static
        # keys the check below accepts any key and treats it as the entity
        # name itself, leaving it to the policy to decide what that entity
        # may do.
        if self.config.static_keys:
            return AuthResult(
                authenticated=False,
                error="Invalid API key",
            )

        # No static keys - API key is used as entity identifier
        # (Actual validation happens via entity/witness in policy)
        roles = _lookup_roles(api_key)
        return AuthResult(
            authenticated=True,
            entity=api_key,
            roles=roles,
        )


class JWTAuthProvider(AuthProvider):
    """
    Authenticates clients via JWT tokens.

    Validates tokens against a JWKS endpoint or static public key,
    verifies claims (issuer, audience), and extracts entity from
    a configurable claim (default: sub).
    """

    def __init__(self, config: JWTAuthConfig):
        self.config = config
        self._jwks_client: Any = None
        self._public_key: str | None = None
        self._jwt: Any = None  # Will be jwt module if available

        # Lazy import jwt to avoid hard dependency
        try:
            import jwt
            from jwt import PyJWKClient

            self._jwt = jwt

            if config.jwks_url:
                self._jwks_client = PyJWKClient(
                    config.jwks_url,
                    cache_jwk_set=True,
                    lifespan=config.jwks_cache_seconds,
                )
            elif config.public_key:
                self._public_key = config.public_key
            elif config.public_key_path:
                # Load public key from file
                key_path = Path(config.public_key_path)
                if key_path.exists():
                    self._public_key = key_path.read_text()
        except ImportError:
            self._jwt = None

    def _extract_claim(self, payload: dict, claim_path: str) -> Any | None:
        """
        Extract a claim from JWT payload, supporting dot notation for nested claims.

        Args:
            payload: The decoded JWT payload
            claim_path: Claim name, supports dot notation (e.g., "realm_access.roles")

        Returns:
            The claim value, or None if not found
        """
        parts = claim_path.split(".")
        value: Any = payload
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
            if value is None:
                return None
        return value

    def authenticate(self, context: grpc.ServicerContext) -> AuthResult:
        """Validate JWT from request metadata."""
        if self._jwt is None:
            return AuthResult(
                authenticated=False,
                error="PyJWT not installed (pip install pyjwt[crypto])",
            )

        metadata = dict(context.invocation_metadata())
        auth_header = metadata.get(self.config.metadata_key)

        if not auth_header:
            return AuthResult(
                authenticated=False,
                error=f"Missing token in metadata key '{self.config.metadata_key}'",
            )

        # Ensure auth_header is a string (gRPC metadata values can be str or bytes)
        if isinstance(auth_header, bytes):
            auth_header = auth_header.decode("utf-8")

        # Extract token from header
        if self.config.token_prefix:
            if not auth_header.startswith(self.config.token_prefix):
                return AuthResult(
                    authenticated=False,
                    error=f"Token must start with '{self.config.token_prefix}'",
                )
            token = auth_header[len(self.config.token_prefix) :]
        else:
            token = auth_header

        try:
            # Get signing key
            if self._jwks_client:
                signing_key = self._jwks_client.get_signing_key_from_jwt(token)
                key = signing_key.key
            elif self._public_key:
                key = self._public_key
            else:
                return AuthResult(
                    authenticated=False,
                    error="No JWKS URL or public key configured",
                )

            # Build validation options
            options: dict[str, Any] = {}
            kwargs: dict[str, Any] = {"algorithms": ["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"]}

            if self.config.issuer:
                kwargs["issuer"] = self.config.issuer
            else:
                options["verify_iss"] = False

            if self.config.audience:
                kwargs["audience"] = self.config.audience
            else:
                options["verify_aud"] = False

            if options:
                kwargs["options"] = options

            # Decode and validate
            payload = self._jwt.decode(token, key, **kwargs)

            # Extract entity from configured claim
            entity = payload.get(self.config.entity_claim)
            if not entity:
                return AuthResult(
                    authenticated=False,
                    error=f"Token missing '{self.config.entity_claim}' claim",
                )

            # Extract roles from token claim (supports dot notation)
            token_roles = self._extract_claim(payload, self.config.roles_claim)
            if not isinstance(token_roles, list):
                token_roles = [token_roles] if token_roles else []
            # Ensure all roles are strings
            token_roles = [str(r) for r in token_roles]

            entity_str = str(entity)

            # Determine roles based on use_central_roles flag
            if self.config.use_central_roles:
                # Explicit override: use central config, fall back to token roles
                roles = _lookup_roles(entity_str, {entity_str: token_roles} if token_roles else None)
            else:
                # Default: use token roles only (respects scope/consent)
                # No fallback - empty token roles means no roles granted
                roles = token_roles

            return AuthResult(authenticated=True, entity=entity_str, roles=roles)

        except self._jwt.ExpiredSignatureError:
            return AuthResult(authenticated=False, error="Token expired")
        except self._jwt.InvalidAudienceError:
            return AuthResult(authenticated=False, error="Invalid audience")
        except self._jwt.InvalidIssuerError:
            return AuthResult(authenticated=False, error="Invalid issuer")
        except self._jwt.InvalidTokenError as e:
            return AuthResult(authenticated=False, error=f"Invalid token: {e}")
        except Exception as e:
            # Catch JWKS fetch errors, connection errors, etc.
            # so the auth chain can fall through to the next provider
            logger.debug(f"JWT auth error: {type(e).__name__}: {e}")
            return AuthResult(authenticated=False, error=f"JWT auth error: {type(e).__name__}")


# -----------------------------------------------------------------------------
# Passthrough (No Authentication)
# -----------------------------------------------------------------------------


class PassthroughAuthProvider(AuthProvider):
    """
    No authentication - allows all requests.

    Entity is extracted from request metadata if present,
    otherwise defaults to "anonymous".
    """

    def __init__(self, config: PassthroughAuthConfig):
        self.config = config

    def authenticate(self, context: grpc.ServicerContext) -> AuthResult:
        """Always succeeds, extracts entity from metadata if present.

        Returns entity=None if no entity metadata is provided, allowing the server
        to distinguish between unauthenticated passthrough and an authenticated
        entity that happens to be named 'anonymous'.
        """
        metadata = dict(context.invocation_metadata())
        entity_raw = metadata.get(self.config.entity_metadata_key)  # None if not present
        # Decode bytes to string if needed (gRPC metadata values can be str or bytes)
        entity: str | None = (
            entity_raw.decode("utf-8") if isinstance(entity_raw, bytes)
            else entity_raw
        )
        roles = _lookup_roles(entity) if entity else []
        return AuthResult(authenticated=True, entity=entity, roles=roles)


# -----------------------------------------------------------------------------
# Auth Chain
# -----------------------------------------------------------------------------


class AuthChain(AuthProvider):
    """
    Tries multiple authentication providers in order.

    Returns success on the first provider that authenticates successfully.
    If all providers fail, returns the error from the last provider.
    """

    def __init__(self, providers: list[AuthProvider]):
        if not providers:
            raise ValueError("AuthChain requires at least one provider")
        self.providers = providers

    def authenticate(self, context: grpc.ServicerContext) -> AuthResult:
        """Try each provider until one succeeds."""
        last_error = "No authentication providers configured"

        for provider in self.providers:
            provider_name = type(provider).__name__
            result = provider.authenticate(context)
            if result.authenticated:
                logger.debug(f"Auth chain: {provider_name} succeeded, entity={result.entity}")
                return result
            logger.debug(f"Auth chain: {provider_name} failed: {result.error}")
            last_error = result.error or "Authentication failed"

        logger.debug(f"Auth chain: all providers failed, last error: {last_error}")
        return AuthResult(authenticated=False, error=last_error)


# =============================================================================
# Factory Functions
# =============================================================================


def create_auth_provider(config: AuthConfigUnion) -> AuthProvider:
    """Create an auth provider from a Pydantic configuration."""
    if isinstance(config, PassthroughAuthConfig):
        return PassthroughAuthProvider(config)

    elif isinstance(config, MTLSAuthConfig):
        return MTLSAuthProvider(config)

    elif isinstance(config, APIKeyAuthConfig):
        return APIKeyAuthProvider(config)

    elif isinstance(config, JWTAuthConfig):
        return JWTAuthProvider(config)

    elif isinstance(config, ChainAuthConfig):
        providers = [create_auth_provider(p) for p in config.providers]
        return AuthChain(providers)

    else:
        raise ValueError(f"Unknown auth config type: {type(config)}")


def load_auth_config(path: str) -> AuthConfigFile:
    """Load authentication configuration from a JSON file."""
    import json

    config_path = Path(path)
    if not config_path.exists():
        # Return default (passthrough) config if file doesn't exist
        return AuthConfigFile()

    with open(config_path) as f:
        data = json.load(f)

    return AuthConfigFile.model_validate(data)


def load_auth_provider(path: str) -> AuthProvider:
    """Load auth config from file and create the provider."""
    config = load_auth_config(path)
    return create_auth_provider(config.auth)
