"""Tests for sasy.auth.providers — all five auth providers."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import jwt
import pytest
from sasy.auth.providers import (
    APIKeyAuthConfig,
    APIKeyAuthProvider,
    APIKeyEntityConfig,
    AuthChain,
    ChainAuthConfig,
    JWTAuthConfig,
    JWTAuthProvider,
    MTLSAuthConfig,
    MTLSAuthProvider,
    PassthroughAuthConfig,
    PassthroughAuthProvider,
    create_auth_provider,
    load_auth_config,
    load_auth_provider,
)

from helpers import FakeServicerContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Generate a valid RSA key pair at import time for JWT tests
def _generate_rsa_keys() -> tuple[str, str]:
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization

    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048
    )
    private_pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ).decode()
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_pem, public_pem


_RSA_PRIVATE_KEY, _RSA_PUBLIC_KEY = _generate_rsa_keys()


def _make_jwt(
    payload: dict,
    key: str = _RSA_PRIVATE_KEY,
    algorithm: str = "RS256",
) -> str:
    return jwt.encode(payload, key, algorithm=algorithm)


# Patch _lookup_roles so tests don't depend on auth_config.yaml
@pytest.fixture(autouse=True)
def _patch_lookup_roles():
    """Return empty roles by default; tests override as needed."""
    with patch(
        "sasy.auth.providers._lookup_roles",
        side_effect=lambda entity, fallback=None: (
            fallback.get(entity, []) if fallback else []
        ),
    ):
        yield


# ===================================================================
# APIKeyAuthProvider
# ===================================================================


class TestAPIKeyAuthProvider:
    """Tests for API-key based authentication."""

    @pytest.mark.parametrize(
        "key,entity,roles",
        [
            ("key-a", "alice", []),
            ("key-b", "bob", []),
        ],
        ids=["alice-key", "bob-key"],
    )
    def test_valid_key_string_entity(
        self, key: str, entity: str, roles: list[str]
    ) -> None:
        config = APIKeyAuthConfig(
            static_keys={"key-a": "alice", "key-b": "bob"},
        )
        provider = APIKeyAuthProvider(config)
        ctx = FakeServicerContext(metadata=[("x-api-key", key)])
        result = provider.authenticate(ctx)

        assert result.authenticated is True
        assert result.entity == entity

    @pytest.mark.parametrize(
        "key,entity,expected_roles",
        [
            (
                "key-admin",
                "admin-user",
                ["admin", "writer"],
            ),
            (
                "key-reader",
                "reader-user",
                ["reader"],
            ),
        ],
        ids=["admin-with-roles", "reader-with-roles"],
    )
    def test_valid_key_with_roles(
        self,
        key: str,
        entity: str,
        expected_roles: list[str],
        _patch_lookup_roles,
    ) -> None:
        """Keys mapped to APIKeyEntityConfig with explicit roles."""
        with patch(
            "sasy.auth.providers._lookup_roles",
            side_effect=lambda e, fallback=None: (
                fallback.get(e, []) if fallback else []
            ),
        ):
            config = APIKeyAuthConfig(
                static_keys={
                    "key-admin": APIKeyEntityConfig(
                        entity="admin-user",
                        roles=["admin", "writer"],
                    ),
                    "key-reader": APIKeyEntityConfig(
                        entity="reader-user",
                        roles=["reader"],
                    ),
                },
            )
            provider = APIKeyAuthProvider(config)
            ctx = FakeServicerContext(
                metadata=[("x-api-key", key)]
            )
            result = provider.authenticate(ctx)

            assert result.authenticated is True
            assert result.entity == entity
            assert result.roles == expected_roles

    def test_invalid_key_rejected(self) -> None:
        config = APIKeyAuthConfig(
            static_keys={"good-key": "alice"},
        )
        provider = APIKeyAuthProvider(config)
        ctx = FakeServicerContext(
            metadata=[("x-api-key", "wrong-key")]
        )
        result = provider.authenticate(ctx)

        assert result.authenticated is False
        assert "Invalid API key" in (result.error or "")

    def test_missing_header_rejected(self) -> None:
        config = APIKeyAuthConfig(
            static_keys={"good-key": "alice"},
        )
        provider = APIKeyAuthProvider(config)
        ctx = FakeServicerContext(metadata=[])
        result = provider.authenticate(ctx)

        assert result.authenticated is False
        assert "Missing API key" in (result.error or "")

    def test_no_static_keys_uses_key_as_entity(self) -> None:
        """When no static_keys configured, the key itself is the entity."""
        config = APIKeyAuthConfig(static_keys={})
        provider = APIKeyAuthProvider(config)
        ctx = FakeServicerContext(
            metadata=[("x-api-key", "my-entity-key")]
        )
        result = provider.authenticate(ctx)

        assert result.authenticated is True
        assert result.entity == "my-entity-key"


# ===================================================================
# JWTAuthProvider
# ===================================================================


class TestJWTAuthProvider:
    """Tests for JWT-based authentication."""

    def test_valid_token(self) -> None:
        payload = {
            "sub": "user-1",
            "roles": ["admin", "reader"],
            "exp": int(time.time()) + 3600,
        }
        token = _make_jwt(payload)
        config = JWTAuthConfig(public_key=_RSA_PUBLIC_KEY)
        provider = JWTAuthProvider(config)
        ctx = FakeServicerContext(
            metadata=[("authorization", f"Bearer {token}")]
        )
        result = provider.authenticate(ctx)

        assert result.authenticated is True
        assert result.entity == "user-1"
        assert "admin" in result.roles
        assert "reader" in result.roles

    def test_expired_token_rejected(self) -> None:
        payload = {
            "sub": "user-1",
            "roles": [],
            "exp": int(time.time()) - 60,
        }
        token = _make_jwt(payload)
        config = JWTAuthConfig(public_key=_RSA_PUBLIC_KEY)
        provider = JWTAuthProvider(config)
        ctx = FakeServicerContext(
            metadata=[("authorization", f"Bearer {token}")]
        )
        result = provider.authenticate(ctx)

        assert result.authenticated is False
        assert "expired" in (result.error or "").lower()

    def test_wrong_key_rejected(self) -> None:
        """Token signed with a different key should fail."""
        # Use a trivially different key — just use HS256 with a secret
        # to guarantee it won't validate against our RSA public key.
        bad_token = jwt.encode(
            {"sub": "x", "exp": int(time.time()) + 3600},
            "wrong-secret",
            algorithm="HS256",
        )
        config = JWTAuthConfig(public_key=_RSA_PUBLIC_KEY)
        provider = JWTAuthProvider(config)
        ctx = FakeServicerContext(
            metadata=[("authorization", f"Bearer {bad_token}")]
        )
        result = provider.authenticate(ctx)

        assert result.authenticated is False

    def test_missing_token_rejected(self) -> None:
        config = JWTAuthConfig(public_key=_RSA_PUBLIC_KEY)
        provider = JWTAuthProvider(config)
        ctx = FakeServicerContext(metadata=[])
        result = provider.authenticate(ctx)

        assert result.authenticated is False
        assert "Missing" in (result.error or "")

    def test_nested_roles_claim(self) -> None:
        """Dot-notation roles_claim like 'realm_access.roles'."""
        payload = {
            "sub": "user-2",
            "realm_access": {"roles": ["viewer"]},
            "exp": int(time.time()) + 3600,
        }
        token = _make_jwt(payload)
        config = JWTAuthConfig(
            public_key=_RSA_PUBLIC_KEY,
            roles_claim="realm_access.roles",
        )
        provider = JWTAuthProvider(config)
        ctx = FakeServicerContext(
            metadata=[("authorization", f"Bearer {token}")]
        )
        result = provider.authenticate(ctx)

        assert result.authenticated is True
        assert result.roles == ["viewer"]

    def test_missing_sub_claim(self) -> None:
        """Token without the entity claim should fail."""
        payload = {
            "roles": ["admin"],
            "exp": int(time.time()) + 3600,
        }
        token = _make_jwt(payload)
        config = JWTAuthConfig(public_key=_RSA_PUBLIC_KEY)
        provider = JWTAuthProvider(config)
        ctx = FakeServicerContext(
            metadata=[("authorization", f"Bearer {token}")]
        )
        result = provider.authenticate(ctx)

        assert result.authenticated is False
        assert "sub" in (result.error or "")


# ===================================================================
# PassthroughAuthProvider
# ===================================================================


class TestPassthroughAuthProvider:
    """Tests for passthrough (no-auth) provider."""

    def test_always_succeeds(self) -> None:
        config = PassthroughAuthConfig()
        provider = PassthroughAuthProvider(config)
        ctx = FakeServicerContext(metadata=[])
        result = provider.authenticate(ctx)

        assert result.authenticated is True

    def test_extracts_entity_from_metadata(self) -> None:
        config = PassthroughAuthConfig()
        provider = PassthroughAuthProvider(config)
        ctx = FakeServicerContext(
            metadata=[("x-entity", "my-agent")]
        )
        result = provider.authenticate(ctx)

        assert result.authenticated is True
        assert result.entity == "my-agent"

    def test_no_entity_returns_none(self) -> None:
        config = PassthroughAuthConfig()
        provider = PassthroughAuthProvider(config)
        ctx = FakeServicerContext(metadata=[])
        result = provider.authenticate(ctx)

        assert result.authenticated is True
        assert result.entity is None


# ===================================================================
# AuthChain
# ===================================================================


class TestAuthChain:
    """Tests for chained authentication."""

    def test_first_fails_second_succeeds(self) -> None:
        """Chain falls through to next provider on failure."""
        api_config = APIKeyAuthConfig(
            static_keys={"secret": "alice"},
        )
        passthrough_config = PassthroughAuthConfig()

        chain = AuthChain(
            [
                APIKeyAuthProvider(api_config),
                PassthroughAuthProvider(passthrough_config),
            ]
        )
        # No API key → first fails; passthrough succeeds
        ctx = FakeServicerContext(metadata=[])
        result = chain.authenticate(ctx)

        assert result.authenticated is True

    def test_all_fail(self) -> None:
        """When every provider fails, chain returns failure."""
        api1 = APIKeyAuthConfig(
            static_keys={"k1": "a"},
        )
        api2 = APIKeyAuthConfig(
            static_keys={"k2": "b"},
        )
        chain = AuthChain(
            [
                APIKeyAuthProvider(api1),
                APIKeyAuthProvider(api2),
            ]
        )
        ctx = FakeServicerContext(
            metadata=[("x-api-key", "wrong")]
        )
        result = chain.authenticate(ctx)

        assert result.authenticated is False

    def test_empty_chain_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            AuthChain([])


# ===================================================================
# MTLSAuthProvider
# ===================================================================


class TestMTLSAuthProvider:
    """Tests for mTLS certificate-based authentication."""

    def test_valid_cn(self) -> None:
        config = MTLSAuthConfig()
        provider = MTLSAuthProvider(config)
        ctx = FakeServicerContext(
            auth_ctx={"x509_common_name": [b"my-service"]}
        )
        result = provider.authenticate(ctx)

        assert result.authenticated is True
        assert result.entity == "my-service"

    def test_no_cert_rejected(self) -> None:
        config = MTLSAuthConfig()
        provider = MTLSAuthProvider(config)
        ctx = FakeServicerContext(auth_ctx={})
        result = provider.authenticate(ctx)

        assert result.authenticated is False

    def test_no_auth_context_rejected(self) -> None:
        config = MTLSAuthConfig()
        provider = MTLSAuthProvider(config)
        ctx = FakeServicerContext(auth_ctx=None)
        # auth_context() returns {} by default when None
        result = provider.authenticate(ctx)

        assert result.authenticated is False

    @pytest.mark.parametrize(
        "source,ctx_key,value",
        [
            ("cn", "x509_common_name", [b"svc-a"]),
            ("san_dns", "x509_dns_name", [b"svc-b.local"]),
            (
                "san_uri",
                "x509_uri_name",
                [b"spiffe://domain/svc-c"],
            ),
        ],
        ids=["cn", "san_dns", "san_uri"],
    )
    def test_entity_sources(
        self,
        source: str,
        ctx_key: str,
        value: list[bytes],
    ) -> None:
        config = MTLSAuthConfig(entity_source=source)
        provider = MTLSAuthProvider(config)
        ctx = FakeServicerContext(auth_ctx={ctx_key: value})
        result = provider.authenticate(ctx)

        assert result.authenticated is True
        assert result.entity is not None

    def test_entity_pattern_extraction(self) -> None:
        """Extract entity from SPIFFE URI using regex."""
        config = MTLSAuthConfig(
            entity_source="san_uri",
            entity_pattern=r"spiffe://[^/]+/(.+)",
        )
        provider = MTLSAuthProvider(config)
        ctx = FakeServicerContext(
            auth_ctx={
                "x509_uri_name": [
                    b"spiffe://example.org/my-workload"
                ]
            }
        )
        result = provider.authenticate(ctx)

        assert result.authenticated is True
        assert result.entity == "my-workload"

    def test_allowed_entities_pattern(self) -> None:
        config = MTLSAuthConfig(
            allowed_entities_pattern=r"^allowed-.*$",
        )
        provider = MTLSAuthProvider(config)

        # Allowed
        ctx_ok = FakeServicerContext(
            auth_ctx={"x509_common_name": [b"allowed-svc"]}
        )
        assert provider.authenticate(ctx_ok).authenticated is True

        # Blocked
        ctx_bad = FakeServicerContext(
            auth_ctx={"x509_common_name": [b"blocked-svc"]}
        )
        assert (
            provider.authenticate(ctx_bad).authenticated is False
        )

    def test_strip_prefix(self) -> None:
        config = MTLSAuthConfig(strip_prefix="client-")
        provider = MTLSAuthProvider(config)
        ctx = FakeServicerContext(
            auth_ctx={"x509_common_name": [b"client-myapp"]}
        )
        result = provider.authenticate(ctx)

        assert result.authenticated is True
        assert result.entity == "myapp"


# ===================================================================
# Factory / load functions
# ===================================================================


class TestCreateAuthProvider:
    """Tests for create_auth_provider and load_auth_provider."""

    @pytest.mark.parametrize(
        "config",
        [
            PassthroughAuthConfig(),
            APIKeyAuthConfig(static_keys={"k": "e"}),
            MTLSAuthConfig(),
            JWTAuthConfig(public_key=_RSA_PUBLIC_KEY),
        ],
        ids=["passthrough", "api_key", "mtls", "jwt"],
    )
    def test_create_each_type(self, config) -> None:
        provider = create_auth_provider(config)
        assert provider is not None

    def test_create_chain(self) -> None:
        config = ChainAuthConfig(
            providers=[
                APIKeyAuthConfig(static_keys={"k": "e"}),
                PassthroughAuthConfig(),
            ]
        )
        provider = create_auth_provider(config)
        assert isinstance(provider, AuthChain)

    def test_load_auth_config_from_file(self, tmp_path: Path) -> None:
        cfg = {
            "auth": {
                "type": "api_key",
                "static_keys": {"my-key": "my-entity"},
            }
        }
        f = tmp_path / "auth.json"
        f.write_text(json.dumps(cfg))
        loaded = load_auth_config(str(f))

        assert isinstance(loaded.auth, APIKeyAuthConfig)
        assert "my-key" in loaded.auth.static_keys

    def test_load_auth_config_missing_file(self) -> None:
        """Missing file returns default passthrough config."""
        loaded = load_auth_config("/nonexistent/path.json")
        assert isinstance(loaded.auth, PassthroughAuthConfig)

    def test_load_auth_provider_from_file(
        self, tmp_path: Path
    ) -> None:
        cfg = {"auth": {"type": "none"}}
        f = tmp_path / "auth.json"
        f.write_text(json.dumps(cfg))
        provider = load_auth_provider(str(f))
        assert isinstance(provider, PassthroughAuthProvider)
