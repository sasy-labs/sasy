"""
Centralized authorization configuration.

Single source of truth for entity -> role mappings.
Auth providers (JWT, API key, mTLS) authenticate and extract the entity name,
then roles are looked up here.

RPC-level authorization is enforced via @require_role() decorators in service code.

Environment variables:
    AUTH_CONFIG_PATH: Path to auth_config.yaml file (auto-loaded on first access)
"""


import os
from pathlib import Path

from pydantic import BaseModel, Field

from sasy.capture import capture_logger

logger = capture_logger(__name__)


class EntityConfig(BaseModel):
    """Configuration for a single entity (service identity or client)."""
    description: str | None = None
    roles: list[str] = Field(default_factory=list)


class AuthorizationConfig(BaseModel):
    """
    Centralized authorization configuration.

    Example YAML:
        trust_domain: observability.local

        entities:
          policy-engine:
            description: Policy engine service
            roles:
              - updates-subscriber
              - graph-reader

          client:
            description: Standard client application
            roles:
              - user
              - graph-writer
              - openai-access
    """
    trust_domain: str = "observability.local"
    entities: dict[str, EntityConfig] = Field(default_factory=dict)

    def get_roles_for_entity(self, entity: str) -> list[str]:
        """Get roles assigned to an entity."""
        if entity in self.entities:
            return self.entities[entity].roles
        return []

    def get_role_mappings(self) -> dict[str, list[str]]:
        """Get entity -> roles mapping for interceptor config."""
        return {
            entity: config.roles
            for entity, config in self.entities.items()
        }

    def has_entity(self, entity: str) -> bool:
        """Check if an entity is defined in the config."""
        return entity in self.entities


# Global config instance
_config: AuthorizationConfig | None = None


def load_auth_config(path: str) -> AuthorizationConfig:
    """
    Load authorization config from YAML or JSON file.

    Sets the global config instance for use by interceptors and auth providers.
    """
    global _config

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Auth config not found: {path}")

    content = config_path.read_text()

    if config_path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(content)
        except ImportError:
            raise ImportError("PyYAML required for YAML config: pip install pyyaml")
    else:
        import json
        data = json.loads(content)

    _config = AuthorizationConfig.model_validate(data)
    logger.info(f"Loaded auth config from {path}: {len(_config.entities)} entities")

    return _config


def get_auth_config() -> AuthorizationConfig | None:
    """
    Get the global authorization config.

    Auto-loads from AUTH_CONFIG_PATH environment variable if not already loaded.
    """
    global _config
    if _config is None:
        env_path = os.environ.get("AUTH_CONFIG_PATH")
        if env_path and Path(env_path).exists():
            load_auth_config(env_path)
    return _config


def set_auth_config(config: AuthorizationConfig) -> None:
    """Set the global authorization config programmatically."""
    global _config
    _config = config


# Default config matching config/auth_config.yaml
# Note: updates-server operates through Neo4j directly and doesn't use this role system
DEFAULT_CONFIG = AuthorizationConfig(
    trust_domain="observability.local",
    entities={
        # Infrastructure services
        "policy-engine": EntityConfig(
            description="Policy engine service",
            roles=["observability-reader"],
        ),
        "reference-monitor": EntityConfig(
            description="HTTP proxy with policy enforcement",
            roles=["credential-reader", "policy-client"],
        ),
        # Application clients (mTLS)
        "client": EntityConfig(
            description="Standard client application",
            roles=["reference-monitor-user", "observability-writer", "openai-access"],
        ),
        # Admin
        "admin": EntityConfig(
            description="Administrative access",
            roles=["admin", "reference-monitor-user", "credential-reader", "credential-writer",
                   "observability-reader", "observability-writer", "observability-admin"],
        ),
    },
)
