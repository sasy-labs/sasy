"""Credential server client API.

Uses the shared ``sasy.config`` for endpoint, TLS, and auth settings.
"""


from sasy.config import get_async_stub, get_config, get_stub
from sasy.proto import credential_server_pb2_grpc as credential_server_grpc
from sasy.proto.credential_server_pb2 import (
    Credential,
    CredentialsRequest,
    SetCredentialsRequest,
)


def _metadata() -> list[tuple[str, str]]:
    return get_config().get_metadata()


def set_credentials(
    entity: str,
    service: str,
    credentials: dict[str, str],
) -> str:
    """Store credentials for an entity/service pair.

    Requires ``credential-writer`` role. The tenant is server-derived
    from the auth context — callers don't pick the tenant on
    the wire.

    Args:
        entity: Entity identifier (use ``"*"`` for global defaults)
        service: Service identifier (e.g. ``"openai"``)
        credentials: Key-value pairs to store
    """
    stub = get_stub(credential_server_grpc.CredentialServerStub)  # type: ignore
    response = stub.SetCredentials(
        SetCredentialsRequest(
            entity=entity,
            service=service,
            credentials=[Credential(key=k, value=v) for k, v in credentials.items()],
        ),
        metadata=_metadata(),
    )
    return response.response  # type: ignore


def get_credentials(
    entity: str,
    service: str,
) -> dict[str, str]:
    """Get credentials for an entity/service pair.

    Requires ``credential-reader`` role. Tenant is server-derived
    from the auth context.

    Args:
        entity: Entity identifier
        service: Service identifier
    """
    stub = get_stub(credential_server_grpc.CredentialServerStub)  # type: ignore
    response = stub.GetCredentials(
        CredentialsRequest(entity=entity, service=service),
        metadata=_metadata(),
    )
    return {c.key: c.value for c in response.credentials}


# ── Async API ─────────────────────────────────────────────



async def get_credentials_async(
    entity: str,
    service: str,
) -> dict[str, str]:
    """Get credentials asynchronously."""
    stub = get_async_stub(credential_server_grpc.CredentialServerStub)  # type: ignore
    response = await stub.GetCredentials(
        CredentialsRequest(entity=entity, service=service),
        metadata=_metadata(),
    )
    return {c.key: c.value for c in response.credentials}
