"""Shared test helpers — importable from any test module."""

from __future__ import annotations


class AbortError(Exception):
    """Raised by FakeServicerContext.abort to simulate gRPC abort."""

    def __init__(self, code, details: str):
        self.code = code
        self.details = details
        super().__init__(f"{code}: {details}")


class FakeServicerContext:
    """
    Minimal stand-in for grpc.ServicerContext.

    Carries invocation metadata and an optional auth_context
    (for mTLS). No network, no real gRPC — just enough to satisfy
    the auth provider interface.
    """

    def __init__(
        self,
        metadata: list[tuple[str, str]] | None = None,
        auth_ctx: dict | None = None,
    ):
        self._metadata = metadata or []
        self._auth_ctx = auth_ctx or {}
        self._code = None
        self._details = None

    def invocation_metadata(self) -> list[tuple[str, str]]:
        return self._metadata

    def auth_context(self) -> dict:
        return self._auth_ctx

    def abort(self, code, details: str = "") -> None:
        self._code = code
        self._details = details
        raise AbortError(code, details)

    async def abort_async(self, code, details: str = "") -> None:
        self.abort(code, details)
