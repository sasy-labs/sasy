"""Canonical gRPC / HTTP metadata header keys used by the SDK.

The client-side mirror of the server's ``sasy_common::headers`` module.
Centralizing the keys keeps the SDK's producer sites (auth hooks and
providers) from drifting from each other — and from the wire keys the
server reads. These are wire values; they must stay byte-stable.
"""

#: API-key credential header (default for the api-key auth hook/provider).
API_KEY = "x-api-key"

#: Bearer / JWT credential header (standard HTTP header name).
AUTHORIZATION = "authorization"

#: User-supplied actor in the caller's domain (free-form; not auth).
ENTITY = "x-entity"

#: Delegated roles asserted by a service-proxy caller.
ROLES = "x-roles"
