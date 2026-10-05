/**
 * Canonical gRPC metadata header keys used by the SDK — the client-side
 * mirror of the server's `sasy_common::headers` module. Centralizing the
 * keys keeps the SDK's auth hooks from drifting from each other and from
 * the wire keys the server reads. These are wire values; keep them
 * byte-stable.
 */

/** API-key credential header. */
export const API_KEY = "x-api-key";

/** Bearer / JWT credential header (standard HTTP header name). */
export const AUTHORIZATION = "authorization";

/** User-supplied actor in the caller's domain (free-form; not auth). */
export const ENTITY = "x-entity";

/** Delegated roles asserted by a service-proxy caller. */
export const ROLES = "x-roles";
