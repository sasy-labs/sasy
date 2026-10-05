import type { AuthHook } from "./types.js";
import { API_KEY, AUTHORIZATION, ENTITY, ROLES } from "./headers.js";

/** No authentication. */
export class NoAuthHook implements AuthHook {
  getMetadata(): [string, string][] {
    return [];
  }
}

/** Static API key authentication (x-api-key header).
 *
 * Supply the complete, independently provisioned key. Its value is opaque:
 * the hook sends it unchanged and never derives credentials from a name or
 * environment setting. */
export class ApiKeyAuthHook implements AuthHook {
  constructor(private readonly apiKey: string) {}

  getMetadata(): [string, string][] {
    return [[API_KEY, this.apiKey]];
  }
}

/** Static Bearer token authentication. */
export class StaticTokenAuthHook implements AuthHook {
  constructor(private readonly token: string) {}

  getMetadata(): [string, string][] {
    return [[AUTHORIZATION, `Bearer ${this.token}`]];
  }
}

/** Entity/role metadata (for local binary with API key auth bypass).
 *
 * Tenant is derived server-side from the authenticated entity (via
 * auth_config), never sent on the wire — so this hook no longer
 * carries a tenant id.
 *
 * The roles header is sent only when roles were given. The server reads
 * a present `x-roles` as the caller's complete role set, so an empty one
 * would say "this caller has no roles" — costing an admin connection the
 * tenant-wide policy scopes — rather than "no roles were forwarded". */
export class EntityAuthHook implements AuthHook {
  constructor(
    private readonly entity: string,
    private readonly roles: string[] = [],
  ) {}

  getMetadata(): [string, string][] {
    const metadata: [string, string][] = [[ENTITY, this.entity]];
    if (this.roles.length > 0) {
      metadata.push([ROLES, this.roles.join(",")]);
    }
    return metadata;
  }
}
