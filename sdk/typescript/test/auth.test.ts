import { describe, expect, test } from "bun:test";

import {
  ApiKeyAuthHook,
  EntityAuthHook,
  NoAuthHook,
} from "../src/auth.js";
import { API_KEY, ENTITY, ROLES } from "../src/headers.js";

const keys = (pairs: [string, string][]) => pairs.map(([k]) => k);

describe("EntityAuthHook", () => {
  test("sends the roles it was given", () => {
    const md = new EntityAuthHook("copilot", ["admin"]).getMetadata();
    expect(md).toEqual([
      [ENTITY, "copilot"],
      [ROLES, "admin"],
    ]);
  });

  test("joins several roles with commas", () => {
    const md = new EntityAuthHook("copilot", ["admin", "auditor"]).getMetadata();
    expect(md).toContainEqual([ROLES, "admin,auditor"]);
  });

  // The server reads a present x-roles as the caller's complete role set, so
  // an empty header would strip an admin connection of the tenant-wide policy
  // scopes (SetPolicy with a default or force scope).
  test("omits the roles header when no roles were given", () => {
    const md = new EntityAuthHook("copilot").getMetadata();
    expect(keys(md)).not.toContain(ROLES);
    expect(md).toEqual([[ENTITY, "copilot"]]);
  });

  test("omits the roles header for an explicitly empty list", () => {
    const md = new EntityAuthHook("copilot", []).getMetadata();
    expect(keys(md)).not.toContain(ROLES);
  });
});

describe("the other hooks", () => {
  test("NoAuthHook sends nothing", () => {
    expect(new NoAuthHook().getMetadata()).toEqual([]);
  });

  test("ApiKeyAuthHook sends the key header", () => {
    expect(new ApiKeyAuthHook("k").getMetadata()).toEqual([[API_KEY, "k"]]);
  });
});

describe("opaque API keys", () => {
  test("passes the complete key unchanged despite a legacy suffix environment", () => {
    const previous = process.env.SASY_API_KEY_SUFFIX;
    process.env.SASY_API_KEY_SUFFIX = "legacy-shared-secret";
    try {
      const fullKey = "client-label-independent_secret-with-hyphens";
      expect(new ApiKeyAuthHook(fullKey).getMetadata()).toEqual([[API_KEY, fullKey]]);
      expect(new NoAuthHook().getMetadata()).toEqual([]);
    } finally {
      if (previous === undefined) delete process.env.SASY_API_KEY_SUFFIX;
      else process.env.SASY_API_KEY_SUFFIX = previous;
    }
  });

  test("the removed options cannot compose a key for a JavaScript caller", () => {
    // JavaScript permits extra arguments even after removal from the TS API.
    const hook = Reflect.construct(ApiKeyAuthHook, ["complete-key", { apiKeySuffix: "old" }]);
    expect(hook.getMetadata()).toEqual([[API_KEY, "complete-key"]]);
  });

  test("no suffix helpers remain on direct or public exports", async () => {
    const auth = await import("../src/auth.js");
    const sdk = await import("../src/index.js");
    for (const module of [auth, sdk]) {
      expect("entityApiKey" in module).toBe(false);
      expect("SasyApiKeySuffixNotConfigured" in module).toBe(false);
    }
    expect("forEntity" in ApiKeyAuthHook).toBe(false);
  });
});
