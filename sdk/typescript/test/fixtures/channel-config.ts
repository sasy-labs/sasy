import assert from "node:assert/strict";
import { configure, getConfig, getClient, resetChannel } from "../../src/channel.js";
import { ApiKeyAuthHook } from "../../src/auth.js";

try {
  process.env.SASY_URL = "configured.example:443";
  process.env.TLS_CA_PATH = "/configured/ca.pem";
  process.env.TLS_CERT_PATH = "/configured/client.pem";
  process.env.TLS_KEY_PATH = "/configured/key.pem";
  process.env.SASY_INSECURE = "true";
  const authHook = new ApiKeyAuthHook("test-key");
  const first = configure({ authHook });
  assert.deepEqual(first, {
    url: "configured.example:443", caPath: "/configured/ca.pem",
    certPath: "/configured/client.pem", keyPath: "/configured/key.pem",
    insecure: true, authHook,
  });
  configure({ url: "explicit.example:443", insecure: false });
  const updated = configure({ authHook: new ApiKeyAuthHook("replacement") });
  assert.equal(updated.url, "explicit.example:443");
  assert.equal(updated.insecure, false);
  assert.equal(updated.caPath, first.caPath);
  assert.equal(updated.certPath, first.certPath);
  assert.equal(updated.keyPath, first.keyPath);
  configure({ caPath: undefined, certPath: undefined, keyPath: undefined, authHook: undefined });
  assert.equal(getConfig().caPath, undefined);
  assert.equal(getConfig().authHook, undefined);

  class Stub {
    constructor(readonly address: string, _credentials: unknown) {}
  }
  const before = getClient(Stub);
  configure({ authHook });
  const after = getClient(Stub);
  assert.notEqual(after, before);
  assert.equal(after.address, "explicit.example:443");
} finally {
  resetChannel();
}
