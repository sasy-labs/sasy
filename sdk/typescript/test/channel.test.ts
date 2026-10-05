import { expect, test } from "bun:test";
import { fileURLToPath } from "node:url";

test("partial configuration preserves the endpoint and TLS settings across auth updates", () => {
  // First-use environment defaults require a fresh module instance. Other tests
  // configure the shared SDK, and resetChannel intentionally preserves config.
  const result = Bun.spawnSync([
    process.execPath,
    fileURLToPath(new URL("./fixtures/channel-config.ts", import.meta.url)),
  ]);
  expect(new TextDecoder().decode(result.stderr)).toBe("");
  expect(result.exitCode).toBe(0);
});
