import { expect, test } from "bun:test";
import { createServer } from "node:net";
import { configure, resetChannel } from "../src/channel.js";
import { health } from "../src/policy.js";

test("health deadline bounds an accepted connection that never answers gRPC", async () => {
  const sockets = new Set<import("node:net").Socket>();
  const server = createServer(socket => { sockets.add(socket); socket.on("close", () => sockets.delete(socket)); });
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
  const port = (server.address() as { port: number }).port;
  configure({ url: `127.0.0.1:${port}`, insecure: true, caPath: undefined, certPath: undefined, keyPath: undefined, authHook: undefined });
  try {
    const start = Date.now();
    await expect(health(150)).rejects.toMatchObject({ code: 4 });
    expect(Date.now() - start).toBeLessThan(2000);
    for (const timeout of [0, -1, NaN, Infinity]) await expect(health(timeout)).rejects.toThrow("positive finite");
  } finally {
    resetChannel();
    for (const socket of sockets) socket.destroy();
    await new Promise<void>(resolve => server.close(() => resolve()));
  }
});
