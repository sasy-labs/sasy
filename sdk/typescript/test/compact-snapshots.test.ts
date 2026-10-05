import { expect, spyOn, test } from "bun:test";
import * as grpc from "@grpc/grpc-js";
import * as capture from "../src/capture.js";
import { configure, resetChannel } from "../src/channel.js";
import { ObservabilityService } from "../src/generated/observability.js";
import { resolveEvents, resetObservability } from "../src/observability.js";
import { session } from "../src/session.js";
import { captureEvent, eventContentHash } from "../src/snapshot-content.js";
import type { EventSnapshot } from "../src/types.js";

async function withServer(handler: (method: string, request: any, callback: any) => void, run: () => Promise<void>) {
  const server = new grpc.Server();
  server.addService(ObservabilityService, Object.fromEntries(["resolveEvents", "resolveSnapshots"].map(method => [
    method, (call: any, callback: any) => handler(method, call.request, callback),
  ])));
  const port = await new Promise<number>((resolve, reject) => server.bindAsync("127.0.0.1:0", grpc.ServerCredentials.createInsecure(), (error, port) => error ? reject(error) : resolve(port)));
  configure({ url: `127.0.0.1:${port}`, insecure: true, certPath: undefined, keyPath: undefined, caPath: undefined });
  resetObservability();
  try { await run(); }
  finally { resetObservability(); resetChannel(); server.forceShutdown(); }
}

const reference = (): EventSnapshot => ({
  event: { id: "input", text: "https://test.invalid/?key=synthetic", role: 1, entity: "actor", derivedFrom: { name: "approve", arguments: "{}" } },
  baseId: "sasy:mv1:known", reuseDependencies: true,
});

test("mixed batch uses compact RPC, sends full outputs, and retains session API", async () => {
  const requests: any[] = [];
  await withServer((method, request, callback) => {
    requests.push({ method, request });
    callback(null, { ids: request.snapshots.map((item: any) => item.reuseDependencies ? item.baseId : "sasy:mv1:output") });
  }, async () => {
    const input = reference();
    const output = { event: { id: "output", text: "produced", tools: [] }, dependencies: [{ source: input.baseId!, destination: "output" }] };
    const before = JSON.stringify([input, output]);
    expect(await session("conversation").resolveEvents([input, output])).toEqual([input.baseId!, "sasy:mv1:output"]);
    expect(requests[0].method).toBe("resolveSnapshots");
    expect(requests[0].request.sessionId).toBe("conversation");
    const [compact, full] = requests[0].request.snapshots;
    expect(compact.event).toEqual({ id: "input", tools: [], text: undefined, agent: undefined, role: undefined, principal: undefined, entity: undefined, derivedFrom: undefined });
    expect(compact.contentHash).toEqual(eventContentHash(captureEvent(input.event)));
    expect(compact.dependencies).toEqual([]);
    expect(full.contentHash).toBeUndefined();
    expect(full.event.text).toBe("produced");
    expect(full.dependencies[0].source).toBe(input.baseId);
    expect(JSON.stringify([input, output])).toBe(before);
    await session("conversation").resolveEvents([input], { compact: false });
    expect(requests[1].method).toBe("resolveEvents");
    expect(requests[1].request.snapshots[0].contentHash).toBeUndefined();
    expect(requests[1].request.snapshots[0].event.text).toBe(input.event.text);
  });
});

test("invalid reuse forms and empty origins fail before any RPC", async () => {
  let count = 0;
  await withServer((_method, _request, callback) => { count++; callback(null, { ids: [] }); }, async () => {
    const invalid = [
      { ...reference(), baseId: undefined },
      { ...reference(), dependencies: [{ source: "other", destination: "input" }] },
      { ...reference(), event: { id: "" } },
    ];
    for (const snapshot of invalid) await expect(resolveEvents([snapshot])).rejects.toThrow();
    expect(await resolveEvents([])).toEqual([]);
    expect(count).toBe(0);
  });
});

test("changed current payload is hashed again, including provenance removal", async () => {
  const hashes: Buffer[] = [];
  await withServer((_method, request, callback) => {
    hashes.push(Buffer.from(request.snapshots[0].contentHash));
    callback(null, { ids: [request.snapshots[0].baseId] });
  }, async () => {
    const input = reference();
    for (const edit of [() => {}, () => { input.event.text += "\nchanged"; }, () => { delete input.event.derivedFrom; }, () => { input.event.entity = "other"; }, () => { input.event.tools = [{ name: "new", arguments: "{}" }]; }]) {
      edit();
      await resolveEvents([input]);
    }
    expect(new Set(hashes.map(hash => hash.toString("hex"))).size).toBe(5);
  });
});

for (const compact of [true, false]) test(`reference acknowledgement cannot substitute another ID (compact=${compact})`, async () => {
  await withServer((_method, _request, callback) => callback(null, { ids: ["sasy:mv1:wrong"] }), async () => {
    await expect(resolveEvents([reference()], "conversation", { compact })).rejects.toThrow("changed a referenced identity");
  });
});

test("invalid or failed replies never populate the capture cache, and no fallback RPC occurs", async () => {
  const methods: string[] = [];
  let mode = "incomplete";
  await withServer((method, request, callback) => {
    methods.push(method);
    if (mode === "error") callback({ code: grpc.status.UNIMPLEMENTED, details: "no compact support" });
    else callback(null, { ids: mode === "incomplete" ? [] : mode === "empty" ? [""] : mode === "wrong" ? ["sasy:mv1:wrong"] : [request.snapshots[0].baseId] });
  }, async () => {
    const spy = spyOn(capture, "captureLength");
    try {
      const input = reference();
      delete input.event.derivedFrom;
      for (mode of ["incomplete", "empty", "wrong", "error"]) {
        await expect(resolveEvents([input])).rejects.toThrow();
      }
      expect(spy).toHaveBeenCalledTimes(4);
      mode = "ok";
      await resolveEvents([input]);
      await resolveEvents([input]);
      expect(spy).toHaveBeenCalledTimes(5);
      expect(methods).toEqual(Array(6).fill("resolveSnapshots"));
    } finally { spy.mockRestore(); }
  });
});

test("caller-supplied contentHash cannot replace the full current payload", async () => {
  await withServer((_method, request, callback) => {
    expect(request.snapshots[0].contentHash).toEqual(eventContentHash(captureEvent(reference().event)));
    callback(null, { ids: [request.snapshots[0].baseId] });
  }, async () => {
    await resolveEvents([{ ...reference(), contentHash: Buffer.alloc(32), unknown: "ignored" } as any]);
  });
});

test("one bad identity prevents cache publication for every reference in a batch", async () => {
  let wrong = true;
  await withServer((_method, request, callback) => {
    const ids = request.snapshots.map((item: any) => item.baseId);
    if (wrong) ids[1] = "sasy:mv1:wrong";
    callback(null, { ids });
  }, async () => {
    const inputs: EventSnapshot[] = ["first", "second"].map(text => ({
      event: { id: text, text }, baseId: `sasy:mv1:${text}`, reuseDependencies: true,
    }));
    const spy = spyOn(capture, "captureLength");
    try {
      await expect(resolveEvents(inputs)).rejects.toThrow("changed a referenced identity");
      wrong = false;
      await resolveEvents(inputs);
      expect(spy).toHaveBeenCalledTimes(4);
      await resolveEvents(inputs);
      expect(spy).toHaveBeenCalledTimes(4);
    } finally { spy.mockRestore(); }
  });
});
