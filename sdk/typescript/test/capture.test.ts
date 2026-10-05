import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import * as grpc from "@grpc/grpc-js";
import { captureText, MAX_CAPTURE_LENGTH } from "../src/capture.js";
import { ApiKeyAuthHook } from "../src/auth.js";
import { configure, resetChannel } from "../src/channel.js";
import { registerEvents, registerEventsWithDependencies, resolveEvents } from "../src/observability.js";
import { session } from "../src/session.js";
import { checkToolCall } from "../src/reference-monitor.js";
import { ObservabilityService } from "../src/generated/observability.js";
import { RMProxyService } from "../src/generated/reference_monitor.js";

const fixtures: { name: string; input: string; expected: string }[] = JSON.parse(readFileSync(new URL("./fixtures/capture-credentials.json", import.meta.url), "utf8"));
for (const fixture of fixtures) test(`shared capture corpus: ${fixture.name}`, () => {
  expect(captureText(fixture.input)).toBe(fixture.expected);
  expect(captureText(fixture.expected)).toBe(fixture.expected);
});

test("capture refuses oversized text before walking it", () => {
  expect(() => captureText("x".repeat(MAX_CAPTURE_LENGTH + 1))).toThrow("limit");
});

test("actual gRPC capture changes only telemetry content; auth and direct authorization remain intact", async () => {
  const raw = "https://api.test/?key=fixture-secret&n=1";
  const clean = "https://api.test/?key=[redacted]&n=1";
  const calls: { request: any; metadata: grpc.Metadata }[] = [];
  const server = new grpc.Server();
  const collect = (call: any, callback: any) => {
    calls.push({ request: call.request, metadata: call.metadata });
    callback(null, { ids: ["fixture-id"] });
  };
  server.addService(ObservabilityService, { registerEvents: collect, registerEventsWithDependencies: collect });
  server.addService(RMProxyService, { checkToolCall: (call: any, callback: any) => {
    calls.push({ request: call.request, metadata: call.metadata });
    callback(null, { authorized: true, transformIds: [] });
  } });
  const port = await new Promise<number>((resolve, reject) => server.bindAsync("127.0.0.1:0", grpc.ServerCredentials.createInsecure(), (error, port) => error ? reject(error) : resolve(port)));
  configure({ url: `127.0.0.1:${port}`, insecure: true, certPath: undefined, keyPath: undefined, caPath: undefined, authHook: new ApiKeyAuthHook("fixture-transport-secret") });
  try {
    const event = { id: "fixture-id", text: raw, agent: raw, principal: raw, entity: "fixture-actor", tools: [{ name: raw, arguments: JSON.stringify({ Authorization: "fixture-secret" }) }], derivedFrom: { name: "tool", arguments: raw } };
    const edge = { source: "fixture-input", destination: "fixture-id", principal: raw };
    const before = JSON.stringify({ event, edge });
    expect(await registerEvents([event], "fixture-session")).toEqual(["fixture-id"]);
    expect(await registerEventsWithDependencies([event], [edge], "fixture-session")).toEqual(["fixture-id"]);
    expect(JSON.stringify({ event, edge })).toBe(before);
    for (const { request, metadata } of calls) {
      expect(metadata.get("x-api-key")).toEqual(["fixture-transport-secret"]);
      expect(request.sessionId).toBe("fixture-session");
      // A recorded message reaches the graph as it is: the same bytes the
      // reference monitor is given below when it decides.
      expect(request.events[0]).toMatchObject({ text: raw, agent: raw, principal: raw, entity: "fixture-actor", tools: [{ name: raw, arguments: JSON.stringify({ Authorization: "fixture-secret" }) }], derivedFrom: { name: "tool", arguments: raw } });
    }
    expect(calls[1].request.edges[0]).toMatchObject(edge);
    const outbound = new Request(raw, { headers: { Authorization: "Bearer fixture-secret" } });
    const args = JSON.stringify({ url: outbound.url, headers: Object.fromEntries(outbound.headers) });
    await checkToolCall({ fnName: "http", args, inputNodeIds: ["fixture-input"] });
    expect(calls[2].request.args).toBe(args);
    expect(calls[2].metadata.get("x-api-key")).toEqual(["fixture-transport-secret"]);
    expect(outbound.url).toBe(raw);
    expect(outbound.headers.get("Authorization")).toBe("Bearer fixture-secret");
  } finally {
    resetChannel();
    server.forceShutdown();
  }
});

for (const [depth, padding] of [[10000, 0], [7, 20000]]) test(`nested header arrays reject with bounded parsing (${depth} levels)`, () => {
  const payload = '{"Authorization":['.repeat(depth) + JSON.stringify("fixture-secret" + "x".repeat(padding)) + ']}'.repeat(depth);
  const parse = JSON.parse;
  const sizes: number[] = [];
  JSON.parse = (value: string) => { sizes.push(value.length); return parse(value); };
  try {
    expect(() => captureText(payload)).toThrow("nesting/work limit");
    expect(sizes.length).toBeLessThanOrEqual(3);
    expect(sizes.filter(size => size > 20).length).toBeLessThanOrEqual(1);
  } finally {
    JSON.parse = parse;
  }
});

test("snapshot RPC preserves full replacement and returned IDs without changing caller objects", async () => {
  const requests: any[] = [];
  const server = new grpc.Server();
  server.addService(ObservabilityService, { resolveSnapshots: (call: any, callback: any) => {
    requests.push(call.request);
    callback(null, { ids: call.request.sessionId === "incomplete" ? [] : ["sasy:mv1:canonical"] });
  } });
  const port = await new Promise<number>((resolve, reject) => server.bindAsync("127.0.0.1:0", grpc.ServerCredentials.createInsecure(), (error, port) => error ? reject(error) : resolve(port)));
  configure({ url: `127.0.0.1:${port}`, insecure: true, certPath: undefined, keyPath: undefined, caPath: undefined, authHook: new ApiKeyAuthHook("fixture-transport-secret") });
  try {
    const snapshot = {
      event: { id: "origin", text: "https://api.test/?key=fixture-secret", tools: [] },
      baseId: "sasy:mv1:previous", dependencies: [{ source: "input-version", destination: "origin" }],
    };
    const before = JSON.stringify(snapshot);
    expect(await session("conversation").resolveEvents([snapshot])).toEqual(["sasy:mv1:canonical"]);
    expect(requests[0].sessionId).toBe("conversation");
    expect(requests[0].snapshots[0]).toMatchObject({
      event: { id: "origin", text: "https://api.test/?key=fixture-secret", tools: [] },
      baseId: "sasy:mv1:previous", dependencies: snapshot.dependencies, reuseDependencies: false,
    });
    expect(requests[0].snapshots[0].event.derivedFrom).toBeUndefined();
    expect(JSON.stringify(snapshot)).toBe(before);
    await expect(resolveEvents([snapshot], "incomplete")).rejects.toThrow("incomplete identities");
  } finally {
    resetChannel();
    server.forceShutdown();
  }
});
