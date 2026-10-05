import { expect, spyOn, test } from "bun:test";
import { readFileSync } from "node:fs";
import * as capture from "../src/capture.js";
import { Event as ProtoEvent } from "../src/generated/observability.js";
import { captureEvent, copyEvent, eventContentHash, SnapshotDigestCache } from "../src/snapshot-content.js";

const fixture = JSON.parse(readFileSync(new URL("../../../tests/fixtures/message-content-hashes.json", import.meta.url), "utf8"));
for (const vector of fixture.vectors) test(`content fingerprint cross-language vector: ${vector.name}`, () => {
  const event = ProtoEvent.fromJSON(vector.event);
  const content = copyEvent(event);
  content.id = undefined;
  content.principal = undefined;
  expect(Buffer.from(ProtoEvent.encode(content).finish()).toString("hex")).toBe(vector.canonical_proto_hex);
  expect(eventContentHash(event).toString("hex")).toBe(vector.sha256);
});

test("content fingerprint retains every policy field, optional presence and tool order", () => {
  const baseline = { id: "origin", tools: [] };
  const empty = eventContentHash(baseline);
  for (const field of ["text", "agent", "entity"] as const) {
    expect(eventContentHash({ ...baseline, [field]: "" })).not.toEqual(empty);
  }
  expect(eventContentHash({ ...baseline, role: 0 })).not.toEqual(empty);
  expect(eventContentHash({ ...baseline, derivedFrom: {} })).not.toEqual(empty);
  expect(eventContentHash({ ...baseline, tools: [{}] })).not.toEqual(empty);
  expect(eventContentHash({ tools: [{ name: "a" }, { name: "b" }] })).not.toEqual(
    eventContentHash({ tools: [{ name: "b" }, { name: "a" }] }),
  );
  expect(eventContentHash({ ...baseline, id: "different", principal: "forged", unknown: "ignored" } as any)).toEqual(empty);
});

test("a recorded message keeps its text, and digesting is over that text", () => {
  // The reference monitor is given these bytes when it decides, so the record
  // the policy engine reasons over has to be the same bytes.
  const raw = "https://test.invalid/?api_key=synthetic-secret";
  const event = { id: "origin", text: raw, tools: [{ arguments: raw }], derivedFrom: { name: "approval", arguments: raw } };
  const before = JSON.stringify(event);
  const prepared = new SnapshotDigestCache().prepare(event);
  const recorded = captureEvent(event);
  expect(recorded.text).toBe(raw);
  expect(recorded.tools[0].arguments).toBe(raw);
  expect(recorded.derivedFrom?.arguments).toBe(raw);
  expect(prepared.contentHash).toEqual(eventContentHash(recorded));
  expect(prepared.contentHash).toEqual(eventContentHash(event));
  expect(JSON.stringify(event)).toBe(before);
});

test("cache only skips capture for an exact current raw fingerprint after commit", () => {
  const cache = new SnapshotDigestCache();
  const spy = spyOn(capture, "captureLength");
  try {
    const event = { text: "https://test.invalid/?key=first-secret", tools: [{ name: "send", arguments: "original" }] };
    const first = cache.prepare(event);
    cache.prepare(event);
    expect(spy).toHaveBeenCalledTimes(4);
    cache.commit([first]);
    expect(cache.prepare(event).contentHash).toEqual(first.contentHash);
    expect(spy).toHaveBeenCalledTimes(4);
    event.tools[0].arguments = "mutated";
    expect(cache.prepare(event).contentHash).not.toEqual(first.contentHash);
    expect(spy).toHaveBeenCalledTimes(6);
    event.tools[0].arguments = "original";
    event.text = "https://test.invalid/?key=second-secret";
    // Two messages that differ only inside a URL query are two messages: with
    // the text recorded as it is, they no longer collapse to one digest.
    expect(cache.prepare(event).contentHash).not.toEqual(first.contentHash);
    expect(spy).toHaveBeenCalledTimes(8);
  } finally { spy.mockRestore(); }
});

test("cache evicts bounded entries and ignores completions preceding clear", () => {
  const cache = new SnapshotDigestCache(2);
  const spy = spyOn(capture, "captureLength");
  try {
    const a = cache.prepare({ text: "a" });
    cache.commit([a, cache.prepare({ text: "b" })]);
    cache.commit([cache.prepare({ text: "c" })]);
    spy.mockClear();
    cache.prepare({ text: "b" });
    cache.prepare({ text: "c" });
    expect(spy).not.toHaveBeenCalled();
    cache.prepare({ text: "a" });
    expect(spy).toHaveBeenCalledTimes(1);
    cache.clear();
    cache.commit([a]);
    cache.prepare({ text: "a" });
    expect(spy).toHaveBeenCalledTimes(2);
  } finally { spy.mockRestore(); }
});

test("a cached input cannot bypass the capture length limit after mutation", () => {
  const cache = new SnapshotDigestCache();
  const event = { text: "valid", tools: [{ arguments: "valid" }] };
  cache.commit([cache.prepare(event)]);
  event.text = "x".repeat(capture.MAX_CAPTURE_LENGTH + 1);
  expect(() => cache.prepare(event)).toThrow("limit");
  event.text = "valid";
  // Deeply nested JSON is recorded as written. The nesting and work limits
  // bounded the credential scanner, which a message's own text no longer goes
  // through, so there is no scanning cost here to bound.
  event.tools[0].arguments = '{"Authorization":['.repeat(10000) + '"synthetic"' + ']}'.repeat(10000);
  const prepared = cache.prepare(event);
  expect(prepared.contentHash).toEqual(eventContentHash(captureEvent(event)));
});

test("an adapter's metadata record is copied, digested and bounded like the text", () => {
  // The metadata is a record of the message, so a message it tells apart from
  // another must reach the digest; and it is bounded as the text is.
  const event = { id: "origin", text: "answer", metadata: '{"s:content":"answer"}' };
  expect(captureEvent(event).metadata).toBe(event.metadata);
  expect(eventContentHash(event)).not.toEqual(eventContentHash({ id: "origin", text: "answer" }));
  const oversized = "x".repeat(capture.MAX_CAPTURE_LENGTH + 1);
  expect(() => captureEvent({ text: "answer", metadata: oversized })).toThrow("limit");
  expect(() => new SnapshotDigestCache().prepare({ text: "answer", metadata: oversized })).toThrow("limit");
});
