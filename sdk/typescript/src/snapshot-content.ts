import { createHash } from "node:crypto";
import { captureLength, MAX_CAPTURE_LENGTH } from "./capture.js";
import { Event as ProtoEvent } from "./generated/observability.js";
import type { Event } from "./types.js";

const CONTENT_DOMAIN = Buffer.from("sasy:event-content:v1\0");
const RAW_DOMAIN = Buffer.from("sasy:event-raw:v1\0");

/** Copy only fields represented by the wire schema, retaining optional presence. */
export function copyEvent(event: Event): ProtoEvent {
  const tool = (value: NonNullable<Event["derivedFrom"]>) => ({
    name: value.name, arguments: value.arguments,
  });
  return {
    id: event.id, text: event.text, metadata: event.metadata,
    agent: event.agent, role: event.role,
    principal: event.principal, entity: event.entity,
    tools: (event.tools ?? []).map(tool),
    derivedFrom: event.derivedFrom === undefined ? undefined : tool(event.derivedFrom),
  };
}

/**
 * Copy an event, checking the size of the text it carries.
 *
 * A message's text, an adapter's metadata record of it and a tool call's
 * arguments are recorded as they are. They are what the policy engine reasons
 * over, and the reference monitor is given the same bytes when it decides, so
 * rewriting them here would leave a policy reading text that neither the model
 * nor the monitor saw. The graph holds the
 * conversation and is sensitive whatever this function does; it is protected
 * as a store rather than by rewriting the record. Transport metadata is
 * different and is still scrubbed where it appears.
 */
export function captureEvent(event: Event): ProtoEvent {
  const result = copyEvent(event);
  if (result.text !== undefined) captureLength(result.text);
  if (result.metadata !== undefined) captureLength(result.metadata);
  for (const tool of [...result.tools, ...(result.derivedFrom ? [result.derivedFrom] : [])]) {
    if (tool.arguments !== undefined) captureLength(tool.arguments);
  }
  return result;
}

/** Hash an already captured event; identity and authenticated principal are excluded. */
export function eventContentHash(event: Event): Buffer {
  const content = copyEvent(event);
  content.id = undefined;
  content.principal = undefined;
  return createHash("sha256").update(CONTENT_DOMAIN).update(ProtoEvent.encode(content).finish()).digest();
}

export interface PreparedDigest {
  rawHash: string;
  contentHash: Buffer;
  epoch: number;
}

/** Stores only fixed-size fingerprints, never message bodies or credentials. */
export class SnapshotDigestCache {
  private readonly entries = new Map<string, Buffer>();
  private epoch = 0;

  constructor(private readonly capacity = 4096) {
    if (!Number.isSafeInteger(capacity) || capacity < 1) throw new Error("Digest cache capacity must be positive");
  }

  prepare(event: Event): PreparedDigest {
    // Take a detached copy before fingerprinting and capture so both inspect
    // the same values, including nested tools and explicit empty fields.
    const raw = copyEvent(event);
    raw.id = undefined;
    raw.principal = undefined;
    for (const text of [raw.text, raw.metadata, ...raw.tools.map(tool => tool.arguments), raw.derivedFrom?.arguments]) {
      if (text !== undefined && text.length > MAX_CAPTURE_LENGTH) {
        throw new Error("telemetry capture text exceeds 16 MiB character limit");
      }
    }
    const rawHash = createHash("sha256").update(RAW_DOMAIN).update(ProtoEvent.encode(raw).finish()).digest("hex");
    const cached = this.entries.get(rawHash);
    return { rawHash, contentHash: cached === undefined ? eventContentHash(captureEvent(raw)) : Buffer.from(cached), epoch: this.epoch };
  }

  /** Publish only after the entire RPC and its returned identities are validated. */
  commit(prepared: PreparedDigest[]): void {
    for (const { rawHash, contentHash, epoch } of prepared) {
      if (epoch !== this.epoch) continue;
      this.entries.delete(rawHash);
      this.entries.set(rawHash, Buffer.from(contentHash));
      while (this.entries.size > this.capacity) this.entries.delete(this.entries.keys().next().value!);
    }
  }

  clear(): void {
    this.entries.clear();
    this.epoch++;
  }
}
