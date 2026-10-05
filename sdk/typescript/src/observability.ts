import { buildMetadata, getClient, resetClient, unaryCall } from "./channel.js";
import {
  ObservabilityClient,
  type Edge,
  type Graph,
  type IDs,
} from "./generated/observability.js";
import type { Event, EventSnapshot, ResolveEventsOptions } from "./types.js";
import { captureEvent, copyEvent, SnapshotDigestCache, type PreparedDigest } from "./snapshot-content.js";

const ensureClient = () => getClient(ObservabilityClient);

const digestCache = new SnapshotDigestCache();

export function resetObservability(): void {
  resetClient(ObservabilityClient);
  digestCache.clear();
}

/** Resolve snapshots atomically, sending checked compact references for unchanged inputs. */
export async function resolveEvents(
  snapshots: EventSnapshot[],
  sessionId?: string,
  options: ResolveEventsOptions = {},
): Promise<string[]> {
  const prepared: PreparedDigest[] = [];
  const compact = options.compact !== false;
  const request = {
    snapshots: snapshots.map((snapshot) => {
      const event = copyEvent(snapshot.event);
      const baseId = snapshot.baseId;
      const dependencies = (snapshot.dependencies ?? []).map((edge) => ({ ...edge }));
      const reuseDependencies = snapshot.reuseDependencies ?? false;
      if (!event.id) throw new Error("Each snapshot requires a stable event.id origin");
      if (reuseDependencies && (!baseId || dependencies.length)) {
        throw new Error("reuseDependencies requires baseId and no dependencies");
      }
      if (compact && reuseDependencies) {
        const digest = digestCache.prepare(event);
        prepared.push(digest);
        return { event: { id: event.id, tools: [] }, baseId, dependencies, reuseDependencies, contentHash: digest.contentHash };
      }
      return { event: captureEvent(event), baseId, dependencies, reuseDependencies, contentHash: undefined };
    }),
    sessionId,
  };
  if (!request.snapshots.length) return [];
  const c = ensureClient();
  const resp = await unaryCall<IDs>((cb) => compact
    ? c.resolveSnapshots(request, buildMetadata(), cb)
    : c.resolveEvents(request, buildMetadata(), cb));
  if (resp.ids.length !== request.snapshots.length || resp.ids.some((id) => !id)) {
    throw new Error("Snapshot resolution returned incomplete identities");
  }
  for (let i = 0; i < request.snapshots.length; i++) {
    const item = request.snapshots[i];
    if (item.reuseDependencies && resp.ids[i] !== item.baseId) {
      throw new Error("Snapshot resolution changed a referenced identity");
    }
  }
  digestCache.commit(prepared);
  return resp.ids;
}

/**
 * Register events in the dependency graph. Returns server-assigned IDs.
 *
 * `sessionId` scopes the write to a session's graph partition (the
 * server keys storage on the auth-derived tenant + this session id).
 * Omit it to land in the per-tenant global partition. Prefer
 * {@link Session.registerEvents} from `session(id)` for ergonomics.
 */
export async function registerEvents(
  events: Event[],
  sessionId?: string,
): Promise<string[]> {
  const c = ensureClient();
  const resp = await unaryCall<IDs>((cb) => c.registerEvents(
      { events: events.map(captureEvent), sessionId },
      buildMetadata(), cb));
  return resp.ids;
}

/** Register dependency edges between events, optionally session-scoped. */
export async function registerDependencies(
  edges: Edge[],
  sessionId?: string,
): Promise<void> {
  const c = ensureClient();
  await unaryCall((cb) => c.registerDependencies(
      { edges, sessionId },
      buildMetadata(), cb));
}

/** Register events and dependencies atomically. Returns server-assigned IDs. */
export async function registerEventsWithDependencies(
  events: Event[],
  edges: Edge[],
  sessionId?: string,
): Promise<string[]> {
  const c = ensureClient();
  const resp = await unaryCall<IDs>((cb) => c.registerEventsWithDependencies(
      { events: events.map(captureEvent), edges, sessionId },
      buildMetadata(), cb));
  return resp.ids;
}

/** Get the backward slice (all causes) from an event.
 *
 * `sessionId` names the shard the `eventId` lives in. Pass it (the
 * session the event was written under) to read that shard directly;
 * omit it (global / `undefined`) to let the server resolve the id via
 * the tenant-wide reverse index. */
export async function backwardSlice(
  eventId: string,
  maxDepth?: number,
  sessionId?: string,
): Promise<Graph> {
  const c = ensureClient();
  return unaryCall<Graph>((cb) => c.backwardSlice(
      { eventId, maxDepth, sessionId },
      buildMetadata(), cb));
}

/** Get the forward slice (all effects) from an event. See
 * {@link backwardSlice} for `sessionId`. */
export async function forwardSlice(
  eventId: string,
  maxDepth?: number,
  sessionId?: string,
): Promise<Graph> {
  const c = ensureClient();
  return unaryCall<Graph>((cb) => c.forwardSlice(
      { eventId, maxDepth, sessionId },
      buildMetadata(), cb));
}
