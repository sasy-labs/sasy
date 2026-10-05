import { checkToolCall as _checkToolCall } from "./reference-monitor.js";
import {
  registerEvents as _registerEvents,
  registerDependencies as _registerDependencies,
  registerEventsWithDependencies as _registerEventsWithDependencies,
  resolveEvents as _resolveEvents,
} from "./observability.js";
import {
  setPolicy as _setPolicy,
  bindPolicy as _bindPolicy,
  bindPolicyByName as _bindPolicyByName,
  endSession as _endSession,
  updatePolicyMetadata as _updatePolicyMetadata,
} from "./policy.js";
import type { ToolCallRequest, ToolCallResponse, Event, EventSnapshot, Edge, ResolveEventsOptions } from "./types.js";
import type {
  PolicyMetadataFact,
  SetPolicyResponse,
  UpdatePolicyMetadataResponse,
} from "./generated/policy_engine.js";

/**
 * A session-scoped view of the SASY SDK. Mirrors the Python
 * `sasy.session(...)` context manager: every call made through this
 * object is tagged with `sessionId`, so the server routes graph writes
 * and policy checks to that session's partition + binding.
 *
 * ```ts
 * const s = sasy.session("conv-123");
 * await s.setPolicy(POLICY_SOURCE);          // pin this session
 * const r = await s.checkToolCall({ fnName, args });
 * await s.registerEvents([event]);
 * await s.endSession();                       // release session (observations kept)
 * ```
 */
export class Session {
  constructor(public readonly sessionId: string) {}

  /** Authorize a tool call within this session. */
  checkToolCall(request: ToolCallRequest): Promise<ToolCallResponse> {
    return _checkToolCall({ ...request, sessionId: this.sessionId });
  }

  /** Register events into this session's graph partition. */
  registerEvents(events: Event[]): Promise<string[]> {
    return _registerEvents(events, this.sessionId);
  }

  /** Resolve immutable message versions within this session. */
  resolveEvents(snapshots: EventSnapshot[], options?: ResolveEventsOptions): Promise<string[]> {
    return _resolveEvents(snapshots, this.sessionId, options);
  }

  /** Register dependency edges into this session's graph partition. */
  registerDependencies(edges: Edge[]): Promise<void> {
    return _registerDependencies(edges, this.sessionId);
  }

  /** Register events + dependencies atomically into this session. */
  registerEventsWithDependencies(
    events: Event[],
    edges: Edge[],
  ): Promise<string[]> {
    return _registerEventsWithDependencies(events, edges, this.sessionId);
  }

  /**
   * Pin this session to a policy (`SetPolicy` scope=session). The
   * server dedupes identical source by content hash, so calling this
   * once at session start is cheap on repeat runs.
   */
  setPolicy(
    policySource: string,
    opts?: { functorSource?: string; backend?: string; policyMetadata?: PolicyMetadataFact[] },
  ): Promise<SetPolicyResponse> {
    return _setPolicy(policySource, { session: this.sessionId }, opts);
  }

  /**
   * Bind this session to an already-installed policy by its content hash,
   * without sending source. Used by the restricted appliance: the daemon
   * resolves a baked profile's hash from the policy-pack manifest and binds
   * it, so the `.dl` source need not be shipped or transmitted.
   */
  bindPolicy(
    contentHash: string,
    opts?: { policyMetadata?: PolicyMetadataFact[] },
  ): Promise<SetPolicyResponse> {
    return _bindPolicy({ session: this.sessionId }, {
      contentHash,
      policyMetadata: opts?.policyMetadata,
    });
  }

  /**
   * Bind this session to an already-installed curated policy by its baked
   * PROFILE NAME (e.g. "security"), without sending source OR its content hash.
   * Used by the restricted appliance: the daemon binds the configured profile by
   * name, so neither the `.dl` source nor a manifest hash need be transmitted.
   */
  bindPolicyByName(
    profileName: string,
    opts?: { policyMetadata?: PolicyMetadataFact[] },
  ): Promise<SetPolicyResponse> {
    return _bindPolicyByName({ session: this.sessionId }, {
      profileName,
      policyMetadata: opts?.policyMetadata,
    });
  }

  /**
   * Append dynamic metadata facts to this session's `PolicyMetadata`
   * EDB (append-only; re-seeds the live evaluator). Used to record a
   * detaint decision — `detaint_approved` / `detaint_denied` on the
   * source node of an `@ask`'d action.
   */
  updatePolicyMetadata(facts: PolicyMetadataFact[]): Promise<UpdatePolicyMetadataResponse> {
    return _updatePolicyMetadata(this.sessionId, facts);
  }

  /** Release the evaluator, binding, metadata and ownership; graph observations remain. */
  endSession(): Promise<void> {
    return _endSession(this.sessionId);
  }
}

/**
 * Open a session-scoped SDK view. Unlike Python's context manager this
 * does not set ambient state — hold the returned {@link Session} and
 * call through it. Call {@link Session.endSession} when done if you
 * want the server to release the evaluator eagerly.
 */
export function session(sessionId: string): Session {
  return new Session(sessionId);
}
