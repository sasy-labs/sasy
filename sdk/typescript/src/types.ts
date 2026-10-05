// ============================================
// SDK-local types (not in proto)
// ============================================

/** Authentication metadata provider for gRPC calls. */
export interface AuthHook {
  getMetadata(): [string, string][];
}

/** Configuration for the SASY client. */
export interface SasyConfig {
  /** Server address (host:port). Default: "localhost:10089" */
  url: string;
  /** Path to a CA certificate (private CA / self-signed). */
  caPath?: string;
  /** Path to client certificate for mTLS. */
  certPath?: string;
  /** Path to client private key for mTLS. */
  keyPath?: string;
  /** Authentication hook. Default: no auth. */
  authHook?: AuthHook;
  /**
   * Opt in to plaintext gRPC (no TLS, no server-cert verification).
   * For isolated/internal networks only. Also settable via
   * `SASY_INSECURE=1`. By default the client uses TLS and verifies the
   * server certificate against the system trust store.
   */
  insecure?: boolean;
}

// ============================================
// Re-exports from generated proto types
// ============================================

export {
  DenialReasonType,
  type AuthorizationRequest,
  type AuthorizationResponse,
  type Action,
  type HttpRequestAction,
  type ToolCallAction,
  type SendMessageAction,
  type ActionResult,
  type PerformanceTiming,
  type DenialTrace,
  type DenialReason,
  type SetPolicyRequest,
  type SetPolicyResponse,
  type PolicyScope,
  type SessionTarget,
  type DefaultTarget,
  type ForceTarget,
  type EndSessionRequest,
  type EndSessionResponse,
  type ValidatePolicyResponse,
  type EvaluatorStatusResponse,
  type PolicyMetadataFact,
} from "./generated/policy_engine.js";

export {
  Role,
  type Edge,
  type Tool,
  type Graph,
  type SliceRequest,
} from "./generated/observability.js";

// Override Event to keep repeated fields optional (callers omit them)
import type { Event as _Event } from "./generated/observability.js";
export type Event = Omit<_Event, "tools"> & {
  tools?: _Event["tools"];
};

/** Complete message contents and explicit incoming dependencies for versioning. */
export interface EventSnapshot {
  event: Event & { id: string };
  baseId?: string;
  dependencies?: import("./generated/observability.js").Edge[];
  /** Reuse a base's dependencies only when its complete contents are unchanged. */
  reuseDependencies?: boolean;
}

export interface ResolveEventsOptions {
  /** Send complete snapshots through the legacy RPC instead of compact references. */
  compact?: boolean;
}

// Override ToolCallRequest to keep inputNodeIds + metadata optional
import type {
  ToolCallRequest as _ToolCallRequest,
  ToolCallResponse,
} from "./generated/reference_monitor.js";
export type { ToolCallResponse };
export type ToolCallRequest = Omit<_ToolCallRequest, "inputNodeIds" | "metadata"> & {
  inputNodeIds?: _ToolCallRequest["inputNodeIds"];
  metadata?: _ToolCallRequest["metadata"];
};

export type { Credential } from "./generated/credential_server.js";
