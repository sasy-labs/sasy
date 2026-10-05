// Configuration
export { configure, getConfig, resetChannel } from "./channel.js";

// Auth hooks
export {
  NoAuthHook,
  ApiKeyAuthHook,
  StaticTokenAuthHook,
  EntityAuthHook,
} from "./auth.js";

// Reference Monitor
export { checkToolCall, resetReferenceMonitor } from "./reference-monitor.js";

// Sessions (mirrors Python sasy.session(...))
export { session, Session } from "./session.js";

// Policy Engine
export {
  setPolicy,
  bindPolicy,
  bindPolicyByName,
  endSession,
  updatePolicyMetadata,
  validatePolicy,
  getEvaluatorStatus,
  checkAuthorization,
  health,
  resetPolicyEngine,
  type PolicyScopeArg,
} from "./policy.js";

// Credentials
export { getCredentials, setCredentials, resetCredentialServer } from "./credentials.js";

// Observability
export {
  registerEvents,
  registerDependencies,
  registerEventsWithDependencies,
  resolveEvents,
  backwardSlice,
  forwardSlice,
  resetObservability,
} from "./observability.js";

// Types
export type {
  SasyConfig,
  AuthHook,
  ToolCallRequest,
  ToolCallResponse,
  DenialTrace,
  DenialReason,
  AuthorizationRequest,
  AuthorizationResponse,
  ActionResult,
  PerformanceTiming,
  Action,
  HttpRequestAction,
  ToolCallAction,
  SendMessageAction,
  SetPolicyRequest,
  SetPolicyResponse,
  PolicyMetadataFact,
  PolicyScope,
  SessionTarget,
  DefaultTarget,
  ForceTarget,
  EndSessionRequest,
  EndSessionResponse,
  ValidatePolicyResponse,
  EvaluatorStatusResponse,
  Credential,
  Event,
  EventSnapshot,
  ResolveEventsOptions,
  Edge,
  Tool,
  Graph,
  SliceRequest,
} from "./types.js";

export { DenialReasonType, Role } from "./types.js";
