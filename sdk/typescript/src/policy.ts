import { buildMetadata, getClient, resetClient, unaryCall } from "./channel.js";
import {
  PolicyEngineClient,
  type PolicyScope,
  type PolicyMetadataFact,
  type SetPolicyRequest,
  type SetPolicyResponse,
  type UpdatePolicyMetadataResponse,
  type ValidatePolicyResponse,
  type EvaluatorStatusResponse,
  type AuthorizationRequest,
  type AuthorizationResponse,
  type HealthResponse,
} from "./generated/policy_engine.js";

const ensureClient = () => getClient(PolicyEngineClient);

export function resetPolicyEngine(): void {
  resetClient(PolicyEngineClient);
}

/**
 * Where a {@link setPolicy} call applies. Mirrors the server's
 * `PolicyScope` oneof:
 *
 * - `{ session: "<id>" }` — pin one session (empty string targets the
 *   per-tenant global session; requires the `admin` role).
 * - `{ default: true }` — set the tenant default for newly-spawned
 *   unpinned sessions (gradual rollout; `admin` only).
 * - `{ force: true }` — set the tenant default AND evict every live
 *   session in the tenant (break-glass; `admin` only).
 */
export type PolicyScopeArg =
  | { session: string }
  | { default: true }
  | { force: true };

function toProtoScope(scope: PolicyScopeArg): PolicyScope {
  if ("session" in scope) {
    return { session: { sessionId: scope.session } };
  }
  if ("default" in scope) {
    return { default: {} };
  }
  return { force: {} };
}

/**
 * Install + bind a policy. Replaces the removed `uploadPolicy`.
 *
 * The server dedupes by content hash, so re-sending identical source
 * (e.g. once per session) is cheap after the first compile.
 *
 * @param policySource - Soufflé source (dot-notation sugar allowed).
 * @param scope - {@link PolicyScopeArg}: bind a session, or set the
 *   tenant default (gradual / force).
 * @param opts.functorSource - optional custom C++ functor source.
 * @param opts.backend - target backend; "" keeps the current one.
 */
export async function setPolicy(
  policySource: string,
  scope: PolicyScopeArg,
  opts?: { functorSource?: string; backend?: string; policyMetadata?: PolicyMetadataFact[] },
): Promise<SetPolicyResponse> {
  const c = ensureClient();
  const request: SetPolicyRequest = {
    policySource,
    functorSource: opts?.functorSource ?? "",
    backend: opts?.backend ?? "",
    scope: toProtoScope(scope),
    // Static per-policy config (e.g. cooldown_days) → PolicyMetadata EDB.
    policyMetadata: opts?.policyMetadata ?? [],
    // Source-mode upload: not a bind-by-hash/name request.
    bindContentHash: "",
    bindProfileName: "",
  };
  return unaryCall<SetPolicyResponse>((cb) => c.setPolicy(request, buildMetadata(), cb));
}

/**
 * Bind an ALREADY-INSTALLED policy by its content hash, without sending
 * source. The server binds the target scope to the pre-installed policy
 * carrying this hash (subject to the restricted policy lock) and never
 * compiles; a miss is rejected with `FAILED_PRECONDITION`. Lets a co-located
 * daemon bind a baked profile without shipping the `.dl` source.
 *
 * @param scope - {@link PolicyScopeArg}: bind a session, or set the tenant
 *   default (gradual / force).
 * @param opts.contentHash - the baked profile's content hash (from the
 *   policy-pack `manifest.json`).
 * @param opts.policyMetadata - static per-policy config (cooldowns,
 *   rule_off/rule_on); not part of the hash.
 */
export async function bindPolicy(
  scope: PolicyScopeArg,
  opts: { contentHash: string; policyMetadata?: PolicyMetadataFact[] },
): Promise<SetPolicyResponse> {
  const c = ensureClient();
  const request: SetPolicyRequest = {
    policySource: "",
    functorSource: "",
    backend: "",
    scope: toProtoScope(scope),
    policyMetadata: opts.policyMetadata ?? [],
    bindContentHash: opts.contentHash,
    bindProfileName: "",
  };
  return unaryCall<SetPolicyResponse>((cb) => c.setPolicy(request, buildMetadata(), cb));
}

/**
 * Bind an ALREADY-INSTALLED curated policy by its baked PROFILE NAME (e.g.
 * "security"), without sending source OR its content hash. The server resolves
 * the name to its content hash via the registry name index and binds the target
 * scope (subject to the restricted policy lock); it never compiles, and a miss
 * is rejected with `FAILED_PRECONDITION`. Lets a co-located daemon bind an
 * embedded policy with just the profile name from its config.
 *
 * @param scope - {@link PolicyScopeArg}: bind a session, or set the tenant
 *   default (gradual / force).
 * @param opts.profileName - the baked profile name (e.g. "security", "deny-all").
 * @param opts.policyMetadata - static per-policy config (cooldowns,
 *   rule_off/rule_on); not part of the policy identity.
 */
export async function bindPolicyByName(
  scope: PolicyScopeArg,
  opts: { profileName: string; policyMetadata?: PolicyMetadataFact[] },
): Promise<SetPolicyResponse> {
  const c = ensureClient();
  const request: SetPolicyRequest = {
    policySource: "",
    functorSource: "",
    backend: "",
    scope: toProtoScope(scope),
    policyMetadata: opts.policyMetadata ?? [],
    bindContentHash: "",
    bindProfileName: opts.profileName,
  };
  return unaryCall<SetPolicyResponse>((cb) => c.setPolicy(request, buildMetadata(), cb));
}

/**
 * Release the server-side evaluator, policy binding, session metadata and
 * ownership. Graph observations remain, but reusing this session id does
 * not retain its former policy or approvals. Use a new id for a new conversation.
 */
export async function endSession(sessionId: string): Promise<void> {
  const c = ensureClient();
  await unaryCall((cb) => c.endSession({ sessionId }, buildMetadata(), cb));
}

/**
 * Append dynamic metadata facts to a session's `PolicyMetadata` EDB.
 * The server unions them in (append-only) and re-seeds the live
 * evaluator so they take effect on the next check — no evict/rebuild.
 * Facts persist (RocksDB) and survive evaluator respawns.
 *
 * Used by the detaint recorder: record `detaint_approved(<node>, "")`
 * when the user approves an `@ask`'d action (the gating source is
 * benign), or `detaint_denied(<node>, "")` when they reject it
 * (remembered as a hard block; deny-precedence in the policy).
 */
export async function updatePolicyMetadata(
  sessionId: string,
  facts: PolicyMetadataFact[],
): Promise<UpdatePolicyMetadataResponse> {
  const c = ensureClient();
  return unaryCall<UpdatePolicyMetadataResponse>((cb) =>
    c.updatePolicyMetadata({ sessionId, facts }, buildMetadata(), cb),
  );
}

export async function validatePolicy(
  policySource: string,
  opts?: { runAnalyses?: boolean; extraReachTargets?: string[] },
): Promise<ValidatePolicyResponse> {
  const c = ensureClient();
  const request = {
    policySource,
    runAnalyses: opts?.runAnalyses ?? false,
    extraReachTargets: opts?.extraReachTargets ?? [],
  };
  return unaryCall<ValidatePolicyResponse>((cb) => c.validatePolicy(request, buildMetadata(), cb));
}

export async function getEvaluatorStatus(): Promise<EvaluatorStatusResponse> {
  const c = ensureClient();
  return unaryCall<EvaluatorStatusResponse>((cb) => c.getEvaluatorStatus({}, buildMetadata(), cb));
}

/** Engine health, optionally bounded by a gRPC deadline in milliseconds. */
export async function health(timeoutMs?: number): Promise<{ healthy: boolean; message: string }> {
  if (timeoutMs !== undefined && (!Number.isFinite(timeoutMs) || timeoutMs <= 0))
    throw new Error("health timeout must be a positive finite number");
  const c = ensureClient();
  return unaryCall<HealthResponse>((cb) => timeoutMs === undefined
    ? c.health({}, buildMetadata(), cb)
    : c.health({}, buildMetadata(), { deadline: Date.now() + timeoutMs }, cb));
}

export async function checkAuthorization(
  request: AuthorizationRequest,
): Promise<AuthorizationResponse> {
  const c = ensureClient();
  return unaryCall<AuthorizationResponse>((cb) => c.checkAuthorization(request, buildMetadata(), cb));
}
