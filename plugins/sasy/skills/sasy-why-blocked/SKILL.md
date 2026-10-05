---
name: sasy-why-blocked
description: Explain why SASY blocked (or allowed) an agent's action and how to fix it. Use when a tool call was denied by SASY, the agent reports [BLOCKED] or a SASY denial, everything is denied, or the user asks why SASY made a decision.
---

Find which of these is happening, from the user's output, code and policy:

1. **Everything is denied.** A new engine denies all actions until a session
   binds a policy. Check that the code runs inside
   `with sasy.session(policy=Path("policy.dl")):`; `policy="policy.dl"` as a
   string is treated as policy *source*, not a filename.
2. **No allow rule matched, or a deny rule fired, or both.** An action is
   authorized only when an `IsAuthorized(idx)` rule matches and no
   `Unauthorized(idx)` rule does. Tell the cases apart with the decision's
   `denial_trace.reasons`: each has a `reason_type`. Several can be present.
   `NOT_ALLOWLISTED` means no allow rule matched; `DENYLISTED` means a deny
   rule fired; `ASK` means a `// @ask` rule requests human approval, which is
   not a policy error (an application without approval handling treats it as
   a denial); `NOT_AUTHENTICATED` and `SYNC_TIMEOUT` mean the request itself
   failed (credentials, or the engine timed out on recorded context), not the
   policy. Do not infer the case from the message text: `denial_reasons` and
   `suggestions` can carry `// @deny_message` and `// @suggestion` text from
   allow rules as well as from deny rules.
   - For `NOT_ALLOWLISTED`, `denial_trace.allow_routes` lists each allow rule as
     `blocked` (cannot match this request, e.g. a different tool name),
     `possible` or `unknown`.
   - For `DENYLISTED`, find the `Unauthorized` rule in the policy and explain
     which recorded input made it fire. Ancestry rules (`CurrentDepends`,
     `ToolResult`, ...) fire on anything the requesting message was computed
     from, including earlier tool results and summaries.
3. **The policy did not compile.** The error appears when the session block is
   entered. Common causes: an undeclared relation (add a `.decl`), or a typo in
   a relation name.
4. **A connection problem, not a denial.** `UNAUTHENTICATED` or TLS errors mean
   the SDK's settings do not match the running engine; see
   https://docs.sasy.ai/local-engine/ troubleshooting.

To reproduce a decision in isolation, rebuild the same inputs with `sasy.record`
and call `sasy.check_tool_call` with the requesting message's ID; print
`authorized`, `denial_reasons`, `suggestions` and `denial_trace`.

Explain the cause in one or two sentences, then propose the smallest fix: a
policy change (with the `write-policy` skill) if the rule is wrong, or a code
change if the agent recorded the wrong inputs. Do not weaken a policy just to
make a correct denial go away; say when the block is the policy working as
intended. Details: https://docs.sasy.ai/policy-language/ (validating and
debugging a policy).
