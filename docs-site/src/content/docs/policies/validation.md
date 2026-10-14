---
title: Validate and debug policies
description: Compile policies, test decisions, interpret denial traces, and use static analysis.
---

Validate in stages: check the policy's syntax, exercise decisions against the
engine and integration, then use diagnostics and static analysis to understand
unexpected results.

## Compile without changing a policy binding

From a source checkout:

```sh
make souffle-validate FILE=my_policy.dl
```

This preprocesses and validates the policy locally without a running engine. It
needs Python and Soufflé on `PATH`; it reports compiler errors and does not
exercise the policy against recorded context.

The SDK also provides `sasy.policy.validate_policy(source)` and TypeScript's
`validatePolicy(source)`, backed by the `ValidatePolicy` RPC. They validate
source without installing it. Binding a policy to a session compiles it as well;
a binding error surfaces when entering `sasy.session(policy=...)`, not when
calling `sasy.configure`.

Pass `Path("my_policy.dl")` to load a Python policy file. A plain string is
interpreted as policy source. Declare policy-specific relations with `.decl`;
the [common relations](/policies/core-relations/) are already declared.

## Test the resulting decisions

Use synthetic observations in an isolated session. For each rule, check a case
that should allow, a case that should deny, and a case with missing prerequisite
evidence. Check the actual arguments and recorded message versions that the
integration supplies.

For ancestry rules, test direct inputs separately from ancestors. For an approval
rule, test a matching approval and one for another resource or different
arguments. If the application only recorded a proposed or failed call, it must
not satisfy a requirement for a successful `ToolResult`.

When testing [relevance gates](/policies/relevance/), compare the gated policy
with an always-enabled variant for every protected action. A false gate yields
an empty ancestry relation; an empty result is not evidence that no ancestor
exists.

In this checkout, build and select the engine explicitly:

```sh
make build-rust-dev
SASY_TEST_ENGINE="$PWD/target/debug/sasy" \
  .venv/bin/python -m pytest -q --run-integration -m integration
```

These are the repository's owned-engine tests. Use the same build and session
isolation for your own policy scenarios. See the
[test guide](https://github.com/sasy-labs/sasy/blob/main/tests/README.md) for
prerequisites and test lanes. Normal CI uses scripted models; live-provider
checks are separate and opt in.

## Read a denial trace

For a Python tool check, inspect `result.authorized`, `result.denial_reasons`,
`result.suggestions`, and `result.denial_trace`. At the policy API layer, each
action result carries its trace. A transport or evaluator failure means no
usable decision was obtained; it is not evidence that a semantic denial rule
matched.

A fired denial rule contributes its
[reason and suggestion annotations](/policies/reference/#feedback-and-denial-reasons).
When no allow rule matches, the trace also describes candidate allow routes:

| `allow_routes` status | Meaning |
| --- | --- |
| `blocked` | Fixed request conditions rule out this allow rule, for example a different tool name. |
| `possible` | Those conditions do not rule it out. This is a hint, not a promise that a suggested edit makes it match. |
| `unknown` | The bounded analysis could not decide. |

This diagnostic analysis does not run functors again or evaluate ancestry
conditions. Rule identifiers and statuses are in `allow_routes`. The text in
`reasons` uses applicable allow-rule `@deny_message` hints; `suggested_fixes`
uses suggestions from routes not classified as blocked. On `IsAuthorized`
allowlist rules, `@tool_pattern` and `@url_pattern` filter suggestions to matching
actions; without them, there is no action-pattern restriction. Suggestions on
`Unauthorized` denylist rules are provided when those rules fire.

A missing metadata row, a JSON path using the wrong field names, and an omitted
ancestry gate can each leave a rule unmatched. Check the recorded inputs and
metadata shapes before interpreting an unmatched rule as a policy decision
about data that was never supplied.

## Static analysis

```sh
python3 scripts/validate_policy.py --analyze my_policy.dl
python3 scripts/validate_policy.py --analyze --json my_policy.dl
```

The script tries a configured engine first and otherwise falls back to local
validation and the `policy-analyze` binary. Add `--no-rpc` to require the local
path. Local analysis may build that binary from the checkout.

The report identifies overlaps between allow and deny rules, redundancy, and
conditional reachability. An intentional broad allow with narrower denials can
be an expected carve-out rather than an error. Inspect the reported conditions
against the policy's intended behavior.

Analysis covers a supported subset of the language. User `@functor` calls are
modeled symbolically, not executed to establish their meaning. Some Soufflé
built-in expressions, such as `strlen(...)`, are explicitly unsupported by the
full analysis pipeline.
An unsupported or inconclusive result does not invalidate an otherwise
compilable policy, and it is not a proof that the policy is correct.

Generated diagnostics reserve relation names beginning with `SasyAllow`; do not
use that prefix for your own relations. See [security limits](/limits/#policy-analysis-and-diagnostics)
for analysis and runtime bounds, and [configuration](/configuration/) for
evaluation deadlines.
