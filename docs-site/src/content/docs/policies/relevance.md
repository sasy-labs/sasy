---
title: Dependency relevance
description: Control when ancestry and pairwise reachability are evaluated.
---

A policy that checks only a tool name does not need to walk the message graph.
A policy that depends on earlier results does. SASY uses relevance gates to avoid
unnecessary dependency traversal.

## What the gates control

An **EDB relation** contains supplied facts, such as `Actions` and `Edge`.
An **IDB relation** is derived by rules, such as `CurrentDepends`.
The built-in relevance mechanism gates these two derived relations:

| Gate | Derived relation |
| --- | --- |
| `CurrentDependsPolicyRelevant()` | `CurrentDepends`: ancestors of current inputs. |
| `ReachableFromPolicyRelevant()` | `ReachableFrom`: pairwise reachability among those ancestors. |

When a gate is false, the corresponding relation is empty. Helpers that depend
on it, including `ToolResultField` and ancestry-based unknown-origin checks, have
no ancestor rows to read. Direct `Current` inputs remain available.
`ReachableFromPolicyRelevant` also implies `CurrentDependsPolicyRelevant`.

These are the two built-in gates. Naming a new relation `SomethingPolicyRelevant`
does not automatically gate another IDB; a custom relation must be joined in its
own rules if it is meant to control their evaluation.

## Automatic behavior

You can write a correct ancestry policy without authoring gates. Preprocessing
supplies default gates when the policy references a gated relation or a common
helper that depends on it. A policy with no consumer needs no default traversal.

The engine then analyzes those defaults and can restrict traversal to actions
whose decision or diagnostic rules need it. It traces dependencies through
policy-defined helper relations as well. When syntax, recursion, or analysis
limits prevent a sufficiently precise result, it retains conservative behavior
and may do more work. It does not turn an unanswered dependency question into a
negative policy fact.

Automatic inference changes only generated defaults. It preserves authored
gate rules. This bounded optimization is separate from the
[static policy analysis report](/policies/validation/#static-analysis).

## Explicit action gates

For a policy whose ancestry rules protect `publish` and `send_email`, write:

```prolog
CurrentDependsPolicyRelevant() :- IsTool(_, "publish").
CurrentDependsPolicyRelevant() :- IsTool(_, "send_email").
```

This enables ancestry when either tool appears in the action batch. The gate is
request-wide, not indexed per action. Add every action whose rules need ancestry,
including rules that use helpers such as `DependsOnUnattributedInput`.

If only `publish` needs pairwise event ordering, use the narrower second gate:

```prolog
ReachableFromPolicyRelevant() :- IsTool(_, "publish").
```

That also enables `CurrentDepends` for `publish`. `ReachableFrom` is more
expensive than collecting ancestors alone, so do not enable it merely to read a
prior tool result.

## Gates affect what a rule can observe

An explicit gate is part of policy behavior. If the gate omits `send_email`, an
ancestry-based denial for `send_email` may never match. A negated ancestry test
may instead succeed because the relation is empty. Neither outcome establishes
that the underlying graph contains no relevant event.

Base gate conditions on the proposed action and other available request facts.
Do not require `CurrentDepends` to establish its own gate: that circular
requirement can leave the relation empty.

Before narrowing a gate, compare decisions with an always-enabled baseline:

```prolog
CurrentDependsPolicyRelevant().
ReachableFromPolicyRelevant().
```

Use a separate policy variant for the baseline. Exercise every protected action,
including mixed action batches, a relevant ancestor, no relevant ancestor, and
missing required evidence. The [validation guide](/policies/validation/) explains
how to distinguish compiler checks, diagnostics, and behavioral tests.
