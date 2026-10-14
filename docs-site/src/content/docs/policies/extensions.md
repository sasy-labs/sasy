---
title: Policy extension points
description: Supply message, action, and policy metadata, and add native functors.
---

Use ordinary policy rules to combine the [core relations](/policies/core-relations/).
When a rule needs information outside those relations, supply metadata or add a
custom functor. Metadata supplies facts; a functor computes a value during policy
evaluation.

## Choose the scope of a fact

| Policy relation | Populate it through | Lifetime |
| --- | --- | --- |
| `MessageMetadata(id, value)` | `Event.metadata` when recording or resolving a message. | Part of the recorded message version. |
| `ActionMetadata(idx, rel, a, b)` | The proposed action's metadata in an authorization request. | This action in this check. |
| `PolicyMetadata(rel, a, b)` | Policy upload metadata, or the session metadata-update API. | Configuration and accumulated facts for the policy binding/session. |

These facts are supplied by the application or adapter. The engine does not
perform the external lookup that a fact describes. Give the producer a defined
meaning and scope before using its facts to authorize an action. Metadata is not
a replacement for the server-derived `Principal` and roles.

## Policy configuration

Separate deployment settings from policy logic. For example, this policy allows
publication only to destinations configured on its binding:

```prolog
.decl AllowedDestination(name: symbol)
AllowedDestination(name) :- PolicyMetadata("allowed_destination", name, _).

IsAuthorized(idx) :-
    Actions(idx, action),
    action = $CallTool("publish", args),
    AllowedDestination(@json_get_str(args, "destination")).
```

After configuring the SDK connection, bind it with Python:

```python
from pathlib import Path
import sasy

with sasy.session(
    policy=Path("publication.dl"),
    policy_metadata=[("allowed_destination", "internal", "")],
):
    result = sasy.check_tool_call("publish", '{"destination":"internal"}')
```

Python accepts `(rel, a)` or `(rel, a, b)` tuples; omitted `b` is empty. The
TypeScript equivalent is `session.setPolicy(source, { policyMetadata:
[{ rel: "allowed_destination", a: "internal", b: "" }] })`. The gRPC field is
`SetPolicyRequest.policy_metadata`.

### Append facts during a session

An application can add a fact after an external event, such as a completed review:

```python
from sasy.policy import update_policy_metadata

update_policy_metadata(session_id, [("reviewed", "synthetic-report", "approved")])
```

`session_id` is the active session's wire ID, available in Python through
`sasy.get_current_session_id()` for an ordinary session. TypeScript exposes
`session.updatePolicyMetadata(facts)`; gRPC exposes `UpdatePolicyMetadata`.
Facts accumulate: this API appends and does not remove or replace earlier facts.
Use per-action metadata for a verdict that must apply to one check, and bind
approval facts to the resource and arguments they actually cover. A policy must
read the fact for it to affect a decision.

## Facts for one proposed action

Pass a lookup result with the action it describes:

```python
result = sasy.check_tool_call(
    "publish",
    '{"destination":"internal"}',
    metadata=[("content_scan", "classification", "public")],
)
```

The engine projects that tuple into `ActionMetadata` with the same index as the
action. A matching allow rule is:

```prolog
IsAuthorized(idx) :-
    Actions(idx, action),
    IsTool(action, "publish"),
    ActionMetadata(idx, "content_scan", "classification", "public").
```

Missing metadata leaves this allow rule unmatched. A denial is returned in
`result.authorized`; the application must check it before dispatch. Required
transforms must also be applied or the action refused.

TypeScript's `session.checkToolCall` accepts `metadata: [{ rel, a, b }]`. At the
gRPC layer these facts are `Action.metadata` for `CheckAuthorization`, or
`ToolCallRequest.metadata` on the reference-monitor tool-check API. They are
separate from the RPC's authentication headers.

## Message metadata

Write the source message's policy-relevant fields into `Event.metadata`. The
server exposes the string as `MessageMetadata(id, value)` and includes it in the
message's content identity. It does not interpret or validate your schema.

Inside an active session, record a custom JSON field with an immutable version:

```python
import json
from sasy.observability import Event, EventSnapshot, resolve_events

node_id, = resolve_events([EventSnapshot(event=Event(
    id="synthetic-report",
    text="A synthetic report",
    metadata=json.dumps({"marker": "reviewed"}),
))])
```

For immutable observations, use Python's `resolve_events` with `EventSnapshot`
objects, or TypeScript's `session.resolveEvents`. The older
`record_events_with_dependencies` / `session.registerEventsWithDependencies`
APIs also accept `Event.metadata`. See [recording and versioning](/instrumentation/#immutable-message-versions)
for the APIs and dependency handling.

For custom events, choose a documented format, usually JSON. Supplied integration
adapters already define their formats: [LangChain records](/reference/langchain-records/)
and [ADK records](/reference/google-adk-records/). Their metadata keys carry type
prefixes such as `s:`; custom metadata does not acquire those prefixes automatically.

## Custom functors

Use a custom functor for an operation that the existing rules and
[built-in functions](/policies/reference/#json-functors) cannot express. Declare
its Datalog signature and provide a matching C++ implementation separately.
This minimal example shows the interface:

```prolog
.functor label_is_internal(label: symbol): unsigned stateful

IsAuthorized(idx) :-
    Actions(idx, action),
    action = $CallTool("publish", args),
    @label_is_internal(@json_get_str(args, "label")) = 1.
```

```cpp
#include <souffle/SouffleInterface.h>

extern "C" souffle::RamDomain label_is_internal(
    souffle::SymbolTable* symbols,
    souffle::RecordTable*,
    souffle::RamDomain label) {
    return symbols->decode(label).rfind("internal:", 0) == 0 ? 1 : 0;
}
```

The `stateful` signature receives the symbol and record tables before its
Datalog arguments. Decode symbol arguments and encode symbol return values
through the symbol table. Keep the implementation's name and types consistent
with its declaration.

For Python file-based policies, put the implementation in a companion
`functors.cpp`; the SDK discovers it when loading the policy. For explicit
uploads, use `sasy.policy.set_session_policy(..., functor_source=cpp_source)`.
TypeScript's `session.setPolicy` accepts `functorSource`; the gRPC field is
`SetPolicyRequest.functor_source`. The engine compiles the source into the
compiled evaluator or the interpreted backend's shared functor library.

Custom functors are native code. Admission requires an administrator unless the
operator explicitly enables non-admin admission; sandbox-required admission
needs working Linux confinement. See [engine options](/cli/#engine-server-options)
and [security limits](/limits/). A restricted build accepts installed profiles,
not new source uploads.

Validate the policy and exercise allowed, denied, and missing-data cases with
the intended backend. A syntax check does not execute the C++ implementation.
See [validation and debugging](/policies/validation/).
