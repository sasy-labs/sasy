---
title: ADK records for policies
description: Read ADK text, function calls, results, and metadata in policies.
---

Use this reference when a policy needs fields from a Google ADK message. For
setup and supported execution, see the [Google ADK guide](/integrations/google-adk/).

## Record representation

ADK groups message parts into a `Content`; the adapter records one node per part.

| Part or resource | What a policy reads |
| --- | --- |
| Text part | Its text in `SentMessage(id, msg)` → `msg.contents`. |
| Function call | Its tool name and arguments; argument JSON uses the original field names. |
| Function result | Its response as ordinary JSON in `msg.contents`. Only a guarded tool's successful return establishes `ToolResult(id, tool, args)` evidence. |
| State or artifact read | Resource details, provenance, and value in `msg.contents`; metadata repeats this text under `s:text`. |
| Any recorded part | The source part as JSON in `MessageMetadata(id, md)`. |

For ordinary tool arguments and results, use paths such as `destination.account`.
Metadata uses prefixed paths such as `s:function_call.s:args.s:destination.s:account`.

## Read part metadata

String dictionary keys in metadata have an `s:` prefix at every level. Common
paths are:

| Part field | Metadata path |
| --- | --- |
| Text | `s:text` |
| Function-call name or ID | `s:function_call.s:name`, `s:function_call.s:id` |
| Function-call arguments | `s:function_call.s:args` |
| Function-response name or ID | `s:function_response.s:name`, `s:function_response.s:id` |
| Function-response value | `s:function_response.s:response` |

For example, a call to `pay(amount=5)` with ID `c1` has metadata shaped like:

```json
{"s:function_call":{"s:args":{"s:amount":5},"s:id":"c1","s:name":"pay"}}
```

Read the call ID with `@json_get_str_path(md, "s:function_call.s:id")` after
joining `MessageMetadata(id, md)`. Read the amount with
`@json_get_int_path(md, "s:function_call.s:args.s:amount")`.

Scope such rules with `Current(id)` for direct action inputs or
`CurrentDepends(id)` for ancestors. A message without metadata has no
`MessageMetadata` row. See [JSON functions](/policies/reference/#json-functors)
and [core relations](/policies/core-relations/) for ancestry rules.

## Value representation and limits

Metadata uses the [same key prefixes and value wrappers as LangChain
records](/reference/langchain-records/#value-representation-and-limits), including
wrappers for bytes, tuples, sets, and ordered dictionaries. Tool arguments and
result text use ordinary JSON instead.

Other serializable Python values in metadata use a `py:<module>:<qualname>`
wrapper around their JSON rendering. Enum wrappers contain the member name under
`enum` and its encoded value under `value`. Such renderings may omit object
state; use ordinary JSON fields for values your policy needs to inspect. Values
that cannot be serialized in the required representation stop the run.

The supported part fields are text, function calls, function responses, and
thought markers. `Part.part_metadata` and other unsupported part fields stop the
turn before recording, so policies cannot read them. Saved artifact provenance
is matched on the stored version's ordinary JSON text.

Metadata participates in snapshot identity. Snapshots can differ across
Pydantic or adapter versions, including older encodings of `OrderedDict`.
