---
title: Built-ins and feedback
description: Additional relations, JSON and text functors, and denial feedback annotations.
---

The [core relations](/policies/core-relations/) cover actions, identity, messages,
and authorization. This page adds the helpers and feedback notation used by more
specific policies. The schema and signatures are in
[`souffle/common_policy.dl`](https://github.com/sasy-labs/sasy/blob/main/souffle/common_policy.dl).

## Context helpers

| Relation | Meaning |
| --- | --- |
| `IsToolCall(action)` | Matches any proposed tool call. |
| `ReachableFrom(source, destination)` | Forward reachability between ancestors of the current inputs. Both endpoints are in the backward slice. Useful for ordering earlier events. |
| `ToolResultFieldName(field)` | A fact you write to select a top-level JSON result field to extract. |
| `ToolResultField(tool, field, value)` | A selected, non-empty string field from a successful tool result in the ancestry. |
| `UnattributedInput(id)` | A direct input or ancestor whose recorded text carries the common unknown-origin marker. |
| `DependsOnUnattributedInput()` | True when any such input is in scope. |
| `UnattributedInputOrigin(id, origin)` | Its non-empty origin label, when the adapter recorded one. |
| `CurrentDependsPolicyRelevant()`, `ReachableFromPolicyRelevant()` | Gates controlling dependency traversal; see [relevance](/policies/relevance/). |

Declare a field to make it available through `ToolResultField`:

```prolog
ToolResultFieldName("classification").

// @deny_message: This publication depends on a confidential document
Unauthorized(idx) :-
    Actions(idx, action),
    IsTool(action, "publish"),
    ToolResultField("read_document", "classification", "confidential").
```

`ToolResultField` omits missing fields and empty strings and handles string values
only. For numbers or nested fields, join `ToolResult` with `SentMessage` and use a
JSON functor directly. The helper does not identify which resource a result
belongs to; join on message IDs and arguments when that distinction matters.

Unknown-origin helpers recognize the unprefixed `provenance` marker in message
text. LangChain uses a prefixed envelope and needs its
[adapter-specific marker rules](/integrations/langchain/#inputs-of-unknown-origin).

## Metadata and edge details

| Relation | Meaning |
| --- | --- |
| `MessageMetadata(id, metadata)` | The adapter's record of the source message, stored as `Event.metadata`. No row exists when metadata was omitted. |
| `ActionMetadata(idx, rel, a, b)` | Facts attached to this proposed action, indexed like `Actions`. |
| `PolicyMetadata(rel, a, b)` | Configuration supplied with the policy, plus any session-scoped facts appended later. |
| `EdgeData(source, destination, data)` | Optional recording context: `data.proximal` is 1 for an immediate predecessor, 0 for false, or -1 if unset; `data.message_index` is the zero-based input position, or -1 if unset. |
| `EdgePrincipal(source, destination, principal)` | Server-stamped identity of the edge writer, when known. |
| `EdgeEntity(source, destination, entity)` | Caller-supplied actor label for the edge, when present. |

See [extension points](/policies/extensions/) for the APIs that populate metadata
and examples of projecting it into policy-specific relations. An edge can exist
without any of these optional edge-detail rows.

## Feedback and denial reasons

Put annotation comments immediately above a rule:

```prolog
// @deny_message: External publication is disabled
// @suggestion: Publish internally
Unauthorized(idx) :-
    Actions(idx, action),
    IsTool(action, "publish"),
    action = $CallTool(_, args),
    @json_get_str(args, "destination") = "external".
```

| Notation | Effect |
| --- | --- |
| `// @deny_message: text` | Human-readable reason for a matching denial rule. |
| `// @suggestion: text` | Suggested remediation. It does not change authorization. |
| `// @ask` | Return an approval request instead of a hard block. |
| `// @tool_pattern: pattern`, `// @url_pattern: pattern` | Filter suggestions on `IsAuthorized` allowlist rules to matching actions. They do not replace conditions in the rule body. |
| `// @broad` | Marks a broad rule for static analysis, such as an intentional allow rule with narrower denials. It does not change evaluation. |

The preprocessor rewrites ordinary `Unauthorized` rules to
`DenialReason(idx, kind, reason, suggestion)`. Unannotated rules use
`"Action is denylisted"`, an empty suggestion, and kind `"block"`. A block reason
implies `Unauthorized`. You can also derive `DenialReason` directly when the
reason needs values from the rule.

Annotations on `IsAuthorized` provide hints when no allow rule matches; they do
not create a denial rule. `@tool_pattern` and `@url_pattern` filter those
allowlist suggestions. Feedback on `Unauthorized` denylist rules is provided
when the rules fire. See [debugging denial traces](/policies/validation/#read-a-denial-trace)
for how those hints and allow-route statuses appear in a response.

### Approval requests

```prolog
// @ask
// @deny_message: Publishing externally needs human approval
Unauthorized(idx) :-
    Actions(idx, action),
    IsTool(action, "publish"),
    action = $CallTool(_, args),
    @json_get_str(args, "destination") = "external".
```

`@ask` emits a reason of kind `"ask"`. It does not itself supply an allowlist
match, and a hard block still wins. The engine returns the request as not
authorized; the supplied integration adapters block the tool. An application
that supports approval must handle that result and arrange a subsequent
authorized check. The annotation does not open a human approval UI by itself.

### HTTP controls and transforms

| Output | Effect |
| --- | --- |
| `ApplyTransform(idx, transform_id)` | Requires the configured credential/header/query transform on an authorized HTTP proxy request. |
| `DenyUnauthorized(idx)` | Explicitly reject an unauthorized HTTP proxy request. |
| `AllowPassthrough(idx)` | Allows the HTTP proxy to forward an unauthorized request unchanged, without transforms. |

Tool-call checks do not honor HTTP passthrough. A caller that receives required
transforms must apply them or refuse to dispatch; the LangChain and ADK tool
adapters refuse transforms they cannot apply. See [configuration](/configuration/)
for the HTTP path and credential configuration.

## JSON functors

A functor is called with `@name(...)` inside a rule. These functions inspect JSON
strings; they do not fetch additional observations.

| Functor | Result |
| --- | --- |
| `@json_get_str(json, field)`, `@json_get_int(json, field)` | A top-level string or integer; `""` or -1 on a miss. |
| `@json_has_field(json, field)` | 1 when the top-level field is present, otherwise 0. |
| `@json_get_str_path(json, path)`, `@json_get_int_path(json, path)` | Nested string or integer using dotted paths and array indices; the same miss values apply. |
| `@json_array_len(json, path)` | Length of the array at a path. |
| `@json_array_get_str(json, path, i)` | String element of an array. |
| `@json_array_get_field_str(json, path, i, field)` | A string field of an array's object element. |
| `@json_object_has_key(json, path, key)` | 1 if the object at the path has this key, otherwise 0. |
| `@json_object_field_int(json, base, key, sub)` | Integer at the nested path assembled from `base`, dynamic `key`, and `sub`. |

For example, `@json_get_str_path(args, "destination.account")` reads a nested
account field, and `@json_get_str_path(result, "items[0].name")` reads an array
entry. Missing or wrongly typed values use the function's miss value; they are
not a positive observation. Do not treat an empty string or -1 as evidence of a
successful lookup when the rule requires one.

Paths split on `.` and `[`; colons in metadata keys need no escaping. A key
containing `.` or `[` cannot be addressed by a path. At the top level, use
`@json_get_str(json, "<whole key>")` to read such a string field.

## Text, URL, time, and semantic functors

| Functor | Result |
| --- | --- |
| `@str_contains(s, sub)`, `@str_starts_with(s, prefix)` | 1 for a match, otherwise 0. |
| `@str_to_lower(s)`, `@str_len(s)` | Lowercased text or its length. |
| `@url_host(url)` | The URL's host. |
| `@registrable_domain(s)` | The registrable domain of the first URL or host, or `""`. |
| `@count_urls(s)` | Number of URL/host fetch targets; may overcount. |
| `@hours_between(t1, t2)` | Absolute whole hours between ISO-style timestamps; -1 if either fails to parse. |
| `@llm_check_fn(prompt, context)` | Optional LLM yes/no predicate, returning 1 or 0. Requires operator configuration; it is not an independent source of recorded facts. |

The compiled backend supplies the configured LLM predicate. The shipped
interpreted functor library does not make live LLM requests and returns 0 on an
uncached input. Choose the backend deliberately when a policy depends on it.

Shell-command and Unicode helpers include `@cmd_invokes`, `@cmd_runs_sub`,
`@cmd_egress_class`, `@normalize_cmd`, `@dangerous_rm`, `@has_deceptive_unicode`,
and `@deceptive_unicode_class`. Their signatures and classification values are
commented in the common schema. Use [custom functors](/policies/extensions/#custom-functors)
when a required operation is not available here.
