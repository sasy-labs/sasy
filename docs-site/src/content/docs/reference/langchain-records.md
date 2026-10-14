---
title: LangChain records for policies
description: Read LangChain message content, tool evidence, and metadata in policies.
---

Use this reference when a policy needs fields from a LangChain message. For setup
and supported execution, see the [LangChain guide](/integrations/langchain/).

## Record representation

| Policy data | What it contains |
| --- | --- |
| `SentMessage(id, msg)` → `msg.contents` | The message's string content, or structured content blocks encoded as JSON using the representation below. |
| Tool-call arguments | Ordinary JSON with the tool's argument names, without metadata key prefixes. |
| `ToolResult(id, tool, args)` | Evidence that a guarded tool ran successfully, with its name and arguments. Supplied historical tool messages do not establish this evidence. |
| `MessageMetadata(id, md)` | The source message as JSON, including content, message type, name, LangChain ID, tool calls, `additional_kwargs`, and response metadata. |

Use message contents and tool evidence for ordinary content and action rules.
Use metadata for fields that message contents do not show.

## Read message metadata

String dictionary keys have an `s:` prefix at every level. For example,
`additional_kwargs={"marker": "exfil"}` appears in metadata as
`{"s:additional_kwargs": {"s:marker": "exfil"}, ...}`.

| Message field | Metadata path |
| --- | --- |
| Message type, such as `human` or `ai` | `s:type` |
| LangChain message ID | `s:id` |
| Message name | `s:name` |
| An `additional_kwargs` entry | `s:additional_kwargs.s:<key>` |
| A response metadata entry | `s:response_metadata.s:<key>` |

This block denies `publish` when a current input or any ancestor carries the
`exfil` marker. Add it alongside your ordinary allow rules:

```prolog
CurrentDependsPolicyRelevant() :- IsTool(_, "publish").

.decl MarkedMessage(id: symbol)
MarkedMessage(id) :-
    Current(id),
    MessageMetadata(id, md),
    @json_get_str_path(md, "s:additional_kwargs.s:marker") = "exfil".
MarkedMessage(id) :-
    CurrentDepends(id),
    MessageMetadata(id, md),
    @json_get_str_path(md, "s:additional_kwargs.s:marker") = "exfil".

Unauthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "publish"),
    MarkedMessage(_).
```

`CurrentDepends` covers ancestors; `Current` covers the direct inputs. A message
without metadata has no `MessageMetadata` row and does not match these rules.
See [core relations](/policies/core-relations/) and
[JSON functions](/policies/reference/#json-functors) for query details,
and [inputs of unknown origin](/integrations/langchain/#inputs-of-unknown-origin)
for rules that recognize messages whose provenance could not be established.

## Value representation and limits

Strings, numbers, booleans, nulls, and lists retain their JSON shapes. Dictionary
keys are prefixed by type: `s:` for strings, `i:` for integers, and `b:` for
base64-encoded bytes. Other supported Python values use these wrappers:

| Python value | JSON representation |
| --- | --- |
| `bytes`, `bytearray` | `{"bytes": "<base64>"}`, `{"bytearray": "<base64>"}` |
| `tuple` | `{"tuple": [...]}` |
| `set`, `frozenset` | `{"set": [...]}`, `{"frozenset": [...]}`; elements are sorted by their encoding. |
| `OrderedDict` | `{"ordered_dict": [["s:key", "value"], ...]}`; entries retain their order and keys retain their type prefixes. |

These shapes also apply inside structured message content. Convert unsupported
values, such as `datetime`, `Decimal`, UUIDs, or enum members, before recording;
values the adapter cannot represent stop the run. Non-finite numbers and invalid
Unicode also stop the run.

Metadata participates in snapshot identity, so an edit to it can invalidate a
carried message reference. Snapshots can differ across Pydantic versions; in
particular, older versions treat an `OrderedDict` value as a `dict`.
