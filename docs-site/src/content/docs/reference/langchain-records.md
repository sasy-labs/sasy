---
title: LangChain record encoding
description: Canonical message records, metadata, and version identity.
---

This reference describes the adapter’s message representation. For setup and
supported execution, use the [LangChain guide](/integrations/langchain/).

## Record representation

A recorded node has two parts, and they answer two different questions.

Its **text** is what the message says, and it is what a policy matches on:
string content as it stands, and structured content — a list of content blocks —
written as the canonical JSON object described below, since a list has no other
faithful spelling as one string.

Beside it, the node's **metadata** is the message it records, whole, as one
canonical JSON object: content and its type, message type, name, the LangChain
`id`, tool calls parsed and unparsed, tool-result status and `tool_call_id`,
every `additional_kwargs` entry and all response metadata. Metadata is a field
of the recorded event, like the text and the role; the engine stores it
verbatim, never interprets it, and — this is the point of it — the content hash
the engine takes over a node covers it, so an edit anywhere in the message
changes the node.

**A policy can read the metadata.** The engine offers it as
`MessageMetadata(id, metadata)` — a relation keyed by message id, with a row
only for a message recorded with metadata. Match on the text for what the model
read; reach for the metadata when what decides the call is a field the text does
not show, such as an `additional_kwargs` entry:

```prolog
Unauthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "publish"),
    CurrentDepends(id),
    MessageMetadata(id, md),
    @json_get_str_path(md, "s:additional_kwargs.s:marker") = "exfil".
```

The `s:` prefixes are part of the encoding described below. See
[the policy language reference](/policy-language/) for the relation itself.

Two messages can therefore share one text — two assistant messages saying the
same thing with different `additional_kwargs`, say. That is not a collision:
their metadata differs, so they are two nodes with two marks.

Every key in the metadata object carries a two-part prefix naming the kind of key it
was — `s:` for a string, `i:` for an integer, `b:` for bytes — and a value JSON
cannot express on its own is written as a one-key object, one name per type:
`{"bytes": "<base64>"}`, `{"bytearray": "<base64>"}`, `{"tuple": [...]}`,
`{"set": [...]}`, `{"frozenset": [...]}`. This is not decoration. Pydantic's
JSON dump is lossy in exactly the places that matter: it writes `b"k"` and `"k"`
as the same key, a tuple and a set as the same list, a UUID or an enum member as
its string form, and every dict key as a string, so `{1: x}` and `{"1": x}` came
out alike. Each of those was two messages sharing one node and one mark — and a
bytes key is not a curiosity here, because `additional_kwargs[b"__openai_role__"]`
and `additional_kwargs["__openai_role__"]` are the difference between a system
message the pinned client transmits as `system` and one it transmits as
`developer`. Because every genuine key carries a prefix and no wrapper name
contains one, no message can be written to imitate a wrapper.

A value of a type the encoding does not know — a `datetime`, a `Decimal`, a
`UUID`, any object of your own — **stops the run** with an error naming the
type, rather than being written down as a string. That is the deliberate
boundary: an unanticipated type fails closed instead of quietly colliding with
something else. Convert such a value before the message is recorded.

The boundary is drawn on the exact type of each value in the message's Pydantic
dump, not on what the value is an instance of. The dump keeps a **subclass of a
scalar type** as it is, so that subclass is refused too: a string enum
(`class Colour(str, Enum)`) is a `str` and an `IntEnum` member is an `int`, so
either would otherwise be written as the plain value it carries — which is the
plain value's node, and so the plain value's mark. Both stop the run and name
the class instead. Pass `member.value` if you mean the plain value.

A subclass of a container is different, because it never reaches the encoder:
Pydantic's dump turns a `list`, `dict` or `tuple` subclass (an `OrderedDict`,
say) into the plain container with the same contents. It is recorded as that
plain container, and nothing is lost by it, since the contents are what the
record holds. An enum built on a container (`class Kind(tuple, Enum)`) is the
exception, because the dump keeps the member itself rather than normalizing it:
it reaches the encoder as a type the encoder does not know, and is refused by
name like any other enum member.

One kind of string is refused as well: text that is not valid Unicode, meaning a
string holding an unpaired surrogate such as `chr(0xD83D) + chr(0xDE00)`. JSON
writes that pair exactly as it writes `"\U0001F600"`, the character the pair
stands for, so the two strings would share one node. No provider can be sent
such a string, so it stops the run rather than being recorded as the character
it resembles.

The metadata holds all of it for one reason: so that the node is decided by the
message and nothing else. The text, role, agent and tool entries beside it are a
projection for policies to read, and none of them is reversible — two messages
can say the same thing, two message types share one role, an absent name and the
literal name `langchain` give one agent, and the recorded tool list is the single
source the client sends rather than every place a call can hide. A field left out
of the whole recorded event would therefore be a field a message could be edited
in without changing its recorded node, which is the same as saying its mark would
still verify after the edit.

Naming the fields a provider reads and recording only those was the earlier
design, and it cannot be made to hold. In the pinned `langchain-openai` alone,
the request path reads `name`, `tool_calls`, `function_call`, `audio` and
`__openai_role__` out of `additional_kwargs` on the chat-completions path,
several more (`tool_outputs`, `refusal`, `acknowledged_safety_checks`,
`__openai_function_call_ids__`, a `computer_call_output` type tag) on the
responses path, and `response_metadata["id"]` to decide which messages are sent
at all. Any such list is a list of the fields an edit *cannot* hide in. The
adapter keeps none, and a new provider key is covered the day it appears.

