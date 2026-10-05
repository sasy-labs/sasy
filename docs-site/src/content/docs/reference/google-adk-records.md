---
title: ADK record encoding
description: Canonical message records, metadata, and version identity.
---

This reference describes the adapter’s message representation. For setup and
supported execution, use the [Google ADK guide](/integrations/google-adk/).

## Record representation

ADK groups the parts of one turn into a single `Content`, and the adapter
records one node per part. A node has two sides.

Its **text**, role and tool entries are the projection a policy matches on: a
text part's text, a tool call as a tool entry with its arguments as ordinary
JSON, and a tool result as the response value written as ordinary JSON. Those
keep the spelling the caller gave them, so a rule reading arguments with
`@json_get_str` reads what was dispatched.

Beside it, the node's **metadata** is the whole `google.genai` part the node
stands for — every field it carries, not only the ones the projection shows.
Metadata is a field of the recorded event, like the text and
the role. The engine stores it verbatim and never interprets it, and the content
hash the engine takes over a node covers it, so a change to any part of the
message changes the node. **A policy can read it**, as
`MessageMetadata(id, metadata)` — a relation keyed by message id, with a row
only for a message recorded with metadata. Its keys carry the type prefixes of
the canonical encoding described just below, so a rule reaches a field with a
prefixed path. A tool-call node, for instance, is recorded as
`{"s:function_call":{"s:args":{"s:amount":5},"s:id":"c1","s:name":"pay"}}`, and
`@json_get_str_path(md, "s:function_call.s:id")` reads the call id that the
node's tool entry — a name and its arguments — does not show. Write
rules against the text for what the model read, and against the metadata for a
field the text does not show.

The metadata is written with the same canonical encoder the [LangChain
adapter](/integrations/langchain/#what-a-node-holds) uses, and so are the
adapter's own comparisons — whether a message in this turn's request is the one
recorded last turn, and whether an event in the retained history is the event
that was observed. Both compare the canonical encoding of the part rather than
Pydantic's JSON dump, which is lossy in ways that would let two different
messages share one record: it writes bytes as base64, so `b"v"` and the string
`"dg=="` come out alike; a tuple and a set as one list; an int dict key and its
decimal string as one key.

The encoding covers JSON's own types plus bytes, bytearrays, tuples, sets and
frozensets, each with a wrapper of its own. Over those types it is exact: two
different values never share one record.

A tool's arguments and its return are arbitrary Python, and a result is recorded
only after the tool has run, so a value of a type the encoding does not know
does not stop the run: it is written under a wrapper naming that type's module
and qualified name, over the rendering Pydantic's JSON mode gives it. A tool
returning `{"when": datetime(2020, 1, 2)}` is recorded as
`{"s:when": {"py:datetime:datetime": "2020-01-02T00:00:00"}}`, which is not the
record of the string `"2020-01-02T00:00:00"` — that is `{"s:when":
"2020-01-02T00:00:00"}`. Only a value with no rendering at all — an object of
your own that Pydantic cannot serialize — stops the run, with an error naming
its type. Text that is not valid Unicode, a lone surrogate, is recorded under a
`surrogates` wrapper holding its bytes, because JSON escapes a lone surrogate
pair exactly as it escapes the character that pair stands for.

The wrapper is a weaker guarantee than the encoding's own types give, and it is
worth being precise about how much weaker. It tells the value apart from every
natively encoded value, and from the values of any type with a different module
and qualified name. It does **not** promise that two values of that one type
differ — a rendering keeps only what the renderer keeps — and it does not
promise that two classes built with the same module and qualified name differ,
which two calls to `Enum("Account", ..., type=str)` in one module produce.

Enum members are the exception the adapter makes exact, because they are common
in tool returns: a member is recorded as its member name alongside the record of
its value, so `{"account": Account.BLOCKED}` becomes
`{"s:account": {"py:yourmodule:Account": {"enum": "BLOCKED", "value":
"blocked"}}}`. That keeps it apart from the plain string `"blocked"`, and keeps
two members apart whose values a JSON dump would write alike. The module and the
qualified name are joined with a colon, which neither of them can contain, so
one wrapper name has one reading.

What the record guarantees: it covers every field of the part, and any two
values that differ as JSON data are recorded differently. Two values that are the
same as JSON and differ only in their Python type — a tuple and a list, say —
are told apart for the types listed above, and not otherwise.

That is safe because of what SASY trusts. Your agent's code is trusted; the data
it reads and what the model returns are not. Untrusted values arrive as text or
JSON, and JSON has no tuples, bytes, enum members or model instances, so only
your own code can create such a pair.

Two consequences worth knowing. A saved artifact's provenance — the claim tying a
stored version to the node that produced it — is matched on the version's JSON
text. And a Pydantic model or dataclass passed as a tool argument is recorded as
the dict it dumps to.

What a policy reads is not this encoding: the event's text and a tool entry's
arguments are the plain JSON dump of the value, under the field names the caller
gave them. A tool that returns a value `json.dumps` itself cannot write — a
`datetime` — therefore still stops the run when that text is recorded.

Three genai fields hold values of any type at all. `FunctionCall.args` and
`FunctionResponse.response` are inside the supported profile, and what is
recorded for them is exactly what this section describes: the plain JSON dump in
the node's tool arguments and text, and the canonical record — wrappers and all
— in the metadata, under `s:function_call.s:args` and
`s:function_response.s:response`. `Part.part_metadata` is the third, and a part
carrying it is outside the supported profile: the adapter refuses the whole turn
before it is observed, so no rule can read a `part_metadata` field. The same
holds for any other part field outside text, function calls, function results
and thought markers.

Events the adapter builds by hand for a state or artifact read carry metadata
like any other event: they go through the same recording path, as one part
holding one text, so their metadata is `{"s:text": "<that text>"}`. The text
already holds the resource, its provenance and its value, so the metadata shows
nothing the text does not. A replayed artifact-provenance claim rebuilds the
same metadata on the event it reconstructs, because the engine decides on the
whole event.

