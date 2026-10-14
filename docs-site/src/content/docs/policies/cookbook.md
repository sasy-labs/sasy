---
title: Policy cookbook
description: Recipes for tool arguments, ancestry, approvals, roles, and unknown-origin input.
---


These recipes use the [core relations](/policies/core-relations/) and
[built-in functions](/policies/reference/). They are rule fragments: combine
them with the rest of your policy's allow and denial rules. No allowlist match
means an action is not authorized. [Validate and test](/policies/validation/)
the resulting policy against the observations your integration records.

## Allow a fixed set of tools

```prolog
IsAuthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "read_public").

IsAuthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "draft_summary").
```

## Allow a tool only for one argument value

```prolog
IsAuthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "publish"),
    a = $CallTool(_, args),
    @json_get_str(args, "destination") = "internal".
```

## Deny when the ancestry contains a confidential read

The rule fires even though a drafting tool rewrote the text in between, because it follows recorded
dependencies rather than searching the output for secret words.

```prolog
.decl ConfidentialInput()
ConfidentialInput() :- CurrentDepends(id), ToolResult(id, "read_confidential", _).

// @deny_message: External publication depends on confidential information
// @suggestion: Publish internally or start a separate task using public inputs
Unauthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "publish"),
    a = $CallTool(_, args),
    @json_get_str(args, "destination") = "external",
    ConfidentialInput().
```

## Deny a combination of sources

Either source alone does not trigger this denial; together they do.

```prolog
.decl ReadSource(name: symbol)
ReadSource(name) :-
    CurrentDepends(id),
    ToolResult(id, "read_document", args),
    name = @json_get_str(args, "name").

// @deny_message: This summary depends on both untrusted and sensitive documents
Unauthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "send_summary"),
    ReadSource("external-note"),
    ReadSource("sensitive-report").
```

## Decide on a field the text does not show

A node's text is what the model
read; the rest of the message is in its metadata. Here a LangChain message whose
`additional_kwargs` carries a marker refuses the publication, although the text
is unremarkable.

```prolog
.decl MarkedDependency(id: symbol)
MarkedDependency(id) :-
    CurrentDepends(id),
    MessageMetadata(id, md),
    @json_get_str_path(md, "s:additional_kwargs.s:marker") = "exfil".

// @deny_message: A message this publication depends on is marked in its metadata
Unauthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "publish"),
    MarkedDependency(_).
```

This recipe inspects ancestors. Add a corresponding `Current(id)` rule when
direct inputs must also be checked. A message without metadata does not match.
See [LangChain records](/reference/langchain-records/) for field paths and
[extension points](/policies/extensions/#message-metadata) for metadata producers.

## Require a prior approval bound to the same arguments

Because `ToolResult` exists only for a call that actually ran and returned without error, a denied or
failed approval cannot satisfy this.

```prolog
.decl ApprovedPayment(request_id: symbol, payee: symbol, amount: symbol)
ApprovedPayment(request_id, payee, amount) :-
    CurrentDepends(approval),
    ToolResult(approval, "approve_payment", approved_args),
    SentMessage(approval, approved_message),
    approved_message.agent = "reviewer",
    request_id = @json_get_str(approved_args, "request_id"),
    payee = @json_get_str(approved_args, "payee"),
    amount = @json_get_str(approved_args, "amount").

IsAuthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "disburse_payment"),
    a = $CallTool(_, args),
    ApprovedPayment(@json_get_str(args, "request_id"),
                    @json_get_str(args, "payee"), @json_get_str(args, "amount")).
```

The `reviewer` condition matches the integration's agent label. If the approval
must come from a particular authenticated writer, also check the result message's
`principal` field; agent labels are not authenticated identity.

## Allow a tool only for a role

The role comes from the engine's authentication configuration, not from the conversation.

```prolog
IsAuthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "disburse_payment"),
    HasRole("treasury").
```

## Allow model traffic to one provider host

Only relevant when the HTTP hooks are enabled (`sasy.instrument(http=True)`); without a rule like this, an
outbound request is denied.

```prolog
IsAuthorized(idx) :-
    Actions(idx, a),
    QueriesHost(a, "api.openai.com").
```

## Treat input of unknown origin conservatively

An adapter records a value it consumed but cannot attribute — a message handed back without its mark, state
another process wrote, a value from a different SASY session — as a message of
unknown origin, and carries on. `DependsOnUnattributedInput()` matches
a recognized unknown-origin marker in a direct input or ancestor.

```prolog
CurrentDependsPolicyRelevant() :- IsTool(_, "publish").

// @deny_message: This action depends on input whose origin was not recorded
Unauthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "publish"),
    DependsOnUnattributedInput().
```

To act only on values that came from another session, read the label instead.
It says where the adapter placed the value; it is not a claim that the value is
trustworthy.

```prolog
// @deny_message: This action depends on a value from another SASY session
Unauthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "publish"),
    UnattributedInputOrigin(_, "another SASY session").
```

Note that the gate rule above names `publish`. A rule that reads the ancestry
for some other tool needs that tool in the gate too, or it never fires:

```prolog
CurrentDependsPolicyRelevant() :- IsTool(_, "publish").
CurrentDependsPolicyRelevant() :- IsTool(_, "send_email").
```

The common helper reads the unprefixed provenance marker. For LangChain, add
its [adapter-specific rules](/integrations/langchain/#inputs-of-unknown-origin).
See [dependency relevance](/policies/relevance/) before narrowing either gate.

Such a message never carries successful-tool evidence, so a rule that requires a
`ToolResult` already refuses it. These relations are for the other direction:
noticing that something unattributed reached the action at all.

## Request human approval

Use `// @ask` on a denial rule to return an approval request. The application
must handle that result before an action can run; supplied adapters block it.
See [approval requests](/policies/reference/#approval-requests) for the notation
and its interaction with allowlist matches and hard blocks.
