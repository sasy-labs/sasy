---
title: Core policy relations
description: Actions, authorization, identity, and the recorded evidence a policy reads.
---

Start with the [small policy examples](/policy-language/). This page explains the
relations used in most policies. They are declared by the engine's common schema;
you do not need to declare them again.

## Proposed actions and decisions

| Relation | Source and meaning |
| --- | --- |
| `Actions(idx, action)` | Input: proposed actions in this check. `idx` is the zero-based action index. |
| `IsTool(action, name)` | Helper: the action is a tool call with this name. |
| `QueriesHost(action, host)` | Helper: the action is an HTTP request to this non-empty host. |
| `IsAuthorized(idx)` | Policy output: put this action on the allowlist. |
| `Unauthorized(idx)` | Policy output: block this action, even if it also has an allowlist match. |
| `Authorized(idx)` | Common-rule output: an authenticated principal exists, the action is allowlisted, and it is not hard-blocked. |

An action is a tagged value with one of these shapes:

| Action | Fields |
| --- | --- |
| `$CallTool(name, args)` | Tool name and ordinary JSON argument string. |
| `$HTTPRequest(url, body, headers_json)` | Request URL, body, and JSON headers. |
| `$SendAttempt(message)` | A proposed message, using the `Message` record described below. |

Bind the same `idx` from `Actions` in your decision rule. A check can contain more
than one action; a decision for index 0 says nothing about index 1.

Graph observations and session policy bindings are scoped to the authenticated
tenant. A session keeps its recorded context across checks; recording a fact does
not execute or authorize a tool. See [session configuration](/configuration/).

No allowlist match means the action is not authorized. `Unauthorized` wins over
an allowlist match. Approval requests and HTTP forwarding controls have additional
behavior described in [built-ins and feedback](/policies/reference/).

## Authenticated identity

| Relation | Meaning |
| --- | --- |
| `Principal(id)` | The authenticated caller of this check, stamped by the server. |
| `PrincipalRole(principal, role)` | Roles associated with that authenticated principal. |
| `HasPrincipal()` | True when this request has an authenticated principal. |
| `HasRole(role)` | True when that principal holds the named role. |
| `TenantId(id)` | Server-derived tenant for the request. |
| `Entity(id)` | A free-form actor label supplied by the caller; it is not authenticated identity. |

For example, require the authenticated caller's role:

```prolog
IsAuthorized(idx) :-
    Actions(idx, action),
    IsTool(action, "disburse_payment"),
    HasRole("treasury").
```

Message records also carry the principal that wrote them. A message's `agent`
label or `entity` field is not a substitute for authenticated identity. See
[authentication](/authentication/) for the source of principals, roles, and tenants.

## Context and dependency direction

| Relation | Meaning |
| --- | --- |
| `Edge(source, destination)` | The destination was computed from the source. |
| `Current(id)` | A recorded message supplied as a direct input of this check. |
| `CurrentDepends(id)` | An ancestor of a current message, following dependencies backward. |
| `SentMessage(id, message)` | The recorded message and its fields. |
| `ToolResult(id, tool, args)` | Successful-tool evidence attached to this recorded result, with the tool's name and arguments. |

An edge from a tool result to a summary means the summary depends on the result.
A check using that summary as a current input can therefore find the tool result
in `CurrentDepends` even if its text was rewritten along the way.

`CurrentDepends` does not include a node merely because it is current. To inspect
both direct inputs and ancestors, declare a helper:

```prolog
.decl InScope(id: symbol)
InScope(id) :- Current(id).
InScope(id) :- CurrentDepends(id).
```

These relations describe the graph the application or adapter recorded. Missing
observations do not prove that an event never happened. `!Relation(...)` tests
absence in the supplied facts and derived results, so a rule using absence needs
complete recording for the question it asks.

Dependency traversal can be restricted by the [relevance gates](/policies/relevance/).
An explicit gate must cover every action whose rules need ancestry.

## Message fields and tool evidence

A `Message` has these fields, available through the preprocessor's dot notation:

| Field | Meaning |
| --- | --- |
| `contents` | Recorded text; the adapter determines how structured content is represented. |
| `tools_json` | JSON list of tool calls carried by this message. |
| `agent` | Free-form agent or conversation label. |
| `agent_role` | Conversation role: `$Assistant()`, `$SystemRole()`, `$UserRole()`, or `$AgentType()`. |
| `entity` | Caller-supplied actor label on the recorded message. |
| `principal` | Server-stamped identity of the writer, or empty when unknown. |

For example, inspect the contents of a successful result:

```prolog
.decl ConfidentialRead(id: symbol)
ConfidentialRead(id) :-
    CurrentDepends(id),
    ToolResult(id, "read_document", _),
    SentMessage(id, message),
    @json_get_str(message.contents, "classification") = "confidential".
```

`ToolResult` establishes that a guarded call returned successfully; a proposed,
denied, failed, or supplied historical call is not that evidence. Its `args`
column holds the call arguments. Read the return value from the result message's
`contents`. The [LangChain](/reference/langchain-records/) and
[ADK](/reference/google-adk-records/) references describe their record shapes.

Continue with [built-ins and feedback](/policies/reference/) for the broader
relation and functor reference, or use the [cookbook](/policies/cookbook/) for
rules that combine these pieces.
