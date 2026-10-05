---
title: LangChain agents
description: What the LangChain adapter checks, how to enable it, and what it does not cover.
---

Call `sasy.instrument()`, then build your agent with LangChain's
`create_agent`: SASY records the inputs behind the agent's tool calls and
enforces a policy before each tool runs. Start with the complete
[LangChain example](/examples/#langchain-information-flow), then use this guide for supported
configurations and multi-turn conversations. Custom LangGraph `StateGraph`
applications need [their own adapter](/instrumentation/). With the LangChain
adapter enabled, every tool call of a `ToolNode` that SASY did not build is
refused inside a SASY session
(see [tools SASY does not protect](#tools-sasy-does-not-protect)). A tool that
graph code calls itself, not through a `ToolNode` (for example the hand-written
tool node of the LangGraph quickstart), is not checked at all.

## At a glance

**What you get**

- Every tool call the agent makes is checked against your policy immediately
  before the tool function runs.
- The check sees the call's [ancestry](/concepts/): every message the requesting
  model output was computed from, through earlier tool results and drafting
  steps, across turns and across sub-agents called from tool bodies.
- A denied call does not run. The model receives an error tool message and can
  carry on.

**How to enable**

1. Follow the [quick start](/get-started/) to run the engine and install the SDK. Run your
   Python entry point with `python your_agent.py` in the environment where you
   installed the SDK and framework.
2. Call `sasy.instrument()` once at startup, after any explicit `sasy.configure(...)` and
   before you build the agent. When LangChain is installed, this makes
   LangChain's `create_agent` build SASY's agent.
3. Build the agent with `create_agent(model, tools)` from `langchain.agents`.
4. Run it inside `with sasy.session(policy=Path("policy.dl")):`.

That is the whole integration, and it is the same for every agent you build this
way. Your model's HTTP requests to its provider are not checked unless you also
pass `http=True`; see [model egress](#model-egress).

## Information-flow example

The [LangChain example](/examples/#langchain-information-flow) uses a scripted
model to read a document, draft a summary, and request publication. Documents
are synthetic and publication writes to an in-memory list; LangChain execution,
dependency recording, and policy checks are real.

In this example, we tighten the security policy from the [quick start](/get-started/):
confidential ancestry blocks external publication, even after a drafting step
rewrites the text. The [example policy](https://github.com/sasy-labs/sasy/blob/main/examples/langchain-information-flow/policy.dl)
contains this rule:

```prolog
.decl ConfidentialInput()
ConfidentialInput() :- CurrentDepends(id), ToolResult(id, "read_confidential", _).

// @deny_message: External publication depends on confidential information
Unauthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "publish"),
    a = $CallTool(_, args),
    @json_get_str(args, "destination") = "external",
    ConfidentialInput().
```

`CurrentDepends` follows the inputs behind the model output requesting
`publish`. It includes the confidential read through the later draft, so the
rule denies the external send. The policy allows internal publication and
external publication based only on public input.

## Try a live model

Both the single-agent and supervisor examples can let a live model choose their
tool calls.
Add `OPENAI_API_KEY=...` to the repository-root `.env`. The demo loads it without
overriding an exported value. The repository's `all` extra includes the
supported provider integration:

```bash
uv run examples/langchain-information-flow/demo.py --live
uv run examples/langchain-information-flow/supervisor.py --live
```

Live mode defaults to `gpt-4.1-mini`. Set `SASY_LIVE_MODEL` in `.env` or pass
`--live-model MODEL` to choose another model.

Only synthetic documents are used. A model may choose not to publish; inspect
the tool results to distinguish a denied request from one it never made.

Live mode enables HTTP checks. The example policy allows `api.openai.com`, and
the provider request, including its API key, passes through your engine. Use an
engine you trust and review that rule before using real confidential documents.
See [model egress](#model-egress) for the boundary and
[supported versions](#supported-versions) for provider restrictions.

## Run a supervisor with parallel workers

The [supervisor example](https://github.com/sasy-labs/sasy/blob/main/examples/langchain-information-flow/supervisor.py)
runs three native LangChain agents. By default, scripted models make the run
repeatable without a provider key. A supervisor delegates
research and a public review in parallel, then attempts to publish their combined
report. The same policy follows confidential information through the research
agent's answer and denies external publication before the simulated effect.
The public reviewer does not acquire its sibling's confidential dependencies.

With the engine and SDK environment configured as for the single-agent example:

```bash
uv run examples/langchain-information-flow/supervisor.py
uv run examples/langchain-information-flow/supervisor.py --public
uv run examples/langchain-information-flow/supervisor.py --async
```

The default confidential/external case is denied; `--public` or
`--destination internal` permits publication. Use `--live` for a real model in
all three agents, including their delegated research and review. This needs
`OPENAI_API_KEY` in `.env` and makes paid provider requests; the document reads
and publication remain simulated. Live model choices can vary between runs.

The console trace labels each agent's model output, tool call and result.

## Build an agent

Use the [SDK usage pattern](/sdk/#langchain) with your configured model and
tools. The [LangChain example](/examples/#langchain-information-flow) supplies a runnable scripted
example. `sasy.instrument()` enables SASY for each supported framework that is installed;
pass `langchain=True` to make a missing or unsupported LangChain an error
instead of a skipped framework (see [supported versions](#supported-versions)).

`sasy.instrumentation.langchain.create_agent(model, tools)` is the explicit,
legacy interface. It takes the same construction arguments and works without
`sasy.instrument()`, but returns a `SasyAgent` with `invoke`/`ainvoke` only.
The instrumented LangChain factory returns a native compiled graph.

Use `await agent.ainvoke(...)` for asynchronous execution. SASY enforcement
requires an active session, either an explicit `sasy.session(...)` or the
opt-in process default. The explicit `SasyAgent` interface raises outside a
session; instrumented native graphs retain native behavior outside one.
Each protected invocation is a separate run; to
continue a conversation, pass the message objects the previous invocation
returned back in. The caller owns and ends the SASY session; concurrent
invocations keep separate observation and dispatch state.

## What `sasy.instrument()` changes

`sasy.instrument()` changes LangChain in two ways, for the rest of the process.
Calling it again changes nothing further.

### `create_agent` builds a SASY agent

`langchain.agents.create_agent` (also importable from `langchain.agents.factory`)
is replaced by a version that builds the agent through SASY's adapter. Code that
ran `from langchain.agents import create_agent` before `sasy.instrument()` gets
the SASY version too: SASY updates every module-level name that refers to
LangChain's original function. A reference kept anywhere else, such as a class
attribute, a dictionary or a `functools.partial`, still holds the original, so
call `sasy.instrument()` before any code stores `create_agent`.

The instrumented factory returns LangChain's native `CompiledStateGraph`.
Agents can be built outside a session and invoked inside one. The following
restrictions apply to protected execution in an active SASY session; unscoped
execution retains native LangChain behavior.
Supported execution methods are `invoke`, `ainvoke`, `batch` and `abatch`.
`get_graph()` exposes its topology. `with_config()` preserves the guarded
invocation boundary; supported settings are `recursion_limit`,
`max_concurrency`, `tags`, `metadata`, `run_name` and `run_id`.

Each invocation, including each batch item, has its own dependency scope.
Checked delegation tools can use `batch` and `abatch`; child runs retain the
caller's ancestry, and their returned answers become dependencies of the tool
result. All items start from the ancestry available before the batch, so an
earlier item's answer does not become an input of a later sibling. Settings
bound with `with_config()` also apply to batches. Batch configuration can be
one supported config or a list of configs, one per input. Explicit callbacks
and checkpoint settings remain unsupported.
`return_exceptions=True` is refused before any batch item runs: failed-run
exceptions can contain data whose provenance is not yet tracked as a result.
The inherited `batch_as_completed` and `abatch_as_completed` methods are also
refused; use `batch` or `abatch` instead.
Returned messages carry verified provenance as described below. You can pass
`{"recursion_limit": 12}` as the second argument to `invoke`/`ainvoke`; the
previous `recursion_limit=12` spelling also works.

A native graph does not enable every LangGraph feature. Direct `stream` and
`astream`, embedding the compiled agent as a custom graph node, callbacks,
checkpoint configuration, and graph-structure edits are refused. Sub-agents
invoked from checked tool bodies are supported. Custom middleware, persistence,
state-based handoffs and Deep Agents are not supported yet.

The replacement takes four arguments:

- `model`: a chat model object (a `BaseChatModel` instance such as
  `ChatOpenAI(...)`), or a model name such as `"openai:gpt-5"`. A name is
  turned into a model object exactly as LangChain's own `create_agent` does it,
  with `langchain.chat_models.init_chat_model`, and the resulting model is then
  checked like any other (see [supported versions](#supported-versions) for the
  model configurations that are refused).
- `tools`: plain functions or `StructuredTool` instances.
- `system_prompt`: a string or a `SystemMessage`. LangChain turns a string into
  a `SystemMessage` with that content, so a string and a `SystemMessage` with
  the same content and no other fields are recorded identically. Any other
  field a `SystemMessage` carries (its own `name`, `id` and so on) is recorded
  as given. A `SystemMessage` is copied when the agent is built: changing
  your message object afterwards does not change what the agent sends, just as
  with a string. Any other type raises `ValueError`.
- `name`: the name of the LangGraph graph that LangChain compiles, as in
  LangChain's `create_agent`. LangChain also puts this name on every message
  the model produces, and SASY records those messages under that agent name
  (in place of the default `langchain`), so a policy that matches on agent
  names sees it.

Every other LangChain `create_agent` argument — `middleware`, `response_format`,
`state_schema`, `context_schema`, `checkpointer`, `store`, `interrupt_before`,
`interrupt_after`, `debug`, `cache` and `transformers` — raises
`ValueError` naming the argument when it is given anything other than its
default value, because the adapter cannot record what that feature does. The
same argument passed at its default (for example `checkpointer=None`) is
accepted. An argument LangChain's `create_agent` does not take at all raises
`TypeError`.

### Tools SASY does not protect

LangGraph runs an agent's tools through a `ToolNode`: the step of the graph that
calls the tool the model asked for. When the LangChain adapter is enabled (you
did not pass `langchain=False` to `sasy.instrument()`), one rule applies inside
a `with sasy.session(...)` or `with sasy.global_session(...)` block: **tools run
only in agents SASY builds.** Every tool call of a `ToolNode` that SASY did not
build is refused with `InstrumentationError` (from
`sasy.instrumentation.langchain`) before any tool-call wrapper or tool body
runs. It does not
matter which tool the model asked for, whether the node knows that tool's name,
whether the tool itself is one SASY wrapped, or whether the node has a
tool-call wrapper (`wrap_tool_call` or `awrap_tool_call`, which is how
middleware reaches it). The error stops the run: LangGraph's
`handle_tool_errors` setting does not turn it into a tool message.

"Agents SASY builds" means the agent that LangChain's `create_agent` returns
after `sasy.instrument()`, and the one `sasy.instrumentation.langchain.create_agent`
returns. Their `ToolNode` runs every call through SASY's own middleware, which
checks it against your policy and reports a tool name the model invented back
to the model without running anything.

This catches an agent built without SASY at its first tool call, instead of
letting it act unchecked. Such agents include one built with LangChain's
original `create_agent` held from before `sasy.instrument()`, one built with
LangGraph's `create_react_agent`, and your own `StateGraph` with a `ToolNode`.
None of them can run tools inside a SASY session; custom LangGraph graphs are
not supported inside a session yet. Build the agent with `create_agent` after
`sasy.instrument()` instead.

If you write [your own adapter](/instrumentation/) for a graph that runs tools
through a `ToolNode`, pass `langchain=False` to `sasy.instrument()`; the check is
then not installed, and `create_agent` is not replaced either.

What the check does not cover:

- Outside a SASY session nothing is refused, and an agent built without SASY
  runs its tools unchecked.
- A tool that your own graph code calls directly, not through a `ToolNode`, is
  not checked. This includes the hand-written tool node of the LangGraph
  quickstart, which calls `tool.invoke` itself, and functional-API (`@task`)
  code that calls a tool.
- The check needs the `langchain` package. With LangGraph installed but not
  `langchain`, the adapter cannot load. `sasy.instrument()` then skips it, so
  `ToolNode` tools run unchecked, and the first LangGraph graph the
  application runs issues a `SasyInstrumentationWarning` saying that LangChain
  and LangGraph agents will NOT be checked; with `langchain=True`,
  `sasy.instrument()` raises an error instead. Install the `langchain` extra, or pass
  `langchain=False` if the application does not use LangChain or LangGraph.

## Compatibility

**Supported scope.** The adapter covers the agent that `create_agent` returns
after `sasy.instrument()`, or that `sasy.instrumentation.langchain.create_agent`
returns. Unsupported configurations are refused; external execution paths need
their own checks:

- Custom middleware, callbacks, checkpointers, streaming delivery, dynamic
  tools, injected state, runtime or store arguments, custom `BaseTool`
  subclasses, state-mutating `Command` results and arbitrary LangGraph nodes.
- Tools declared `return_direct=True`, which end the run on their own result.
  The run would then have no final model answer, and the text a caller reads
  back would not depend on the tool result it came from. Setup refuses them.
- Tools whose arguments are Pydantic models, and tools taking a
  `RunnableConfig`, runtime, store or callbacks parameter. Setup refuses them.
- Anything the model provider does on its own side: provider web search, code
  execution, provider-side tools.
- Calling your original tool objects directly. Only the agent `create_agent`
  returns is guarded.
- The model's own HTTP request to its provider, unless you pass `http=True` to
  `sasy.instrument()` (see [Model egress](#model-egress)).

**Tool outcomes**

- When a call is denied, the model receives an error tool message that names the
  tool and carries your policy's denial reasons and suggestions. Write them for
  the model to read, and keep out of them anything the model should not see.
- A denial inside a sub-agent does not fail the outer delegation. The inner
  agent normally answers anyway, and the outer tool result looks successful.
- A tool that returns the string `"Error: ..."` without raising counts as a
  successful result, and so produces `ToolResult` evidence.

### Model egress

The adapter checks tools. The HTTP request your model client sends to its
provider (its "egress") is not checked by default. SASY's HTTP hooks check it
when you enable them with `sasy.instrument(http=True)`; they are off by default.
With them on, requests made with `httpx` or `requests` go through the engine,
which checks each one against the policy and can inject credentials (add an API
key the engine holds to the request). The engine then sees each request,
including your provider API key, so enable the hooks only with an engine you
operate. Your policy must then allow the host, or the model's request is
denied:

```prolog
IsAuthorized(idx) :-
    Actions(idx, a),
    QueriesHost(a, "api.openai.com").
```

## What is recorded and enforced

Model outputs depend on every actual input message, including the system
message. Structured message content and tool-call specifications are preserved.
A tool result depends on the model output that requested it. Later model calls
consume those results, preserving information flow through multiple processing
steps. The adapter does not add dependencies merely because another message is
available in a session.

Tools are ordinary functions or `StructuredTool` instances. Authorization sees
the arguments after Pydantic validation, coercion, and function defaults. The
adapter checks and dispatches the same detached argument objects, preventing an
external mutation during an asynchronous check from changing what runs.

Denials and unsupported required transforms produce error tool messages without
executing the function. (A transform is a change the policy requires before an
action runs, such as credential injection; the adapter cannot apply one to a
tool call, so a policy that requires one for a tool call is reported as an
error. The reference monitor applies transforms to HTTP requests routed by
`sasy.instrument(http=True)`.) Recording failures and reference-monitor failures stop
the run. Error results do not emit successful `ToolResult` provenance. Successful
results include the tool name and normalized dispatch arguments.

Within one `invoke`, a LangChain message ID identifies one message. Across
invocations it does not: a message in a new `invoke` is treated as new even if
it reuses an old LangChain ID, unless it is one of the message objects a
previous `invoke` returned (see Multi-turn conversations below). Before each model call, the
adapter checks the actual consumed messages and resolves them in one batch:
compact references for known generated messages, full snapshots for new
observations and for user and system messages. It uses the returned
immutable version IDs. Repeated unchanged snapshots keep their IDs without graph
or synchronization writes. Version comparison covers the whole message, because
[the recorded node holds the whole message](#what-a-node-holds): content and its
type, role, name, the LangChain `id`, tool calls parsed and unparsed,
tool-result status and `tool_call_id`, every `additional_kwargs` entry and all
response metadata. Two different messages therefore never produce one recorded
node, and no edit to a message leaves the node it was recorded as unchanged.
Changes to the fields outside the graph's role/name/content/tool-call projection
are additionally reported as such before the round trip, rather than only as a
rejected version.

Generated messages re-observed as inputs must still match their recorded
computation. Unexplained content edits and changes to recorded tool-call
specifications are rejected. Existing tool results retain their actual dispatch
inputs and arguments. Server version identity survives retries and restart, and
a message's identity travels with the message object itself, as described under
[carrying messages between invocations](#carrying-messages-between-invocations).

Model observation alone is not network mediation. SASY's HTTP hooks (off by
default; enabled with `sasy.instrument(http=True)`) cover supported
`httpx`/`requests` provider paths; authorize the specific endpoint. Custom transports need their own
dispatch checks. Localhost
HTTP bypass behavior also applies. This adapter does not redact streamed tokens.

## Carrying messages between invocations

When an invocation ends, the adapter marks each message it recorded: it writes
that message's graph node, and the alias the node was recorded under, into the
message's `response_metadata` under the reserved key `sasy`. A mark is
bookkeeping, not content. It is removed from the copy of the conversation handed
to the model, so no provider ever receives it, and it is ignored when the
adapter compares auxiliary message fields.

Pass those same message objects back — as `result["messages"]` plus a new user
message, carried through your own state, or restored from a store your
application keeps — and the next invocation restores each message's identity. It
sends the mark to the engine as a compact reference: a short form that names an
existing immutable version instead of recording the message again. The engine
accepts it only if that version exists for that alias, in this session, for this
principal, and its stored content hash equals the hash of the content the
adapter just computed. Multi-turn chat therefore keeps its history: a tool result
read in turn one is still an ancestor of an action attempted in turn three.

A mark that does not verify buys nothing. An invented node, a mark copied onto
different text, an edited message, a mark taken from another session, and a tool
result whose mark claims a tool that never ran all fail the check, because to
pass it the message would have to *be* the recorded message. What the engine
hashes is [the recorded node](#what-a-node-holds), which holds the whole
message, plus its role, agent and tool entries, so changing anything about a
recorded message makes its mark fail — including a tool result's `tool_call_id`,
the message's own `id`, and an `additional_kwargs` entry no released provider
client has ever read. Nothing about a message is outside the hash. Tool calls
carried on an admitted assistant message are history: this run will not dispatch
them.

Identity is checked on top of that, not instead of it: the engine accepts a mark
only as a version of the alias that mark names, so a message cannot inherit
another message's node by taking its id either.

A message is recorded as being of unknown origin only when the engine checks
its mark and rejects it. If the engine cannot be reached, times out or errors,
nothing has been decided, so the invocation stops with an error rather than
carrying on with ancestry it simply failed to look up. The adapter reads the
engine's `INVALID_ARGUMENT` answer as the rejection. The engine also returns
that status for a few transport-level problems with the request itself, such as
a text containing a NUL byte, so a message rejected for one of those is recorded
as being of unknown origin rather than stopping the run.

### What a node holds

The adapter records message content for policies and a canonical snapshot in
metadata so the engine can detect changes to the source message. See
[record encoding](/reference/langchain-records/) for the field representation,
supported value types, and version-comparison rules.

### Inputs of unknown origin

Where the adapter cannot establish where a message came from, it continues and
says so instead of refusing. An assistant or tool message with no mark, or with
a mark that does not verify, is recorded as a node of its own holding a JSON
object `{"s:langchain_message": ..., "s:provenance": "unattributed external
input", "s:value": ...}` — the same encoding as above, prefixes and all — with
the user role, no ancestors and no tool evidence. Its metadata is the whole
message, exactly as for a recorded message.
`langchain_message` is the whole message, as above; `value` repeats its content
so a policy can read what the run consumed without unpacking the message. The
model call that reads it depends on that node, so a policy can see that the run
consumed text of unknown origin and decide what to allow. A supplied tool result
is never evidence that the tool ran.

The `s:provenance` key is what a policy reads to tell this text from the content of
a recorded message. The two shapes also cannot produce one node, whatever a
message's content is made to spell: the metadata is the message either way, and
these two events carry different texts for the same message. So an argument value
the model can read back — a secret pasted into a historical search query, say —
is text a policy can match on either way, not something only the provider sees.

The common `UnattributedInput` helper reads the unprefixed `provenance` key.
For LangChain’s canonical envelope, add these rules to your policy before using
`DependsOnUnattributedInput()`:

```prolog
CurrentDependsPolicyRelevant() :- IsTool(_, "publish").

UnattributedInput(id) :-
    Current(id),
    SentMessage(id, msg),
    @json_get_str(msg.contents, "s:provenance") = "unattributed external input".
UnattributedInput(id) :-
    CurrentDepends(id),
    SentMessage(id, msg),
    @json_get_str(msg.contents, "s:provenance") = "unattributed external input".

Unauthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "publish"),
    DependsOnUnattributedInput().
```

These rules cover both direct inputs and ancestors. Change `publish` to the
sink your policy protects, and retain its ordinary allow rules. The generic
helper alone does not recognize LangChain’s prefixed marker. This
canonical envelope has no origin field, so match its provenance marker rather
than relying on `UnattributedInputOrigin` to identify it.

Which tool calls a message records is the rule its provider client applies, not
the sum of the places a call can hide. `langchain-openai` uses one source and
ignores the rest: the parsed and the unparseable tool calls together when the
message carries any, otherwise the raw `additional_kwargs["tool_calls"]`,
otherwise the legacy `function_call`. The recorded tool list follows that rule,
so a message carrying both a parsed call and a raw one records the parsed call
alone — the call the provider is actually asked to make. The ignored source is
still part of the recorded content, where adding it breaks the message's mark;
it is simply not a requested tool.

Two hand-offs still lose provenance. **Flattened text**: if you carry only the
text of an answer (`str(result["messages"][-1].content)`) into a new
`HumanMessage`, there is nothing to verify, and it is recorded as an ordinary
fresh user input with no ancestors. **Rewritten messages**: shortening a
message's content, or rebuilding it without its `response_metadata`, invalidates
or drops its mark, and the message becomes an unattributed external input.
Whole messages dropped by `trim_messages` or `RemoveMessage` change nothing for
the messages that remain.

## Supported versions

The adapter is tested against, and the extra pins, LangChain 1.4.0,
langchain-core 1.6.3, LangGraph 1.2.11 and langgraph-prebuilt 1.1.0. Setup
refuses a different version rather than silently losing a hook it needs. When
LangChain is installed at another version, `sasy.instrument()` skips the
adapter. The first time the application then calls
`langchain.agents.create_agent` or runs a LangGraph graph (`invoke`, `ainvoke`,
`stream` or `astream`), SASY issues a `SasyInstrumentationWarning` (a Python
warning, also logged, once per process) saying that LangChain agents will NOT
be checked by SASY. An application that has LangChain installed but never uses
it gets no warning. Pass `langchain=True` to make this an error, raised before
anything is changed, or `langchain=False` if your application does not use
LangChain, which also silences the warning. The example's live mode uses
langchain-openai 1.6.2.

The factory intentionally does not accept custom middleware, callbacks, injected
state/runtime/store arguments, dynamic tools, checkpointers, streaming delivery,
or arbitrary graph nodes. It rejects
custom `BaseTool` subclasses, tools declared `return_direct=True` and
state-mutating `Command` results. These paths
need additional observation and dispatch contracts before they can be supported.
Provider-retained conversations, previous-response IDs, remote prompt references,
automatic previous-response reuse, and raw provider body overrides are rejected.
Their hidden inputs cannot be resolved into the observed graph by this adapter.
Model configuration is checked both at setup and before each model dispatch.

A tool must consume only the arguments it was given. If a tool reads a file, a
database or another message on its own, record those inputs yourself or the
policy will not see them. Your tool code and your model implementation are
trusted code: SASY checks what a tool is asked to do, not what it then does.

### One invocation is the tracked unit

Dependencies are tracked inside a single `invoke` or `ainvoke`, and across the
delegations that an invocation starts itself. Five consequences follow:

- **Sub-agents called through tools are tracked.** A guarded agent may be
  invoked inside the body of a guarded tool of another guarded agent (the "agent
  as a tool" or supervisor pattern), in the same `sasy.session(...)`. Every model
  call of the inner agent is recorded as depending on the tool call that started
  it, whatever the inner agent was handed — a fresh question, a conversation
  restored from marks, or text of unknown origin — so everything the outer agent
  had read is in the inner check's ancestry: the set of messages an action was
  computed from. The tool result the outer agent then reads depends on the inner
  agent's final answer, so what the sub-agent found reaches the outer agent's
  later checks. A run always ends on a model answer, which is why a tool declared
  `return_direct=True` is refused at setup. Delegation nests up to
  eight levels deep; deeper delegation raises an error. Invoking a guarded agent
  anywhere else inside a run — with no guarded tool body around it — raises an
  error, because the inner input would have no observed cause.
- **One tool body is one computation.** Different tool calls made by the same
  model message (for example two parallel `delegate` calls) never share
  dependencies: one call's sub-agents contribute nothing to the other call's
  ancestry. Within a single tool body the rule is deliberately coarse: a body may
  run a small workflow, calling agent A and passing A's answer as plain text to
  agent B. Plain text carries no identity, so every inner run started by a body
  depends on the tool call plus every inner answer already returned to that body,
  and the tool's result depends on all of them. Timing decides this, not your code: a
  sub-agent depends on every sub-agent answer that had already come back to the
  same tool body when it started. Two sub-agents started together with
  `asyncio.gather` are independent. Started one after the other, the second
  depends on the first, even if you never passed the first answer to it. A tool is checked once,
  before its body runs. A body that delegates and then acts on the answer itself
  is not checked again for that action unless the action is its own checked
  call: an HTTP request through SASY's HTTP instrumentation, or a direct
  reference-monitor check. Those checks do include the answers already returned
  to the body. A check made *inside* a sub-agent belongs to that sub-agent's run
  instead, not to the body — including the HTTP request the sub-agent's model
  sends, which is checked with that run's own inputs and the tool call that
  started it. Prefer a separate guarded tool for the action, so the model's
  request for it is checked with the delegated answer in its ancestry. Two limits
  follow from Python's context variables, which carry the SASY session and the
  body's inputs. Work a body starts and leaves running is refused if it makes a
  check, or starts a sub-agent, after the body has returned, and a sub-agent
  still running when its body returns is stopped at its next model call or tool
  call. A tool call that has already passed its check when the body returns is
  not interrupted; it runs to completion and its result is then refused. Await
  the sub-agents a body starts before returning. Work a body hands
  to its own thread pool with a plain `executor.submit(fn)` has no session and no
  inputs at all. Use `asyncio.to_thread(fn)`, which copies the context, or
  capture the context in the body and run the function inside it:
  `executor.submit(contextvars.copy_context().run, fn)`.
- **Side channels need explicit registration.** The tool-body rule covers
  answers returned to the body that started the sub-agent. It does not cover an
  answer that leaves a body another way — a global variable, a file, a database
  or a cache — and is read by a different tool call or a later invocation. Inside
  the tool body that reads it, register the message objects you read:

  ```python
  from sasy.instrumentation.langchain import consume_messages

  def recall() -> str:
      """Return the answer kept from an earlier run."""
      consume_messages([saved_answer])   # a message returned by a guarded agent
      return str(saved_answer.content)
  ```

  A message whose mark verifies keeps the ancestry it already has. An assistant
  or tool message without a verified mark is recorded as an unattributed external
  input; a user or system message without one is recorded as a new input, whose
  ancestor is the tool call that started the run when this is a delegated run,
  and which has none at the top level. A message this run has already recorded
  keeps what it has, so handing back a copy that lost its `response_metadata`
  does not downgrade it. That holds as long as it is the same message: register
  a *different* message under an identity the run recorded — the same `id`,
  other text — and the new one is recorded as an unattributed external input of
  its own instead of being taken for the first. Nothing the body read is missing
  from the graph, and the new text inherits nothing the first message
  established. Registering the same replacement twice records it once. The
  identity the replacement is given is a fresh one the run mints, not a name
  derived from the message: a derived name is one that could be worked out and
  occupied in advance, and the run would then find the identity already taken.
  The call only adds dependencies, and only to the tool body that makes it.
- **An answer handed on as plain text loses its history.** If you pass an
  agent's answer into another invocation as the text of a new user message, SASY
  records it as a fresh user input with no ancestors; pass the message objects
  instead. A delegated run that opens a different `sasy.session(...)` is a
  partial case: sessions do not share message identities, so that run is
  recorded as independent work, its own input has no ancestors, and what the
  outer agent had read is not in its checks. Its answer is not lost, though. The
  calling tool body records the answer in its own session as an unattributed
  external input — the same node shape used for any message whose origin cannot
  be established — and the tool's result depends on that node. A policy sees
  that the result came from outside, with the answer's text, but not which tools
  produced it. Checks made inside that run, including the HTTP request its model
  sends, resolve against that run and its own session. Keep a rule that depends
  on tool evidence inside one session.
- **Later turns keep their history when messages are passed back.** An
  invocation accepts earlier assistant and tool messages. Each one is admitted
  with the ancestry it already has when its mark verifies, and recorded as an
  unattributed external input when it does not.

A denied or failed sub-agent run adds no successful-tool evidence: the denied
call is recorded as a failed tool result. The denial does not travel out of the
sub-agent, though. LangChain hands the inner agent an error tool message, the
inner agent normally carries on and answers anyway, and the outer agent then
receives an ordinary successful `delegate` result whose ancestry holds the
failed call rather than a successful one. If a delegation must fail when an
inner action is denied, check the delegating tool itself as well.

Pass the agent returned by `create_agent` to the rest of your application. The
factory guards copies of your tools, so the original tool objects you still hold
are not checked if you call them directly.

### Provider-side tools and context

The factory accepts any LangChain chat model. What a provider does on its own
side is out of scope: if a model or option makes the provider search the web,
run code or call its own tools, SASY does not check that action and its results
do not appear as dependencies of the model's answer.

One group of options is rejected, because it changes the input the model
receives after the adapter has recorded it: for the tested client,
`ChatOpenAI`, these are retained conversations, previous-response IDs, remote
prompts, replacement inputs or instructions, provider-side tool lists and raw
body overrides. The adapter does not know the equivalent options of other
provider clients; review them yourself.

### Tool shapes refused at setup

Arguments are recorded as JSON, so a tool that cannot present its arguments that
way is refused when you build the agent, with an error naming the tool and the
argument. Use JSON-native argument types: strings, numbers, booleans, lists and
dictionaries.

- An argument declared as a Pydantic model. A model instance has no JSON form in
  the check, so the call would stop the run when its arguments are serialized.
  Take the fields as arguments of their own, or a JSON string the tool parses.
- An injected-state argument, including the optional form
  `Optional[Annotated[int, InjectedState("key")]]` with a default. The optional
  form reaches the model as an ordinary parameter and fails validation at every
  call, so the body never runs and no check is ever made.

A tool that takes a `RunnableConfig`, `runtime`, `store`, `callbacks` or
`run_manager` parameter is refused at setup for the same reason: the adapter
checks the arguments the tool is actually given.

### A tool the model invented

If the model asks for a tool that does not exist, the framework has nothing to
run. Nothing runs and nothing is checked, and the model is told — `Error: no
tool named X is available.` — so it can correct itself, as it would without the
adapter. The failed result is recorded with no successful-tool evidence.

A dispatch that does not match the model output the adapter recorded is a
different matter and still stops the run.

A tool result counts as successful when LangChain marks it so. A tool that
returns the text "Error: ..." without raising is recorded as a successful
result. When a call is denied, the model is told which tool was denied, the
reasons the policy gave, and any suggestions it carried.

The [`examples/langchain-information-flow`](https://github.com/sasy-labs/sasy/tree/main/examples/langchain-information-flow) example includes allowed and denied
publication, transitive dependencies through a drafting tool, a deterministic
scripted model, and optional live model execution. All data and deliveries are
synthetic. Its policy explicitly permits the live provider to process that data.
