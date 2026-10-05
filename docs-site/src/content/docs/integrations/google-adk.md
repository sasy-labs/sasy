---
title: Google ADK
description: What the Google ADK adapter checks, how to enable it, and what it does not cover.
---

## At a glance

**What you get**

- Every function tool is checked against your policy immediately before its
  Python callable runs, with ADK's default argument values already filled in.
- Agent transfers, `AgentTool` children and native task children are checked
  before they start.
- The check sees the call's [ancestry](/concepts/) across agents, including
  values passed through session state (`output_key`, `{key}` templates,
  `tool_context.state`) and text artifacts.
- A denied tool does not run. The model receives an error result beginning with
  `[BLOCKED]`.

**How to enable**

1. Follow the [quick start](/get-started/) to run the engine and install the SDK. Run your
   Python entry point with `python your_agent.py` in the environment where you
   installed the SDK and framework. Google ADK
   **2.9.1** exactly is required.
2. Call `sasy.instrument()` once at startup. It enables the ADK adapter when
   Google ADK is installed.
3. Run your existing `Runner` inside `with sasy.session(policy=...):`, using
   `runner.run_async`.

Your agents, tools and models do not change, and the adapter is the same for
every ADK application. Outside an active SASY session, the patches pass through
to native ADK behavior. The supported paths and restrictions below apply while
SASY is active, including when you opt into a
[process-wide default session](/configuration/#session-lifetime-and-background-work).

## Run the example

[`examples/adk-separation-of-duties`](https://github.com/sasy-labs/sasy/tree/main/examples/adk-separation-of-duties)
runs three agents — a requester, a reviewer and a disburser — in an ADK
`Workflow` with scripted models, and allows the payment only when its ancestry
contains a matching approval. `main.py` there is the complete runnable file for
the outline below.

## Enable it

Configure the SASY endpoint and authentication as described in the
[local engine guide](/local-engine/), then enable the framework patches:

```python
import sasy

sasy.instrument()

# runner is an ordinary google.adk.runners.Runner.
with sasy.session(policy=policy_source):
    async for event in runner.run_async(
        user_id=user_id,
        session_id=adk_session_id,
        new_message=user_content,
    ):
        show(event)
```

`policy_source` is the policy to bind; `runner`, `user_id`, `adk_session_id` and
`user_content` are ADK's own values, and `show` stands for whatever your
application does with each event.

- Call `sasy.instrument` once, at startup. It enables SASY for each supported
  framework that is installed: ADK, and also LangChain and Langroid if they are
  present. Pass `adk=True` to make a missing or unsupported ADK an error
  rather than skipped, and `langchain=False` or `langroid=False` to skip a
  framework you do not use.
- If Google ADK is installed at a version other than 2.9.1,
  `sasy.instrument()` skips the ADK adapter. The first time the application
  then calls `run`, `run_async` or `run_live` on an ADK `Runner`, SASY issues a
  `SasyInstrumentationWarning` (a Python warning, also logged, once per
  process) saying that ADK agents will NOT be checked by SASY. An application
  that has ADK installed but never runs it gets no warning. Pass `adk=True` to
  make this an error, raised before anything is changed, or `adk=False` if
  your application does not use ADK, which also silences the warning.
- HTTP checks are off by default. Pass `http=True` to route requests made with
  `httpx` or `requests` through the engine, which checks each one against the
  policy and can inject credentials (API keys the engine holds) into it. That
  includes the model's request to Google, so the policy must then allow
  Google's model endpoint. The example turns HTTP checks on only for its live
  model; its scripted models make no requests.
- Keep your existing runner, model and tool objects.
- Use `run_async`. The synchronous `Runner.run` runs on a thread that does not
  carry the SASY session.
- Give each top-level ADK conversation its own SASY session. Use
  `end_on_exit=False` when the session has to survive between turns, and end it
  explicitly when the conversation finishes.
- Separate conversations may run at the same time. Two turns of the *same*
  conversation at the same time are rejected.
- Keep the runner to continue a conversation. Importing a stored history into a
  new runner is not supported, because the adapter cannot reconstruct where
  those messages came from.
- Use ADK's `InMemorySessionService` or `DatabaseSessionService` on its current
  v1 schema. A v0 database has to be migrated first, and another session service
  is untested.
- For database sessions also install `'sqlalchemy[asyncio]>=2,<3'` and an async
  database driver, and use its URL scheme, for example
  `sqlite+aiosqlite:///sessions.db` with ADK's bundled SQLite driver.

## Compatibility

**Not supported** — the run stops with an error, before it starts or when the
path is reached:

- `Runner.run` (the synchronous entry point), live audio and video, remote
  agents and A2A or MCP toolsets (the two protocols ADK uses to reach agents and
  tools in other processes), plugins, code execution, compaction, provider
  context caches, resumable apps, Jinja instruction templates, custom `BaseAgent`
  subclasses, custom execution overrides on tools, and importing an existing
  session history into a new `Runner`.
- Credential transforms on tool calls: a decision that requires one blocks the
  tool. The reference monitor applies transforms to HTTP requests routed
  by `sasy.instrument(http=True)`.
- Model egress (the model's own requests to Google), unless you pass
  `http=True`, and even then when the GenAI client does not use `httpx`.

**Session boundaries**

[Ancestry and tool evidence do not cross SASY sessions.](#values-from-another-sasy-session)
A value carried into a new session is marked as unattributed input. Policies
should handle that marker explicitly; missing ancestry does not establish that
the value is safe.

## Supported execution

The adapter supports text `LlmAgent` execution, ordinary text streaming,
synchronous or asynchronous function tools, local `transfer_to_agent` handoffs,
and standard `SequentialAgent`, `ParallelAgent` and `LoopAgent` composition. Gemini's
generate-content API and scripted models are the model paths the adapter is
tested against. Other model providers are untested: they may work, with no
guarantee. Function-tool subclasses may
retain their own declarations, but must use ADK's standard execution methods;
custom execution overrides require separate instrumentation. Agent names must
be unique throughout the configured tree, so parallel agents cannot share an
output or tool-call identity.

The adapter also supports a root `Workflow` with a static serial chain of
standard single-turn `LlmAgent` nodes, or one fan-out into disjoint agent
chains that meet at a `JoinNode`, optionally followed by more serial agents.
Each parallel branch keeps its own ancestry until the join combines the exact
outputs of its predecessors. A later agent then sees the joined ancestry,
including tool results from all branches.
An exclusive conditional fork can instead select one of two or more disjoint agent
chains through a guarded router tool immediately after `START`. The selected
chain may reconverge at an ordinary agent. Its actions inherit the router's
decision and the selected branch's history, including state reads made by the
router. Unselected branches do not run. This profile requires at least two
distinct nonempty string routes and a single finalizer; other routes and merges
are rejected.
Graph nodes that run arbitrary functions or tools, dynamic nodes, callbacks,
retries, and schema coercion are also rejected.
State templates, `output_key`, tool-context state access, and supported text
artifacts use the same instrumentation in serial and exclusive conditional
Workflows. In a Workflow with parallel branches, these operations are supported
before the fork and after the `JoinNode`. Concurrent branches can also save and
load text artifacts through ADK's built-in `FileArtifactService`, including
artifact instruction templates. Each read depends on the exact published
version it consumed. Mutable state, filename listings, `LoadArtifactsTool`,
and in-memory or custom artifact services remain restricted in concurrent
branches.

When one model message asks for several tools at once, the adapter records one
node per call, and each tool's result depends on all of those call nodes. ADK
combines the results of such a batch into a single event before storing them.
A tool in a batch may write state: each call's writes are recorded against that
combined event, with the same provenance a single call's writes get. Forwarding
an `AgentTool` child's writes is the one exception, and is refused in a batch.

Streaming chunks pass through normally. SASY observes the completed model output
before downstream tool dispatch; it does not record an event or make an RPC for
each chunk. Cancellation before completion does not create a completed-output
observation. Bidirectional live audio/video is a separate, unsupported path.

Local transfers are checked before the transfer tool sets its target. A denied
transfer does not start the receiving agent. The receiving model's dependencies
come from the messages it actually consumes, including ADK's presentation of
another agent's messages.

Tool authorization runs after ADK's argument preprocessing and tool callbacks.
The check includes default argument values and uses a private copy of exactly
the arguments passed to the callable. Denials and required transforms prevent
execution; this adapter does not apply transforms. Tool error results contain a
`[BLOCKED]` explanation that the model can consume. Denied calls, required
transforms and tool returns containing an `error` field are recorded as feedback
without successful `ToolResult` evidence. Later framework response events do not
turn that feedback into evidence of execution. Recording or policy-service
failures abort execution instead of allowing a tool without evidence.

A model response that carries a provider error code or error message stops the
run. It is never recorded as a completed model output, so it cannot dispatch a
tool or count as a child agent's successful reply.

A before-model callback may drop messages or rewrite the request. Messages it
keeps unchanged keep their IDs. Content it changes is recorded as a new message
computed from the callback's inputs and the state it read. A callback may not add
tools or function declarations, and so cannot open a dispatch path the adapter
does not check.
Callbacks and tool implementations must not consume
unrecorded external context or perform unmediated protected actions; such custom
orchestration needs its own instrumentation. A callback's fabricated response
is not an observed model or tool execution and fails correlation. Generated
system instructions are recorded as callback derivations. Before-model callbacks receive the
request message IDs; after-model callbacks receive the completed output IDs.
Tool preparation and before-tool callbacks receive the initiating message IDs,
and after-tool callbacks also receive the actual result IDs. These task-local
scopes carry into nested HTTP checks. After-model callbacks on partial streaming
chunks are rejected because those chunks have no individual graph identity.

Before each model call the adapter records the exact messages the model is about
to receive and gets back a permanent ID for each. A message that has not changed
keeps its ID, and a short content hash is sent in place of its text, so nothing
new is stored. This includes ADK's *presentation* of another agent's messages —
the context text ADK writes when it shows one agent's output to another. A
message counts as an input only if it is actually in the request, not because it
sits in the ADK session. If recorded history was edited between turns, the run
stops: an edit on its own does not say what the new content was computed from.
Native history and message presentations retain their exact source event identities,
so identical replies from different agents remain distinct. Unknown provenance or
edits to a bound presentation stop execution.
Tool-call IDs must be unique for each agent within the retained conversation;
different agents may reuse an ID.

### What a record holds

The adapter records message content for policies and a canonical snapshot in
metadata so the engine can detect changes to the source message. See
[record encoding](/reference/google-adk-records/) for the field representation,
supported value types, and version-comparison rules.

### Child agents and tasks

Ordinary `AgentTool(child)` calls and configured `LlmAgent` children with
`mode="single_turn"` or `mode="task"` are observed automatically. Each delegation
is checked under the child's tool name before launch. Native task completion is
also checked as `finish_task`; include these action names in the policy where
appropriate. Calls made by the child still receive their own tool checks. The
delegation target is verified again after the check returns, so a child swapped
while authorization was pending is never launched.

The child input depends on the exact initiating call. The parent's return value
depends on the specific completed child response that ADK returns, carrying that
response's ancestors. Parallel siblings keep separate scopes; merely sharing a
session does not make one sibling's approval evidence available to another.
Denial, cancellation and failed completion do not create successful delegation
evidence. Repeated identical replies retain distinct identities.

These native task modes currently support plain text input/output and standard
execution without task retries, `parallel_worker`, custom task schemas or
resumption. Your own calls to `Context.run_node` are not supported. The adapter
handles ADK's node machinery only for the child-agent modes listed above and
the qualified root `Workflow` graphs described earlier.

`include_contents="none"` may still retain messages from the current invocation.
When selective history matters, inspect the finalized request or filter its
message objects in a before-model callback; the adapter records what is actually
sent.

Agent names are application-assigned labels, not authenticated human identities.
SASY stamps the authenticated writer separately. Do not treat an ADK `user_id`,
message role, or remote agent label as independent authentication.

## Boundaries

The adapter rejects unsupported ADK versions and missing dispatch hooks. Before
a conversation runs, it checks for unsupported custom agents, remote/toolset
tools, structured instruction values, combined or structured
`static_instruction` configurations, code execution,
plugins, compaction, provider context caches and resumable apps. Unobserved state
changes and unsupported artifact operations also abort. Jinja rendering is
rejected where it happens rather than up front: ADK 2.9.1 has no setting that
turns it on, and the only way to reach it is to call
`instructions_utils.inject_session_state` with `use_jinja2=True` yourself, which
stops the run at that call. These paths need explicit
provenance instrumentation before they can be supported.

Bidirectional audio/video, remote A2A/MCP execution, and importing an existing
unobserved session history are not supported. Reusing a runner's recorded
conversation is supported; reconstructing its correlation from persistent
history is not. Cancellation restores the caller's context; a cancelled run does not establish evidence for unfinished work.

HTTP `extra_body` overrides are rejected in both finalized model requests and
Gemini client defaults, including clients created through `client_kwargs`.
These options can replace the input body after framework observation and would
otherwise invalidate its provenance.

Model network egress is a separate action boundary. SASY's HTTP
instrumentation, which is off by default and enabled with
`sasy.instrument(http=True)`, covers supported `httpx`/`requests` transports. Google GenAI can
select `aiohttp`, which these hooks do not cover; the live example explicitly
supplies `types.HttpOptions(httpx_async_client=httpx.AsyncClient())`. The adapter
propagates model input IDs into model HTTP checks and the originating tool-call
IDs into HTTP checks made inside function tools. Other transports need their own
interceptor; framework observation alone does not enforce network egress.

## State and artifact inputs

ADK injects state through instruction templates and writes model responses with
`output_key`. SASY records those reads and writes, preserving the producing
messages when a later agent reads a value—even with `include_contents="none"`:

```python
writer = LlmAgent(name="writer", model=model, output_key="decision")
reader = LlmAgent(
    name="reader", model=model,
    instruction="Use this decision: {decision}",
    include_contents="none",
)
# Put writer before reader in a SequentialAgent.
```

An instruction may also be a callable provider, synchronous or asynchronous,
that ADK calls with a read-only context before each model call:

```python
def instruction(context):
    return "Use this decision: " + context.state.get("decision", "none")

reader = LlmAgent(name="reader", model=model, instruction=instruction)
```

The provider's `state` mapping records the key and value at each read, exactly
as an instruction template does, and the resulting system instruction depends on
those keys and nothing else. A provider that reads no key yields an instruction
with no state dependencies, so an approval stored under a key it never read does
not reach a later tool check. Testing whether a key exists is a presence read.
The provider must return text. Its context offers state, the invocation id, the
agent name and the user id; the session, the invocation's user content, custom
metadata and credentials stay blocked, because those carry no observed
provenance of their own. `global_instruction` providers work the same way.

Only referenced keys become dependencies. `{key?}` records an absent optional
value; unrelated keys do not supply approval ancestry. Unprefixed keys belong
to the ADK session; `app:`, `user:` and `temp:` select application, user and
invocation scopes. A `temp:` key belongs to one invocation: the adapter records
it under the invocation that wrote it, and ADK keeps it out of the event it
stores. Writing a `temp:` key and a stored key in the same call is supported and
recorded as a write of both. Initial values supplied outside an observed computation are
labeled as unattributed external inputs, not successful tool results. Later
unobserved edits, including nested container edits, stop execution.

Function tools may receive ADK's injected `ToolContext`. Its `state` mapping
captures the key and value at each read. Reads are synchronous and do not make
an RPC individually; SASY flushes pending observations before a protected tool
or HTTP action and before recording the output. Callbacks use separate scopes,
so their reads also reach the actions and derived messages that consume them.
The runtime context handle is excluded from JSON action arguments; resolved
data arguments and defaults remain in the check.

Testing whether a key exists (`"decision" in tool_context.state`) is recorded as
a presence read, without reading the value. When an agent's `output_key` or a
tool assignment created the key, that presence read depends on the message that
created it, exactly as a read of the value would.

Assign a JSON value to a state key to record a write. Nested values returned by
reads are immutable: copy, edit and reassign the whole key when needed. State
writes require the exact injected context: the `ToolContext` in the function-tool
body, or the `CallbackContext` of the callback making the write. Direct edits to
`actions.state_delta` are rejected. A successful writer's recorded result becomes
part of the written value's provenance only after the tool completes. A tool that returns an `error` field still commits its
state writes, because ADK persists them; the written value depends on what the
tool consumed but carries no successful-tool evidence. A tool that raises or is
cancelled has its state writes rolled back.

Callbacks write state through `callback_context.state[key] = value`, in
before/after agent, model and tool callbacks. The written value depends on the
callback's own inputs plus the reads it made before the write, and on nothing
else. A before-model callback's inputs are the request messages, an after-model
callback's the model output, a before-tool callback's the messages that initiated
the call, and an after-tool callback's those plus the tool result. An agent
callback consumes no message, so its own state reads are its whole input;
artifact access from a callback is rejected. A callback write never carries successful-tool evidence, even when the tool
it surrounds succeeds. SASY compares the state change ADK emits on the resulting
event against its record of the writes; any other key in that change stops the
run. For an agent callback that comparison runs as the callback returns, before
ADK builds the event, so a key written straight into `actions.state_delta` there
never reaches the session service and cannot outlive the stopped run as a
`user:` or `app:` value. A callback that raises has its writes rolled back, and a written value
becomes readable elsewhere only once ADK has emitted the event that carries it. When
a callback writes the same key an agent fills through its `output_key`, ADK keeps
the `output_key` value, because it is applied last; SASY records that value with
the model output as its producer, and the callback's superseded write gains no
reader.
State writes in the callbacks of parallel branches are rejected, exactly as tool
state operations there are. An agent callback that returns content is rejected:
that content would replace the agent's own output and has no observed producer.
Artifact writes in callbacks remain rejected.

Text artifact support works through ADK's artifact service interface. The
in-memory and file-backed services are tested; other `BaseArtifactService`
implementations must preserve the version and metadata semantics below. Tools may use
`await tool_context.save_artifact(name, types.Part(text=...))` and
`await tool_context.load_artifact(name)`, or the corresponding service methods
with the current application, user and session identifiers. Each saved version
retains its producer; a load adds only that version to subsequent result and
action dependencies. Instruction templates such as `{artifact.report.txt}` and
the native `LoadArtifactsTool` retain the same provenance. Before-model
callbacks may select or reorder the artifact contents already placed in a
request, as intact objects; callbacks cannot load or save artifacts themselves. Listing
filenames does not consume every file's contents. ADK's artifact store has no
rollback: an artifact saved before its tool raises, is cancelled or returns an
error stays readable. A later reader of that version depends on what the failed
tool consumed, without successful-tool evidence.

The built-in `FileArtifactService` supports overlapping saves and loads in
qualified Workflow branches. SASY reserves the actual native version number,
records its write dependencies, and lets ADK atomically publish the content and
metadata together. A read of the latest version selects a published version
once and loads that exact version, even if another save finishes meanwhile.
Reserved but unpublished versions are not readable. A reader can consume a
published write before its producing tool returns; that read inherits the
write's inputs without assuming the tool completed successfully. Cancelling a
save cannot undo a file operation already running, but any version it publishes
already carries its recorded write ancestry. Canonical file and scope names
are required so aliases cannot give the same stored file different identities.
Filename listings remain serial; after a join, a listing retains the observed
dependencies of the names it returns without consuming file contents. Listings
containing a filename with no published version are rejected, including names
left behind by failed saves.

Other services retain serial access and catalog checks. If an
error or a cancellation lands after ADK stored the version but before SASY
completed the save's bookkeeping, SASY then
refuses every artifact operation that uses that artifact service object and
reports an incomplete save. The refusal follows the service object, not the
session, so later SASY sessions sharing that object are refused too; a new
service instance, in practice a restart, is what clears it. Another
writer sharing the same backend can change the store while this session runs.
SASY reads the stored filenames and version numbers when an artifact operation
starts and checks them again against the result that operation returns, so a
version or filename that appeared or vanished outside this session stops the run
rather than being read under the wrong version number or listed with no recorded
dependency.
Artifact storage is observed internal transport; tool dispatch is the action
boundary checked by the reference monitor (RM), the SASY service that authorizes
each action.

Producers travel with the artifact. Each save asks ADK to store, under a
reserved key in the artifact's own custom metadata, the identifiers of the
messages SASY recorded for that version together with a digest, a fixed-size
fingerprint, of exactly the text it recorded. A backend that keeps its data on
disk keeps that provenance with it. When a load finds a version this process
never saw, such as one written by another runner or before a restart, SASY
rebuilds that text from the content the load actually returned, requires the
digest to match it, and asks the observation service whether those identifiers
still hold exactly that text in the current SASY session. Only then does the
load depend on the real producer. Stored provenance that is absent, malformed,
names messages holding different text, or no longer matches the content leaves
the content an unattributed external input, and so does a backend that drops
custom metadata. The stored record is written when the artifact is
saved, before the saving tool has finished, so it names the message recorded for
that version at that point: a reader in another process inherits what the tool
had read, without evidence that the tool completed. Completion records a second
message for the same version, holding the same text and depending on both that
first message and the tool's result. A record that names the second message
instead — which only something with write access to the store's metadata can
leave there — verifies the same way, and the load then carries the completion
evidence. That second message exists only for a tool that did finish, so either
record leaves the provenance accurate. Metadata your own code passes to
`save_artifact` stays rejected, so the reserved entry is the only claim about an
artifact's provenance. For backends other than the built-in file service, a
save also stops the run if the backend does not number the new version one above
the highest version it already stored.

An `AgentTool` child receives ADK's copied state with per-key provenance: copying
the dictionary does not count as reading every value. Only keys the child reads
become inputs, including through nested delegations. ADK drops `temp:` keys from
this copy.

The child may also assign state keys and use `output_key`. When the child
finishes, ADK forwards the child's state change back into the parent's state. A
state change, which ADK calls a delta, is the set of key/value pairs an event
reports as newly written. SASY matches every forwarded entry against the child's
own observed writes: the key must have a completed write inside that child
holding exactly that value, or the run stops before the parent changes. The
forwarded value then keeps the child's real producers, so a later reader of the
key depends on what the child consumed, and gains a child tool's successful
result as evidence only when that tool really succeeded. The delegation's own
result is not a producer of what the child wrote, and a forwarded key nobody
reads adds no dependency. SASY applies the whole forwarded change when the child
completes, where ADK applies each change as its event arrives, so a child that
fails partway leaves the parent value untouched. Forwarding is rejected for
`temp:` and ADK-internal (`_adk`) keys, and for a delegation the model requested
alongside other tool calls in the same response, which would need the forwarded
write ordered explicitly against those calls' own writes. The native artifact
forwarder supports reads and filename lists, preserving the selected artifact's
ancestry; forwarded artifact writes are not yet supported.

### Values from another SASY session

Retain the runner and services for continued conversations. Each SASY session
records its own separate graph. Services outlive a session, so a later session
can read a key or artifact an earlier one produced: `app:` and `user:` state and
a shared artifact store are the common cases. Such a value enters the reading
session as an unattributed external input, the same message shape used for
values that were already there when observation began, with an added `origin`
field naming that it came from another SASY session. The producing session's
identifier is not recorded, because it may belong to a different tenant or
entity and naming it would carry that identity across. The reading session never
reuses the other graph's message identities; it records the value again under
its own. Reading the same value twice in one session yields one message, so a
policy sees one input rather than two.

Nothing else crosses. The other session's ancestry stays behind, and so does any
evidence it holds: an approval a tool granted there does not authorize an
approval-requiring action here. Taint (a dependency on untrusted content) does
not cross either, so a value that was tainted in the producing session arrives
here with a clean, parentless history. A policy that denies on taint should
treat external inputs conservatively, for example by refusing to let an input
labeled `"origin": "another SASY session"` reach a sensitive action, rather than
reading its empty ancestry as evidence that the value is safe.

Resource access must remain in its observed task. Application-created tasks,
parallel agents outside the qualified Workflow artifact path and concurrent
invocations sharing mutable services need
explicit dependency and ordering support; unsupported access is rejected. ADK's
in-memory artifact backend is intended for development, not durable production
storage.

Jinja templates, callback artifact writes,
binary artifacts, caller-supplied artifact metadata, artifact deletion and
version-metadata methods, and experimental dynamic-instruction routing for
artifact tools remain unsupported.
Credential, memory and UI methods on `ToolContext` also require their own
observation and action boundaries. Access to the raw context session is blocked.

## Application-created child tasks

Native agent communication is handled by the adapter. Tasks created inside your
own tool or orchestration code need explicit registration of their inputs and
outputs. Register each child response with the observation API, using its real
consumed inputs. Then add only the response the parent actually consumes:

```python
from sasy.instrumentation.adk_tasks import consume_events_async
from sasy.proto.observability_pb2 import EventSnapshot

# Inside an observed ADK tool or callback. Your custom orchestration has already
# registered response_event and its dependencies, returning response_id.
await consume_events_async([
    EventSnapshot(
        event=response_event,
        base_id=response_id,
        reuse_dependencies=True,
    )
])
```

The helper verifies the unchanged response against the server in the current
SASY session. Its ancestry then reaches subsequent action checks, state writes
and the parent output. Use `consume_events` in synchronous code. Starting a child
Python task does not authorize it to mutate its parent's dependency scope; await
its response and consume it in the parent task. Before passing the parent's
current inputs into custom orchestration, use
`await sasy.instrumentation.dependencies.resolve_inputs_async()` to include any
pending state reads. The synchronous counterpart is `resolve_inputs`.

A child response carries its own ancestors. Registering every sibling result as
a parent dependency would incorrectly supply context the parent never consumed.
This helper covers response consumption, not arbitrary task bodies. A custom
executor still needs its own task-local input scope and checks at its protected
action boundaries; inheriting the parent's context is insufficient. A value read from
another SASY session arrives as a labeled external input; see
[values from another SASY session](#values-from-another-sasy-session). These
helpers do not carry ancestry between sessions.

## OpenTelemetry

The adapter adds SASY message IDs, session/entity IDs and RM decisions to ADK's
native model and tool spans. Configure the shared SASY exporter before running
agents:

```python
from sasy.instrumentation.otel import OTelConfig, configure_otel

configure_otel(OTelConfig(service_name="my-adk-agent"))
sasy.instrument()
```

Enrichment adds identifiers and bounded outcomes, not prompt or argument copies.
ADK's own telemetry configuration controls its other attributes. Multipart
outputs get child span links for every output message. `sasy.dispatch.outcome`
describes model/tool dispatch; the native span status covers the surrounding ADK
pipeline, including callback failures. Session/entity scope is captured before
batch export so concurrent sessions do not share trace ownership.

## Payment example

`examples/adk-separation-of-duties` runs requester, reviewer and payer agents with
a real SASY policy. Approval is bound to the request ID, payee and exact amount.
The default scripted models need no provider credentials; `--live` exercises
Gemini 3.8 Flash with the same synthetic tools and an in-memory ledger. Set
`SASY_ADK_LIVE_MODEL` to choose another Gemini model. No money moves.

The example demonstrates application role separation. Its configured `reviewer`
label is not evidence that an independent person approved the payment.
