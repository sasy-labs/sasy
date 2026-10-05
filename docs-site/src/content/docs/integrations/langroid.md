---
title: Langroid
description: What the Langroid adapter checks, how to enable it, and what it does not cover.
---

## At a glance

**What you get**

- Langroid agent responses and task handoffs are recorded across cooperating
  agents. Custom tools and Langroid's built-in orchestration tools are checked
  against your policy immediately before they run.
- Model inputs, tool inputs and the framework's own history truncation are
  recorded, so a check sees the call's ancestry: every message the requesting
  model output was computed from.
- A reply holding several tool calls is recorded call by call. Langroid runs
  them all and hands back one combined result; each tool's own result is
  recorded as a node of its own, carrying the call it came from, and the
  combined result is recorded as computed from all of them. A rule that finds a
  read through its tool result therefore sees every read in the reply, however
  many calls the model chose to make. Two calls with the same name and the same
  arguments are two calls: each gets its own result, because the second may
  return different data from the first.
- A tool handler may return another tool call, which Langroid runs at once. The
  first tool's result is that returned call, recorded as such, and the second
  call is checked with the first tool's result among what it was computed from
  — so a rule that denies a send computed from a secret read still sees the
  read. The second tool's result is recorded as computed from the first's, and
  the reply is recorded as computed from the end of the chain. A chain of any
  length is recorded the same way.
- A tool call may not name its own handler. A model's tool JSON can carry an
  extra `_handler` field, and older Langroid ran the agent method named there
  while the call was checked and recorded under its own name. A call that
  names a handler other than the one its tool class declares is refused, and
  neither method runs. The handler a tool class declares for itself keeps
  working.
- With `sasy.instrument(http=True)`, outbound HTTP requests made with `httpx`
  or `requests` go through the same check. HTTP checks are off by default.

**How to enable**

1. Follow the [quick start](/get-started/) to run the engine and install the SDK. Run your
   Python entry point with `python your_agent.py` in the environment where you
   installed the SDK and framework.
   The extra accepts Langroid 0.67.1 through 0.67.8; 0.67.8 is the version the
   test suite runs against. 0.67.1 is the floor because earlier versions let a
   tool call redirect itself to another agent method.
2. Call `sasy.instrument()` once at application
   startup. This enables the Langroid adapter when Langroid is installed,
   along with the adapter for any other supported framework that is installed.
   If Langroid is installed but the adapter cannot import it, `sasy.instrument()`
   skips the adapter. The first time the application then runs a Langroid
   `Task` or asks a `ChatAgent` for an LLM response, SASY issues a
   `SasyInstrumentationWarning` (a Python warning, also logged, once per
   process) saying that Langroid agents will NOT be checked by SASY. An
   application that has Langroid installed but never uses it gets no warning.
   If even those Langroid classes cannot be loaded, the warning comes from
   `sasy.instrument()` itself. Pass `langroid=True` to make a missing or
   unloadable Langroid an error, or `langroid=False` if your application does
   not use Langroid, which also silences the warning. The adapter does not check which Langroid version
   is installed, so install it through the extra. HTTP checks are off by
   default; pass `http=True` to have the engine check the agent's `httpx` and
   `requests` calls and inject credentials (API keys the engine holds) into
   them. The policy must then allow each host the agent calls, including the
   model provider's.
3. Run each conversation inside its own `with sasy.session(policy=...):` block.

Keep `tool_policy_fail_closed=True`, the default: with it, a tool whose decision
could not be obtained does not run. It is set in code only, through
`sasy.configure(tool_policy_fail_closed=...)`. No environment variable and no
`.env` file changes it, so a file dropped next to your application cannot turn
the gate off, and `sasy.instrument()` logs a warning when it is off. The other
instrumentation settings do read the environment, under names that start with
`SASY_` (`SASY_LOG_DENIALS`, `SASY_LOG_POLICY_DECISIONS`,
`SASY_LOG_POLICY_DECISIONS_TRANSFORMS_ONLY`, `SASY_LOG_TRANSFORMS`); the two
OpenTelemetry settings keep the names that ecosystem uses, `OTEL_ENABLED` and
`OTEL_SERVICE_NAME`.

**Not covered**

- File attachments.
- History restored into a new run. A message this session never recorded has no
  inputs the adapter can defend, so a task continued from one stops at
  `task.init()`. Record the message first (below). "Continued" means a task
  started with no message of its own: `Task(agent, restart=False)`, or a second
  `task.run()` on the same agent. `Task(agent)` on its own clears the history
  and starts fresh, so it is not affected. Restored history that is only read
  as model input is treated by role: a USER or SYSTEM message with no parent is
  recorded as a fresh root, because that is what it is, while a restored
  assistant or tool message stops the next protected call.
- An edit your application makes to a message, unless you record it (below).
- A decision that requires a credential transform the tool adapter cannot apply:
  the tool is blocked rather than run unprotected. HTTP transforms are applied
  separately, by the reference-monitor proxy, to requests routed by
  `sasy.instrument(http=True)`.
- HTTP requests, including the model's requests to its provider, unless you
  pass `http=True`.
- Networking libraries other than `httpx` and `requests`. The HTTP hooks
  deliberately let `localhost`, `127.0.0.1` and `::1` through: those requests
  are neither checked nor recorded, so a service reachable under one of those
  exact names is invisible to your policy. The same machine under another
  spelling (`127.0.0.2`, `0.0.0.0`, a decimal address) is checked like any
  other host.
- Request bodies the hooks cannot read: an `httpx` request whose body is an
  async iterator, or a `requests` request whose body is a file object or a
  generator. These stop with an error from the conversion, which does not say
  why; nothing is sent unchecked.
- Langroid's synthetic "no answer" document, which a task injects when a step
  produces nothing and `allow_null_result=True`, and its `__SEND__:` /
  `__PASS__` routing, which rewrites a message that was already recorded. Both
  stop the run rather than record a computation that did not happen.
- An allowlist policy has to name Langroid's own orchestration tools
  (`done_tool`, `pass_tool`, `send_tool`, `forward_tool` and the rest) if your
  agents use them. They go through the same check as your tools, and a denied
  one stalls the task loop.

The [Langroid example](https://github.com/sasy-labs/sasy/tree/main/examples/langroid-information-flow)
in `examples/langroid-information-flow` is a three-agent team: two helpers read
documents and a third agent, which can never read a file, sends the email. One
document carries an injected instruction, and the agent that sends obeys it. The
same program runs under two policy files — multi-level security, and a
toxic-flow rule — so only the policy decides what is allowed. Its two helpers
also run as concurrent tasks in one session, which is where a check on what a
message was computed from differs from a session-wide flag: a secret another
agent has already read is not a dependency of a message that never saw it.
[Your first policy](/quickstart/) shows the plain configure-and-check shape.

## Threads inside a handler

The adapter keeps the session and the current inputs in context variables
(Python's `contextvars`), which hold a separate value per thread and per asyncio
task. A worker thread started inside a handler therefore begins with the
defaults: no session, and no inputs. Carry the context into the worker:

```python
import contextvars
from concurrent.futures import ThreadPoolExecutor

def fetch_all(self, msg: FetchAll) -> str:
    with ThreadPoolExecutor() as pool:
        running = [pool.submit(contextvars.copy_context().run, fetch, url)
                   for url in msg.urls]
        pages = [task.result() for task in running]
    return summarize(pages)
```

One copy per worker: a single `Context` object cannot be entered by two threads
at once. In async code, `asyncio.to_thread(...)` carries the context already.

Without a copied context or a process-wide default, automatic Langroid and
HTTP instrumentation in the worker is inactive. Carry the context, or open a
`sasy.session(...)` block inside the worker. Keep the owning session open until
its workers finish; an inherited closed session raises rather than bypassing
checks. See [session lifetime](/configuration/#session-lifetime-and-background-work).

## Calling a handler yourself

Handlers are checked where Langroid dispatches them, inside a responder:
`agent_response`, `llm_response`, `task.run` and their async forms. Calling
`agent.handle_message(...)` or `agent.handle_tool_message(...)` yourself is
refused as soon as it reaches a tool handler, because outside a responder there
is no record of what the call was computed from, and a check with no ancestry
answers a different question than the one your run is asking.

Passing a plain string to a responder (`agent.agent_response("some text")`)
records the string as a fresh user message with no ancestry — which is what a
string from outside the conversation is. To keep the ancestry, pass the
`ChatDocument`, or record the message your code composed (below).

## Recording an edit your application makes

If your code changes a message after it was recorded, record the change before
the message is used again:

```python
from sasy.instrumentation.langroid import record_message_update

message.content = revised_text
new_id = record_message_update(message, input_ids=[other_consumed_id])
```

The helper includes the previous version of `message` plus the input IDs you
supply. Pass `input_ids=[]` when the edit consumed only the previous message;
use `record_message_update_async` in async code. The message must already have a
recorded version in the current session.

An edit clears the evidence that a tool completed successfully, unless the
operation explicitly supplies `derived_from` for a completed tool. Do not copy
that evidence from an earlier result: a rule that requires a successful
`ToolResult` would then accept work that never ran.

Unexpected content, or a changed parent, fails before the next protected
dispatch. For a message in the form the model provider sees (an `LLMMessage`),
the adapter compares the fields it recorded and refuses on the spot. For a
`ChatDocument`, the changed message goes to the engine as an unchanged one and
the engine refuses it, because its content no longer matches what was recorded
under that version. Either way the run stops. Neither check can say what the new
content was computed from, which is why an application edit has to be recorded
explicitly. Custom handlers that read files, external state or extra messages
must record those inputs too.

## Recording a message your application composes

A `ChatDocument` your code builds from earlier results — the question for the
next task, assembled from a previous task's answer — was produced by no
responder the adapter observes. Record it, with the version IDs of what it was
computed from, before handing it to a task:

```python
from sasy.instrumentation.langroid import record_message, recorded_version_id

answer_id = recorded_version_id(answer_document)   # None if never recorded
question = ChatDocument(content=f"Does {answer} ...", metadata=ChatDocMetaData(sender=Entity.USER))
record_message(question, input_ids=[answer_id])
task.run(question)
```

Without this, a composed message with no parent is recorded as a fresh user
input and the policy sees none of its ancestry. `record_message` refuses a
message that already has a version; that is an edit.

## Subclassing `Task`

The adapter records the result of `Task.result` on Langroid's `Task`. A
subclass that overrides `result` replaces that method, so the override must
record the document it builds:

```python
from sasy.instrumentation.langroid import is_instrumented, record_task_result

class MyTask(lr.Task):
    def result(self, status=None):
        result_doc = ...
        if is_instrumented():
            record_task_result(self, result_doc)
        return result_doc
```

`is_instrumented()` is true once `sasy.instrument()` has patched Langroid in
this process, so the same class runs unchanged without SASY.

## What counts as a recorded tool call

A message's tool calls are read from the message itself: the provider's function
and tool-call fields, plus any call written into the text, in Langroid's JSON
form (`{"request": "<tool>", ...}`) or its XML form (`<tool>...</tool>`). No
agent's tool registry is consulted, so every agent that reads a message records
the same calls for it. That is what lets a parent task emit a call its sub-task
handles: both record the same message, and neither refuses the other's reading
of it. Two consequences:

- A JSON object with a `request` field is recorded as a tool call even when no
  agent could handle it. It is what the message asked for; whether anything runs
  is still decided by the check at dispatch.
- A call whose fields are unusable — a known tool with the wrong types — is
  recorded as it was written and left to Langroid, which feeds its own
  validation error back to the model. Recording it does not end the run.

## What the adapter does with in-place edits

Each responder receives a detached copy of its input message; the caller's
object stays as it was recorded, and the responder's output is recorded from
what it returns. A value a handler writes onto its input — on the tool object it
was given, for another agent to read later — therefore does not reach that
agent. Pass it in the returned message, or keep it on the agent that produced
it.

Langroid itself writes `request=<tool name>` into a tool call's arguments when
an agent handles it. The adapter leaves that tag out of the recorded arguments,
so a message reads the same before and after handling.
