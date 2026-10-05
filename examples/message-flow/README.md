# Information flow: untrusted input and sensitive data

This example shows a policy preventing a simulated email when its contents or
destination could have been influenced by both an untrusted note and a sensitive
report. SASY follows that influence through the message and action dependency
graph, including the summary between the document reads and the send.

## Run the example

[Start a local engine](https://docs.sasy.ai/local-engine/) and, from the
repository root, run the scripted scenarios:

```bash
uv run examples/message-flow/demo.py
```

For a real model, add `OPENAI_API_KEY=...` to the repository-root `.env`, then run:

```bash
uv run examples/message-flow/demo.py --live
```

Live mode defaults to `gpt-4.1-mini` and makes paid provider requests. Set
`SASY_LIVE_MODEL` in `.env` or pass `--model` to choose another model.
`--sensitive-only` selects the allowed baseline. The scripted run needs no
provider key; live results depend on the model's choices.

A scripted run prints a narrated walkthrough: the policy, then each run with
SASY's decisions, waiting for Enter between steps (`--no-pause` turns that off).
`--only-note` shows only the run with the external note. Use `--trace` to print
every recorded event instead, or `--quiet` to show only the scenario summaries.

## What the system does

[`demo.py`](demo.py) runs a small scripted agent twice by default. Each run starts with the
user asking it to summarize a report and send the summary to
`reviewer@example.net`:

| Run | Inputs and behavior | Policy decision |
| --- | --- | --- |
| Sensitive only | Read the report, summarize it, send to the user's reviewer | Allow |
| Untrusted + sensitive | Also read an external note that redirects the send to `audit@partner.example`; the agent follows it | Deny |

Without `--live`, the agent's responses and action choices are scripted;
`--live` lets a model choose them. The document store and email delivery are
simulated in both modes, while SASY's dependency records and policy decisions
are real.

## The policy

[`policy.dl`](policy.dl) permits reads of the two named documents and allows
`send_summary`, except when its recorded history includes reads of **both**
`external-note` and `sensitive-report`:

```prolog
Unauthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "send_summary"),
    DependsOnUntrusted(),
    DependsOnSensitive().
```

The two dependency predicates follow `CurrentDepends`, the events in the current
action's input history, to the named document-read results. This derives the
untrusted and sensitive labels that could have influenced the send, including
through the summary. The policy does not need to detect the injected instruction
or scan the outgoing message for secrets. A denial takes precedence over an
allow rule and stops delivery.

## Manual instrumentation

This example illustrates how to instrument an agent with custom orchestration.
Each scenario binds the policy to a fresh session:

```python
with sasy.session(policy=Path(__file__).with_name("policy.dl")):
    ...
```

For agents built with supported frameworks, enable `sasy.instrument()` once and
use only this session context manager around the run. The framework integration
records dependencies and checks actions automatically. The calls below show
what custom orchestration must supply itself.

The `record` helper registers each message and edges from the messages it
consumed. The scripted summary consumes the instruction and document results,
so its dependency edges carry both sources forward:

```python
record_events_with_dependencies(
    [message],
    [Edge(source=source.id, destination=message.id) for source in inputs],
)
```

At the dispatch site, the `dispatch` helper records the tool request, checks it
before running the handler, then records the result or denial as another message:

```python
tool = Tool(name=name, arguments=json.dumps(arguments))
request = record("Tool request", inputs, tool=tool, role=Role.LLM)
decision = sasy.check_tool_call(name, tool.arguments, input_node_ids=[request.id])
if not decision.authorized:
    output = "[BLOCKED] " + "; ".join(decision.denial_reasons)
elif name == "read_document":
    output = DOCUMENTS[arguments["name"]]
elif name == "send_summary":
    sent.append(arguments.copy())  # Simulated delivery.
    output = "Simulated delivery completed"
result = record(output, [request], tool=tool, result=True)
```

Record message-producing and processing operations, including application glue
outside framework calls, and check each action before its handler runs. Missing
dependencies make rules over input history incomplete.

## Read more

For framework-integrated information flow, see the
[LangChain examples](../langchain-information-flow/) for drafting and cooperating
agents and the [Langroid example](../langroid-information-flow/) for a
three-agent team with an MLS comparison. The
[ADK example](../adk-separation-of-duties/) shows approval across agents; see
[adding a framework](https://docs.sasy.ai/instrumentation/) for custom adapters.
