# Information flow through LangChain agents

These examples show SASY blocking external publication when an agent's decision
could have been influenced by confidential information. The rule still applies
after drafting or delegation to another agent. Documents and publication are
simulated; the LangChain agents, SASY dependency graph, and policy checks are
real. Scripted models make the outcomes repeatable without provider keys.

## Run the examples

Complete the repository [Quick start](../../README.md#quick-start) first, then
run from the repository root:

```bash
uv run examples/langchain-information-flow/demo.py
uv run examples/langchain-information-flow/supervisor.py
```

In the single-agent walkthrough, only the confidential external publication
is denied; the supervisor shows the same rule across cooperating agents. To
run either example with a real model, add `OPENAI_API_KEY=...` to the
repository-root `.env` and run:

```bash
uv run examples/langchain-information-flow/demo.py --live
uv run examples/langchain-information-flow/supervisor.py --live
```

Live mode defaults to `gpt-4.1-mini` and makes paid provider requests. Set
`SASY_LIVE_MODEL` in `.env` or pass `--live-model MODEL` to choose another model.
Both programs load `.env` without overriding exported values. The supervisor
uses the live model for all three agents and records the messages exchanged
through its research and review tools in the same SASY session. Document reads
and publication remain simulated; live model choices can vary between runs.

For other scenarios, use `demo.py --destination internal` or `demo.py --public`
to allow a single-agent publication. Use `supervisor.py --public` for an allowed
cooperating-agent run, or `supervisor.py --async` for asynchronous agents and
tools. A scripted run prints a narrated walkthrough of the policy and each
scenario, pausing between steps in a terminal (`--no-pause` turns this off).
`--trace` prints agent-labeled model outputs, tool calls, and tool responses
instead; `--quiet` prints only the final delivery summary.

## What the system does

- [`demo.py`](demo.py): one agent reads a public or confidential document, calls
  a drafting tool, then requests publication to an internal or external channel.
- [`supervisor.py`](supervisor.py): a supervisor delegates research and public
  review to two worker agents in parallel, receives their summaries, then
  requests publication of a combined report.

All agents are built with LangChain's `create_agent` after `sasy.instrument()`.
Cooperating agents run in the same SASY session; instrumentation records the
messages and tool results that feed each action and checks tools before they run.
Under this information-flow policy, parallel workers do not affect one another's
authorization until they communicate. The supervisor's combined report depends
on both workers' returned answers.

## The policy

Both programs use [`policy.dl`](policy.dl). Reads, drafting, and delegation are
allowed. Publication to either channel is allowed unless it targets `external`
and the action's input history contains a `read_confidential` result:

```prolog
ConfidentialInput() :- CurrentDepends(id), ToolResult(id, "read_confidential", _).

Unauthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "publish"),
    a = $CallTool(_, args),
    @json_get_str(args, "destination") = "external",
    ConfidentialInput().
```

`CurrentDepends` follows the dependencies behind the current action. The rule
checks for a confidential read anywhere in that history, even across tool and
agent boundaries. It does not require an attacker or untrusted input.

| Inputs and destination | Expected result |
| --- | --- |
| Confidential → external (default) | Denied; no delivery |
| Confidential → internal | Allowed; one simulated delivery |
| Public → external | Allowed; one simulated delivery |

The [walkthrough](https://docs.sasy.ai/first-agent/) explains the output and rule;
the [LangChain guide](https://docs.sasy.ai/integrations/langchain/) covers the
integration and supported configurations.
