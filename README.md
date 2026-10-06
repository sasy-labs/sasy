<p align="center">
<a href="https://sasy.ai/">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs-site/public/assets/sasy-wordmark-dark.svg">
    <img src="docs-site/public/assets/sasy-wordmark-light.svg" alt="SASY — Seamless Agent Security" width="340">
  </picture>
</a>
</p>

<p align="center">
  <a href="https://docs.sasy.ai/"><img src="https://img.shields.io/badge/docs-docs.sasy.ai-6366f1" alt="Docs"></a>
  <a href="https://pypi.org/project/sasy/"><img src="https://img.shields.io/pypi/v/sasy?cacheSeconds=300" alt="PyPI version"></a>
  <a href="https://www.npmjs.com/package/sasy-js"><img src="https://img.shields.io/npm/v/sasy-js?cacheSeconds=300" alt="npm version"></a>
  <a href="https://github.com/sasy-labs/sasy/actions/workflows/ci.yml"><img src="https://github.com/sasy-labs/sasy/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://arxiv.org/abs/2602.16708"><img src="https://img.shields.io/badge/arXiv-2602.16708-b31b1b" alt="Paper"></a>
</p>

<p align="center">
  <a href="https://docs.sasy.ai/"><b>Docs</b></a> ·
  <a href="https://blog.sasy.ai/"><b>Blog</b></a> ·
  <a href="https://sasy.ai/"><b>Website</b></a>
</p>

SASY enforces policies on what AI agents do. You write rules as a Datalog policy;
SASY compiles it and checks every tool call before it runs, against the history of
messages and tool results that led to that call, even through summaries, tools and
handoffs between agents.

Your agent records each message and tool result in a dependency graph, and asks
the SASY engine about each proposed tool call before it runs; the engine allows it
or returns the denial reason to the agent. For LangChain, Google ADK and Langroid,
`sasy.instrument()` does this automatically. Any other agent makes the same two
calls with SASY's primitives: `sasy.record` for each message and
`sasy.check_tool_call` before each tool runs:

[![SASY architecture: instrumentation connects the agent to its tools and services, records messages and dependencies in the engine's dependency graph, and checks proposed actions against developer policies. Allowed calls execute; denied calls return a reason to the agent.](docs-site/public/diagrams/architecture.svg)](https://docs.sasy.ai/architecture/)

**We actively maintain SASY and provide support for policy formalization and integration. If you have a system or use case where you'd like to use SASY, please reach out.**


## Quick start

For convenience, SASY includes skills for coding agents (Claude Code, Codex,
Cursor and others) that help with setup, framework integration, policy writing
and explaining denials. Install them with:

```bash
npx skills add sasy-labs/sasy
```

Then ask your agent to walk you through a SASY demo or integrate enforcement into your system.

### Use SASY in your project

You need uv and a running Docker installation. No checkout of this repository is
needed. In your project:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv add sasy            # in a project with a pyproject.toml
# uv pip install sasy  # in a plain directory without one
sasy engine start
```

`sasy engine start` starts the policy engine, with a default configuration set up
at `~/.sasy`, automatically read by the Python SDK.
See [local engine](https://docs.sasy.ai/local-engine/) for configuration options.
Without installing the SDK, you can also run `uvx sasy engine start`.

Write your rules in Datalog using the [policy guide](https://docs.sasy.ai/policy-language/),
or describe your policy in natural language and use the
[`write-policy` skill](plugins/policy-compiler/skills/write-policy/) to translate
it into enforceable Datalog.
Then enable SASY before building your agent and run each task in a session.
For example, with your LangChain `model` and `tools`:

```python
from pathlib import Path
import sasy
from langchain.agents import create_agent

sasy.instrument()
agent = create_agent(model, tools)
with sasy.session(policy=Path("policy.dl")):
    result = agent.invoke({"messages": [{"role": "user", "content": "Your task"}]})
```

Google ADK and Langroid agents are also
supported; see the [ADK](https://docs.sasy.ai/integrations/google-adk/) and
[Langroid](https://docs.sasy.ai/integrations/langroid/) integration guides.

### Your own agent loop

To integrate SASY with your own orchestration, record messages and dependencies
produced by your system, and check the authorization of actions before they
execute.

```python
import json
from pathlib import Path
import sasy

with sasy.session(policy=Path("policy.dl")):
    task = sasy.record("Summarize the report and email the team", role="user")
    report = sasy.record(report_text, result_of=("read_document", {"name": "report"}))
    request = sasy.record(model_reply, role="llm", inputs=[task, report],
                          tool_calls=[("send_email", args)])
    decision = sasy.check_tool_call("send_email", json.dumps(args),
                                    input_node_ids=[request])
    if decision.authorized:
        result = send_email(**args)
        sasy.record(result, inputs=[request], result_of=("send_email", args))
    else:
        print(decision.denial_reasons)
```

`sasy.record` returns the message's ID; its `inputs` become edges in the dependency
graph over which policies operate.
The built-in framework adapters make these calls for you; see
[add a framework](https://docs.sasy.ai/instrumentation/) to integrate yours.
See [SDK usage patterns](https://docs.sasy.ai/sdk/) for connection options and
framework examples, and [use the SDK directly](https://docs.sasy.ai/quickstart/)
for a custom agent loop.

## Example: information flow

**Task:** A coordinator asks two summarizer agents to review a sensitive report
and an external note, then email a summary to the intended reviewer. The note
tries to redirect the email.

**Policy:** Deny a send if **both sensitive and untrusted inputs** could have
influenced it, even through a summary. The core rule in the
[example policy](examples/message-flow/policy.dl) is:

```prolog
Unauthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "send_summary"),
    DependsOnUntrusted(),
    DependsOnSensitive().
```

The full policy defines these predicates using the current action's
[dependency graph](https://docs.sasy.ai/dependency-graph/). The diagram shows two
timings: when the note's summary arrives before the send, the redirected send is
blocked; before it arrives, the intended send is allowed.

[![Two variants of a coordinator and two summarizer agents. When the untrusted note's summary arrives before the send, sensitive and untrusted inputs both influence it and the redirected email is blocked. When the note's summary has not arrived, the intended email is allowed.](docs-site/public/diagrams/information-flow.svg)](https://docs.sasy.ai/get-started/#see-the-decision)

The runnable [message-flow example](examples/message-flow/) demonstrates the
same rule with a scripted single agent in two runs. It uses synthetic documents,
simulated email, and a local engine; no model-provider key is needed. From a clone
of this repository, with the engine running, run
`uv run examples/message-flow/demo.py`.
To run it with a real model, add `OPENAI_API_KEY` to `.env` and run:

```bash
uv run examples/message-flow/demo.py --live
```

See the [example guide](examples/message-flow/) for model and scenario options.

## What you can enforce

SASY’s policy language, [Datalog](https://docs.sasy.ai/policy-language/), expresses
rules over facts and relationships. Policies can express, for example:

- **Information flow:** restrict publication of outputs derived from confidential
  documents, including after a drafting step rewrites them.
- **Approvals:** require a review in a payment's ancestry that matches its request
  ID, payee, and amount. See the [ADK example](examples/adk-separation-of-duties/).
- **Access and workflow rules:** check roles, require a completed step before the
  next action, or apply different policies to the same agent. The
  [Langroid example](examples/langroid-information-flow/) compares multi-level
  security and toxic-flow policies in a three-agent program.

## Develop SASY

`make docker-build` builds the engine
image from source; `uv run sasy engine stop`, then
`uv run sasy engine start --image sasy`, runs that build.
[Build from source](https://docs.sasy.ai/building/) covers Nix and native
toolchains; [CONTRIBUTING.md](CONTRIBUTING.md) covers tests, documentation, pull
requests and releases.

## Learn more

SASY includes a Rust engine and Python and TypeScript SDKs.

[See the paper](https://arxiv.org/abs/2602.16708).

- [Documentation](https://docs.sasy.ai/), [local engine setup](https://docs.sasy.ai/local-engine/),
  and [building from source](https://docs.sasy.ai/building/).
- [Security model](https://docs.sasy.ai/limits/): what SASY trusts and guarantees.
- [Blog](https://blog.sasy.ai/) for research notes and policy examples.
- [Contributing](CONTRIBUTING.md) and [engine crates](crates/README.md).
- [Skills for coding agents](plugins/): setup, framework integration, denial
  explanations and policy authoring.
- [Security](SECURITY.md) for private vulnerability reporting.

For Claude Code, see the separate [sasy-guard](https://github.com/sasy-labs/sasy-guard)
project and [its documentation](https://guard.sasy.ai/).

SASY is licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Cite this work

If you find this work useful in your own research, please consider citing the following:

```bibtex
@misc{palumbo2026formalpolicyenforcementrealworld,
  title={Formal Policy Enforcement for Real-World Agentic Systems},
  author={Nils Palumbo and Sarthak Choudhary and Jihye Choi and Guy Amir and Prasad Chalasani and Somesh Jha},
  year={2026},
  eprint={2602.16708},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  url={https://arxiv.org/abs/2602.16708},
}
```
