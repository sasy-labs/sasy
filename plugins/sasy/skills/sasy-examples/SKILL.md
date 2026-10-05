---
name: sasy-examples
description: Run and explain the SASY example demos (message-flow, LangChain, Google ADK, Langroid) - which example shows what, how to run one, its options, and how to read its output. Use when the user wants to see SASY working, asks for a demo or a walkthrough, or asks how to run an example or what its output means.
---

The examples live in the `examples/` directory of the SASY repository, so they
need a clone. From the clone's root, start a local engine once (it needs
Docker; use the `sasy-setup` skill if this fails):

```bash
git clone https://github.com/sasy-labs/sasy
cd sasy
uv run sasy engine start
```

Run every example from the repository root. None needs a model-provider key:
the models are scripted, and documents, email, publishing and payments are
simulated. The SASY engine and its allow or deny decisions are real.

| Example | Command | What it shows |
| --- | --- | --- |
| message-flow (start here) | `uv run examples/message-flow/demo.py` | An agent summarizes a confidential report and emails it. When it has also read an untrusted note that redirects the email, SASY blocks the send. |
| LangChain | `uv run examples/langchain-information-flow/demo.py` | Publishing to the external channel is denied when the call depends on a confidential document, even after a summary rewrote it. `supervisor.py` in the same directory applies the rule across cooperating agents. |
| Google ADK | `uv run examples/adk-separation-of-duties/main.py` | Separation of duties: a payment is disbursed only with a matching request and approval from the right agents. |
| Langroid | `uv run examples/langroid-information-flow/demo.py --policy mls` | The same program under two policies (`mls` or `toxic-flow`), including an inbox document that hides an instruction to forward a merger plan. |

## Running an example

A plain run prints a narrated walkthrough: what the agent is asked to do, the
policy, then each run with SASY's decision on every tool call and a short
explanation, and a summary table. In a terminal it waits for Enter between
steps. When you run an example for the user from a non-interactive shell, it
does not wait; read the output and walk the user through it.

Options every example has (`--help` lists all of them, with examples):

- `--trace` replaces the narration with a verbose trace: the recorded events
  (message-flow), the model and tool callbacks (LangChain), the completed ADK
  events, or Langroid's full output.
- `--quiet` prints only the final summary.
- `--no-pause` skips the Enter prompts.
- `--live` lets a real model choose the actions. It makes paid provider
  requests and needs a key in the repository-root `.env`: `OPENAI_API_KEY`, or
  `GOOGLE_API_KEY` for the ADK example. Langroid also needs `--model NAME`.
  Live results depend on the model.

Options for one example:

- message-flow: `--only-note` shows only the run with the untrusted note.
- ADK: `--scenario NAME` runs one scenario (`approved`, `wrong-payee`,
  `wrong-amount`, `wrong-request`, `missing-approval`, `same-agent`,
  `denied-approval` or `approval-error`).
- LangChain: `--public` and `--destination internal|external` select one run;
  `supervisor.py --async` runs the agents asynchronously.
- Langroid: `--policy mls|toxic-flow` picks the policy; `--scenario` runs one of
  `1`, `2`, `3`, `4` or `read-up`.

## Reading the output

Each event line names who acted: `user`, `agent` (or the agent's name), or
`model`. After every tool call, a `SASY` line says `✓ allowed` or `✗ denied`.
A denial quotes the reason SASY returned, which is the policy's
`// @deny_message` when the rule has one. After each
run, a short paragraph explains which part of the call's history made the
decision. Each example's policy is the `.dl` file in its directory.

When the user wants to go further, hand off:

- why a specific call was denied: `sasy-why-blocked`;
- what a policy file means: `read-policy`;
- change the policy or write one for their own agent: `write-policy`;
- add SASY to their own code: `sasy-setup`, then the skill for their framework.

Do not claim that an example sends real email, publishes anything or moves
money.
