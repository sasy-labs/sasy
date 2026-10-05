# Payment approval across ADK agents

This example shows SASY requiring evidence of approval before an agent can make
a simulated payment. The request and approval must be in the payment's actual
input history and match its request ID, payee, and amount. The agents use real
ADK runners and a real SASY engine; documents and ledger entries are synthetic,
and no payment service is contacted.

## Run the example

Complete the repository [Quick start](../../README.md#quick-start), then run
from the repository root:

```bash
uv run examples/adk-separation-of-duties/main.py
uv run examples/adk-separation-of-duties/main.py --scenario wrong-payee
```

A plain run prints a narrated walkthrough: the policy, then the approved payment
and several denied ones, with SASY's decision on each tool call, waiting for
Enter between steps (`--no-pause` turns that off). `--scenario NAME` narrates
one scenario. Use `--trace` to print every completed ADK event and the JSON
ledger summary instead, or `--quiet` to print only the summary.

For a live Gemini run, add `GOOGLE_API_KEY=...` to the repository-root `.env`,
then run:

```bash
uv run examples/adk-separation-of-duties/main.py --live
```

The live run defaults to `gemini-3.8-flash` and makes paid provider requests;
payments remain simulated. Set `SASY_ADK_LIVE_MODEL` in `.env` to select a
different Gemini model. The demo loads `.env` without overriding exported values.

## What the system does

[`main.py`](main.py) runs three agents in a serial ADK `Workflow`:

1. `requester` calls `submit_payment` with a payment request.
2. `reviewer` calls `approve_payment` with the approved details.
3. `payer` calls `disburse_payment`, which adds a payment to an in-memory ledger
   only if SASY authorizes the call.

SASY's ADK instrumentation records the tool results and the messages passed
between agents. Those dependencies show which request and approval could have
influenced the payer's action. The default run uses scripted models. With
`--live`, the agents use Gemini on the same synthetic invoice; SASY also checks
the model's HTTP requests to Google, which the example policy allows.

## The policy

[`payment_policy.dl`](payment_policy.dl) allows requests and approvals. It allows
a disbursement only when its input history contains successful `submit_payment`
and `approve_payment` results from the configured `requester` and `reviewer`
agents. Both results must match **all three payment fields** in the proposed
call. Its `ApprovedPayment` relation collects that matching evidence:

```prolog
IsAuthorized(idx) :-
    Actions(idx, a),
    a = $CallTool("disburse_payment", args),
    ApprovedPayment(@json_get_str(args, "request_id"),
                    @json_get_str(args, "payee"), @json_get_str(args, "amount")).
```

An approval for a different payment does not satisfy the rule. Nor does approval
text alone: the policy requires the recorded tool result in this action's
history.

## Scenarios

Use `--scenario NAME` to run any row below; `approved` is the default.

| Scenario | Policy evidence | Result |
| --- | --- | --- |
| `approved` | Matching request and approval from the expected agents | One simulated payment |
| `wrong-payee`, `wrong-amount`, `wrong-request` | The payment differs from the approved details | Denied; empty ledger |
| `missing-approval`, `same-agent` | Required evidence from the reviewer is absent | Denied; empty ledger |
| `denied-approval`, `approval-error` | The approval tool did not produce successful approval evidence | Denied; empty ledger |

The dependency graph lets authorization depend on completed work that influenced
the action, rather than accepting the payer's claim that a payment was approved.
The approved run prints one payment under `dispatched`; refused cases print an
empty list. See the [ADK guide](https://docs.sasy.ai/integrations/google-adk/)
for supported configurations.
