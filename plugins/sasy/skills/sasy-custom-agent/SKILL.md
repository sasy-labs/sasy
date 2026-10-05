---
name: sasy-custom-agent
description: Add SASY policy enforcement to agent code that does not use a supported framework, using sasy.record, sasy.check_tool_call and sasy.session. Use for hand-written agent loops, custom orchestration or direct LLM API calls.
---

SASY decides; the application enforces. Add calls at exactly two points of the
user's agent loop, inside one session per task. Make sure a SASY engine is
running first (the `sasy-setup` skill).

**The primitives**

- `with sasy.session(policy=Path("policy.dl")):` binds the policy and scopes
  everything below. Pass a `Path`; a plain string is treated as policy source.
- `sasy.record(text, role=..., inputs=[...], tool_calls=[...], result_of=...)`
  records one message and returns its ID. `role` is `"system"`, `"user"`,
  `"llm"` or `"agent"` (the default; also used for tool results). `inputs` are
  IDs of the messages it was computed from; they become the dependency edges
  policies trace. `tool_calls` lists `(name, arguments)` a message requests;
  `result_of=(name, arguments)` marks a tool result. `sasy.record_async` is the
  async form.
- `sasy.check_tool_call(name, json.dumps(args), input_node_ids=[request_id])`
  asks the engine whether the call may run, given the ancestry of the message
  that requested it. A denial is a result, not an exception.

**Where to call them**

1. Record every message the loop produces or consumes: the user task, each
   model reply (with `inputs` set to everything sent to the model, and
   `tool_calls`), and each tool result (with `inputs=[request]` and
   `result_of`). Missing edges hide ancestry from policies.
2. Immediately before running a tool, check it with the requesting message's ID.
   Run the tool only when `decision.authorized` is true **and**
   `decision.transform_ids` is empty; a decision with transforms allows the call
   only with those credential transforms applied. Otherwise do not run it, and
   return `decision.denial_reasons` to the model as the tool's result.

```python
with sasy.session(policy=Path("policy.dl")):
    task = sasy.record(user_text, role="user")
    request = sasy.record(reply_text, role="llm", inputs=[task],
                          tool_calls=[(name, args)])
    decision = sasy.check_tool_call(name, json.dumps(args), input_node_ids=[request])
    if decision.authorized and not decision.transform_ids:
        result = run_tool(name, args)
        sasy.record(str(result), inputs=[request], result_of=(name, args))
    else:
        result = "Blocked: " + "; ".join(decision.denial_reasons)
```

Then write the policy with the `write-policy` skill and run the agent against
both an allowed and a denied case. Details:
https://docs.sasy.ai/quickstart/ and https://docs.sasy.ai/instrumentation/.
