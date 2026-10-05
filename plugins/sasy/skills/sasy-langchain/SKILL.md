---
name: sasy-langchain
description: Protect a LangChain agent with SASY so every tool call is checked against a policy before it runs. Use when the user's agent is built with LangChain or LangGraph and they want SASY enforcement.
---

1. Make sure SASY is installed with the LangChain extra (`sasy[langchain]`,
   which pins the supported LangChain/LangGraph versions) and an engine is
   running; use `sasy-setup` if not.
2. Call `sasy.instrument()` once at startup, after any `sasy.configure(...)` and
   **before** the agent is built. It makes LangChain's `create_agent` build a
   SASY-checked agent.
3. Build the agent with `create_agent(model, tools)` from `langchain.agents`.
4. Run each task inside `with sasy.session(policy=Path("policy.dl")):`.

```python
from pathlib import Path
import sasy
from langchain.agents import create_agent

sasy.instrument()
agent = create_agent(model, tools)
with sasy.session(policy=Path("policy.dl")):
    result = agent.invoke({"messages": [{"role": "user", "content": task}]})
```

A denied call does not run; the model receives an error tool message.

**Check these in the user's code** before calling it done:

- Tools must run through the agent `create_agent` builds. Inside a SASY session,
  a `ToolNode` SASY did not build is refused, and a tool that graph code calls
  directly (not through a `ToolNode`) is **not checked at all**. A custom
  LangGraph `StateGraph` needs the `sasy-custom-agent` approach instead.
- The model's own HTTP calls are not checked unless `sasy.instrument(http=True)`;
  then the policy must allow the model provider's host.
- Interacting agents that must be judged together share one session.

Then write the policy (`write-policy`) and test one allowed and one denied
call. Details: https://docs.sasy.ai/integrations/langchain/
