---
name: sasy-adk
description: Protect a Google ADK agent application with SASY so function tools, agent transfers and child agents are checked against a policy. Use when the user's agents are built with Google ADK and they want SASY enforcement.
---

1. Make sure SASY is installed with the ADK extra (`sasy[adk]`); it requires
   **Google ADK 2.9.1 exactly**. An engine must be running; use `sasy-setup` if
   not.
2. Call `sasy.instrument()` once at startup. It enables the ADK adapter when
   ADK is installed. Agents, tools and models do not change.
3. Run the existing `Runner` inside `with sasy.session(policy=...):`, using
   `runner.run_async`.

```python
from pathlib import Path
import sasy

sasy.instrument()
# ... build agents and an ordinary google.adk.runners.Runner as before ...
with sasy.session(policy=Path("policy.dl")):
    async for event in runner.run_async(user_id=user_id, session_id=session_id,
                                        new_message=message):
        ...
```

What is checked: every function tool immediately before its callable runs
(with ADK defaults filled in), agent transfers, `AgentTool` children and native
task children. Ancestry follows values through session state (`output_key`,
`{key}` templates, `tool_context.state`) and text artifacts. A denied tool does
not run; the model receives a result beginning with `[BLOCKED]`.

Before calling it done, compare the user's application with the supported
execution paths and boundaries in the guide (for example, application-created
child tasks and values from another SASY session need care). Then write the
policy (`write-policy`) and test an allowed and a denied case. Details:
https://docs.sasy.ai/integrations/google-adk/
