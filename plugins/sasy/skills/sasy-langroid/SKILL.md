---
name: sasy-langroid
description: Protect a Langroid multi-agent program with SASY so tool calls are checked against a policy before they run. Use when the user's agents are built with Langroid and they want SASY enforcement.
---

1. Make sure SASY is installed with the Langroid extra (`sasy[langroid]`, which
   accepts Langroid 0.67.1 to 0.67.8; the adapter does not check the version
   itself, so install it through the extra). An engine must be running; use
   `sasy-setup` if not.
2. Call `sasy.instrument()` once at startup. Pass `langroid=True` so a missing
   or unloadable Langroid is an error instead of a warning that agents will not
   be checked.
3. Run each conversation inside its own `with sasy.session(policy=...):` block.

```python
from pathlib import Path
import sasy

sasy.instrument(langroid=True)
# ... build ChatAgents and Tasks as before ...
with sasy.session(policy=Path("policy.dl")):
    task.run(user_message)
```

What is checked: custom tools and Langroid's built-in orchestration tools,
immediately before they run. Agent responses, task handoffs, model inputs and
history truncation are recorded, so a check sees the full ancestry; each tool
in a multi-tool reply is recorded and checked separately.

Before calling it done:

- Keep `tool_policy_fail_closed=True` (the default), so a tool whose decision
  could not be obtained does not run.
- If the user's code calls a handler directly, edits messages itself, composes
  messages outside Langroid or subclasses `Task`, follow the matching section
  of the guide; those paths need explicit recording.
- With `sasy.instrument(http=True)`, `httpx`/`requests` calls are checked too,
  and the policy must allow the model provider's host.

Then write the policy (`write-policy`) and test an allowed and a denied case.
Details: https://docs.sasy.ai/integrations/langroid/
