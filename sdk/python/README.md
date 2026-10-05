# sasy

Python client for [SASY](https://github.com/sasy-labs/sasy). It records the inputs
behind an agent's actions and asks the policy engine for a decision before each
protected tool runs.

The engine runs separately; follow [local engine setup](https://docs.sasy.ai/local-engine/)
before running an agent.

## Install

Install in a Python 3.11 or newer environment, then start the local engine with
Docker running:

```bash
pip install sasy
sasy engine start
```

For repository examples, use the
[checkout setup](https://docs.sasy.ai/local-engine/#repository-examples).

## Protect an agent

Local setup saves a selected connection profile in `~/.sasy`. The Python SDK
uses it from any project unless you configure another endpoint. Enable
instrumentation before building any agents:

```python
import sasy

sasy.configure()
sasy.instrument()
```

Run each task inside `with sasy.session(policy=...)`. The session records its
messages and binds the policy used to check tool calls. Cooperating agents share
the same session.

Supported frameworks: [LangChain](https://docs.sasy.ai/integrations/langchain/),
[Google ADK](https://docs.sasy.ai/integrations/google-adk/), and
[Langroid](https://docs.sasy.ai/integrations/langroid/).

For a complete runnable program, follow [framework examples](https://docs.sasy.ai/examples/)
or try the [LangChain examples](https://github.com/sasy-labs/sasy/tree/main/examples/langchain-information-flow).

Instrumentation enables supported installed frameworks. An unusable framework
is skipped with a warning; pass its flag as `True`, such as
`sasy.instrument(langchain=True)`, to require it. HTTP checks are off by default.
See [instrumentation options](https://docs.sasy.ai/instrumentation/)
for selection and HTTP behavior.

## Use the client directly

```python
import json
from pathlib import Path

import sasy

sasy.configure()
with sasy.session(policy=Path("read-only.dl")):
    decision = sasy.check_tool_call("Read", json.dumps({"file_path": "notes.txt"}))
    if decision.authorized:
        # Perform the protected action here.
        print(Path("notes.txt").read_text())
```

Supply your own policy file. For checks that depend on earlier messages, your
integration must also record those dependencies. See the
[adapter guide](https://docs.sasy.ai/instrumentation/).

See [configuration](https://docs.sasy.ai/configuration/),
[authentication](https://docs.sasy.ai/authentication/), and the
[security model](https://docs.sasy.ai/limits/) for setup and supported boundaries.
