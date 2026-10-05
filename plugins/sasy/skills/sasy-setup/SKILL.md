---
name: sasy-setup
description: Install the SASY Python SDK, start a local SASY engine and confirm it enforces a policy. Use when the user asks to install, set up or start SASY, or when another SASY skill needs a running engine.
---

Set SASY up in the user's project so their agent code can use it. Do the steps;
do not hand them to the user.

1. **Check prerequisites.** Check that Python 3.11 or newer is available and
   Docker or a compatible provider is installed and started: `docker info`
   must succeed. If it is not running, tell the user how to start their
   provider and stop there.
2. **Install the SDK** into the project's environment. Prefer uv, which is much
   faster than pip:
   - a uv project (has `pyproject.toml`): `uv add sasy`
   - a virtual environment without `pyproject.toml`: `uv pip install sasy`
   - no uv: `pip install sasy`
   Add a framework extra when the project uses one: `sasy[langchain]`,
   `sasy[adk]` or `sasy[langroid]` (they pin the supported versions).
3. **Start the engine:** `sasy engine start` (or `uvx sasy engine start`
   without the SDK). It runs the engine image in Docker on `127.0.0.1:10089`,
   creates its keys and certificates on first start, and saves the client
   settings in `~/.sasy`, which the SDK uses in every project. It returns once
   the engine accepts TLS connections. `sasy engine status` shows it;
   `sasy engine stop` stops it and keeps its data.
4. **Prove it works** with a two-line policy and two checks, run in the
   project's environment:

   ```python
   import json, sasy
   policy = 'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "read").'
   with sasy.session(policy=policy):
       assert sasy.check_tool_call("read", json.dumps({})).authorized
       assert not sasy.check_tool_call("write", json.dumps({})).authorized
   print("SASY engine is enforcing policies")
   ```

5. Report what was installed, that the engine is running, and point to the next
   step: `sasy-langchain`, `sasy-adk`, `sasy-langroid` or `sasy-custom-agent`
   to protect an agent, and `write-policy` to write its policy.

If a step fails, read its error rather than retrying blindly. `UNAUTHENTICATED`
or a TLS error usually means settings from an older engine; see
https://docs.sasy.ai/local-engine/ (troubleshooting and profiles). A new engine
denies everything until a session binds a policy; that is expected.
