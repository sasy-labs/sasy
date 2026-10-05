# sasy-js

TypeScript and JavaScript client for [SASY](https://github.com/sasy-labs/sasy), a
policy engine that authorizes each tool call an agent wants to make using the
chain of messages that led to it.

This package is the client only. It has no framework adapter and does not
intercept tools by itself: you call `session.checkToolCall(...)` where your own
code dispatches a tool, and run the tool only if the decision is authorized. It
needs a running SASY engine ([start a local engine](https://docs.sasy.ai/local-engine/)).
Node.js 20 or newer.

Install from npm:

```bash
npm install sasy-js
```

To build from a checkout of the repository, run these commands from the root
with Node.js, npm and `protoc`:

```bash
npm ci --ignore-scripts --no-audit --no-fund
npm run build:sdk
```

See [the instrumentation contract](https://docs.sasy.ai/instrumentation/) for
what to record and where to check, and
[authentication](https://docs.sasy.ai/authentication/) for keys and roles.

## Check an action in a session

With an authenticated engine running, save this as an ES module in your application. Set `SASY_API_KEY` to your client key and point `caPath` at the certificate authority file that signs the engine's certificate:

```js
import { randomUUID } from "node:crypto";
import * as sasy from "sasy-js";

const apiKey = process.env.SASY_API_KEY;
if (!apiKey) throw new Error("Set SASY_API_KEY to your client key");
sasy.configure({
  url: process.env.SASY_URL ?? "localhost:10089",
  caPath: "certs/server.crt",
  authHook: new sasy.ApiKeyAuthHook(apiKey),
});

const session = sasy.session(randomUUID());
try {
  const binding = await session.setPolicy(
    'IsAuthorized(idx) :- Actions(idx, action), IsTool(action, "Read").',
  );
  if (!binding.accepted) throw new Error(binding.message);
  const decision = await session.checkToolCall({
    fnName: "Read",
    args: JSON.stringify({ file_path: "notes.txt" }),
  });
  console.log(decision.authorized); // true; no tool is executed here
} finally {
  await session.endSession();
}
```

Keep observations and checks on this session object; it supplies `sessionId` to every request. Creating it does not set scope for top-level SDK functions. Record input messages before using their IDs in `checkToolCall({ fnName, args, inputNodeIds })`. `endSession()` releases the evaluator, policy binding, session metadata and ownership. Stored graph observations remain; use a new session ID for a new conversation.

## Developing this package

The tests also require [Bun](https://bun.sh/):

```bash
npm run typecheck --workspace=sasy-js
npm run lint --workspace=sasy-js
npm test --workspace=sasy-js
```

The source is in `sdk/typescript`; the package is published as `sasy-js`.
