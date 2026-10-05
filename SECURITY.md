# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through GitHub: open the repository's
**Security** tab and choose **Report a vulnerability**. That creates a private
advisory that only the maintainers can see. Please do not open a public issue or
pull request containing an undisclosed vulnerability.

Include the affected package and version or commit, the configuration needed to
reproduce the behavior, the impact you expect, and the evidence you have. Remove
live credentials and personal data from logs and attachments. We will coordinate
investigation, a fix where appropriate, and public disclosure with you. This
policy does not promise a response time or a bounty.

Until a supported-version policy is published, fixes target the current
development branch. Reporting an issue in an older release is still useful; do
not assume that older releases receive backported fixes.

Reports about sasy-guard, the separate Claude Code integration, belong in
https://github.com/sasy-labs/sasy-guard.

## What SASY defends against, and what it does not

SASY decides whether a proposed tool call may run. The decision is only as good
as what the application tells it, so the trust model is:

**Trusted.** The engine process and the host it runs on. The policy author and
the admin key that can replace a policy. The application code and the framework
adapter that record messages and place the check before each tool runs. Tool
implementations themselves: a tool that does more than its name suggests is
outside what a policy can see.

**Not trusted.** Model output, tool results, documents and any other content
that flows through the agent. A policy can deny an action whose ancestry — the
set of messages it was computed from — includes such content. SASY does not
judge whether content is an attack; it restricts what the agent may do after
reading it.

**Out of scope.** A tool dispatch path with no check in front of it: the engine
never hears about it and cannot deny it. An agent that can run arbitrary code
outside its tools. Content the application never recorded. Anything that already
reached the model provider: a tool result containing a secret is in the model's
context before any later check applies.

**The operator's responsibility.** Recorded message content is stored
unencrypted in the engine's data directory; protect that directory as you would
the messages themselves. Policy functors written in C++ and in-process evaluator
plugins run native code with the engine's privileges. On Linux, policy
compilation is confined with bubblewrap, a sandboxing tool, when bubblewrap is
installed and usable; if it is absent, or `DISABLE_BWRAP=1` is set, compilation
runs unconfined, so install it on any host that compiles policies you did not
write. macOS has no equivalent boundary and is for development. If you enable the HTTP hooks, your
model-provider requests, including the provider credential, pass through the
engine: use only an engine you operate.

The current gaps are listed at https://docs.sasy.ai/limits/.
