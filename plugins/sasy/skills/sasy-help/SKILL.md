---
name: sasy-help
description: Explain what SASY is and list the SASY skills and what each one does. Use when the user asks what SASY is, what the SASY skills can do, which SASY skills are available, or how to get started with SASY.
---

SASY enforces policies on what AI agents do. You write rules in Datalog; SASY
checks every tool call before it runs, against the history of messages and
tool results that led to it. A Python SDK instruments the agent; a local engine
(run with `sasy engine start`) makes the decisions. Docs: https://docs.sasy.ai/

For documentation lookup, https://docs.sasy.ai/llms.txt provides an index of
pages. Follow the relevant page links; use https://docs.sasy.ai/llms-full.txt
when you need the full documentation as plain text.

When asked what the SASY skills can do, list exactly these, with the example
request for each:

| Skill | What it does | Ask, for example |
| --- | --- | --- |
| `sasy-help` | Explains SASY and lists these skills | "What can the SASY skills do?" |
| `sasy-setup` | Installs the SDK, starts the engine and checks it works | "Set up SASY in this project." |
| `sasy-examples` | Runs and explains the example demos | "Walk me through a SASY demo." |
| `sasy-langchain` | Protects a LangChain agent | "Protect my LangChain agent with SASY." |
| `sasy-adk` | Protects a Google ADK application | "Protect my ADK agents with SASY." |
| `sasy-langroid` | Protects a Langroid program | "Protect my Langroid agent with SASY." |
| `sasy-custom-agent` | Adds SASY to your own agent loop | "Protect my own agent loop with SASY." |
| `write-policy` | Writes or revises a SASY policy | "Write a SASY policy that blocks emailing confidential documents." |
| `read-policy` | Explains an existing policy | "Explain this SASY policy." |
| `sasy-why-blocked` | Explains why SASY blocked an action | "Why did SASY block this tool call?" |

Do not invent other skills or describe them differently. To get started,
suggest `sasy-setup`, then the skill for the user's framework, then
`write-policy`. To see SASY work first, suggest `sasy-examples`.
