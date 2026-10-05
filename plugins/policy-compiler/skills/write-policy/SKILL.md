---
name: write-policy
description: Write or revise a SASY Soufflé policy using the public schema and local validator. Use for policy-authoring requests, not hosted translation-service tasks.
---

Read the user's policy requirements and the schema: `policies/common_policy.dl` in a SASY checkout, or otherwise the relations table at https://docs.sasy.ai/policy-language/ (the same file is at https://github.com/sasy-labs/sasy/blob/main/policies/common_policy.dl). The schema is the contract: use its existing relations and functors, and inspect the integration's actual message fields and dependency edges before depending on them.

Write the policy at the requested path. Explain which observations it requires, what is allowed and denied, and which unknown states fail closed. Keep any custom C++ functor source separate; admission of that source requires an administrator unless the operator explicitly opts in.

Validate it. In a SASY checkout, run `make souffle-validate FILE=path/to/policy.dl` (local preprocessing and Soufflé validation). Without a checkout, start an engine (`sasy engine start`, see the `sasy-setup` skill) and bind the policy: entering `with sasy.session(policy=Path("policy.dl")):` compiles and validates it, and fails with the compiler's errors. Pass a `Path`; a string is treated as policy source. A syntax pass does not establish enforcement correctness: exercise representative allowed, denied and missing-data cases against the configured engine when available, and distinguish untested assumptions in the result.

The Python `sasy.policy.api` and TypeScript policy APIs can bind the policy to a session. Upload or change a running deployment only when that operation is in the user's authorized scope. This skill does not call a hosted translator, capture private trajectories, or claim formal equivalence proofs.
