---
name: read-policy
description: Explain a SASY Datalog policy, its message dependencies, authorization behavior and limitations using the public schema.
---

Read the requested policy, its includes, the schema (`policies/common_policy.dl` in a SASY checkout, or the relations table at https://docs.sasy.ai/policy-language/), and any companion functor source. Trace the authorization rules to the observations they consume. Separate policy decisions from transport authentication and application execution.

Describe the allowed and denied actions, the role of dependency edges, missing-data behavior, and any custom functor assumptions. Use small concrete allowed and denied examples when helpful. A rule matching stored text is only as accurate as the integration's event fields and dependency graph.

Use `make souffle-validate FILE=path/to/policy.dl` in a checkout, or bind the policy to a session on a running engine, when validation helps. Report unavailable tools or untested runtime behavior honestly. Reading a policy does not authorize uploading it or changing a running service.
