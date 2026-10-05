# Policies

`common_policy.dl` is the shared schema every SASY policy builds on. It declares
the relations the engine fills in before each check — the proposed actions, the
recorded messages and the dependencies between them, the authenticated caller
and its roles — and the relations your rules write to, `IsAuthorized` and
`Unauthorized`. It also declares the built-in functions (called functors) a rule
may call, such as `@json_get_str` for reading a tool argument.

The file itself is a symbolic link to `souffle/common_policy.dl`, where the
engine's build keeps it. On a system without symbolic links, read it at that
path instead.

You do not include it: the policy preprocessor puts it in front of every policy
before Soufflé, the Datalog compiler, sees it. Write only your own rules.

Each relation is explained in plain words, with worked rules, at
https://docs.sasy.ai/policy-language/. Complete example policies live under
`examples/`.
