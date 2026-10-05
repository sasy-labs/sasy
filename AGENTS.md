# Working in this repository (for coding agents and humans)

- Follow `CONTRIBUTING.md` for the toolchain, tests and licensing terms.
- Any behavior change updates the matching page under `docs-site/src/content/docs/`
  in the same pull request. Describe what the system does now and what it does not
  do; do not replace a measured result with a statement of intent.
- Test against the engine built from this checkout (`make build-rust-dev`, then
  name `target/debug/sasy` explicitly). `tests/README.md` lists the test lanes and the
  dependencies each one needs.
- Use synthetic data in tests and examples. Never commit credentials, graph stores
  or real user content.
- Policy-authoring skills for coding agents live in `plugins/policy-compiler/skills/`.
- Published documentation is at `docs.sasy.ai`, built from `docs-site/`.
- sasy-guard, the Claude Code integration, is a separate project:
  https://github.com/sasy-labs/sasy-guard, with its site at `guard.sasy.ai`. Its
  daemon is not part of this repository.
