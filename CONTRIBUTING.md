# Contributing to SASY

SASY includes the Rust policy engine and the Python and TypeScript SDKs.

Start with a focused issue or pull request that describes the behavior you want
to change. For changes to policy semantics, authentication or stored data,
explain what they mean for existing callers and deployments. For private
vulnerability reports follow `SECURITY.md`; for participation in project spaces
follow `CODE_OF_CONDUCT.md`.

Questions and proposals: open a GitHub issue. A good first contribution is an
adapter for another agent framework; the four steps are at
https://docs.sasy.ai/instrumentation/#integrate-once-reuse-across-agents and
`sdk/python/sasy/instrumentation/langchain.py` is the model to follow. No
contributor licence agreement is required.

## Development

The Python gRPC stubs under `sdk/python/sasy/proto` are generated from
`proto/*.proto` and committed. After changing a `.proto` file, regenerate them
with the pinned generator so the result is reproducible; CI fails if the
committed stubs differ from the `.proto` files:

```bash
uv pip install --python .venv/bin/python 'grpcio-tools==1.80.0'
PYTHON=.venv/bin/python scripts/generate_proto.sh
```

This works inside the Nix shell as well; the shell provides `uv` and `protoc`
but does not pin `grpcio-tools` itself.

Use Python 3.11 or newer with uv, Rust 1.95.0 (the CI toolchain), and protoc. Building
policies also needs Soufflé 2.5, a C++ compiler, and Python. The TypeScript
SDK uses Node.js/npm and Bun for its tests. See the package READMEs
for build commands. The Starlight docs under `docs-site/` describe platform
prerequisites in more detail.

```sh
uv sync --all-packages
cargo test --locked -p sasy-auth -p sasy-credential -p sasy-refmon \
  -p sasy-graph -p sasy-policy -p sasy-binary
make lint-rust
```

The [test guide](tests/README.md) gives commands for the fast Python suite, the
integration tests (which start their own engine), and the policy toolchain
checks.

`make lint-rust` checks formatting and runs Clippy on every Rust target, treating
warnings as errors. `make check-rust` is a faster local compiler check without
linking. CI pins Rust so changes to lint rules arrive through deliberate toolchain
updates. Run the build and relevant tests too; compiler checks do not run tests.

For TypeScript SDK changes, run `npm ci`, `npm run build:sdk`,
`npm run typecheck --workspace=sasy-js`, `npm run lint --workspace=sasy-js`,
and `npm test --workspace=sasy-js`. CI also validates the workflow
definitions and builds both SDK packages on pull requests, checking each in an
independent consumer environment.

Run tests relevant to the behavior you change. Tests marked `ignore` by Rust or
skipped by pytest are not evidence that an integration works: run the required
toolchain, platform, and service checks explicitly. Use independent dependencies in each worktree; do not link `node_modules`
to another checkout.

## Documentation

Public documentation lives in `docs-site/`. With Node.js 24 or newer, run
`make docs` to build it and serve the build at the local URL Astro prints
(`http://localhost:4321` by default); stop it with Ctrl-C. `make docs-build`
builds without serving. While editing, `npm run dev` from `docs-site` reloads
pages as you save.

## Pull requests

- Describe the problem, resulting behavior, and any compatibility limits.
- Add regression coverage when a behavior change needs it. Prefer realistic
  state transitions and public interfaces to tests that duplicate the code.
- State the commands you ran and any failures, ignored tests, or missing
  prerequisites. Do not count an old binary's results as validation of new code.
- Update the relevant Starlight pages in the same change. Describe how the
  system works now, including its limits. Build the site with `make docs-build` (Node.js 24 or newer) before you open the pull request.
- Keep generated files, local credentials, graph stores, and unrelated cleanup
  out of the change unless they are required deliverables.

A maintainer reviews the committed change before merging it. Changes made after
a review need another review, including changes made while resolving a merge
conflict.

## Releasing

The engine image and the Python SDK are released together, at one version: the
SDK's `sasy engine start` pulls `ghcr.io/sasy-labs/sasy:<SDK version>`.

1. Set the new version in `Cargo.toml` (`[workspace.package]` and the internal
   crate versions), `sdk/python/pyproject.toml`, `sdk/typescript/package.json`,
   `nix/package.nix`, the image tag in `Makefile` and the image tags in
   `docs-site/src/content/docs/local-engine.mdx`. Refresh `Cargo.lock` with
   `cargo update --workspace`. `python3 scripts/check_release_version.py
   sasy-vX.Y.Z` lists any version it missed.
2. Merge that change, then push the tag `sasy-vX.Y.Z` from the merged commit.
   The SDK release workflow calls `engine-release.yml` to run the core gate,
   qualify native linux/amd64 and linux/arm64 images, and publish their immutable
   version tag. It publishes the SDK to PyPI only after anonymous pulls verify
   that tag and both qualified architectures. A tag that disagrees with the
   tree is rejected, and a published image version is never replaced.
3. The first time only, an organization admin makes the `sasy` package public
   under the organization's Packages > sasy > Package settings. The image
   SDK publication remains blocked until the image is publicly pullable.
   If publication or visibility verification fails, fix the cause and rerun
   only the failed jobs. Rerunning the complete workflow intentionally fails
   the existing-image guard after an image version has been published.

## License

All original contributions to SASY must be provided under Apache-2.0.
Inclusion of pre-existing third-party material requires maintainer approval,
a compatible license, and preservation of its required license and attribution
notices. Identify the origin and license of any third-party material you propose
adding.
