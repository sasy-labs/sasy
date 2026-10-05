# Testing SASY

Run these commands from the repository root. The default Python suite tests SDK
contracts and local build/configuration helpers without starting an engine or
contacting hosted services.

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e 'sdk/python[langroid,langchain,adk]' 'langroid==0.67.8' pytest pytest-asyncio pytest-timeout 'langchain-openai==1.6.2'
SASY_REQUIRE_FRAMEWORKS=1 .venv/bin/python -m pytest -q
```

## Engine integration

Build the engine from this checkout, then select the integration lane explicitly:

```sh
make build-rust-dev
SASY_TEST_ENGINE="$PWD/target/debug/sasy" \
  .venv/bin/python -m pytest -q --run-integration -m integration
```

These tests start their own loopback engines with temporary stores, generated TLS
certificates and independent authentication keys. They cover authorization,
credentials, message dependencies, isolation between tenants (a tenant is one
isolated customer or environment; sessions and default policies belong to a
tenant), policy replacement and restart behavior. Tests of the optional
LLM-backed policy predicate use a local mock provider. No model-provider
credentials or shared running services are needed.

The engine tests require the policy toolchain described in the
[build guide](https://docs.sasy.ai/building/). A missing or unusable
`SASY_TEST_ENGINE` fails the selected integration lane. The fixture stops its
processes and removes temporary state when the tests finish.

The framework tests exercise native Langroid tasks and message edits, LangChain
information flow, and ADK separation of duties, native delegation and resource
provenance against the engine with scripted model responses. Provider calls are
separate, opt-in tests using synthetic data. Set `OPENAI_API_KEY` and
`SASY_LIVE_MODEL` for LangChain, or `GOOGLE_API_KEY` and `SASY_ADK_LIVE_MODEL` for
ADK, then select the corresponding file:

```sh
SASY_RUN_LIVE_FRAMEWORK_TESTS=1 SASY_TEST_ENGINE="$PWD/target/debug/sasy" \
  .venv/bin/python -m pytest -q --run-integration -k live \
  tests/integration/test_langchain_integration.py
# For ADK, select tests/integration/test_adk_integration.py instead.
```

Choose a tool-capable model available to your account. These tests make billable
provider requests; normal CI does not enable them. See the example READMEs for
the supported framework configurations and how to run each scenario directly.

## Policy toolchain

The CSV harness test sends hostile quoted/multiline message fields through the
actual Soufflé parser. Select it explicitly after installing Soufflé:

```sh
.venv/bin/python -m pytest -q --run-toolchain -m toolchain
```

The CI workflow `.github/workflows/ci-core.yml` also runs, on Linux, the
sandboxed policy compilation and runtime checks and the comparison between the
compiled and interpreted Soufflé backends. A passing run on macOS says nothing
about the Linux sandbox, which has no macOS equivalent.

## Other suites

Check Python SDK types and lint with:

```sh
uv pip install --python .venv/bin/python --group typing 'ruff==0.16.6'
.venv/bin/python -m mypy sdk/python/sasy
.venv/bin/ruff check sdk/python
```

The typing group installs stubs and the OpenTelemetry SDK used by optional
instrumentation. It does not require a model-provider account.

Rust unit and integration tests live with their crates; TypeScript tests live in
`sdk/typescript/test`. The message-flow demo's unit and owned-engine checks run
with the Python lanes above; neither calls a model provider. Other checks:

```sh
npm ci --ignore-scripts --no-audit --no-fund
npm run build:sdk
npm run typecheck --workspace=sasy-js
npm run lint --workspace=sasy-js
npm test --workspace=sasy-js
cargo test --locked --workspace
make lint-rust
```

The TypeScript runner requires Bun. Rust tests marked `#[ignore]` need extra
prerequisites and a separate `cargo test -- --ignored` run;
`.github/workflows/ci-core.yml` shows which ones CI runs and with what installed.

`make lint-rust` runs `cargo fmt --check` and Clippy for all workspace targets
with warnings treated as errors. CI uses Rust 1.95.0 and the locked dependencies.
Use `make check-rust` for a quick compiler check during development; it does not
replace the test suite.

## Performance measurements

Two scripts measure the cost of recording messages against an engine they start
themselves. They are measurement tools, not tests; run each with `--help` for its
options and use a release build (`make build-rust`) for any number you
intend to quote.

```sh
.venv/bin/python scripts/benchmark_message_versions.py --help
.venv/bin/python scripts/benchmark_snapshot_references.py --help
```
