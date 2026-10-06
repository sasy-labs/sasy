SASY_BIN ?= $(SASY_RELEASE_TARGET_DIR)/release/sasy
SASY_ADDR ?= 127.0.0.1:10089
SASY_RELEASE_TARGET_DIR ?= target/full-release
SASY_IMAGE ?= sasy
# Prebuilt release image; override with another release tag or digest as needed.
SASY_GHCR_IMAGE ?= ghcr.io/sasy-labs/sasy:0.5.1
SASY_DOCKER_PORT ?= 10089
SASY_SETUP_URL ?= $(SASY_ADDR)
export SASY_BIN SASY_ADDR FILE

.DEFAULT_GOAL := help
.PHONY: help init-config install build-rust build-rust-release build-rust-dev sdk-js certs serve docker-build docker-serve docker-serve-ghcr souffle-validate docs docs-build
help:
	@echo "make docker-serve-ghcr   Pull the prebuilt GHCR engine and serve it locally (recommended)"
	@echo "make docker-build        Build the engine image, including the policy compiler"
	@echo "make docker-serve        Prepare local config and TLS, then serve the Docker engine on localhost:$(SASY_DOCKER_PORT)"
	@echo "make build-rust          Build the optimized engine at $(SASY_BIN). Needs Rust, a C++ compiler, protoc and Soufflé"
	@echo "make build-rust-dev      Build the development engine at target/debug/sasy"
	@echo "make build-rust-release  Alias for make build-rust"
	@echo "make install             Create .venv and install the Python SDK from sdk/python"
	@echo "make init-config         Create local auth (with new random API keys), role and transform config plus .env, keeping existing settings"
	@echo "make certs               Write a self-signed TLS certificate for localhost into certs/"
	@echo "make serve               Run the engine on $(SASY_ADDR) with the local config"
	@echo "make souffle-validate FILE=policy.dl   Check a policy file without a running engine"
	@echo "make sdk-js              Build the TypeScript SDK"
	@echo "make check-rust          Compile-check every Rust target without linking"
	@echo "make lint-rust           Run cargo fmt --check and Clippy with warnings as errors"
	@echo "make docs                Build the documentation site and serve the build locally until stopped; Astro prints the URL"
	@echo "make docs-build          Build the documentation site in docs-site/ without serving it"

install:
	uv venv .venv
	uv pip install --python .venv/bin/python -e sdk/python

# Local API keys are generated, never copied: the engine refuses the example keys.
init-config:
	mkdir -p config/auth data
	@if test -e config/auth/apikey.json; then \
		echo "Keeping existing config/auth/apikey.json"; \
	else \
		(umask 077 && python3 -c 'import json, secrets; config = json.load(open("config/auth/apikey.example.json")); config["static_keys"] = {secrets.token_urlsafe(32): role for role in config["static_keys"].values()}; json.dump(config, open("config/auth/apikey.json", "w"), indent=2); print("Created config/auth/apikey.json with new random keys")'); \
	fi
	@test -e config/auth/jwt.json || cp config/auth/jwt.example.json config/auth/jwt.json
	@test -e config/auth_config.yaml || cp config/auth_config.example.yaml config/auth_config.yaml
	@test -e config/transforms.json || cp config/transforms.example.json config/transforms.json
	@python3 scripts/setup_local_env.py --url "$(SASY_SETUP_URL)"

build-rust:
	cargo build --locked --release --manifest-path Cargo.toml -p sasy-binary --bin sasy --target-dir "$(SASY_RELEASE_TARGET_DIR)"

build-rust-release: build-rust

build-rust-dev:
	cargo build --locked --manifest-path Cargo.toml -p sasy-binary --bin sasy

# Check all Rust targets without linking; useful for a quick local iteration.
.PHONY: check-rust lint-rust
check-rust:
	cargo check --locked --manifest-path Cargo.toml --workspace --all-targets

# CI uses the same formatter and warning-free Clippy gate.
lint-rust:
	cargo fmt --manifest-path Cargo.toml --all --check
	cargo clippy --locked --manifest-path Cargo.toml --workspace --all-targets -- -D warnings

sdk-js:
	npm ci --ignore-scripts --no-audit --no-fund
	npm run build:sdk

certs:
	mkdir -p certs
	test -f certs/server.key || openssl req -x509 -newkey rsa:2048 -nodes -keyout certs/server.key -out certs/server.crt -days 365 -subj /CN=localhost -addext 'subjectAltName=DNS:localhost,IP:127.0.0.1'
	chmod 600 certs/server.key

serve:
	@test -f config/auth/apikey.json -a -f config/auth_config.yaml -a -f config/transforms.json || { echo "Missing local config: run make init-config (https://docs.sasy.ai/local-engine/)." >&2; exit 1; }
	@test -f certs/server.crt -a -f certs/server.key || { echo "Missing local TLS files: run make certs before make serve." >&2; exit 1; }
	"$$SASY_BIN" serve --addr "$$SASY_ADDR" --auth-provider config/auth/apikey.json --auth-config config/auth_config.yaml --transforms config/transforms.json --tls-cert certs/server.crt --tls-key certs/server.key --data-dir data/graph

docker-build:
	docker build -t "$(SASY_IMAGE)" .

# Pull before setup and reuse the same TLS, authentication and storage as source builds.
docker-serve-ghcr:
	docker pull "$(SASY_GHCR_IMAGE)"
	$(MAKE) docker-serve SASY_IMAGE="$(SASY_GHCR_IMAGE)"

# Keep credentials and certificates outside the image and data across restarts.
docker-serve: SASY_SETUP_URL = localhost:$(SASY_DOCKER_PORT)
docker-serve: init-config certs
	docker run --rm --init -p "127.0.0.1:$(SASY_DOCKER_PORT):10089" \
		--user "$$(id -u):$$(id -g)" -e HOME=/data -w /data \
		-v "$(CURDIR)/config:/config:ro" -v "$(CURDIR)/certs:/certs:ro" -v "$(CURDIR)/data:/data" \
		"$(SASY_IMAGE)"

souffle-validate:
	@test -n "$$FILE" || { echo 'Set FILE to a policy path'; exit 1; }
	python3 scripts/validate_policy.py --no-rpc "$$FILE"

docs-build:
	cd docs-site && npm ci && npm run build

# Serves the production build, so what you read is what a deployment would show.
docs: docs-build
	cd docs-site && npm run preview
