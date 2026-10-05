# Engine release rehearsal

The manually dispatched `Engine release rehearsal` workflow currently runs only
in private `sasy-labs/sasy` or `nilspalumbo/sasy-test` staging repositories. It builds Linux x86-64
and ARM64 on native runners from the locked Nix package used by the Dockerfile.
Each imported engine bundle and final container image must pass real policy
compilation, allow/deny, dependency-graph and policy-rebinding checks before the
workflow publishes test images and creates a draft GitHub Release. Downloaded
release assets are checked against SHA256SUMS. This is packaging qualification,
not the complete security, sandbox or core regression suite.

Images live in GitHub Container Registry, not in Git history. The workflow uses
its repository-scoped GitHub token and publishes only unique `rehearsal-*` tags
under `ghcr.io/sasy-labs/sasy` or `ghcr.io/nilspalumbo/sasy-test`, matching
the repository where the workflow runs; it never advances `latest`. New GHCR packages
are private by default. Production Docker Hub publishing is configured separately.
The multi-platform tag includes both `linux/amd64` and `linux/arm64`.

## Native Linux bundles

These are full Nix runtime closures, **not standalone relocatable executables**.
They include the engine, Souffle, compiler, headers, preprocessing and shared
libraries. Install Nix and xz first, download the matching architecture's
`.nar.xz` and `.path` assets plus `prepare_rehearsal_store.py` and SHA256SUMS, then verify and import:

```sh
# Use arm64 in both filenames on ARM64 Linux.
sha256sum --check --ignore-missing SHA256SUMS
xz -dc sasy-linux-amd64.nar.xz | sudo env NIX_REMOTE=local "$(command -v nix-store)" --import > imported-paths
sudo python3 prepare_rehearsal_store.py imported-paths --nix-store "$(command -v nix-store)"
engine=$(cat sasy-linux-amd64.path)
"$engine" --help
```

Import into a root-managed Nix store: the engine requires its pinned runtime
manifest and toolchain files to be root-owned and read-only. Import can reuse
existing store items; the helper verifies all selected contents, removes write
permissions and sets root ownership only within those items, then verifies their
contents again. It does not follow symlinks for ownership changes. On Ubuntu 24.04,
AppArmor must permit the package's Bubblewrap executable to create user
namespaces. The rehearsal installs a profile for that exact executable, following
[Ubuntu's documented per-application permission](https://documentation.ubuntu.com/release-notes/24.04/#unprivileged-user-namespace-restrictions).
It leaves the system-wide restriction enabled; native qualification still runs
with SASY's sandbox active. The `.runtime-manifest.path` asset identifies the
manifest containing the pinned Bubblewrap path.

Keep the imported engine alive across Nix garbage collection by registering it
as a GC root or installing its store package into a profile. Supply TLS, auth
configuration and data paths as described in the setup guide before `serve`.
Do not run imports from an untrusted release. macOS and installation without Nix
are not covered by this rehearsal.

## Container images

The draft release includes per-architecture image digests and the combined
manifest. Authenticate to GHCR to pull a private test image. Use the image with
the same `/config`, `/certs` and writable `/data` mounts as the local Docker setup.
The image supplies its runtime toolchain and runs without root by default.
The test runs with the runner's non-root UID to access its owned temporary files.
Container policy compilation uses the image's existing container confinement;
this test does not certify nested Bubblewrap or host sandbox behavior.

If either architecture fails, there is no combined manifest or draft release.
An individual successfully qualified architecture may already have its uniquely
tagged test image. Fix the failure and dispatch another run; artifacts from an
older run are never substituted for a failing architecture.

Failed candidates are retained as private, unqualified workflow artifacts for
one day. They are diagnostic inputs, not release assets or substitutes for passing
qualification.
