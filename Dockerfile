# Build the same locked Nix engine package as nix/build.sh. Docker is the only
# host build prerequisite; the Python SDK and examples run on the host.
FROM nixos/nix:2.28.5@sha256:e2bff159b88c242722022a273d3832a4392f01dba35b796204d852b89042526e AS builder

WORKDIR /build
COPY . .
# Ordinary Docker builders cannot create Nix's nested build sandbox. This does
# not alter the engine derivation or the native package's runtime sandbox.
RUN nix --extra-experimental-features 'nix-command flakes' build --print-build-logs \
        --option sandbox false --no-update-lock-file --max-jobs 1 --cores 2 \
        'path:/build#docker-root' \
    && mkdir -p /rootfs \
    && cp -a result/. /rootfs/ \
    && mkdir -p /rootfs/nix/store \
    && nix-store -qR "$(readlink -f result)" > /store-paths \
    && while IFS= read -r store_path; do cp -a "$store_path" /rootfs/nix/store/ || exit 1; done < /store-paths \
    && chown -R 0:0 /rootfs/nix \
    && chmod -R a-w /rootfs/nix/store \
    && chmod 0755 /rootfs/data \
    && chown 10001:10001 /rootfs/data \
    && chmod 1777 /rootfs/tmp

FROM scratch
COPY --from=builder /rootfs/ /
ENV PATH=/bin \
    HOME=/data \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
# Preserve the trusted-policy Docker setup without requiring privileged Docker
# flags. Compilation is confined by the container, not nested Bubblewrap.
# The native Nix package retains its sandbox behavior.
ENV DISABLE_BWRAP=1
USER 10001:10001
WORKDIR /data
EXPOSE 10089
# With /config and /certs mounted (make docker-serve), serve that setup.
# Otherwise create keys, config and TLS in /data/local on first start, reuse
# them later, and serve with those. `sasy local-init /data/local` prints the
# client settings again at any time.
CMD ["/bin/sh", "-c", "if [ -e /config/auth/apikey.json ]; then exec sasy serve --addr 0.0.0.0:10089 --auth-provider /config/auth/apikey.json --auth-config /config/auth_config.yaml --transforms /config/transforms.json --tls-cert /certs/server.crt --tls-key /certs/server.key --data-dir /data/graph; fi; sasy local-init /data/local > /dev/null && exec sasy serve --addr 0.0.0.0:10089 --auth-provider /data/local/auth/apikey.json --auth-config /data/local/auth_config.yaml --transforms /data/local/transforms.json --tls-cert /data/local/tls/server.crt --tls-key /data/local/tls/server.key --data-dir /data/graph"]
