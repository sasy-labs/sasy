{ pkgs, sasy }:

# A filesystem layout around the same engine package used by native Nix builds.
# Docker copies this root and its complete store closure into a scratch image.
pkgs.runCommand "sasy-docker-root" { } ''
  mkdir -p "$out/bin" "$out/etc/ssl/certs" "$out/data" "$out/tmp"
  ln -s ${sasy}/bin/sasy "$out/bin/sasy"
  # Souffle's subprocesses use /bin/sh, independently of the launcher's PATH.
  ln -s ${pkgs.bash}/bin/sh "$out/bin/sh"
  ln -s ${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt "$out/etc/ssl/certs/ca-certificates.crt"
  printf '%s\n' 'root:x:0:0:root:/root:/bin/sh' 'sasy:x:10001:10001:SASY:/data:/bin/sh' > "$out/etc/passwd"
  printf '%s\n' 'root:x:0:' 'sasy:x:10001:' > "$out/etc/group"
''
