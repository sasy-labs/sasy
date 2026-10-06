{ pkgs, toolchain }:
let
  lib = pkgs.lib;
  root = ../.;
  # This reviewed list is also the launcher's staging allowlist. Never select
  # a whole private repository tree, its credentials or build output.
  sourceFiles = builtins.fromJSON (builtins.readFile ./source-files.json);
  src = lib.fileset.toSource {
    inherit root;
    fileset = lib.fileset.unions (map (path: root + "/${path}") sourceFiles);
  };
  assets = pkgs.stdenv.mkDerivation {
    pname = "sasy-souffle-assets";
    version = "0.5.1";
    src = lib.fileset.toSource {
      inherit root;
      fileset = lib.fileset.unions (map (path: root + "/${path}")
        (builtins.filter (lib.hasPrefix "souffle/") sourceFiles));
    };
    nativeBuildInputs = [ pkgs.souffle ];
    buildPhase = ''
      runHook preBuild
      cd souffle
      wordSize=$(${pkgs.souffle}/bin/souffle --version | sed -n 's/^Word size: \([0-9][0-9]*\) bits/\1/p')
      test "$wordSize" = 32 -o "$wordSize" = 64
      $CXX -std=c++17 -O2 -o souffle-interpreted interpreted_shim.cpp
      $CXX -std=c++17 -O2 -shared -fPIC -DRAM_DOMAIN_SIZE="$wordSize" \
        -I${pkgs.souffle}/include functors.cpp -o libfunctors${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}
      runHook postBuild
    '';
    installPhase = ''
      runHook preInstall
      mkdir -p "$out/share/sasy/souffle"
      cp sugar.py evaluator_shim.cpp evaluator_protocol.h json_string_codec.h \
        functors_common.cpp functors.cpp common_policy.dl souffle-interpreted \
        libfunctors${pkgs.stdenv.hostPlatform.extensions.sharedLibrary} "$out/share/sasy/souffle/"
      runHook postInstall
    '';
  };
  closure = pkgs.closureInfo { rootPaths = toolchain.runtime ++ [ assets ]; };
  manifest = pkgs.runCommand "sasy-runtime-toolchain.json" { nativeBuildInputs = [ pkgs.jq ]; } ''
    jq -n \
      --rawfile paths ${closure}/store-paths \
      --argjson path '${builtins.toJSON (map (p: "${p}/bin") toolchain.runtime)}' \
      --argjson tools '${builtins.toJSON {
        bwrap = "${pkgs.bubblewrap}/bin/bwrap";
        prlimit = "${pkgs.util-linux}/bin/prlimit";
        true = "${pkgs.coreutils}/bin/true";
        python3 = "${pkgs.python311}/bin/python3";
        # Souffle uses popen, whose /bin/sh must come from this closure.
        sh = "${pkgs.bash}/bin/sh";
        souffle = "${pkgs.souffle}/bin/souffle";
        souffle-interpreted = "${assets}/share/sasy/souffle/souffle-interpreted";
        "g++" = "${toolchain.cxx}/bin/g++";
      }}' \
      '{version: 1, store_paths: ($paths | split("\n") | map(select(length > 0))), path: $path, tools: $tools}' > "$out"
  '';
in pkgs.rustPlatform.buildRustPackage {
  pname = "sasy";
  version = "0.5.1";
  inherit src;
  cargoLock.lockFile = ../Cargo.lock;
  cargoBuildFlags = [ "--locked" "--package" "sasy-binary" "--bin" "sasy" ];
  # Engine integration tests require namespace and RPC fixtures; run them via
  # the installed-package qualification, outside the Nix build sandbox.
  doCheck = false;
  nativeBuildInputs = toolchain.nativeBuildInputs ++ [ pkgs.makeWrapper ];
  buildInputs = toolchain.buildInputs;
  env = toolchain.env // {
    CARGO_BUILD_JOBS = "2";
    # RocksDB 8.10 bundled by the locked librocksdb-sys release relies on
    # transitive fixed-width integer includes removed by GCC 15. Supply the
    # standard header during this package build only; runtime flags are unchanged.
    CXXFLAGS = "-include cstdint";
  };
  postInstall = ''
    wrapProgram "$out/bin/sasy" \
      --set SASY_SOUFFLE_ASSETS ${assets}/share/sasy/souffle \
      --set SASY_SOUFFLE_INCLUDE ${pkgs.souffle}/include \
      --set SASY_SOUFFLE ${pkgs.souffle}/bin/souffle \
      --set SASY_CXX ${toolchain.cxx}/bin/g++ \
      --prefix PATH : ${lib.makeBinPath toolchain.runtime} \
      ${lib.optionalString pkgs.stdenv.isLinux "--set SASY_NIX_RUNTIME_MANIFEST ${manifest}"}
  '';
  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    "$out/bin/sasy" --help > /dev/null
    runHook postInstallCheck
  '';
  passthru = { inherit assets; runtimeManifest = manifest; };
  meta = {
    description = "SASY full policy engine with its runtime compilation toolchain";
    license = lib.licenses.asl20;
    mainProgram = "sasy";
    platforms = lib.platforms.linux;
  };
}
