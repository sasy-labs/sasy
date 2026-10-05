{ pkgs }:
let
  # Both platforms expose the compiler through this stable command name.
  cxx = pkgs.writeShellScriptBin "g++" ''
    exec ${pkgs.stdenv.cc}/bin/c++ "$@"
  '';
  runtime = [ cxx pkgs.stdenv.cc pkgs.souffle pkgs.python311 pkgs.coreutils pkgs.bash ]
    ++ pkgs.lib.optionals pkgs.stdenv.isLinux [ pkgs.bubblewrap pkgs.util-linux ];
in {
  inherit cxx runtime;
  nativeBuildInputs = [ pkgs.rustPlatform.bindgenHook pkgs.pkg-config pkgs.cmake pkgs.protobuf ];
  buildInputs = [ pkgs.openssl ];
  env = {
    LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
    SOUFFLE_PREFIX = "${pkgs.souffle}";
    SOUFFLE_INCLUDE = "${pkgs.souffle}/include";
    SASY_SOUFFLE_INCLUDE = "${pkgs.souffle}/include";
    SASY_SOUFFLE = "${pkgs.souffle}/bin/souffle";
    SASY_CXX = "${cxx}/bin/g++";
  };
}
