{
  description = "SASY full engine and development toolchain";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  outputs = { nixpkgs, ... }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      linuxSystems = [ "aarch64-linux" "x86_64-linux" ];
      forSystems = nixpkgs.lib.genAttrs systems;
      environment = system:
        let pkgs = import nixpkgs { inherit system; };
        in { inherit pkgs; toolchain = import ./nix/toolchain.nix { inherit pkgs; }; };
    in {
      packages = nixpkgs.lib.genAttrs linuxSystems (system:
        let
          args = environment system;
          sasy = import ./nix/package.nix args;
          docker-root = import ./nix/docker-root.nix { inherit (args) pkgs; inherit sasy; };
        in { inherit sasy docker-root; default = sasy; });
      devShells = forSystems (system:
        let inherit (environment system) pkgs toolchain;
        in {
          default = pkgs.mkShell {
            nativeBuildInputs = toolchain.nativeBuildInputs;
            buildInputs = toolchain.buildInputs;
            packages = with pkgs; [
              rustc cargo rustfmt clippy clang libclang mcpp uv nodejs_24 bun
              git gnumake jq
            ] ++ toolchain.runtime;
            inherit (toolchain) env;
          };
        });
    };
}
