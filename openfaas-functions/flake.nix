{
  inputs = {
    nixpkgs.url = "github:Nixos/nixpkgs/nixos-22.11";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
    cargo2nix = {
      url = "github:cargo2nix/cargo2nix/release-0.11.0";
      inputs.rust-overlay.follows = "rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
    pre-commit-hooks = {
      url = "github:cachix/pre-commit-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.nixpkgs-stable.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
    alejandra = {
      url = "github:kamadorueda/alejandra/3.0.0";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs:
    with inputs;
      flake-utils.lib.eachDefaultSystem (
        system: let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
            overlays = [cargo2nix.overlays.default];
          };

          rustChannel = "nightly";
          rustProfile = "minimal";
          rustVersion = "2022-11-05";
          target = "x86_64-unknown-linux-gnu";

          rustPkgsEcho = pkgs.rustBuilder.makePackageSet {
            inherit rustChannel rustProfile target rustVersion;
            packageFun = import ./echo/Cargo.nix;
            rootFeatures = [];
          };
        in rec {
          formatter = alejandra.defaultPackage.${system};
          checks = {
            pre-commit-check = pre-commit-hooks.lib.${system}.run {
              src = ./.;
              settings.statix.ignore = ["Cargo.nix"];
              hooks = {
                # Nix
                alejandra.enable = true;
                statix.enable = true;
                deadnix = {
                  enable = true;
                  excludes = ["Cargo.nix"];
                };
                # Rust
                rustfmt.enable = true;
                clippy.enable = true;
                cargo-check.enable = true;
              };
            };
          };
          devShells = pkgs.mkShell {
            inherit (self.checks.${system}.pre-commit-check) shellHook;
            default = rustPkgsEcho.workspaceShell {
              packages = with pkgs; [
                docker
                faas-cli
                just
                pkg-config
                openssl
                rust-analyzer
                lldb
                (rustfmt.override {asNightly = true;})
                cargo2nix.packages.${system}.cargo2nix
              ];
            };
          };
        }
      );
}
