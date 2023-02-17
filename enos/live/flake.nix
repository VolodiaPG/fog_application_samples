{
  description = "Application packaged using poetry2nix";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.poetry2nix = {
    url = "github:nix-community/poetry2nix";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        inherit (poetry2nix.legacyPackages.${system}) mkPoetryEnv mkPoetryApplication;

        overlay = self: super:{
          experiments = self.poetry2nix.mkPoetryEnv {
            projectDir = ./.;
            python = self.python311;
            overrides = self.poetry2nix.overrides.withDefaults (newattr: oldattr: {
                cryptography = oldattr.cryptography.overridePythonAttrs (
                    old: {
                      cargoDeps =
                        super.rustPlatform.fetchCargoTarball {
                          src = old.src;
                          sourceRoot = "${old.pname}-${old.version}/src/rust";
                          name = "${old.pname}-${old.version}";
                          sha256 = "sha256-0x+KIqJznDEyIUqVuYfIESKmHBWfzirPeX2R/cWlngc=";
                        };
                    }
                  );
                rfc3986-validator = oldattr.rfc3986-validator.overridePythonAttrs
                  (
                    old: {
                      buildInputs = (old.buildInputs or [ ]) ++ [ oldattr.setuptools oldattr.setuptools-scm oldattr.pytest-runner ];
                    }
                  );
                pathspec = oldattr.pathspec.overridePythonAttrs
                  (
                    old: {
                      buildInputs = (old.buildInputs or [ ]) ++ [ oldattr.flit-scm oldattr.pytest-runner ];
                    }
                  );
                ncclient = oldattr.ncclient.overridePythonAttrs
                  (
                    old: {
                      buildInputs = (old.buildInputs or [ ]) ++ [ oldattr.six ];
                    }
                  );
                jupyter-server-terminals = oldattr.jupyter-server-terminals.overridePythonAttrs
                  (
                    old: {
                      buildInputs = (old.buildInputs or [ ]) ++ [ oldattr.hatchling ];
                    }
                  );
                jupyter-events = oldattr.jupyter-events.overridePythonAttrs
                  (
                    old: {
                      buildInputs = (old.buildInputs or [ ]) ++ [ oldattr.hatchling ];
                    }
                  );
                jupyter-server = oldattr.jupyter-server.overridePythonAttrs
                  (
                    old: {
                      buildInputs = (old.buildInputs or [ ]) ++ [ oldattr.hatch-jupyter-builder oldattr.hatchling ];
                    }
                  );
              });
          };
        };

        pkgs = import nixpkgs {
          inherit system;
          overlays = [overlay];
        };
        lib = nixpkgs.lib;

        dockerImage = pkgs.dockerTools.buildImage {
          name = "enos_deployment";
          tag = "latest";
          copyToRoot = pkgs.buildEnv {
            name = "image-root";
                pathsToLink = [
                    "/bin"
                    "/"
                ];
                paths = with pkgs; [
                    # Linux toolset
                    coreutils
                    gnused
                    bashInteractive

                    # My toolset
                    just
                    jq
                    openssh
                    curl
                    frp

                    # Environment to run enos and stuff
                    experiments
                ];
            };
             runAsRoot = ''
              #!${pkgs.runtimeShell}
              ${pkgs.dockerTools.shadowSetup}
              groupadd -g 1000 enos
              useradd -u 1000 -g 1000 enos
              mkdir -p /home/enos
              chown enos:enos -R /home/enos

              mkdir /tmp
              mkdir -p /usr/bin
              ln -s ${pkgs.coreutils}/bin/env /usr/bin/env
            '';
          config = {
            Env = [ "RUN=python" "HOME=/root"]; 
          };
        };
      in
      {
        packages = {
          docker = dockerImage;
          default = pkgs.experiments;
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [ 
            just
            jq
            poetry2nix.packages.${system}.poetry
            experiments
          ];
        };
      });
}
