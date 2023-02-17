{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";
  };

  outputs = { self, nixpkgs, ... }@inputs:
    let
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      lib = nixpkgs.lib;

      vm = nixpkgs.lib.nixosSystem {
        inherit system;
        modules = [
          "${nixpkgs}/nixos/modules/profiles/qemu-guest.nix"
          "${nixpkgs}/nixos/modules/profiles/all-hardware.nix"
          ./configuration.nix
        ];
      };
    in
    {
      formatter.x86_64-linux = nixpkgs.legacyPackages.x86_64-linux.nixpkgs-fmt;

      nixosConfigurations."vm" = vm;

      packages.x86_64-linux."vm-g5k" = import "${nixpkgs}/nixos/lib/make-disk-image.nix" {
        inherit lib pkgs;
        config = vm.config;
        diskSize = "auto";
        additionalSpace = "2048M"; # Space added after all the necessary 
        format = "qcow2-compressed";
      };


      devShells.${system} = {
        default = pkgs.mkShell {
          buildInputs = with pkgs; [
            just
            nixos-rebuild
            qemu
          ];
        };
      };
    };
}
