{
  description = "isQ Compiler.";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";
    flake-utils.url  = "github:numtide/flake-utils";    
  };
  outputs = { self, nixpkgs, flake-utils, ... }: flake-utils.lib.eachDefaultSystem (system:
    let
    overlay = final: prev: {
      isqc = prev.lib.makeScope prev.newScope (self: with self; {
        isqc1 = prev.haskellPackages.callCabal2nix "isqc1" ./. {};
      });
    };

    pkgs = import nixpkgs {inherit system; overlays = [overlay]; };
    in rec {
      packages.isqc1 = pkgs.isqc.isqc1;
      defaultPackage = packages.isqc1;
      overlays.default = overlay;
      devShell = pkgs.mkShell{
        buildInputs = with pkgs; [hpack];
        inputsFrom = with pkgs; [ (isqc.isqc1.envFunc {})];
      };
    }
  );
}
