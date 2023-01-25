{
  description = "isQ Compiler.";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";
    flake-utils.url  = "github:numtide/flake-utils";    
    gitignore = {
      url = "github:hercules-ci/gitignore.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = { self, nixpkgs, flake-utils, gitignore, ... }: flake-utils.lib.eachDefaultSystem (system:
    let
    inherit (gitignore.lib) gitignoreSource;
    src = gitignoreSource ./.;
    overlay = final: prev: {
      isqc = prev.lib.makeScope prev.newScope (self: with self; {
        isqc1 = prev.haskellPackages.callCabal2nix "isqc1" src {};
      });
    };
    shell = pkgs.haskellPackages.shellFor{
        nativeBuildInputs = with pkgs; [
          haskellPackages.hpack 
          haskellPackages.haskell-language-server 
          haskellPackages.cabal-install];
        packages = p:  [ pkgs.isqc.isqc1];
      };
    pkgs = import nixpkgs {inherit system; overlays = [overlay]; };
    in rec {
      packages.isqc1 = pkgs.isqc.isqc1;
      defaultPackage = packages.isqc1;
      overlays.default = overlay;
      #devShell = pkgs.isqc.isqc1.env;
      devShell = pkgs.mkShell{
        inputsFrom = [ shell];
      };
    }
  );
}
