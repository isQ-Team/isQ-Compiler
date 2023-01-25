{
  description = "isQ Compiler.";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";
    flake-utils.url  = "github:numtide/flake-utils";
    
  };
  outputs = { self, nixpkgs, flake-utils, ... }: flake-utils.lib.eachDefaultSystem (system:
    let
    overlays = [];
    pkgs = import nixpkgs {inherit system overlays;};
    legacyPackages = nixpkgs.legacyPackages;
    in
    rec {
      packages.hello = pkgs.hello;
      packages.default = packages.hello;
      packages.codium = nixpkgs.legacyPackages."${system}".vscodium;
    }
  );
}
