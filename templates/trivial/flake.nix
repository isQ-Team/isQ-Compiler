{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
    flake-utils.url = "github:numtide/flake-utils";
    isq-compiler.url = "github:arclight-quantum/isQ-Compiler";
  };

  outputs = { self, nixpkgs, isq-compiler, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
      in
      {
        devShell = with pkgs; mkShell {
          buildInputs = [ isq-compiler.legacyPackages.${system}.isqc ];
        };
      }
    );
}
