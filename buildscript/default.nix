{pkgs? import ./pkgs.nix}:
let
  frontend = import ../frontend {inherit pkgs;};
  mlir = import ../mlir {inherit pkgs;};
  simulator = import ../simulator {inherit pkgs;};
in
pkgs.buildEnv rec {
  name = "isqv2-tools";
  paths = [ frontend mlir simulator ];
}