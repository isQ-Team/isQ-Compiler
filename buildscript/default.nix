{pkgs? import ./pkgs.nix}:
let
  frontend = import ../frontend {inherit pkgs;};
  mlir = import ../mlir {inherit pkgs;};
  simulator = import ../simulator {inherit pkgs;};
  mlir14 = pkgs.callPackage ../mlir/vendor/mlir.nix {};
in
pkgs.buildEnv rec {
  name = "isqv2-tools";
  paths = [ frontend mlir simulator mlir14 ];
}