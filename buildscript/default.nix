{ pkgs ? import ./pkgs.nix }:
let
  frontend = import ../frontend { inherit pkgs; };
  mlir = import ../mlir { inherit pkgs; };
  simulator = import ../simulator { inherit pkgs; };
  isqc = import ../isqc { inherit pkgs; };
  mlir14 = pkgs.callPackage ../mlir/vendor/mlir.nix { };
in
pkgs.buildEnv rec {
  name = "isqv2";
  paths = [ frontend mlir simulator mlir14 isqc ];
  nativeBuildInputs = [ pkgs.makeWrapper ];
  postBuild = ''
    wrapProgram $out/bin/isqc --set ISQV2_ROOT $out
  '';
}
