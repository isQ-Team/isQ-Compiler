{pkgs? import ./pkgs.nix}:
let 
isqv2 = import ./. {inherit pkgs;};
in
pkgs.dockerTools.buildLayeredImage {
  contents =  [isqv2];
  name = "isqv2";
}