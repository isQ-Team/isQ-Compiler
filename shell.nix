{ pkgs ? import ./pkgs.nix }:
let buildenv = (import ./default.nix { inherit pkgs; });
in 
pkgs.mkShell ({
  buildInputs = [ buildenv ];

} // buildenv.passthru.environmentVars)