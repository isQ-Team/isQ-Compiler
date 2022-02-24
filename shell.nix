{ pkgs ? import ./buildscript/pkgs.nix }:
let buildenv = (import ./buildscript/default.nix {});
in 
pkgs.mkShell ({
  buildInputs = [ buildenv ];

} // buildenv.passthru.environmentVars)