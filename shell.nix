{ pkgs ? import ./buildscript/pkgs.nix }:
let buildenv = (import ./buildscript/devDependencies.nix {});
in 
pkgs.mkShell ({
  buildInputs = [ buildenv ];

} // buildenv.passthru.environmentVars)