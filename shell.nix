<<<<<<< HEAD
{ pkgs ? import ./buildscript/pkgs.nix }:
let buildenv = (import ./buildscript/devDependencies.nix {});
in 
pkgs.mkShell ({
  buildInputs = [ buildenv ];

} // buildenv.passthru.environmentVars)
=======
let
  flake = (builtins.getFlake (builtins.toString ./.));
in
flake.devShell.${builtins.currentSystem}
>>>>>>> merge
