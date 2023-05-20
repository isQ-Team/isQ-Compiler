{ pkgs ? import ../buildscript/pkgs.nix }:
let project = pkgs.haskell.packages.ghc8107.callPackage ./cabal.nix { };
in project
