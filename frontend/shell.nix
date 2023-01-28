{ pkgs ? import ../buildscript/pkgs.nix }:
(import ./default.nix { inherit pkgs; }).env
