let
  moz_overlay = import (builtins.fetchTarball https://github.com/mozilla/nixpkgs-mozilla/archive/master.tar.gz);
  pkgs = import <nixpkgs> { overlays = [ moz_overlay ]; };
  lib = (import <nixpkgs/lib>);
  rustChannel = (pkgs.rustChannelOf { rustToolchain = ./rust-toolchain; });
  callPackage = lib.callPackageWith pkgs;
  mlir = pkgs.callPackage ./vendor/mlir.nix {};
  old_pkgs = import (builtins.fetchTarball {
      url = "https://github.com/NixOS/nixpkgs/archive/528d35bec0cb976a06cc0e8487c6e5136400b16b.tar.gz";
  }) {};
  old_cmake = pkgs.cmake;
in
pkgs.mkShell rec {
  buildInputs = with pkgs; [
      llvmPackages_13.bintools
      llvmPackages_13.clang
      old_cmake
      rustChannel.rust
      llvmPackages_latest.lld
      eigen
      mlir
      ninja
      doxygen
      graphviz
      libxml2
  ];
  RUST_SRC_PATH = "${rustChannel.rust-src}";
}