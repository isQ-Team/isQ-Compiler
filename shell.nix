let
  moz_overlay = import (builtins.fetchTarball https://github.com/mozilla/nixpkgs-mozilla/archive/master.tar.gz);
  pkgs = import (builtins.fetchTarball https://github.com/NixOS/nixpkgs/archive/c6019d8efb5.tar.gz) { overlays = [ moz_overlay ]; };
  lib = pkgs.lib;
  rustChannel = (pkgs.rustChannelOf { rustToolchain = ./rust-toolchain; });
  callPackage = lib.callPackageWith pkgs;
  mlir = pkgs.callPackage ./vendor/mlir.nix {};
in
pkgs.mkShell rec {
  buildInputs = with pkgs; [
      llvmPackages_13.bintools
      llvmPackages_13.clang
      llvmPackages_13.lldb
      cmake
      rustChannel.rust
      llvmPackages_13.lld
      eigen
      mlir
      ninja
      doxygen
      graphviz
      libxml2
  ];
  RUST_SRC_PATH = "${rustChannel.rust-src}";
}