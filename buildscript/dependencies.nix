{pkgs? import ./pkgs.nix}:
let
  lib = pkgs.lib;
  rustChannel = (pkgs.rustChannelOf { rustToolchain = ../rust-toolchain; });
  callPackage = lib.callPackageWith pkgs;
  mlir = pkgs.callPackage ./vendor/mlir.nix {};
  rust = rustChannel.rust.override (old: { extensions = ["rust-src" "rust-analysis"]; });
in
{
  inherit pkgs;
  packages= with pkgs; [
      llvmPackages_13.clang
      llvmPackages_13.lldb
      cmake
      rust
      llvmPackages_13.lld
      binutils
      eigen
      mlir
      ninja
      doxygen
      graphviz
      libxml2
      stack
      xz
      gnumake
      python3
      ghc
  ];
  rustSrcPath = "${rustChannel.rust-src}/lib/rustlib/src/rust/src";
}
