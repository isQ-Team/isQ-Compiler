{pkgs? import ./pkgs.nix}:
let
  lib = pkgs.lib;
  rustChannel = (pkgs.rustChannelOf { rustToolchain = ./rust-toolchain; });
  callPackage = lib.callPackageWith pkgs;
  mlir = pkgs.callPackage ./vendor/mlir.nix {};
  rust = rustChannel.rust.override (old: { extensions = ["rust-src" "rust-analysis"]; });
in
pkgs.buildEnv rec {
  name = "isqv2-tools";
  paths = with pkgs; [
      llvmPackages_13.clang
      llvmPackages_13.lldb
      cmake
      rust
      llvmPackages_13.lld
      eigen
      mlir
      ninja
      doxygen
      graphviz
      libxml2
  ];
  passthru = {
    environmentVars = {
      RUST_SRC_PATH = "${rustChannel.rust-src}/lib/rustlib/src/rust/src";
    };
  };
}
