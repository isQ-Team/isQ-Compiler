{pkgs? import ./pkgs.nix, is_container? false}:
let
  lib = pkgs.lib;
  rustChannel = (pkgs.rustChannelOf { rustToolchain = ../rust-toolchain; });
  callPackage = lib.callPackageWith pkgs;
  mlir = pkgs.callPackage ../mlir/vendor/mlir.nix {};
  rust = rustChannel.rust.override (old: { extensions = ["rust-src" "rust-analysis"]; });
  frontend-deps = (import ../frontend/shell.nix {});
in
pkgs.buildEnv rec {
  name = "isqv2-tools";
  paths = with pkgs; [
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
      cabal2nix
      crate2nix
      cabal-install
      git
      clang-tools
      which
  ] ++ frontend-deps.nativeBuildInputs;
  passthru = {
    environmentVars = {
      RUST_SRC_PATH = "${rustChannel.rust-src}/lib/rustlib/src/rust/src";
    };
  };
  extraPrefix = if is_container then [ "/opt/isqdeps" ] else ["/"];
}
