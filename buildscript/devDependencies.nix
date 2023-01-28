{ pkgs ? import ./pkgs.nix, is_container ? false }:
let
  lib = pkgs.lib;
  rustChannel = (pkgs.rustChannelOf { rustToolchain = ../rust-toolchain; });
  callPackage = lib.callPackageWith pkgs;
  mlir = pkgs.callPackage ../mlir/vendor/mlir.nix { };
  rust = rustChannel.rust.override (old: { extensions = [ "rust-src" "rust-analysis" ]; });
  frontend-deps = (import ../frontend/shell.nix { });
  nix-bundle = pkgs.nix-bundle.overrideAttrs (attrs: {
    postInstall = attrs.postInstall + ''
      sed -i "s/g++/g++ -static-libstdc++/" $out/share/nix-bundle/nix-user-chroot/Makefile
    '';
  });
in
pkgs.buildEnv rec {
  name = "isqv2-tools";
  paths = with pkgs; [
    llvmPackages_13.clang
    llvmPackages_13.lldb
    llvmPackages_13.libclang
    cmake
    rust
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
    #clang-tools
    which
    nix-bundle
  ] ++ frontend-deps.nativeBuildInputs;
  passthru = {
    environmentVars = {
      RUST_SRC_PATH = "${rustChannel.rust-src}/lib/rustlib/src/rust/src";
      LIBCLANG_PATH = pkgs.lib.makeLibraryPath [ pkgs.llvmPackages_13.libclang.lib ];
      BINDGEN_EXTRA_CLANG_ARGS =
        # Includes with normal include path
        (builtins.map (a: ''-I"${a}/include"'') [
          pkgs.glibc.dev
        ])
        # Includes with special directory paths
        ++ [
          ''-I"${pkgs.llvmPackages_13.libclang.lib}/lib/clang/${pkgs.llvmPackages_13.libclang.version}/include"''
        ];

    };
  };
  extraPrefix = if is_container then [ "/opt/isqdeps" ] else [ "/" ];
}
