{pkgs? import ../buildscript/pkgs.nix }:
let
rustChannel = (pkgs.rustChannelOf { rustToolchain = ./rust-toolchain; });
rustPlatform = pkgs.makeRustPlatform {
  cargo = rustChannel.rust;
  rustc = rustChannel.rust;
};
in
with pkgs;
rustPlatform.buildRustPackage rec {
  pname = "isq-simulator";
  version = "0.1.0";
  nativeBuildInputs = [ llvmPackages_13.bintools ];
  src = nix-gitignore.gitignoreSource [] ./.;
  cargoLock = {
    lockFile = ./Cargo.lock;
  };
  postInstall = ''
mkdir -p $out/share/isq-simulator
llvm-link ${src}/src/facades/qir/shim/qir_builtin/shim.ll \
${src}/src/facades/qir/shim/qsharp_core/shim.ll  \
${src}/src/facades/qir/shim/qsharp_foundation/shim.ll \
${src}/src/facades/qir/shim/isq/shim.ll -o $out/share/isq-simulator/isq-simulator.bc
echo "#!/usr/bin/env bash" > $out/bin/isq-simulator-stub
echo "echo $out/share/isq-simulator/isq-simulator.bc" >> $out/bin/isq-simulator-stub
chmod +x $out/bin/isq-simulator-stub
  '';
}