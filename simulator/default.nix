{pkgs? import ../buildscript/pkgs.nix, build_cuda_plugin? true }:
let
rustChannel = (pkgs.rustChannelOf { rustToolchain = ./rust-toolchain; });
rustPlatform = pkgs.makeRustPlatform {
  cargo = rustChannel.rust;
  rustc = rustChannel.rust;
};
cudaPlugin = import ./cuda-plugin {inherit pkgs;};
in
with pkgs;
rustPlatform.buildRustPackage rec {
  pname = "isq-simulator";
  version = "0.1.0";
  nativeBuildInputs = [ llvmPackages_13.bintools ] ;
  buildInputs = (if build_cuda_plugin then [(pkgs.lib.getLib cudaPlugin)] else []);
  src = nix-gitignore.gitignoreSource [] ./.;
  cargoLock = {
    lockFile = ./Cargo.lock;
  };
  buildNoDefaultFeatures = true;
  buildFeatures = if build_cuda_plugin then [ "cuda" ] else [] ;
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