{ rust-bin
, mlir
, makeRustPlatform
, gitignoreSource
, plugins ? [ ]
, lib
}:
let
  rust = rust-bin.fromRustupToolchainFile ./rust-toolchain;
  rustPlatform = makeRustPlatform {
    cargo = rust;
    rustc = rust;
  };
  llvm_tools = mlir;
  pluginDeps = lib.catAttrs "derivation" plugins;
  pluginFeatures = lib.catAttrs "feature" plugins;
  pluginExports = builtins.foldl' (x: y: x // y) { } (lib.catAttrs "exports" plugins);
in
rustPlatform.buildRustPackage ((pluginExports) // rec {
  pname = "isq-simulator";
  version = "0.1.0";
  nativeBuildInputs = [ llvm_tools ];
  #buildInputs = (builtins.concatLists [
  #  (addInput build_cuda_plugin (pkgs.lib.getLib cudaPlugin))
  #  (addInput build_qcis_plugin routingPlugin)
  #]);
  buildInputs = pluginDeps;
  src = gitignoreSource ./.;
  cargoLock = {
    lockFile = ./Cargo.lock;
  };
  buildFeatures = pluginFeatures;
  buildNoDefaultFeatures = true;
  #buildFeatures = builtins.concatLists [
  #  (addInput build_cuda_plugin "cuda")
  #  (addInput build_qcis_plugin "qcis")
  #];
  postInstall = ''
    mkdir -p $out/share/isq-simulator
    ${llvm_tools}/bin/llvm-link ${src}/src/facades/qir/shim/qir_builtin/shim.ll \
    ${src}/src/facades/qir/shim/qsharp_core/shim.ll  \
    ${src}/src/facades/qir/shim/qsharp_foundation/shim.ll \
    ${src}/src/facades/qir/shim/isq/shim.ll -o $out/share/isq-simulator/isq-simulator.bc
    echo "#!/usr/bin/env bash" > $out/bin/isq-simulator-stub
    echo "echo $out/share/isq-simulator/isq-simulator.bc" >> $out/bin/isq-simulator-stub
    chmod +x $out/bin/isq-simulator-stub
  '';
}
)
