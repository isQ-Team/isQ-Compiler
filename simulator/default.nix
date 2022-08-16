{pkgs? import ../buildscript/pkgs.nix, build_cuda_plugin? true, build_qcis_plugin? true }:
let
rustChannel = (pkgs.rustChannelOf { rustToolchain = ./rust-toolchain; });
rustPlatform = pkgs.makeRustPlatform {
  cargo = rustChannel.rust;
  rustc = rustChannel.rust;
};
addInput = flag: dep: if flag then [(dep)] else [];
cudaPlugin = import ./plugins/cuda-plugin {inherit pkgs;};
routingPlugin = import ./plugins/python-routing-plugin {inherit pkgs;};
llvm_tools = pkgs.llvmPackages_13.llvm;
in
with pkgs;
rustPlatform.buildRustPackage ((if build_qcis_plugin then {QCIS_ROUTE_BIN_PATH = "${routingPlugin}/bin/qcis-routing"; } else {}) // rec {
  pname = "isq-simulator";
  version = "0.1.0";
  nativeBuildInputs = [ llvm_tools ] ;
  buildInputs = (builtins.concatLists [
    (addInput build_cuda_plugin (pkgs.lib.getLib cudaPlugin))
    (addInput build_qcis_plugin routingPlugin)
  ]);
  src = nix-gitignore.gitignoreSource [] ./.;
  cargoLock = {
    lockFile = ./Cargo.lock;
  };
  buildNoDefaultFeatures = true;
  buildFeatures = builtins.concatLists [
    (addInput build_cuda_plugin "cuda")
    (addInput build_qcis_plugin "qcis")
  ];
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
})
