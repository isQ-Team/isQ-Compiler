{ enabledPlugins ? [ "qcis" ]
, lib
, vendor ? null
, gitignoreSource ? vendor.gitignoreSource
, mlir ? vendor.mlir
, callPackage
, isQVersion
, isQVersionHook
, isQRustPackages
}:
let
  rustPlatform = vendor.rustPlatform;
  llvm_tools = mlir;
  availablePlugins = {
    qcis = {
      feature = "qcis";
      package = callPackage ./plugins/python-routing-plugin { };
    };
    cuda = {
      feature = "cuda";
      package = callPackage ./plugins/cuda-plugin { };
    };
  };
  plugins = map (x: availablePlugins."${x}") enabledPlugins;
  pluginDeps = lib.catAttrs "package" plugins;
  pluginFeatures = lib.catAttrs "feature" plugins;
  pluginExports = builtins.foldl' (x: y: x // y) { } (lib.catAttrs "exports" plugins);
in
((isQRustPackages.workspace."isq-simulator" { }).override {
  features = pluginFeatures;
}).overrideAttrs (final: prev: ((pluginExports) // rec {
  buildInputs = prev.buildInputs ++ pluginDeps;
  nativeBuildInputs = prev.nativeBuildInputs ++ [ llvm_tools ];
  postInstall = ''
    src=${final.src};
    simulator_stub_path=$bin/share/isq-simulator/isq-simulator.bc
    mkdir -p $bin/share/isq-simulator
    ${llvm_tools}/bin/llvm-link $src/src/facades/qir/shim/qir_builtin/shim.ll \
    $src/src/facades/qir/shim/qsharp_core/shim.ll  \
    $src/src/facades/qir/shim/qsharp_foundation/shim.ll \
    $src/src/facades/qir/shim/isq/shim.ll -o $simulator_stub_path
    echo "#!/usr/bin/env bash" > $bin/bin/isq-simulator-stub
    echo "echo $simulator_stub_path" >> $bin/bin/isq-simulator-stub
    chmod +x $bin/bin/isq-simulator-stub
  '';
  passthru.availablePlugins = availablePlugins;
}
))
