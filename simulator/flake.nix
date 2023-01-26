{
  description = "isQ-IR Simulator";
  inputs = {
    isqc-base.url = "path:../base";
    gitignore.url = "github:hercules-ci/gitignore.nix";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };
  outputs = { self, isqc-base, gitignore, rust-overlay, ...}: 
  isqc-base.lib.isqc-components-flake {
    inherit self;
    overlay = isqc-base.lib.isqc-override (pkgs: final: prev: {
        isq-simulator-plugin-cuda = (final.callPackage ./plugins/cuda-plugin {});
        isq-simulator-plugin-qcis = (final.callPackage ./plugins/python-routing-plugin {});
        isq-simulator =  (final.callPackage ./default.nix {
          gitignoreSource = gitignore.lib.gitignoreSource;
          plugins = [
            { feature = "cuda"; derivation = final.isq-simulator-plugin-cuda;}
            { feature = "qcis"; derivation = final.isq-simulator-plugin-qcis; exports= {QCIS_ROUTE_BIN_PATH = "${final.isq-simulator-plugin-qcis}/bin/qcis-routing";};}
          ];
          });
    });
    components = ["isq-simulator-plugin-cuda" "isq-simulator-plugin-qcis"  "isq-simulator"];
    defaultComponent = "isq-simulator";
    preOverlays = [rust-overlay.overlays.default];
    #shell = {pkgs}: pkgs.mkShell.override {stdenv = pkgs.llvmPackages.stdenv;} {
    #  inputsFrom = [pkgs.isqc.isq-opt];
    #  nativeBuildInputs = [pkgs.clang-tools];
    #};
  };
}
