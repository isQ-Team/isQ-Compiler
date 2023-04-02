{
  description = "isQ-IR Middleend";
  inputs = {
    isqc-base.url = "path:../base";
    gitignore.url = "github:hercules-ci/gitignore.nix";
    vendor.url = "path:../vendor";
  };
  outputs = { self, isqc-base, gitignore, vendor, ... }:
    isqc-base.lib.isqc-components-flake {
      inherit self;
      overlay = isqc-base.lib.isqc-override (pkgs: final: prev: {
        isq-opt = (final.callPackage ./default.nix {
          inherit (final.vendor) mlir;
          gitignoreSource = gitignore.lib.gitignoreSource;
        });
      });
      components = [ "isq-opt" ];
      defaultComponent = "isq-opt";
      depComponentOverlays = [ vendor.overlays.default ];
      shell = { pkgs }: pkgs.mkShell.override { stdenv = pkgs.llvmPackages.stdenv; } {
        inputsFrom = [ pkgs.isqc.isq-opt ];
        nativeBuildInputs = [ pkgs.clang-tools pkgs.nlohmann_json ];
      };
    };
}
