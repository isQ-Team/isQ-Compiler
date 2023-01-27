{
  description = "isQ-IR Middleend";
  inputs = {
    isqc-base.url = "path:../base";
    gitignore.url = "github:hercules-ci/gitignore.nix";
    mlir.url = "path:../base/vendor/mlir";
  };
  outputs = { self, isqc-base, gitignore, mlir, ... }:
    isqc-base.lib.isqc-components-flake {
      inherit self;
      overlay = isqc-base.lib.isqc-override (pkgs: final: prev: {
        isq-opt = (final.callPackage ./default.nix { gitignoreSource = gitignore.lib.gitignoreSource; });
      });
      components = [ "isq-opt" ];
      defaultComponent = "isq-opt";
      depComponentOverlays = [ mlir.overlays.default ];
      shell = { pkgs }: pkgs.mkShell.override { stdenv = pkgs.llvmPackages.stdenv; } {
        inputsFrom = [ pkgs.isqc.isq-opt ];
        nativeBuildInputs = [ pkgs.clang-tools ];
      };
    };
}
