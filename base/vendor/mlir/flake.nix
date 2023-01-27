{
  description = "MLIR";
  inputs = {
    isqc-base.url = "path:../../";
  };
  outputs = { self, isqc-base }: isqc-base.lib.isqc-components-flake {
    inherit self;
    overlay = isqc-base.lib.isqc-override (pkgs: final: prev: {
      mlir = (final.callPackage ./mlir.nix { });
    });
    components = [ "mlir" ];
    defaultComponent = "mlir";
  };
}
