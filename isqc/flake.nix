{
  description = "isQ Compiler Driver";
  inputs = {
    isqc-base.url = "path:../base";
    gitignore.url = "github:hercules-ci/gitignore.nix";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };
  outputs = { self, isqc-base, gitignore, rust-overlay, ... }:
    isqc-base.lib.isqc-components-flake {
      inherit self;
      overlay = isqc-base.lib.isqc-override (pkgs: final: prev: {
        isqc-driver = (final.callPackage ./default.nix {
          gitignoreSource = gitignore.lib.gitignoreSource;
        });
      });
      components = [ "isqc-driver" ];
      defaultComponent = "isqc-driver";
      preOverlays = [ rust-overlay.overlays.default ];
    };
}
