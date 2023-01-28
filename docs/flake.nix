{
  description = "isQ Documents";
  inputs = {
    isqc-base.url = "path:../base";
    gitignore.url = "github:hercules-ci/gitignore.nix";
  };
  outputs = { self, isqc-base, gitignore }:
    isqc-base.lib.isqc-components-flake {
      inherit self;
      overlay = isqc-base.lib.isqc-override (pkgs: final: prev: {
        isqc-docs = (final.callPackage ./default.nix {
          gitignoreSource = gitignore.lib.gitignoreSource;
        });
      });
      components = [ "isqc-docs" ];
      defaultComponent = "isqc-docs";
    };
}
