{
  description = "isQ Compiler Frontend";
  inputs = {
    isqc-base.url = "path:../base";
    gitignore.url = "github:hercules-ci/gitignore.nix";
  };
  outputs = { self, gitignore, isqc-base, ... }:
    let
      inherit (gitignore.lib) gitignoreSource;
      src = gitignoreSource ./.;
    in
    isqc-base.lib.isqc-components-flake {
      inherit self;
      overlay = isqc-base.lib.isqc-override (pkgs: final: prev:
        let
          isqc1 = (pkgs.haskellPackages.callCabal2nix "isqc1" src { });
          inherit (pkgs.haskell.lib) justStaticExecutables;
        in
        {
          isqc1 = justStaticExecutables isqc1;
        });
      shell = { pkgs }:
        let
          hs_shell = pkgs.haskellPackages.shellFor {
            nativeBuildInputs = with pkgs; [
              haskellPackages.hpack
              haskellPackages.haskell-language-server
              haskellPackages.cabal-install
            ];
            packages = p: [ pkgs.isqc.isqc1 ];
          };
        in
        pkgs.mkShell {
          inputsFrom = [ hs_shell ];
        };
      components = [ "isqc1" ];
      defaultComponent = "isqc1";
    };
}
