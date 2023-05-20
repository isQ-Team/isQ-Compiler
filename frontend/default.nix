{ pkgs, vendor? null, gitignoreSource? vendor.gitignoreSource}:
let
src = gitignoreSource ./.;
isqc1 = (pkgs.haskellPackages.callCabal2nix "isqc1" src { });
inherit (pkgs.haskell.lib) justStaticExecutables;
isqc1Static = justStaticExecutables isqc1;
in isqc1Static.overrideAttrs (prev: final: {
  passthru.isQDevShell = 
    let
      hs_shell = pkgs.haskellPackages.shellFor {
        nativeBuildInputs = with pkgs; [
          haskellPackages.hpack
          haskellPackages.haskell-language-server
          haskellPackages.cabal-install
        ];
        packages = p: [ isqc1Static ];
      };
    in
    pkgs.mkShell {
      inputsFrom = [ hs_shell ];
    };
})
