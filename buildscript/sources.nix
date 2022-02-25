{pkgs? import ./pkgs.nix }:
let 
  source_files = (pkgs.nix-gitignore.gitignoreSource [] ../.);
in
{
  source_pkg = pkgs.buildEnv rec {
    name = "isqv2-sources";
    paths = [ source_files ];
    extraPrefix = [ "/opt/isqsources/" ];
  };
  inherit source_files;
}

