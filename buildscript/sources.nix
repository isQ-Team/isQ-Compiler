{dependencies? import ./dependencies.nix {} }:
let 
  source_files = (dependencies.pkgs.nix-gitignore.gitignoreSource [] ../.);
in
{
  source_pkg = dependencies.pkgs.buildEnv rec {
    name = "isqv2-sources";
    paths = [ source_files ];
    extraPrefix = [ "/opt/isqsources/" ];
  };
  inherit source_files;
}

