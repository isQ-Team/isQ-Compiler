{pkgs? import ./pkgs.nix, target? import ./default.nix {inherit pkgs;} }:
with pkgs;
let
nix-user-chroot = ./nix-user-chroot;
target_path = (builtins.toString target);
startup_script = writeScript "startup" ''
    #!/bin/sh
    ISQ_PATH=$(dirname "$0")
    $ISQ_PATH/${nix-user-chroot} $ISQ_PATH/nix ${target_path}/bin/isqc "$@"
  '';
maketar = { targets }:
    stdenv.mkDerivation {
      name = "maketar";
      buildInputs = [ perl ];
      exportReferencesGraph = map (x: [("closure-" + baseNameOf x) x]) targets;
      buildCommand = ''
        storePaths=$(perl ${pathsFromGraph} ./closure-*)
        cp ${startup_script} /build/isqc
        tar -cf - \
          --owner=0 --group=0 --mode=u+rw,uga+r \
          --hard-dereference \
          $storePaths > /build/temp.tar
        tar -C /build -rf /build/temp.tar isqc
        cat /build/temp.tar | gzip -9 > $out
      '';
    };
in 
maketar {targets = [ target_path nix-user-chroot ];}
