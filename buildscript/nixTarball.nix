{pkgs? import ./pkgs.nix, target? import ./default.nix {inherit pkgs;} }:
with pkgs;
let
nix-user-chroot = ./nix-user-chroot;
target_path = (builtins.toString target);
startup_script = writeScript "startup" ''
    #!/bin/sh
    if [ -z "$@" ]; then
    echo isQv2 Toolchain wrapper.
    echo Usage: $0 [TOOL_NAME]
    echo Tools directory: .${target_path}/bin/
    else
    .${nix-user-chroot} ./nix ${target_path}/bin/"$@"
    fi
  '';
maketar = { targets }:
    stdenv.mkDerivation {
      name = "maketar";
      buildInputs = [ perl ];
      exportReferencesGraph = map (x: [("closure-" + baseNameOf x) x]) targets;
      buildCommand = ''
        storePaths=$(perl ${pathsFromGraph} ./closure-*)
        cp ${startup_script} /build/run
        tar -cf - \
          --owner=0 --group=0 --mode=u+rw,uga+r \
          --hard-dereference \
          $storePaths > /build/temp.tar
        tar -C /build -rf /build/temp.tar run
        cat /build/temp.tar | bzip2 -z > $out
      '';
    };
in 
maketar {targets = [ target_path nix-user-chroot ];}