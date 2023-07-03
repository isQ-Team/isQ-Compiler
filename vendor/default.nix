{ caterpillar, ... }:
pkgs: final: prev:
let
  llvmPackages = pkgs.llvmPackages_16;
in
{
  caterpillar = (final.callPackage caterpillar { });
  mlir = (final.callPackage ./mlir { });
  nix-user-chroot = (final.callPackage ./nix-user-chroot { });
  buildTarball = { name, drv, entry, fileName ? name }:
    let
      drv_path = (builtins.toString drv);
      maketar = { targets }: with pkgs;
        let
          entryScriptPath = writeScript "${name}-entry" ''
            #!/bin/sh
            NIX_CHROOT_PATH=$(dirname "$0")
            $NIX_CHROOT_PATH/${final.vendor.nix-user-chroot}/bin/nix-user-chroot $NIX_CHROOT_PATH/nix ${entry} "$@"
          '';
          tarballName = "${fileName}.tar.gz";
        in
        stdenvNoCC.mkDerivation {
          name = "${name}-maketar";

          buildInputs = [ perl ];
          exportReferencesGraph = map (x: [ ("closure-" + baseNameOf x) x ]) targets;
          buildCommand = ''
            mkdir -p $out
            storePaths=$(perl ${pathsFromGraph} ./closure-*)
            export BUILD=`pwd`/;
            cp ${entryScriptPath} $BUILD/${name}
            chmod +x $BUILD/${name}
            echo $storePaths
            tar -cvf - \
              --owner=0 --group=0 --mode=u+rw,uga+r \
              --hard-dereference \
              $storePaths > $BUILD/temp.tar
            tar -C $BUILD -rf $BUILD/temp.tar ${name}
            cat $BUILD/temp.tar | gzip -9 > $out/${tarballName}
          '';
        };
    in
    maketar { targets = [ drv final.vendor.nix-user-chroot ]; };
  inherit llvmPackages;
  clang-tools = pkgs.clang-tools_16;
  stdenvLLVM = pkgs.overrideCC llvmPackages.stdenv (llvmPackages.stdenv.cc.override { inherit (llvmPackages) bintools; });
}
