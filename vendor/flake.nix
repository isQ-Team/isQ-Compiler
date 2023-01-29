{
  description = "isQ vendor flakes";
  inputs = {
    isqc-base.url = "path:../base";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };
  outputs = { self, isqc-base, rust-overlay }: isqc-base.lib.isqc-components-flake {
    inherit self;
    overlay = isqc-base.lib.isqc-override
      (pkgs: final: prev: {
        vendor = {
          mlir = (final.callPackage ./mlir { });
          nix-user-chroot = (final.callPackage ./nix-user-chroot { });
          buildTarball = { name, drv, entry }:
            let
              drv_path = (builtins.toString drv);
              maketar = { targets }: with pkgs;
                let
                  entryScriptPath = writeScript "${name}-entry" ''
                    #!/bin/sh
                    NIX_CHROOT_PATH=$(dirname "$0")
                    $NIX_CHROOT_PATH/${final.vendor.nix-user-chroot}/bin/nix-user-chroot $NIX_CHROOT_PATH/nix ${entry} "$@"
                  '';
                in
                stdenv.mkDerivation {
                  name = "${name}-maketar";
                  buildInputs = [ perl ];
                  exportReferencesGraph = map (x: [ ("closure-" + baseNameOf x) x ]) targets;
                  buildCommand = ''
                    storePaths=$(perl ${pathsFromGraph} ./closure-*)
                    cp ${entryScriptPath} /build/${name}
                    chmod +x /build/${name}
                    echo $storePaths
                    tar -cvf - \
                      --owner=0 --group=0 --mode=u+rw,uga+r \
                      --hard-dereference \
                      $storePaths > /build/temp.tar
                    tar -C /build -rf /build/temp.tar ${name}
                    cat /build/temp.tar | gzip -9 > $out
                  '';
                };
            in
            maketar { targets = [ drv final.vendor.nix-user-chroot ]; };
        };
      });
    preOverlays = [ rust-overlay.overlays.default ];
    # we define packages in extras.
    extra = { system, pkgs }: {
      legacyPackages = pkgs.isqc.vendor;
    };
  };
}
