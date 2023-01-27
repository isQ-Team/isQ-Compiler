{
  description = "isQ Compiler Base package";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";
    flake-utils = {
      url = "github:numtide/flake-utils";
    };
  };
  outputs = { self, nixpkgs, flake-utils }:
    let
      base-overlay = final: prev:
        let pkgs = final; in {
          isqc = prev.lib.makeScope prev.newScope (self: {
            buildISQCEnv =
              { isqc1 ? self.isqc1
              , isq-opt ? self.isq-opt
              , isqc-driver ? self.isqc-driver
              , isq-simulator ? self.isq-simulator
              }: pkgs.buildEnv {
                name = "isqc";
                paths = [ isqc1 isq-opt isqc-driver isq-simulator isq-opt.mlir ];
                nativeBuildInputs = [ pkgs.makeWrapper ];
                postBuild = ''
                  wrapProgram $out/bin/isqc --set ISQ_ROOT $out
                '';
              };
            #mlir = self.callPackage ./mlir.nix {};
          });
        };
    in
    {
      overlays.default = base-overlay;
      #packages.mlir = pkgs.isqc.mlir;
      lib.isqc-override = f: final: prev: {
        isqc = prev.isqc.overrideScope' (f final);
      };
      lib.isqc-components-flake =
        { self
        , overlay
        , preOverlays ? [ ]
        , depComponentOverlays ? [ ]
        , systems ? flake-utils.lib.defaultSystems
        , shell ? null
        , extraShells ? obj: { }
        , components ? [ ]
        , defaultComponent ? null
        , extra ? obj: { }
        , skipBaseOverlay ? false
        }:
        (flake-utils.lib.eachSystem systems (system':
        let
          pkgs = import nixpkgs {
            overlays = preOverlays ++ (if skipBaseOverlay then [ ] else [ base-overlay ]) ++ depComponentOverlays ++ [ overlay ];
            system = system';
            config.allowUnfree = true; # TODO: Remove the CUDA specific part.
          };
          packages = pkgs.lib.listToAttrs (map (component: { name = component; value = pkgs.isqc.${component}; }) components);
          evalShell = shell: shell (builtins.intersectAttrs (builtins.functionArgs shell) { inherit pkgs; system = system'; });
          devShellDefault = (if shell == null then {
            default = if defaultComponent == null then pkgs.mkShell { } else pkgs.isqc.${defaultComponent};
          } else {
            default = evalShell shell;
          });
          devShellExtra = builtins.mapAttrs (name: value: value) (evalShell extraShells);
          outputs = ({
            legacyPackages = packages;
          }) // (if defaultComponent == null then { } else {
            defaultPackage = pkgs.isqc.${defaultComponent};
          }) // (rec {
            devShells = devShellDefault // devShellExtra;
            devShell = devShells.default;
          }) // (evalShell extra);
        in
        outputs

        )) // { overlays.default = overlay; };
    };
}
