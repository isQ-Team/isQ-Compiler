{ nixpkgs, flake-utils }:
rec {
  isqc-base-overlay = final: prev:
    let pkgs = final; in {
      isqc = prev.lib.makeScope prev.newScope (self: { });
    };
  isqc-override = f: final: prev: {
    isqc = prev.isqc.overrideScope' (f final);
  };
  isqc-flake =
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
        overlays = preOverlays ++ (if skipBaseOverlay then [ ] else [ isqc-base-overlay ]) ++ depComponentOverlays ++ [ overlay ];
        system = system';
        # config.allowUnfree = true; # TODO: Remove the CUDA specific part.
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

    )) // { overlays.default = if skipBaseOverlay then overlay else nixpkgs.lib.composeExtensions isqc-base-overlay overlay; };
}
      