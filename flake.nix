{
  description = "isQ Compiler";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    mlir = {
      url = "path:./base/vendor/mlir";
      inputs.isqc-base.follows = "isqc-base";
    };
    isqc-base = {
      url = "path:./base";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
    isqc-driver = {
      url = "path:./isqc";
      inputs.isqc-base.follows = "isqc-base";
      inputs.rust-overlay.follows = "rust-overlay";
    };
    isq-simulator = {
      url = "path:./simulator";
      inputs.isqc-base.follows = "isqc-base";
      inputs.rust-overlay.follows = "rust-overlay";
      inputs.mlir.follows = "mlir";
    };
    isq-opt = {
      url = "path:./mlir";
      inputs.isqc-base.follows = "isqc-base";
      inputs.mlir.follows = "mlir";
    };
    isqc1 = {
      url = "path:./frontend";
      inputs.isqc-base.follows = "isqc-base";
    };
    pre-commit-hooks.url = "github:cachix/pre-commit-hooks.nix";
  };
  nixConfig = {
    bash-prompt-prefix = "(nix-isqc:$ISQC_DEV_ENV)";

  };
  outputs = { self, nixpkgs, flake-utils, isqc-base, isqc-driver, isq-simulator, isq-opt, isqc1, rust-overlay, mlir, pre-commit-hooks }:
    let lib = nixpkgs.lib; in
    isqc-base.lib.isqc-components-flake rec {
      inherit self;
      skipBaseOverlay = true;

      overlay = lib.composeManyExtensions ((map (component: component.overlays.default) [
        isqc-base
        mlir
        isqc1
        isq-opt
        isqc-driver
        isq-simulator
      ]) ++ [
        (isqc-base.lib.isqc-override (pkgs: final: prev: {
          isqc = (final.buildISQCEnv { });
        }))
      ]);
      #overlay = isqc-base.overlays.default;
      #overlay = final: prev: prev;
      components = [ "isqc1" "isq-opt" "isqc-driver" "isq-simulator" "isqc" ];
      defaultComponent = "isqc";
      preOverlays = [ rust-overlay.overlays.default ];
      shell = { pkgs, system }: pkgs.mkShell.override { stdenv = pkgs.llvmPackages.stdenv; } {
        inputsFrom = map (flake: flake.devShell.${system}) [ isqc1 isq-opt isqc-driver isq-simulator ];
        # https://github.com/NixOS/nix/issues/6982
        nativeBuildInputs = [ pkgs.bashInteractive pkgs.nixpkgs-fmt ];
        ISQC_DEV_ENV = "dev";
        inherit (self.checks.${system}.pre-commit-check) shellHook;
      };
      extraShells = { pkgs, system }:
        let defaultShell = shell { inherit pkgs; inherit system; };
        in {
          codium = defaultShell.overrideAttrs (finalAttrs: previousAttrs: {
            nativeBuildInputs = previousAttrs.nativeBuildInputs ++ [ pkgs.vscodium ];
            ISQC_DEV_ENV = "codium";
          });
        };
      extra = { pkgs, system }: {
        formatter = pkgs.nixpkgs-fmt;
        checks = {
          pre-commit-check = pre-commit-hooks.lib.${system}.run {
            src = ./.;
            hooks = {
              nixpkgs-fmt.enable = true;
              update-flake-lock = {
                enable = true;
                name = "Update local flake locks.";
                entry = "nix flake lock --update-input isqc-base --update-input mlir --update-input isqc1 --update-input isq-opt --update-input isqc-driver --update-input isq-simulator";
                language = "system";
                pass_filenames = false;


              };
            };
          };
        };
      };
    };
}
