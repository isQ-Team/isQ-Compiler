{
  description = "isQ Compiler";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    vendor = {
      url = "path:./vendor";
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
      inputs.vendor.follows = "vendor";
    };
    isq-opt = {
      url = "path:./mlir";
      inputs.isqc-base.follows = "isqc-base";
      inputs.vendor.follows = "vendor";
    };
    isqc1 = {
      url = "path:./frontend";
      inputs.isqc-base.follows = "isqc-base";
    };
    isqc-docs = {
      url = "path:./docs";
      inputs.isqc-base.follows = "isqc-base";
    };
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
    pre-commit-hooks.url = "github:cachix/pre-commit-hooks.nix";
  };
  nixConfig = {
    bash-prompt-prefix = "(nix-isqc:$ISQC_DEV_ENV)";

  };
  outputs = { self, nixpkgs, flake-utils, isqc-base, isqc-driver, isq-simulator, isq-opt, isqc1, rust-overlay, vendor, pre-commit-hooks, flake-compat, isqc-docs }:
    let lib = nixpkgs.lib; in
    (isqc-base.lib.isqc-components-flake rec {
      inherit self;
      skipBaseOverlay = true;

      overlay = lib.composeManyExtensions ((map (component: component.overlays.default) [
        isqc-base
        vendor
        isqc1
        isq-opt
        isqc-driver
        isq-simulator
        isqc-docs
      ]) ++ [
        (isqc-base.lib.isqc-override (pkgs: final: prev: rec {
          isqc = (final.buildISQCEnv { });
          isqcTarball = final.vendor.buildTarball {
            name = "isqc";
            drv = isqc;
            entry = "${isqc}/bin/isqc";
          };
          devEnvCodium = pkgs.vscode-with-extensions.override {
            vscodeExtensions = with pkgs.vscode-extensions; [
              llvm-vs-code-extensions.vscode-clangd # clangd
              haskell.haskell # haskell
              rust-lang.rust-analyzer # rust
              arrterian.nix-env-selector # nix env
              jnoortheen.nix-ide # nix
            ];
            vscode = pkgs.vscodium;
          };
        }))
      ]);
      #overlay = isqc-base.overlays.default;
      #overlay = final: prev: prev;
      components = [ "isqc1" "isq-opt" "isqc-driver" "isq-simulator" "isqc" "isqc-docs" "isqcTarball" ];
      defaultComponent = "isqc";
      preOverlays = [ rust-overlay.overlays.default ];
      shell = { pkgs, system }: pkgs.mkShell.override { stdenv = pkgs.llvmPackages.stdenv; } {
        inputsFrom = map (flake: flake.devShell.${system}) [ isqc1 isq-opt isqc-driver isq-simulator isqc-docs ];
        # https://github.com/NixOS/nix/issues/6982
        nativeBuildInputs = [ pkgs.bashInteractive pkgs.nixpkgs-fmt pkgs.rnix-lsp pkgs.rust-analyzer ];
        ISQC_DEV_ENV = "dev";
        shellHook = self.checks.${system}.pre-commit-check.shellHook + ''
          mkdir -p $PWD/.build
          export ISQ_ROOT=$PWD/.build
        '';
      };
      extraShells = { pkgs, system }:
        let defaultShell = shell { inherit pkgs; inherit system; };
        in {
          codium = defaultShell.overrideAttrs (finalAttrs: previousAttrs: {
            nativeBuildInputs = previousAttrs.nativeBuildInputs ++ [ pkgs.isqc.devEnvCodium ];
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
                entry = "make lock";
                language = "system";
                pass_filenames = false;


              };
            };
          };
        };
      };
    }) // {
      templates = {
        trivial = {
          path = ./templates/trivial;
          description = "Basic flake with isQ.";
        };
      };
      defaultTemplate = self.templates.trivial;
    };
}
