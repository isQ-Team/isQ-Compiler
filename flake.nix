{
  description = "isQ Compiler";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";
    flake-utils.url  = "github:numtide/flake-utils";
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
    };
    isq-opt = {
      url = "path:./mlir";
      inputs.isqc-base.follows = "isqc-base";
    };
    isqc1 = {
      url = "path:./frontend";
      inputs.isqc-base.follows = "isqc-base";
    };
  };
  nixConfig = {
    bash-prompt-prefix = "(nix-isqc:$ISQC_DEV_ENV)";

  };
  outputs = { self, nixpkgs, flake-utils, isqc-base, isqc-driver, isq-simulator, isq-opt, isqc1, rust-overlay, mlir}:
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
    ]) ++ [(isqc-base.lib.isqc-override (pkgs: final: prev: {
      isqc = (final.buildISQCEnv {});
    }))]);
    #overlay = isqc-base.overlays.default;
    #overlay = final: prev: prev;
    components = ["isqc1" "isq-opt" "isqc-driver" "isq-simulator" "isqc"];
    defaultComponent = "isqc";
    preOverlays = [rust-overlay.overlays.default];
    shell = {pkgs, system}: pkgs.mkShell.override {stdenv = pkgs.llvmPackages.stdenv;} {
      inputsFrom = map (flake: flake.devShell.${system}) [isqc1 isq-opt isqc-driver isq-simulator];
      # https://github.com/NixOS/nix/issues/6982
      nativeBuildInputs = [ pkgs.bashInteractive ];
      ISQC_DEV_ENV = "dev";
    };
    extraShells = {pkgs, system}: 
      let defaultShell = shell {inherit pkgs; inherit system;};
      in {
        codium = defaultShell.overrideAttrs (finalAttrs: previousAttrs: {
          nativeBuildInputs = previousAttrs.nativeBuildInputs ++ [pkgs.vscodium];
          ISQC_DEV_ENV = "codium";
        });
      };
  };
}
