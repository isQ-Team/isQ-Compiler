{
  description = "isQ Compiler";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";
    flake-utils.url  = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    isqc-base = {
      url = "path:./base";
      #inputs.nixpkgs.follows = "nixpkgs";
      #inputs.flake-utils.follows = "flake-utils";
    };
    isqc-driver = {
      url = "path:./isqc";
      inputs.isqc-base.follows = "isqc-base";
      #inputs.rust-overlay.follows = "rust-overlay";
    };
    isq-simulator = {
      url = "path:./simulator";
      inputs.isqc-base.follows = "isqc-base";
      #inputs.rust-overlay.follows = "rust-overlay";
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
  outputs = { self, nixpkgs, flake-utils, isqc-base, isqc-driver, isq-simulator, isq-opt, isqc1, rust-overlay}:
  let lib = nixpkgs.lib; in
  isqc-base.lib.isqc-components-flake {
    inherit self;
    skipBaseOverlay = true;
    
    overlay = lib.composeManyExtensions (map (component: component.overlays.default) [
      isqc-base
      isqc1
      isq-opt
      isqc-driver
      isq-simulator
    ]);
    #overlay = isqc-base.overlays.default;
    #overlay = final: prev: prev;
    components = ["isqc1" "isq-opt" "isqc-driver" "isq-simulator"];
    #defaultComponent = "isqc-driver";
    preOverlays = [rust-overlay.overlays.default];
    
  };
}
