{
  description = "isQ Compiler Base package";
    inputs = {
      nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";
      flake-utils = {
        url  = "github:numtide/flake-utils";
      };
    };
  outputs = { self, nixpkgs, flake-utils }: 
    let
      base-overlay = final: prev: {
          isqc = prev.lib.makeScope prev.newScope (self: {
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
        lib.debug-nixpkgs = {}: import nixpkgs {};
        lib.isqc-components-flake = 
          {self, 
           overlay, 
           preOverlays? [], 
           depComponentOverlays? [],
           systems? flake-utils.lib.defaultSystems,
           shell? null,
           components? [],
           defaultComponent? null,
           extra? {},
           skipBaseOverlay? false}:
          (flake-utils.lib.eachSystem systems (system':
            let pkgs = import nixpkgs {
              overlays = preOverlays ++ (if skipBaseOverlay then [] else [base-overlay]) ++ depComponentOverlays ++ [overlay];
              system = system';
            };
            packages = pkgs.lib.listToAttrs (map (component: {name=component; value=pkgs.isqc.${component}; }) components);
            
            outputs = ({
              legacyPackages = packages;
            }) // (if defaultComponent==null then {} else {
              defaultPackage = pkgs.isqc.${defaultComponent};
            }) // (if shell == null then {} else {
              devShell = shell {inherit pkgs;};
            }) // extra;
            in outputs

          )) // {overlays.default = overlay;};
    };
}
