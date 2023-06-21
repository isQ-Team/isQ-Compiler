{
  description = "isQ Compiler";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
    rust-overlay.inputs.flake-utils.follows = "flake-utils";
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
    gitignore.url = "github:hercules-ci/gitignore.nix";
    gitignore.inputs.nixpkgs.follows = "nixpkgs";
    # Flakified vendor dependencies.
    caterpillar = {
      url = "github:arclight-quantum/caterpillar";
      flake = false;
    };
  };
  nixConfig = {
    bash-prompt-prefix = "(nix-isqc:$ISQC_DEV_ENV)";
    extra-substituters = [ "https://arclight-quantum.cachix.org" ];
    extra-trusted-public-keys = [ "arclight-quantum.cachix.org-1:DiMhc4M3H1Z3gBiJMBTpF7+HyTwXMOPmLVkjREzF404=" ];
  };
  outputs = { self, nixpkgs, flake-utils, rust-overlay, flake-compat, gitignore, ... }@flakeInputs:
    let
      lib = nixpkgs.lib;
      base = import ./base/isq-flake.nix { inherit nixpkgs; inherit flake-utils; };
    in
    (base.isqc-flake rec {
      inherit self;
      overlay =
        (base.isqc-override (pkgs: final: prev: rec {
          versionInfo =
            let json = import ./scripts/get-sem-ver.nix;
            in
            rec {
              version = json.versionJSON.version;
              dirty = !(self?rev);
              # TODO: use real revision when Nix supports stuff like dirtyRev.
              rev = if (self ? rev) then self.rev else "unknown";
              frozen = json.versionJSON.frozen;
              metadata = lib.concatStringsSep "." (builtins.concatLists [
                (lib.optional (!frozen) rev)
                (lib.optional dirty "dirty")
              ]);
              semVer = if metadata == "" then version else "${version}+${metadata}";
            };
          isQVersionHook = pkgs.stdenvNoCC.mkDerivation {
            name = "isq-version-hook";
            version = versionInfo.version;
            semver = versionInfo.semVer;
            rev = versionInfo.rev;
            frozen = if versionInfo.frozen then "1" else "0";
            buildCommand = ''
              mkdir -p $out/nix-support
              substituteAll ${./scripts/isq-version-hook.sh} $out/nix-support/setup-hook
            '';
          };
          vendorPkgs = {
            inherit (vendor) mlir;
            inherit (vendor) nix-user-chroot;
          };
          vendor = (import ./vendor flakeInputs pkgs final prev) // rec {
            gitignoreSource = gitignore.lib.gitignoreSource;
            rust = pkgs.rust-bin.fromRustupToolchainFile
              ./vendor/rust-toolchain;
            rustPlatform =
              pkgs.makeRustPlatform {
                cargo = rust;
                rustc = rust;
              };
          };
          isqc-driver = (final.callPackage ./isqc { isQVersion = versionInfo; });
          isq-opt = (final.callPackage ./mlir { isQVersion = versionInfo; });
          isq-simulator = (final.callPackage ./simulator { isQVersion = versionInfo; });
          isq-simulator-cuda = (final.callPackage ./simulator { enabledPlugins = [ "qcis" "cuda" ]; });
          isqc1 = (final.callPackage ./frontend { isQVersion = versionInfo; });
          isqc-docs = (final.callPackage ./docs { isQVersion = versionInfo; });
          isqc = (final.buildISQCEnv { });


          buildISQCEnv =
            { isqc1 ? final.isqc1
            , isq-opt ? final.isq-opt
            , isqc-driver ? final.isqc-driver
            , isq-simulator ? final.isq-simulator
            }: pkgs.buildEnv {
              name = "isqc";
              paths = [ ./sysroot isqc1 isq-opt isqc-driver isq-simulator isq-opt.mlir ];
              nativeBuildInputs = [ pkgs.makeWrapper ];
              postBuild = ''
                wrapProgram $out/bin/isqc --set ISQ_ROOT $out --set LLVM_ROOT ${final.vendor.mlir}
              '';
            };

          isqcTarball = final.vendor.buildTarball {
            name = "isqc";
            drv = isqc;
            entry = "${isqc}/bin/isqc";
          };
          isqcImage = pkgs.dockerTools.streamLayeredImage {
            contents = [ isqc ];
            name = "isqc-image";
          };
          isqcImageWithUbuntu = pkgs.dockerTools.streamLayeredImage {
            contents = [
              (isqc.override {
                extraPrefix = "/opt/isqc";
                postBuild = ''
                  wrapProgram $out/opt/isqc/bin/isqc --set ISQ_ROOT $out/opt/isqc/
                '';
              })
            ];
            fromImage = pkgs.dockerTools.pullImage {
              imageName = "ubuntu";
              imageDigest = "sha256:965fbcae990b0467ed5657caceaec165018ef44a4d2d46c7cdea80a9dff0d1ea";
              sha256 = "P7EulEvNgOyUcHH3fbXRAIAA3afHXx5WELJX7eNeUuM=";
              finalImageName = "ubuntu";
              finalImageTag = "latest";
            };
            config = {
              Env = [ "PATH=/opt/isqc/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" ];
            };
            name = "isqc-ubuntu-image";
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
        }));
      #overlay = isqc-base.overlays.default;
      #overlay = final: prev: prev;
      components = [
        "isqc1"
        "isq-opt"
        "isqc-driver"
        "isq-simulator"
        "isq-simulator-cuda"
        "isqc"
        "isqc-docs"
        "isqcTarball"
        "isqcImage"
        "isqcImageWithUbuntu"
        "vendorPkgs"
        "isQVersionHook"
      ];
      defaultComponent = "isqc";
      preOverlays = [ rust-overlay.overlays.default ];
      shell = { pkgs, system }: pkgs.mkShell.override { stdenv = pkgs.isqc.vendor.stdenvLLVM; } {
        inputsFrom = map (component: if component ? passthru && component.passthru ? isQDevShell then component.passthru.isQDevShell else component) (with pkgs.isqc; [ isqc1 isq-opt isqc-driver isq-simulator isqc-docs ]);
        # https://github.com/NixOS/nix/issues/6982
        nativeBuildInputs = [ pkgs.bashInteractive pkgs.nixpkgs-fmt pkgs.rnix-lsp pkgs.isqc.vendor.rust pkgs.isqc.isQVersionHook ];
        ISQC_DEV_ENV = "dev";
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
