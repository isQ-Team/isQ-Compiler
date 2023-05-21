{ vendor? null, gitignoreSource? vendor.gitignoreSource }:
let
  rustPlatform = vendor.rustPlatform;
in
rustPlatform.buildRustPackage rec {
  pname = "isqc-frontend";
  version = "0.1.0";
  src = gitignoreSource ./.;
  cargoLock = {
    lockFile = ./Cargo.lock;
  };
  doCheck = false; # TODO: move tests out of the crate.
}
