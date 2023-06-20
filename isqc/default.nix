{ vendor ? null
, gitignoreSource ? vendor.gitignoreSource
, isQVersion
}:
let
  rustPlatform = vendor.rustPlatform;
in
rustPlatform.buildRustPackage rec {
  pname = "isqc-driver";
  inherit (isQVersion) version;
  src = gitignoreSource ./.;
  cargoLock = {
    lockFile = ./Cargo.lock;
  };
  doCheck = false; # TODO: move tests out of the crate.
}
