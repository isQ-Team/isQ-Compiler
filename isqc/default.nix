{ rust-bin
, makeRustPlatform
, vendor ? null
, gitignoreSource ? vendor.gitignoreSource
}:
let
  rust = rust-bin.fromRustupToolchainFile ./rust-toolchain;
  rustPlatform = makeRustPlatform {
    cargo = rust;
    rustc = rust;
  };
in
rustPlatform.buildRustPackage rec {
  pname = "isqc-driver";
  version = "0.1.0";
  src = gitignoreSource ./.;
  cargoLock = {
    lockFile = ./Cargo.lock;
  };
  doCheck = false; # TODO: move tests out of the crate.
}
