{ rust-bin, pkgsStatic, fetchFromGitHub, lib }:
let
  inherit (pkgsStatic) makeRustPlatform;
  target = pkgsStatic.rust.toRustTarget pkgsStatic.stdenv.targetPlatform;
  rust = rust-bin.stable.latest.default.override {
    targets = [ target ];
  };
  rustPlatform = makeRustPlatform {
    rustc = rust;
    cargo = rust;
  };
  src = fetchFromGitHub {
    owner = "nix-community";
    repo = "nix-user-chroot";
    sha256 = "8w2/Ncfcg6mMRFgMZg3CBBtAO/FI6G6hDMyaLCS3hwk=";
    rev = "1.2.2";
  };
in
rustPlatform.buildRustPackage {
  pname = "nix-user-chroot";
  version = "1.2.2";
  inherit src;
  cargoLock = {
    lockFile = "${src}/Cargo.lock"; # IFD here
  };
  doCheck = false;
}
  