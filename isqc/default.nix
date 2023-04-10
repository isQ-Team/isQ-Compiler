<<<<<<< HEAD
{pkgs? import ../buildscript/pkgs.nix }:
let
rustChannel = (pkgs.rustChannelOf { rustToolchain = ./rust-toolchain; });
rustPlatform = pkgs.makeRustPlatform {
  cargo = rustChannel.rust;
  rustc = rustChannel.rust;
};
in
with pkgs;
rustPlatform.buildRustPackage rec {
  pname = "isqc";
  version = "0.1.0";
  src = nix-gitignore.gitignoreSource [] ./.;
  cargoLock = {
    lockFile = ./Cargo.lock;
  };
}
=======
{ rust-bin, makeRustPlatform, gitignoreSource }:
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
>>>>>>> merge
