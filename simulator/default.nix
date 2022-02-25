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
  pname = "isq-simulator";
  version = "0.1.0";

  src = nix-gitignore.gitignoreSource [] ./.;
  cargoLock = {
    lockFile = ./Cargo.lock;
  };
}