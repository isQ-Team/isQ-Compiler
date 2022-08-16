let
  moz_overlay = import (builtins.fetchTarball {
    url = "https://github.com/mozilla/nixpkgs-mozilla/archive/f233fdc4ff6ba2ffeb1e3e3cd6d63bb1297d6996.tar.gz";
    sha256 = "1rzz03h0b38l5sg61rmfvzpbmbd5fn2jsi1ccvq22rb76s1nbh8i";
  });
  pkgs = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/c6019d8efb5.tar.gz";
    sha256 = "1havpwch8wkbhw0y2q3rnx4z0dz66msxb1agynrgvkw4qmm2hbpj";
  }) { overlays = [ moz_overlay ]; };
in
pkgs