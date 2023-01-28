{ pkgs ? import ../buildscript/pkgs.nix }:
with pkgs;
let
  isq = import ../. { inherit pkgs; };
in
stdenv.mkDerivation {
  name = "isq-tests";
  buildInputs = [ isq coreutils bash ];
  src = ./.;
  inherit coreutils;
  inherit isq;
  inherit bash;
  buildCommand = ''
    cp -r $src/* .
    chmod -R 777 .
    substituteInPlace clean.sh --replace "#!/usr/bin/env bash" "#!$bash/bin/bash"
    substituteInPlace run.sh --replace "#!/usr/bin/env bash" "#!$bash/bin/bash"
    substituteInPlace program.sh --replace "#!/usr/bin/env bash" "#!$bash/bin/bash"
    substituteInPlace testmain.sh --replace "#!/usr/bin/env bash" "#!$bash/bin/bash"
    substituteInPlace subroutines/hasError.sh --replace "#!/usr/bin/env bash" "#!$bash/bin/bash"
    g++ catpath.cpp -o catpath
    ./run.sh
    cat Testing_Report.txt >> $out
  '';
}
