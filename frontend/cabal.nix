{ mkDerivation, aeson, alex, array, base, bytestring, containers
, deepseq, extra, happy, hpack, lens, lib, math-functions, mtl
, parsec, pretty-simple, text
}:
mkDerivation {
  pname = "isqc-frontend";
  version = "0.1.0.0";
  src = ./.;
  isLibrary = false;
  isExecutable = true;
  libraryToolDepends = [ hpack ];
  executableHaskellDepends = [
    aeson array base bytestring containers deepseq extra lens
    math-functions mtl parsec pretty-simple text
  ];
  executableToolDepends = [ alex happy ];
  prePatch = "hpack";
  homepage = "https://github.com/gjz010/isqv2-frontend#readme";
  license = lib.licenses.bsd3;
}
