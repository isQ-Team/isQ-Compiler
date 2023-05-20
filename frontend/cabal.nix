{ mkDerivation
, aeson
, alex
, array
, base
, bytestring
, containers
, deepseq
, directory
, extra
, filepath
, happy
, hpack
, hspec
, lens
, lib
, math-functions
, mtl
, multimap
, parsec
, pretty-simple
, split
, text
}:
mkDerivation {
  pname = "isqc-frontend";
  version = "0.1.0.0";
  src = ./.;
  isLibrary = true;
  isExecutable = true;
  libraryHaskellDepends = [
    aeson
    array
    base
    bytestring
    containers
    deepseq
    directory
    extra
    filepath
    lens
    math-functions
    mtl
    multimap
    parsec
    pretty-simple
    split
    text
  ];
  libraryToolDepends = [ alex happy hpack ];
  executableHaskellDepends = [
    aeson
    array
    base
    bytestring
    containers
    deepseq
    directory
    extra
    filepath
    lens
    math-functions
    mtl
    multimap
    parsec
    pretty-simple
    split
    text
  ];
  executableToolDepends = [ alex happy ];
  testHaskellDepends = [
    aeson
    array
    base
    bytestring
    containers
    deepseq
    directory
    extra
    filepath
    hspec
    lens
    math-functions
    mtl
    multimap
    parsec
    pretty-simple
    split
    text
  ];
  testToolDepends = [ alex happy ];
  prePatch = "hpack";
  homepage = "https://github.com/gjz010/isqv2-frontend#readme";
  license = lib.licenses.bsd3;
}
