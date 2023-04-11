{ python3, stdenv, bash }:
let
  python-deps = pypi: with pypi; [ numpy networkx ];
  # use pypy for performance.
  python = python3.withPackages python-deps;
in
stdenv.mkDerivation rec {
  name = "isq-simulator-plugin-qcis";
  buildInputs = [ python bash ];
  PYTHONPATH = "${python}/${python.sitePackages}";
  src = ./.;
  inherit python;
  pythonpath = PYTHONPATH;
  installPhase = ''
    runHook preInstall
    mkdir -p $out/bin
    mkdir -p $out/lib
    cp -r $src/src $out/lib/isq-simulator-plugin-qcis
    cp $src/bin/qcis-routing $out/bin/
    substituteAllInPlace $out/bin/qcis-routing
    runHook postInstall
  '';
}
