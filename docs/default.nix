{ mkdocs
, python3Packages
, stdenvNoCC
, vendor ? null
, gitignoreSource ? vendor.gitignoreSource
}:
stdenvNoCC.mkDerivation {
  pname = "isqc-docs";
  version = "0.1.0";
  buildInputs = [ mkdocs python3Packages.mkdocs-material ];
  src = gitignoreSource ./.;
}
