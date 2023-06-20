{ mkdocs
, python3Packages
, stdenvNoCC
, vendor ? null
, gitignoreSource ? vendor.gitignoreSource
, isQVersion
}:
stdenvNoCC.mkDerivation {
  pname = "isqc-docs";
  inherit (isQVersion) version;
  buildInputs = [ mkdocs python3Packages.mkdocs-material ];
  src = gitignoreSource ./.;
}
