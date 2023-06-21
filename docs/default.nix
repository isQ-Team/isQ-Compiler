{ mkdocs
, python3Packages
, stdenvNoCC
, vendor ? null
, gitignoreSource ? vendor.gitignoreSource
, isQVersion
, isQVersionHook
}:
stdenvNoCC.mkDerivation {
  pname = "isqc-docs";
  inherit (isQVersion) version;
  buildInputs = [ mkdocs python3Packages.mkdocs-material ];
  nativeBuildInputs = [ isQVersionHook ];
  src = gitignoreSource ./.;
}
