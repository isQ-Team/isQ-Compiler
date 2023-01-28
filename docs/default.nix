{ mkdocs, gitignoreSource, stdenvNoCC }:
stdenvNoCC.mkDerivation {
  pname = "isqc-docs";
  version = "0.1.0";
  buildInputs = [ mkdocs ];
  src = gitignoreSource ./.;
}
