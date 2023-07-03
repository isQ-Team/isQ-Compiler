{ mkdocs
, python3
, stdenvNoCC
, vendor ? null
, gitignoreSource ? vendor.gitignoreSource
, isQVersion
, isQVersionHook
, fetchurl
}:
let
  python = python3.override {
    packageOverrides = self: super: {
      "verspec" = super.buildPythonPackage rec {
        pname = "verspec";
        version = "0.1.0";
        src = fetchurl {
          url = "https://files.pythonhosted.org/packages/a4/ce/3b6fee91c85626eaf769d617f1be9d2e15c1cca027bbdeb2e0d751469355/verspec-0.1.0-py3-none-any.whl";
          sha256 = "0c8wy40a3kc1sisrcnx8pbdyc0g864mawsd48m64dj9wcgapf63l";
        };
        format = "wheel";
        doCheck = false;
        buildInputs = [ ];
        checkInputs = [ ];
        nativeBuildInputs = [ ];
        propagatedBuildInputs = [ ];
      };
      "mike" = super.buildPythonPackage rec {
        pname = "mike";
        version = "1.1.2";
        src = fetchurl {
          url = "https://files.pythonhosted.org/packages/66/8a/f226f8c512a4e3ee36438613fde32d371262e985643d308850cf4bdaed15/mike-1.1.2-py3-none-any.whl";
          sha256 = "1ji72lnzmij9bykq13r4gni4rw0hz1blz0qgy66xfd4qfql7qc2c";
        };
        format = "wheel";
        doCheck = false;
        buildInputs = [ ];
        checkInputs = [ ];
        nativeBuildInputs = [ ];
        propagatedBuildInputs = [
          self."jinja2"
          self."mkdocs"
          self."pyyaml"
          self."verspec"
          self.setuptools
        ];
      };
    };
  };
  pythonMkdocs = python.withPackages (p: with p; [ mkdocs-material mkdocs mike ]);
in
stdenvNoCC.mkDerivation {
  pname = "isqc-docs";
  inherit (isQVersion) version;
  buildInputs = [ pythonMkdocs ];
  nativeBuildInputs = [ isQVersionHook ];
  src = gitignoreSource ./.;
}
