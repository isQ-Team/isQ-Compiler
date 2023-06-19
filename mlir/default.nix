{ cmake
, ninja
, doxygen
, graphviz
, python3
, which
, git
, lld
, eigen
, llvmPackages_16
, nlohmann_json
, mkShell
, clang-tools
, vendor ? null
, gitignoreSource ? vendor.gitignoreSource
, mlir ? vendor.mlir
, caterpillar ? vendor.caterpillar
, fmt
}:
let
  stdenv = vendor.stdenvLLVM;
  isq-opt =
    stdenv.mkDerivation {
      name = "isq-opt";
      nativeBuildInputs = [ cmake ninja doxygen graphviz python3 which git lld ];
      buildInputs = [
        eigen
        mlir
        nlohmann_json
        caterpillar
        fmt
      ];
      src = gitignoreSource ./.;
      cmakeFlags = [ "-DISQ_OPT_ENABLE_ASSERTIONS=1" ];
      passthru.isQDevShell = mkShell.override { stdenv = stdenv; } {
        inputsFrom = [ isq-opt ];
        nativeBuildInputs = [ vendor.clang-tools ];
      };
      inherit mlir;
    };
in
isq-opt
