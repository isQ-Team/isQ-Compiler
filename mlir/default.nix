{ cmake
, ninja
, doxygen
, graphviz
, python3
, which
, git
, lld
, eigen
, llvmPackages
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
  isq-opt =
    llvmPackages.stdenv.mkDerivation {
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
      passthru.isQDevShell = mkShell.override { stdenv = llvmPackages.stdenv; } {
        inputsFrom = [ isq-opt ];
        nativeBuildInputs = [ clang-tools ];
      };
      inherit mlir;
    };
in
isq-opt
