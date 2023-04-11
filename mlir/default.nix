{ mlir, cmake, ninja, doxygen, graphviz, python3, which, git, lld, eigen, gitignoreSource, llvmPackages, nlohmann_json }:
llvmPackages.stdenv.mkDerivation {
  name = "isq-opt";
  nativeBuildInputs = [ cmake ninja doxygen graphviz python3 which git lld ];
  buildInputs = [ eigen mlir nlohmann_json ];
  src = gitignoreSource ./.;
  cmakeFlags = [ "-DISQ_OPT_ENABLE_ASSERTIONS=1" ];
  inherit mlir;
}
