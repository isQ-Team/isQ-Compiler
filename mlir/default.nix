{mlir, cmake, ninja, doxygen, graphviz, python3, which, git, lld, eigen, gitignoreSource, llvmPackages}:
llvmPackages.stdenv.mkDerivation {
  name = "isq-opt";
  nativeBuildInputs = [ cmake ninja doxygen graphviz python3 which git lld ];
  buildInputs = [ eigen mlir ];
  src = gitignoreSource ./.;
  cmakeFlags = [ "-DISQ_OPT_ENABLE_ASSERTIONS=1" ];
  propagatedBuildInputs = [ mlir ];
  inherit mlir;
}