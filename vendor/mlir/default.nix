{ cmake, fetchurl, ninja, python3, git, vendor }:
let
  stdenv = vendor.stdenvLLVM;
in
stdenv.mkDerivation {
  pname = "llvm-mlir";
  version = "16.0.6";
  #builder = ./builder.sh;
  src = fetchurl {
    url = "https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/llvm-project-16.0.6.src.tar.xz";
    sha256 = "13h2qd9brdpyn4i89rxpiw7k1f849f6a5kybsy39xkhp3l472pnf";
  };
  buildInputs = [ cmake ninja python3 git ];
  cmakeFlags = with stdenv; [
    "-DLLVM_ENABLE_PROJECTS=llvm;mlir;lld"
    "-DLLVM_BUILD_EXAMPLES=OFF"
    "-DLLVM_TARGETS_TO_BUILD=X86"
    "-DLLVM_ENABLE_ASSERTIONS=ON"
    "-DBUILD_SHARED_LIBS=ON"
    "-DCMAKE_C_COMPILER=clang"
    "-DCMAKE_CXX_COMPILER=clang++"
    "-DLLVM_ENABLE_LLD=ON"
  ];
  cmakeBuildType = "RelWithDebInfo";
  separateDebugInfo = true;
  postFixup = ''
    mkdir -p $debug
  '';
  cmakeDir = "../llvm";
}
