{ llvmPackages_13, lld_13, cmake, fetchurl, ninja, python3, git }:
let
  stdenv = llvmPackages_13.stdenv;
in stdenv.mkDerivation {
  pname = "mlir";
  version = "14.0.0rc1";
  #builder = ./builder.sh;
  src = fetchurl {
    url = "https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.0-rc1/llvm-project-14.0.0rc1.src.tar.xz";
    sha256 = "3de5fb12e2c43ba4964fabb1baddea870b652d2c971aaff1172ca0c57bb1f54a";
  };
  buildInputs = [cmake ninja python3 git lld_13];
  cmakeFlags = with stdenv; [
    "-DLLVM_ENABLE_PROJECTS=llvm;mlir;lld"
    "-DLLVM_BUILD_EXAMPLES=OFF"
    "-DLLVM_TARGETS_TO_BUILD=X86;NVPTX;AMDGPU"
    "-DLLVM_ENABLE_ASSERTIONS=ON"
    "-DBUILD_SHARED_LIBS=ON"
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
    "-DCMAKE_C_COMPILER=clang"
    "-DCMAKE_CXX_COMPILER=clang++"
    "-DLLVM_ENABLE_LLD=ON"
  ];
  cmakeDir = "../llvm";
}
