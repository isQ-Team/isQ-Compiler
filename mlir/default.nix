<<<<<<< HEAD
{pkgs? import ../buildscript/pkgs.nix }:
let 
  lib = pkgs.lib;
  callPackage = lib.callPackageWith pkgs;
  mlir = pkgs.callPackage ./vendor/mlir.nix {};
in
with pkgs;
llvmPackages_13.stdenv.mkDerivation {
  name = "isq-qir";
  nativeBuildInputs = [ cmake ninja doxygen graphviz python3 which git lld_13 ];
  buildInputs = [ eigen mlir ];
  src = nix-gitignore.gitignoreSource [] ./.;
  cmakeFlags = [ "-DISQ_OPT_ENABLE_ASSERTIONS=1" ];
}
=======
{ mlir, cmake, ninja, doxygen, graphviz, python3, which, git, lld, eigen, gitignoreSource, llvmPackages, nlohmann_json }:
llvmPackages.stdenv.mkDerivation {
  name = "isq-opt";
  nativeBuildInputs = [ cmake ninja doxygen graphviz python3 which git lld ];
  buildInputs = [ eigen mlir nlohmann_json ];
  src = gitignoreSource ./.;
  cmakeFlags = [ "-DISQ_OPT_ENABLE_ASSERTIONS=1" ];
  inherit mlir;
}
>>>>>>> merge
