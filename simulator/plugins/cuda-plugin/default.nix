{ stdenv, cudaPackages_11_5, addOpenGLRunpath, makeWrapper }:
let cudatoolkit = cudaPackages_11_5.cudatoolkit;
in
stdenv.mkDerivation {
  name = "isq-simulator-plugin-cuda";
  buildInputs = [ cudatoolkit addOpenGLRunpath makeWrapper ];
  src = ./.;
  preBuild = ''
    export CUDA_PATH=${cudatoolkit}
  '';
  installPhase = ''
    runHook preInstall
    mkdir -p $out/bin
    mkdir -p $out/lib
    mkdir -p $out/include
    cp qsim_test $out/bin/
    cp libqsim_kernel.so $out/lib/libqsim_kernel.so
    cp qsim_kernel.h $out/include/qsim_kernel.h
    moveToOutput bin "''${!outputBin}"
    runHook postInstall
  '';
  postFixup = ''
    addOpenGLRunpath $out/lib/libqsim_kernel.so
  '';
  outputs = [ "out" "dev" "bin" ];
}
