<<<<<<< HEAD
{pkgs? import ../../../buildscript/pkgs.nix}:
pkgs.stdenv.mkDerivation{
    name = "isqv2-simulator-cuda-plugin";
    buildInputs = with pkgs; [cudaPackages.cudatoolkit_11_5 linuxPackages.nvidia_x11 addOpenGLRunpath makeWrapper];
    shellHook = ''
       export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit_11_5}
       # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
       export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
       export EXTRA_CCFLAGS="-I/usr/include"
    '';
    src = ./.;
    installPhase = ''
=======
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
>>>>>>> merge
    runHook preInstall
    mkdir -p $out/bin
    mkdir -p $out/lib
    mkdir -p $out/include
    cp qsim_test $out/bin/
    cp libqsim_kernel.so $out/lib/libqsim_kernel.so
    cp qsim_kernel.h $out/include/qsim_kernel.h
    moveToOutput bin "''${!outputBin}"
    runHook postInstall
<<<<<<< HEAD
    '';
    postFixup = ''
      addOpenGLRunpath $out/lib/libqsim_kernel.so
    '';
    outputs = [ "out" "dev" "bin"  ];
=======
  '';
  postFixup = ''
    addOpenGLRunpath $out/lib/libqsim_kernel.so
  '';
  outputs = [ "out" "dev" "bin" ];
>>>>>>> merge
}
