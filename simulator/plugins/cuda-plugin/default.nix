{stdenv, cudaPackages_11_5, linuxPackages, addOpenGLRunpath, makeWrapper}:
stdenv.mkDerivation{
    name = "isq-simulator-plugin-cuda";
    buildInputs = [cudaPackages_11_5.cudatoolkit linuxPackages.nvidia_x11 addOpenGLRunpath makeWrapper];
    shellHook = ''
       export CUDA_PATH=${cudaPackages_11_5.cudatoolkit}
       export EXTRA_LDFLAGS="-L/lib -L${linuxPackages.nvidia_x11}/lib"
       export EXTRA_CCFLAGS="-I/usr/include"
    '';
    src = ./.;
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
    outputs = [ "out" "dev" "bin"  ];
}
