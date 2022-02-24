{pkgs? import ./pkgs.nix}:
let 
toolchain = import ./default.nix { is_container = true; };
sources = import ./sources.nix {};
ubuntu = pkgs.dockerTools.pullImage {
  imageName = "ubuntu";
  imageDigest = "sha256:669e010b58baf5beb2836b253c1fd5768333f0d1dbcb834f7c07a4dc93f474be";
  sha256 = "0x5zbfm5mqx9wv36mv2y6fyv266kh2c13rhd4crkj25zrcsm4ib5";
  finalImageName = "ubuntu";
  finalImageTag = "latest";
};


dev_image = pkgs.dockerTools.buildLayeredImage {
  fromImage = ubuntu;
  contents =  [ toolchain sources.source_pkg ];
  name = "isqv2-dev";
  config = {
    Cmd = [ "/bin/bash" "-c" "export PATH=/opt/isqdeps/bin:$PATH;export LANG=C.UTF-8;bash"];
    WorkingDir = sources.source_files;
  };
  #diskSize = 1024;
};

in
dev_image