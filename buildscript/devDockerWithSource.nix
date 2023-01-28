{ pkgs ? import ./pkgs.nix, include_sources ? true }:
let
  toolchain = import ./devDependencies.nix { is_container = true; };
  sources = import ./sources.nix { };
  ubuntu = pkgs.dockerTools.pullImage {
    imageName = "ubuntu";
    imageDigest = "sha256:669e010b58baf5beb2836b253c1fd5768333f0d1dbcb834f7c07a4dc93f474be";
    sha256 = "0x5zbfm5mqx9wv36mv2y6fyv266kh2c13rhd4crkj25zrcsm4ib5";
    finalImageName = "ubuntu";
    finalImageTag = "latest";
  };

  source_files = (pkgs.nix-gitignore.gitignoreSource [ ] ../.);
  source_pkg = pkgs.buildEnv rec {
    name = "isqv2-sources";
    paths = [ source_files ];
    extraPrefix = [ "/opt/isqsources/" ];
  };



  dev_image = pkgs.dockerTools.buildLayeredImage {
    fromImage = ubuntu;
    contents = if include_sources then [ toolchain source_pkg ] else [ toolchain ];
    name = if include_sources then "isqv2-dev-with-source-dont-publish-this" else "isqv2-dev";
    config = {
      Cmd = [ "/bin/bash" "-c" "export PATH=/opt/isqdeps/bin:$PATH;export LANG=C.UTF-8;bash" ];
      WorkingDir = if include_sources then source_files else "/";
    };
    #diskSize = 1024;
  };


in
dev_image
