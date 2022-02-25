Setting up Development Environment
============================

isQ v2 uses [Nix](https://nixos.org/download.html) to manage its dependencies. However, there are multiple ways to setup a development environment.

Nix on Linux (Recommended)
----------------------------

The easiest way to setup develop environment is to build the environment provided by `shell.nix`.

```bash
# Multi-user Nix installation.
sh <(curl -L https://nixos.org/nix/install) --daemon
# And the development environment is just a click away.
nix-shell
```

TODO: Nix on Mac?


Docker Image
----------------------------

We also provide a (Docker development image)[https://hub.docker.com/repository/docker/gjz010/isqv2-dev], shipping required build tools installed in an Ubuntu container to save the trouble of learning how to install or use Nix.

### Quick start

```bash
# In host machine: starts a bash.
docker run -it isqv2-dev:${ISQV2_DEVELOPMENT_IMAGE_HASH}
# In container: initialization work.
apt update
apt install ca-certificates
git clone ${ISQV2_GIT}
```

To build subprojects:

Frontend:
```
cd frontend
ghc Setup.hs
./Setup configure
./Setup build
```

MLIR:
```bash
cd mlir
mkdir build && cd build
cmake -GNinja ..
ninja
```

Simulator:
```bash
cd simulator
cargo build
```




### Building the Docker Image

Simply use Nix to build the Docker image.

```bash
nix build -f buildscript/devDocker.nix
```

The result image will have all tools ready. If you want to distribute source with the Docker image, use

```bash
nix build -f buildscript/devDockerWithSource.nix
```

