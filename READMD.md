isQ Quantum Compiler v2
============================

Monolithic repo.


Building
============================

There are multiple ways to setup a development environment.

Nix on Linux (Recommended)
----------------------------

```bash
# Install nix here: https://nixos.org/download.html
sh <(curl -L https://nixos.org/nix/install) --daemon
# And the development environment is just a click away.
nix-shell
```

TODO: Nix on Mac?


Docker Image
----------------------------

A Docker image, with required build tools installed in an Ubuntu container, is provided. However, some steps are requires to make building work.

```bash
# In host machine: starts a bash.
docker run -it isqv2-dev
# In container: these commands are required to make sure stack runs.
apt update
apt install ca-certificates
```