FROM nixos/nix
RUN mkdir /isqv2
COPY *.nix /isqv2/
WORKDIR /isqv2
RUN nix-env -i -f default.nix
CMD nix-shell