let
  flake = (builtins.getFlake (builtins.toString ./.));
in
flake.legacyPackages.${builtins.currentSystem}.isqc
