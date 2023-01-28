let
  flake = (builtins.getFlake (builtins.toString ./.));
in
flake.devShells.${builtins.currentSystem}.codium
