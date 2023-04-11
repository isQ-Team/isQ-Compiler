let
  flake = (builtins.getFlake (builtins.toString ./.));
in
flake.devShell.${builtins.currentSystem}
