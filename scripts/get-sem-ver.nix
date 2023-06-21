#!/usr/bin/env -S nix  eval --json -f
rec {
  versionJSON = builtins.fromJSON (builtins.readFile ../version.json);
}
