{dependencies? import ./dependencies.nix {}, is_container? false }:
dependencies.pkgs.buildEnv rec {
  name = "isqv2-tools";
  paths = dependencies.packages;
  passthru = {
    environmentVars = {
      RUST_SRC_PATH = dependencies.rustSrcPath;
    };
  };
  extraPrefix = if is_container then [ "/opt/isqdeps" ] else ["/"];
}
