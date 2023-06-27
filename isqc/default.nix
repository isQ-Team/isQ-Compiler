{ vendor ? null
, gitignoreSource ? vendor.gitignoreSource
, isQVersion
, isQVersionHook
, isQRustPackages
}:
(isQRustPackages.workspace."isqc" { }).overrideAttrs (final: prev: {
  nativeBuildInputs = prev.nativeBuildInputs ++ [ isQVersionHook ];
})
