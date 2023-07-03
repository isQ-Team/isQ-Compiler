{ vendor ? null
, gitignoreSource ? vendor.gitignoreSource
, isQVersion
, isQVersionHook
, isQRustPackages
}:
(isQRustPackages.workspace."isqc" { })
