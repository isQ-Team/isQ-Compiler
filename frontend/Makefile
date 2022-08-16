setup: cabal.nix isqc-frontend.cabal
cabal.nix: package.yaml
	cabal2nix --hpack . > cabal.nix
isqc-frontend.cabal: package.yaml
	hpack