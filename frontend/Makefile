all: build
configure: isqc1.cabal
	runhaskell Setup.hs configure
build: isqc1.cabal
	runhaskell Setup.hs build
isqc1.cabal: package.yaml
	hpack


.PHONY: all configure build clean

clean:
	rm -rf isqc1.cabal dist