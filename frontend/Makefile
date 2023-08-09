all: configure build
configure: isqc1.cabal
	runhaskell Setup.hs configure --enable-tests
build: isqc1.cabal
	runhaskell Setup.hs build
test: isqc1.cabal
	runhaskell Setup.hs test
isqc1.cabal: package.yaml
	hpack


.PHONY: all configure build clean test

clean:
	rm -rf isqc1.cabal dist
