all: frontend mlir isqc simulator isq-simulator.bc

.PHONY: check-env frontend mlir isqc simulator isq-simulator.bc all lock develop

check-env:
ifndef ISQ_ROOT
	$(error ISQ_ROOT is undefined)
endif
	$(info $(shell mkdir -p $(ISQ_ROOT)/bin))

frontend: check-env
	cd frontend && stack build --allow-different-user && \
	cp .stack-work/dist/x86_64-linux-tinfo6/Cabal-3.2.1.0/build/isqc1/isqc1 ${ISQ_ROOT}/bin/

mlir: check-env
	cd mlir && mkdir build && cd build && cmake ../ && make && cp tools/isq-opt ${ISQ_ROOT}/bin/

isqc: check-env
	cd isqc && cargo build && cp target/debug/isqc ${ISQ_ROOT}/bin/

simulator: check-env
	cd simulator &&	cargo build && cp target/debug/simulator ${ISQ_ROOT}/bin/

isq-simulator.bc: check-env
	linker=`which llvm-link` && if [ $$linker == "" ]; then linker=$(ISQ_ROOT)/bin/llvm-link; fi \
	&& cd simulator && eval $$linker src/facades/qir/shim/qir_builtin/shim.ll src/facades/qir/shim/qsharp_core/shim.ll \
	src/facades/qir/shim/qsharp_foundation/shim.ll src/facades/qir/shim/isq/shim.ll -o ${ISQ_ROOT}/bin/isq-simulator.bc

develop:
	@exec nix develop
lock:
	cd vendor/mlir && nix flake lock --update-input isqc-base
	cd simulator && nix flake lock --update-input isqc-base --update-input mlir
	cd isqc && nix flake lock --update-input isqc-base
	cd mlir && nix flake lock --update-input isqc-base --update-input mlir
	cd frontend && nix flake lock --update-input isqc-base
	nix flake lock --update-input isqc-base --update-input mlir --update-input isqc1 --update-input isq-opt --update-input isqc-driver --update-input isq-simulator