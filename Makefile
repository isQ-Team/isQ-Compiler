all: frontend mlir isqc simulator isq-simulator.bc

.PHONY: check-env frontend mlir isqc simulator isq-simulator.bc all lock develop run clean

check-env:
ifndef ISQ_ROOT
	$(error ISQ_ROOT is undefined)
endif
	$(info $(shell mkdir -p $(ISQ_ROOT)/bin))

frontend: check-env
	cd frontend && make all
	cd ${ISQ_ROOT}/bin && \
	rm -f isqc1 && \
	ln -s ../../frontend/dist/build/isqc1/isqc1 isqc1

mlir: check-env
	cd mlir && mkdir -p build && cd build && cmake ../ -GNinja && ninja
	cd ${ISQ_ROOT}/bin && \
	rm -f isq-opt && \
	ln -s ../../mlir/build/tools/isq-opt isq-opt

isqc: check-env
	cd isqc && cargo build;
	cd ${ISQ_ROOT}/bin && \
	rm -f isqc && \
	ln -s ../../isqc/target/debug/isqc isqc

simulator: check-env
	cd simulator &&	cargo build
	cd ${ISQ_ROOT}/bin && \
	rm -f simulator && \
	ln -s ../../simulator/target/debug/simulator simulator

isq-simulator.bc: check-env
	mkdir -p ${ISQ_ROOT}/share/isq-simulator;
	linker=`which llvm-link` && if [ $$linker == "" ]; then linker=$(ISQ_ROOT)/bin/llvm-link; fi \
	&& cd simulator && eval $$linker src/facades/qir/shim/qir_builtin/shim.ll src/facades/qir/shim/qsharp_core/shim.ll \
	src/facades/qir/shim/qsharp_foundation/shim.ll src/facades/qir/shim/isq/shim.ll -o ${ISQ_ROOT}/share/isq-simulator/isq-simulator.bc

clean: check-env
	rm .build -r
	mkdir -p .build
develop:
	@exec nix develop
lock:
	cd vendor/mlir && nix flake lock --update-input isqc-base
	cd simulator && nix flake lock --update-input isqc-base --update-input mlir
	cd isqc && nix flake lock --update-input isqc-base
	cd mlir && nix flake lock --update-input isqc-base --update-input mlir
	cd frontend && nix flake lock --update-input isqc-base
	nix flake lock --update-input isqc-base --update-input mlir --update-input isqc1 --update-input isq-opt --update-input isqc-driver --update-input isq-simulator