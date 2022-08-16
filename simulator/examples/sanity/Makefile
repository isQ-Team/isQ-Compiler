all: sanity.so

sanity.so: qir/Program.ll sanity.c ../../src/facades/qir/shim/qir_builtin/shim.ll ../../src/facades/qir/shim/qsharp_core/shim.ll ../../src/facades/qir/shim/qsharp_foundation/shim.ll
	clang -shared -fPIC -o sanity.so sanity.c qir/Program.ll ../../src/facades/qir/shim/qir_builtin/shim.ll ../../src/facades/qir/shim/qsharp_core/shim.ll ../../src/facades/qir/shim/qsharp_foundation/shim.ll

clean:
	rm sanity.so