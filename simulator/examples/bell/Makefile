all: bell.so

bell.so: bell.ll bell.c ../../src/facades/qir/shim/qir_builtin/shim.ll ../../src/facades/qir/shim/qsharp_core/shim.ll ../../src/facades/qir/shim/qsharp_foundation/shim.ll
	clang -shared -fPIC -o $@ $^

clean:
	rm bell.so