all: test1.so

test1.so: test1.ll test1.c ../../src/facades/qir/shim/qir_builtin/shim.ll ../../src/facades/qir/shim/qsharp_core/shim.ll ../../src/facades/qir/shim/qsharp_foundation/shim.ll ../../src/facades/qir/shim/isq/shim.ll
	clang -shared -fPIC -o $@ $^

clean:
	rm test1.so