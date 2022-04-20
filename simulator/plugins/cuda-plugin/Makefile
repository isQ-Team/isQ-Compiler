all: libqsim_kernel.so qsim_test
qsim_test: test.cpp libqsim_kernel.so
	nvcc test.cpp -o $@ -O3 -L. -lqsim_kernel -Xcompiler -fno-rtti -std=c++11
libqsim_kernel.so: qsim_kernel.cu qsim_kernel.h
	nvcc qsim_kernel.cu -o $@ -shared -O3 -Xcompiler -static-libstdc++ -Xcompiler -Wall -Xcompiler -Werror -Xcompiler -Wextra -Xcompiler -Wconversion -Xcompiler -fno-rtti -arch=all
clean:
	rm -f libqsim_kernel.so qsim_test
.PHONY: clean