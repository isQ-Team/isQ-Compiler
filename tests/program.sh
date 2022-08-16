#!/usr/bin/env bash
#第一个参数源文件，第二个参数是输出的编译结果，第三个参数是打印的输出信息
echo Compiling $1

${isq}/bin/isqc compile $1 -o $2 2> $3