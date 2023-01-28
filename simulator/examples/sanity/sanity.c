#include <stdio.h>
#include <stdbool.h>
#ifndef __has_declspec_attribute         // Optional of course.
  #define __has_declspec_attribute(x) 0  // Compatibility with non-clang compilers.
#endif

#if __has_declspec_attribute(dllexport)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

double Qir__Emission__CallableTest__Interop(_Bool shouldFail);
double Qir__Emission__RangeTest__Interop();

DLLEXPORT void isq_simulator_entry(){
  printf("QIR Sanity check started.\n");
  Qir__Emission__CallableTest__Interop(false);
  Qir__Emission__RangeTest__Interop();
}