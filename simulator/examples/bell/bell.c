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


void bell_main();
<<<<<<< HEAD
DLLEXPORT void __isq__entry(){
=======
DLLEXPORT void isq_simulator_entry(){
>>>>>>> merge
  bell_main();
}