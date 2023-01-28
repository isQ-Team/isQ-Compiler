#include <stdio.h>
#ifndef __has_declspec_attribute         // Optional of course.
  #define __has_declspec_attribute(x) 0  // Compatibility with non-clang compilers.
#endif

#if __has_declspec_attribute(dllexport)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

void Microsoft__Quantum__Qir__Emission__DemonstrateTeleportationUsingPresharedEntanglement();


DLLEXPORT void isq_simulator_entry(){
  printf("QIR Simulation started.\n");
    Microsoft__Quantum__Qir__Emission__DemonstrateTeleportationUsingPresharedEntanglement();
    
}