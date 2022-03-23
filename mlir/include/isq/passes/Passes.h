#ifndef _ISQ_PASSES_PASSES_H
#define _ISQ_PASSES_PASSES_H
#include "isq/Operations.h"
namespace isq{
namespace ir{
namespace passes{
extern const char* ISQ_FAMOUS_GATE;
//void registerQuantumGatePass();
void registerDecorateFolding();
void registerQSD();
void registerExpandDecomposition();
void registerLowerToQIRRep();
void registerQIR2LLVM();
void registerPureGateDetect();
void registerRecognizeFamousGates();

bool isFamousGate(DefgateOp op, const char* famous_gate);
}
}
}
#endif