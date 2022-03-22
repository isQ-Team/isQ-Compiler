#ifndef _ISQ_PASSES_PASSES_H
#define _ISQ_PASSES_PASSES_H
namespace isq{
namespace ir{
namespace passes{

//void registerQuantumGatePass();
void registerDecorateFolding();
void registerQSD();
void registerExpandDecomposition();
void registerLowerToQIRRep();
void registerQIR2LLVM();
void registerPureGateDetect();
}
}
}
#endif