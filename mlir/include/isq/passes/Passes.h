#ifndef _ISQ_PASSES_PASSES_H
#define _ISQ_PASSES_PASSES_H
#include "isq/Operations.h"
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
void registerRecognizeFamousGates();
void registerSQRot2U3();

llvm::SmallString<32> getFamousName(const char* famous_gate);
bool isFamousGate(DefgateOp op, const char* famous_gate);
void emitBuiltinGate(mlir::OpBuilder& builder, const char* famous_gate, mlir::ArrayRef<mlir::Value*> qubits, mlir::ArrayRef<mlir::Value> params = {}, mlir::ArrayAttr ctrl = nullptr, bool adjoint = false);
}
}
}
#endif