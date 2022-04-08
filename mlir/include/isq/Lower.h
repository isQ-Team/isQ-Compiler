#ifndef _ISQ_LOWER_H
#define _ISQ_LOWER_H

#include <memory>
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "./QSynthesis.h"
#include "./Operations.h"
#include "./QTypes.h"
namespace isq {
namespace ir {

namespace lower{
    
// Lower from isQ IR to isQ-QIR, where everything is in its reference form.
// This is mainly done by inserting qubit-initialization steps and gate decomposition.
// This allows easy inspecting of QIR form as well as easy conversion to real QIR.
std::unique_ptr<mlir::Pass> createLowerToQIRPass();
// Lower from isQ-QIR to QIR.
// Simply performing type conversion and regular lowering is enough.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

// Lower from isQ IR to QCIS, convert CNOT -> H CZ H, inv T -> TD, inv S -> SD
std::unique_ptr<mlir::Pass> createLowerToQCISPass();

using mlir::PatternRewriter;
using mlir::ModuleOp;
using mlir::Value;
class QIRExternQuantumFunc{
public:
    QStateType getQStateType(::mlir::MLIRContext* ctx);
    QIRQubitType getQIRQubitType(::mlir::MLIRContext* ctx);
    QIRResultType getQIRResultType(::mlir::MLIRContext* ctx);
    mlir::IndexType getIndexType(::mlir::MLIRContext* ctx);
    mlir::IntegerType getI1Type(::mlir::MLIRContext* ctx);
    mlir::IntegerType getI64Type(::mlir::MLIRContext* ctx);
    mlir::Float64Type getF64Type(::mlir::MLIRContext* ctx);
    void printInt(::mlir::Location loc, PatternRewriter& rewriter, ModuleOp module, Value i);
    void printFloat(::mlir::Location loc, PatternRewriter& rewriter, ModuleOp module, Value f);
    Value allocQubit(::mlir::Location loc, PatternRewriter& rewriter, ModuleOp module);
    void releaseQubit(::mlir::Location loc, PatternRewriter& rewriter, ModuleOp module, Value q);
    // Obtain some boolean value at once.
    Value measureQubit(::mlir::Location loc, PatternRewriter& rewriter, ModuleOp module, Value q);
    void reset(::mlir::Location loc, PatternRewriter &rewriter, ModuleOp module, Value q);
    void callQIRGate(::mlir::Location loc, PatternRewriter &rewriter, ModuleOp module, ::mlir::StringRef gateName, ::mlir::ArrayRef<Value> arguments);
    std::string getMainFuncName();
};

}

} 
} 

#endif // _ISQ_LOWER_H
