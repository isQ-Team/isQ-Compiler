#include "isq/lowering/moduleLower.h"
#include "isq/lowering/quantumFunc.h"
#include "isq/IR.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <iostream>
using namespace std;
using namespace mlir;


LogicalResult moduleOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const{
    
    
    auto loc = op->getLoc();

    MLIRContext *context = rewriter.getContext();

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    for (auto &r: op->getRegions()){
        for (auto &b : r.getBlocks()){
            for (auto &p : b.getOperations()){
                auto dop = cast<isq::ir::DeclareQOpOp>(p);
                auto op_name = dop.sym_name();
                if (op_name.equals(StringRef("measure"))){
                    auto func_name = LLVMQuantumFunc::getOrInsertMeasure(rewriter, parentModule);
                }else if (op_name.equals(StringRef("reset"))){
                    auto func_name = LLVMQuantumFunc::getOrInsertReset(rewriter, parentModule);
                }
            }
        }
    }

    rewriter.eraseOp(op);
    return mlir::success();
}