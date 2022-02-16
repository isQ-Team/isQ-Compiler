#include "isq/lowering/defgateLower.h"
#include "isq/lowering/quantumFunc.h"
#include "isq/IR.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <iostream>
using namespace std;
using namespace mlir;


LogicalResult defgateOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const{
    

    auto loc = op->getLoc();

    MLIRContext *context = rewriter.getContext();

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    auto defgateop = cast<isq::ir::DefgateOp>(op);

    auto func_name = LLVMQuantumFunc::getOrInsertGate(rewriter, parentModule, defgateop);

    rewriter.eraseOp(op);
    
    return mlir::success();
}