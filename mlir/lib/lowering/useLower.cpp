#include "isq/lowering/useLower.h"
#include "isq/lowering/quantumFunc.h"
#include "isq/IR.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <iostream>
using namespace std;
using namespace mlir;


LogicalResult useGateOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const{
    

    auto loc = op->getLoc();

    MLIRContext *context = rewriter.getContext();

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    /*
    auto useOp = cast<isq::ir::UseGateOp>(op);

    auto res = useOp.getResult();
    for (auto user : res.getUsers()) {
        user->replaceUsesOfWith(res, operands[0]);
    }*/

    rewriter.eraseOp(op);
    return mlir::success();
}