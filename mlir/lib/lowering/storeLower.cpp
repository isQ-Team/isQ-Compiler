#include "isq/lowering/storeLower.h"
#include "isq/lowering/quantumFunc.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <iostream>
using namespace mlir;
using namespace std;

LogicalResult StoreOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                ConversionPatternRewriter &rewriter) const {

    auto storeOp = cast<mlir::AffineStoreOp>(op);
    auto eletype = storeOp.getMemRefType();

    if (eletype.getElementType().isa<isq::ir::QStateType>()){
        //cout << "remove store qbit" << endl;
        rewriter.eraseOp(op);
        return mlir::success();
    }

    return mlir::failure();

}