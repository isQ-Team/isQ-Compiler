#include "isq/QTypes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include <mlir/IR/PatternMatch.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
struct RemoveQStateTempStore : public mlir::OpRewritePattern<mlir::AffineStoreOp> {
    RemoveQStateTempStore(mlir::MLIRContext *context)
        : OpRewritePattern<mlir::AffineStoreOp>(context, 1) {

        }

    mlir::LogicalResult match(mlir::AffineStoreOp op) const {
        bool is_qstate = op.getValueToStore().getType().isa<isq::ir::QStateType>();
        bool is_from_function_parameter = op.getValueToStore().isa<mlir::BlockArgument>();
        if(is_qstate && is_from_function_parameter){
            return mlir::success();
        }else{
            return mlir::failure();
        }
    }
/*
    void rewrite(mlir::AffineStoreOp op, mlir::PatternRewriter& rewriter){
        for(auto use: op->getUsers()){

        }
    }
*/
};
