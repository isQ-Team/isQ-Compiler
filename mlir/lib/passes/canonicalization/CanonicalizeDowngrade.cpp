
#include "isq/Enums.h"
#include "isq/Operations.h"
#include "isq/passes/canonicalization/CanonicalizeDowngrade.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include <mlir/Support/LogicalResult.h>
namespace isq{
namespace ir{
namespace passes{
namespace canonicalize{

MergeDowngrade::MergeDowngrade(mlir::MLIRContext* ctx): mlir::OpRewritePattern<isq::ir::DowngradeGateOp>(ctx, 1){}
mlir::LogicalResult MergeDowngrade::matchAndRewrite(isq::ir::DowngradeGateOp op,  mlir::PatternRewriter &rewriter) const{
    if(auto parent = mlir::dyn_cast_or_null<isq::ir::DowngradeGateOp>(op.getArgs().getDefiningOp())){
        rewriter.updateRootInPlace(op, [&](){
            op.setOperand(parent->getOperand(0));
        });
        return mlir::success();
    }
    return mlir::failure();
}

EliminateUselessDowngrade::EliminateUselessDowngrade(mlir::MLIRContext* ctx): mlir::OpRewritePattern<isq::ir::DowngradeGateOp>(ctx, 1){}
mlir::LogicalResult EliminateUselessDowngrade::matchAndRewrite(isq::ir::DowngradeGateOp op,  mlir::PatternRewriter &rewriter) const{
    if(op.getArgs().getType() == op.getResult().getType()){
        rewriter.replaceOp(op, op.getArgs());
        return mlir::success();
    }
    return mlir::failure();
    
}



}
}
}
}
