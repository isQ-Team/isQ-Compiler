
#include "isq/Enums.h"
#include "isq/Operations.h"
#include "isq/passes/canonicalization/CanonicalizeDecorateGates.h"
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

MergeDecorate::MergeDecorate(mlir::MLIRContext* ctx): mlir::OpRewritePattern<isq::ir::DecorateOp>(ctx, 1){}
mlir::LogicalResult MergeDecorate::matchAndRewrite(isq::ir::DecorateOp op,  mlir::PatternRewriter &rewriter) const{
    if(auto parent = mlir::dyn_cast_or_null<isq::ir::DecorateOp>(op.getArgs().getDefiningOp())){
        llvm::SmallVector<bool> ctrl;
        bool adj = false;
        auto new_range = op.getCtrl().getAsValueRange<mlir::BoolAttr>();
        auto old_range = parent.getCtrl().getAsValueRange<mlir::BoolAttr>();
        ctrl.append(new_range.begin(), new_range.end());
        ctrl.append(old_range.begin(), old_range.end());
        adj = parent.getAdjoint() ^ op.getAdjoint();
        llvm::SmallVector<mlir::Attribute> ctrls;
        for(auto c: ctrl){
            ctrls.push_back(mlir::BoolAttr::get(rewriter.getContext(), c));
        }
        auto ctrl_array = mlir::ArrayAttr::get(rewriter.getContext(), ctrls);
        rewriter.replaceOpWithNewOp<isq::ir::DecorateOp>(op.getOperation(),op.getResult().getType(), parent.getArgs(), adj, ctrl_array);
        return mlir::success();
    }
    return mlir::failure();
}

EliminateUselessDecorate::EliminateUselessDecorate(mlir::MLIRContext* ctx): mlir::OpRewritePattern<isq::ir::DecorateOp>(ctx, 1){}
mlir::LogicalResult EliminateUselessDecorate::matchAndRewrite(isq::ir::DecorateOp op,  mlir::PatternRewriter &rewriter) const{
    if(op.getCtrl().size()==0 && op.getAdjoint()==false){
        rewriter.replaceOp(op, op.getArgs());
        return mlir::success();
    }
    return mlir::failure();
    
}

AdjointHermitian::AdjointHermitian(mlir::MLIRContext* ctx): mlir::OpRewritePattern<isq::ir::DecorateOp>(ctx, 1){}
mlir::LogicalResult AdjointHermitian::matchAndRewrite(isq::ir::DecorateOp op,  mlir::PatternRewriter &rewriter) const{
    if((op.getArgs().getType().cast<GateType>().getHints() & GateTrait::Hermitian)==GateTrait::Hermitian && op.getAdjoint()){
        rewriter.startRootUpdate(op);
        op.setAdjointAttr(rewriter.getBoolAttr(false));
        rewriter.finalizeRootUpdate(op);
        return mlir::success();
    }
    return mlir::failure();
    
}


}
}
}
}
