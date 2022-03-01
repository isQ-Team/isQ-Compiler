
#include "isq/Operations.h"
#include "isq/passes/canonicalization/MergeDecorateGates.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
namespace isq{
namespace ir{
namespace passes{
namespace canonicalize{

MergeDecorate::MergeDecorate(mlir::MLIRContext* ctx): mlir::OpRewritePattern<isq::ir::DecorateOp>(ctx, 1){}
mlir::LogicalResult MergeDecorate::matchAndRewrite(isq::ir::DecorateOp op,  mlir::PatternRewriter &rewriter) const{
    if(auto parent = mlir::dyn_cast_or_null<isq::ir::DecorateOp>(op.args().getDefiningOp())){
        llvm::SmallVector<bool> ctrl;
        bool adj = false;
        auto new_range = op.ctrl().getAsValueRange<mlir::BoolAttr>();
        auto old_range = parent.ctrl().getAsValueRange<mlir::BoolAttr>();
        ctrl.append(new_range.begin(), new_range.end());
        ctrl.append(old_range.begin(), old_range.end());
        adj = parent.adjoint() ^ op.adjoint();
        llvm::SmallVector<mlir::Attribute> ctrls;
        for(auto c: ctrl){
            ctrls.push_back(mlir::BoolAttr::get(rewriter.getContext(), c));
        }
        auto ctrl_array = mlir::ArrayAttr::get(rewriter.getContext(), ctrls);
        rewriter.replaceOpWithNewOp<isq::ir::DecorateOp>(op.getOperation(),op.getResult().getType(), parent.args(), adj, ctrl_array);
        return mlir::success();
    }
    return mlir::failure();
}

}


}
}
}
