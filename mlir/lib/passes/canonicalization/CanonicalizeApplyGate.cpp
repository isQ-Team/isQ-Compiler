
#include "isq/Enums.h"
#include "isq/Operations.h"
#include "isq/QTypes.h"
#include "isq/passes/canonicalization/CanonicalizeApplyGate.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Support/LogicalResult.h>
namespace isq{
namespace ir{
namespace passes{
namespace canonicalize{

NoDowngradeApply::NoDowngradeApply(mlir::MLIRContext* ctx): mlir::OpRewritePattern<isq::ir::ApplyGateOp>(ctx, 1){}
mlir::LogicalResult NoDowngradeApply::matchAndRewrite(isq::ir::ApplyGateOp op,  mlir::PatternRewriter &rewriter) const{
    if(auto parent = mlir::dyn_cast_or_null<isq::ir::DowngradeGateOp>(op.gate().getDefiningOp())){
        rewriter.updateRootInPlace(op, [&](){
            op.setOperand(0, parent->getOperand(0));
        });
        return mlir::success();
    }
    return mlir::failure();
}

CorrectSymmetryApplyOrder::CorrectSymmetryApplyOrder(mlir::MLIRContext* ctx): mlir::OpRewritePattern<isq::ir::ApplyGateOp>(ctx, 1){}
mlir::LogicalResult CorrectSymmetryApplyOrder::matchAndRewrite(isq::ir::ApplyGateOp op,  mlir::PatternRewriter &rewriter) const{
    if(op.args().size()==0) return mlir::failure();
    if((op.gate().getType().cast<GateType>().getHints() & GateTrait::Symmetric) != GateTrait::Symmetric){
        return mlir::failure();
    }
    auto prev_op = op.args()[0].getDefiningOp<ApplyGateOp>();
    if(!prev_op) return mlir::failure();
    for(auto arg: op.args()){
        if(arg.getDefiningOp()!=prev_op){
            return mlir::failure();
        }
    }
    // reconstruct operations.
    ::mlir::SmallVector<int> curr_to_index_in_prev_output(op.args().size());
    ::mlir::SmallVector<int> prev_output_to_index_in_curr(prev_op.getResults().size(), -1);
    for(auto i=0; i<op.args().size(); i++){
        auto curr_arg = op.args()[i];
        for(auto j=0; j<prev_op.getResults().size(); j++){
            if(curr_arg == prev_op.getResult(j)){
                curr_to_index_in_prev_output[i]=j;
                prev_output_to_index_in_curr[j]=i;
            }
        }
    }
    // check if already ordered.
    bool already_ordered = true;
    auto j = 0;
    for(auto i=0; i<prev_output_to_index_in_curr.size(); i++){
        auto output = prev_output_to_index_in_curr[i];
        if(output==-1) continue;
        if(output!=j){
            already_ordered=false;
            break;
        }
        j++;
    }
    if(already_ordered){
        assert(j==op.args().size());
        return mlir::failure();
    }
    auto new_apply = rewriter.clone(*op);
    j = 0;
    rewriter.startRootUpdate(new_apply);
    for(auto i=0; i<prev_output_to_index_in_curr.size(); i++){
        auto output = prev_output_to_index_in_curr[i];
        if(output==-1) continue;
        new_apply->setOperand(j+1, prev_op.getResult(i));
        prev_output_to_index_in_curr[i] = j;
        j++;
    }
    assert(j == op.args().size());
    rewriter.finalizeRootUpdate(new_apply);
    for(auto i=0; i<curr_to_index_in_prev_output.size(); i++){
        op.getResult(i).replaceAllUsesWith(new_apply->getResult(prev_output_to_index_in_curr[curr_to_index_in_prev_output[i]]));
    }
    rewriter.eraseOp(op);
    return mlir::success();
    
}

CancelUV::CancelUV(mlir::MLIRContext* ctx): mlir::OpRewritePattern<isq::ir::ApplyGateOp>(ctx, 1){}
mlir::LogicalResult CancelUV::matchAndRewrite(isq::ir::ApplyGateOp op,  mlir::PatternRewriter &rewriter) const{
    if(op.args().size()==0) return mlir::failure();
    auto prev_op = op.args()[0].getDefiningOp<ApplyGateOp>();
    if(!prev_op) return mlir::failure();
    if(op.args().size()!=prev_op.args().size()){
        return mlir::failure();
    }
    for(auto i=0; i<op.args().size(); i++){
        if(op.args()[i] != prev_op->getResult(i)){
            return mlir::failure();
        }
    }
    return tryCancel(op, prev_op, rewriter);
}

CancelUUAdj::CancelUUAdj(mlir::MLIRContext* ctx): CancelUV(ctx){}
mlir::LogicalResult CancelUUAdj::tryCancel(isq::ir::ApplyGateOp curr, isq::ir::ApplyGateOp prev, mlir::PatternRewriter& rewriter) const{
    bool erase = false;
    if(auto curr_decorate = curr.gate().getDefiningOp<DecorateOp>()){
        if(curr_decorate.adjoint() && curr_decorate.args()==prev.gate()){
            erase = true;
        }
    }else if(auto prev_decorate = prev.gate().getDefiningOp<DecorateOp>()){
        if(prev_decorate.adjoint() && prev_decorate.args()==curr.gate()){
            erase = true;
        }
    }
    if(erase){
        rewriter.replaceOp(curr, prev.args());
        return mlir::success();
    }
    return mlir::failure();
}

CancelHermitianUU::CancelHermitianUU(mlir::MLIRContext* ctx): CancelUV(ctx){}
mlir::LogicalResult CancelHermitianUU::tryCancel(isq::ir::ApplyGateOp curr, isq::ir::ApplyGateOp prev, mlir::PatternRewriter& rewriter) const{
    if(curr.gate()==prev.gate() && (curr.gate().getType().cast<GateType>().getHints() & GateTrait::Hermitian) == GateTrait::Hermitian){
        rewriter.replaceOp(curr, prev.args());
        mlir::SmallVector<mlir::Operation*> oo(prev->getUsers().begin(), prev->getUsers().end());
        for(auto user: oo){
            if(llvm::isa<mlir::AffineStoreOp>(user)){
                rewriter.eraseOp(user);
            }
        }
        rewriter.eraseOp(prev);
        return mlir::success();
    }
    return mlir::failure();
}

}
}
}
}
