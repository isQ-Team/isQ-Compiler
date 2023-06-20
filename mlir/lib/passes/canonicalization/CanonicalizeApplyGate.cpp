
#include "isq/Enums.h"
#include "isq/Operations.h"
#include "isq/QTypes.h"
#include "isq/passes/Passes.h"
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
    if(auto parent = mlir::dyn_cast_or_null<isq::ir::DowngradeGateOp>(op.getGate().getDefiningOp())){
        rewriter.updateRootInPlace(op, [&](){
            op.setOperand(0, parent->getOperand(0));
        });
        return mlir::success();
    }
    return mlir::failure();
}

CorrectSymmetryApplyOrder::CorrectSymmetryApplyOrder(mlir::MLIRContext* ctx): mlir::OpRewritePattern<isq::ir::ApplyGateOp>(ctx, 1){}
mlir::LogicalResult CorrectSymmetryApplyOrder::matchAndRewrite(isq::ir::ApplyGateOp op,  mlir::PatternRewriter &rewriter) const{
    if(op.getArgs().size()==0) return mlir::failure();
    if((op.getGate().getType().cast<GateType>().getHints() & GateTrait::Symmetric) != GateTrait::Symmetric){
        return mlir::failure();
    }
    auto prev_op = op.getArgs()[0].getDefiningOp<ApplyGateOp>();
    if(!prev_op) return mlir::failure();
    for(auto arg: op.getArgs()){
        if(arg.getDefiningOp()!=prev_op){
            return mlir::failure();
        }
    }
    // reconstruct operations.
    ::mlir::SmallVector<int> curr_to_index_in_prev_output(op.getArgs().size());
    ::mlir::SmallVector<int> prev_output_to_index_in_curr(prev_op.getResults().size(), -1);
    for(auto i=0; i<op.getArgs().size(); i++){
        auto curr_arg = op.getArgs()[i];
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
        assert(j==op.getArgs().size());
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
    assert(j == op.getArgs().size());
    rewriter.finalizeRootUpdate(new_apply);
    for(auto i=0; i<curr_to_index_in_prev_output.size(); i++){
        op.getResult(i).replaceAllUsesWith(new_apply->getResult(prev_output_to_index_in_curr[curr_to_index_in_prev_output[i]]));
    }
    rewriter.eraseOp(op);
    return mlir::success();
    
}

CancelUV::CancelUV(mlir::MLIRContext* ctx): mlir::OpRewritePattern<isq::ir::ApplyGateOp>(ctx, 1){}
mlir::LogicalResult CancelUV::matchAndRewrite(isq::ir::ApplyGateOp op,  mlir::PatternRewriter &rewriter) const{
    if(op.getArgs().size()==0) return mlir::failure();
    auto prev_op = op.getArgs()[0].getDefiningOp<ApplyGateOp>();
    if(!prev_op) return mlir::failure();
    if(op.getArgs().size()!=prev_op.getArgs().size()){
        return mlir::failure();
    }
    for(auto i=0; i<op.getArgs().size(); i++){
        if(op.getArgs()[i] != prev_op->getResult(i)){
            return mlir::failure();
        }
    }
    return tryCancel(op, prev_op, rewriter);
}

CancelUUAdj::CancelUUAdj(mlir::MLIRContext* ctx): CancelUV(ctx){}
mlir::LogicalResult CancelUUAdj::tryCancel(isq::ir::ApplyGateOp curr, isq::ir::ApplyGateOp prev, mlir::PatternRewriter& rewriter) const{
    bool erase = false;
    if(auto curr_decorate = curr.getGate().getDefiningOp<DecorateOp>()){
        if(curr_decorate.getAdjoint() && curr_decorate.getArgs()==prev.getGate()){
            erase = true;
        }
    }else if(auto prev_decorate = prev.getGate().getDefiningOp<DecorateOp>()){
        if(prev_decorate.getAdjoint() && prev_decorate.getArgs()==curr.getGate()){
            erase = true;
        }
    }
    if(erase){
        rewriter.replaceOp(curr, prev.getArgs());
        return mlir::success();
    }
    return mlir::failure();
}

CancelHermitianUU::CancelHermitianUU(mlir::MLIRContext* ctx): CancelUV(ctx){}
mlir::LogicalResult CancelHermitianUU::tryCancel(isq::ir::ApplyGateOp curr, isq::ir::ApplyGateOp prev, mlir::PatternRewriter& rewriter) const{
    if(curr.getGate()==prev.getGate() && (curr.getGate().getType().cast<GateType>().getHints() & GateTrait::Hermitian) == GateTrait::Hermitian){
        rewriter.replaceOp(curr, prev.getArgs());
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

CancelRemoteCZ::CancelRemoteCZ(mlir::MLIRContext* ctx): mlir::OpRewritePattern<isq::ir::ApplyGateOp>(ctx){}


// 0 for CZ, 1 for other diagonal, 2 for nothing.
static int isCZGate(ApplyGateOp apply){
    int fallback = 2;
    if((apply.getGate().getType().cast<GateType>().getHints() & GateTrait::Diagonal) == GateTrait::Diagonal){
        fallback = 1;
    }
    auto usegate = llvm::dyn_cast_or_null<UseGateOp>(apply.getGate().getDefiningOp());
    if(!usegate) return fallback;
    auto defgate = llvm::dyn_cast_or_null<DefgateOp>(mlir::SymbolTable::lookupNearestSymbolFrom(usegate, usegate.getName()));
    if(!defgate) return fallback;
    if(isFamousGate(defgate, "cz")){
        return 0;
    }
    return fallback;
}
// 0 for CX, 1 for sq antidiagonal, 2 for nothing.
static int isCXorADGate(ApplyGateOp apply){
    int fallback = 2;
    if((apply.getGate().getType().cast<GateType>().getHints() & GateTrait::Antidiagonal) == GateTrait::Antidiagonal){
        fallback = 1;
    }
    auto usegate = llvm::dyn_cast_or_null<UseGateOp>(apply.getGate().getDefiningOp());
    if(!usegate) return fallback;
    auto defgate = llvm::dyn_cast_or_null<DefgateOp>(mlir::SymbolTable::lookupNearestSymbolFrom(usegate, usegate.getName()));
    if(!defgate) return fallback;
    if(isFamousGate(defgate, "cnot")){
        return 0;
    }
    return fallback;
}

static mlir::Value getCorrespondingResult(mlir::Value value, isq::ir::ApplyGateOp apply){
    for(auto i=0; i<apply.getArgs().size(); i++){
        if(value==apply.getArgs()[i]){
            return apply.getResult(i);
        }
    }
    assert(0 && "not operand!");
}
static mlir::Value getCorrespondingArg(mlir::Value value, isq::ir::ApplyGateOp apply){
    for(auto i=0; i<apply.getNumResults(); i++){
        if(value==apply->getResult(i)){
            return apply.getArgs()[i];
        }
    }
    assert(0 && "not result!");
}
mlir::LogicalResult CancelRemoteCZ::matchAndRewrite(isq::ir::ApplyGateOp op, mlir::PatternRewriter &rewriter) const {
    const int IS_CZ = 0;
    const int IS_DIAG = 1;
    const int NEITHER = 2;
    auto is_cz = isCZGate(op);
    if(is_cz!=IS_CZ) return mlir::failure();
    mlir::Value fst = op.getArgs()[0];
    mlir::SmallPtrSet<mlir::Operation*, 16> ops;
    while(true){
        auto fst_def = mlir::dyn_cast_or_null<ApplyGateOp>(fst.getDefiningOp());
        if(!fst_def) break;
        auto is_cz = isCZGate(fst_def);
        if(is_cz==NEITHER) break;
        if(is_cz==IS_CZ){
            ops.insert(fst_def);
        }
        fst = getCorrespondingArg(fst, fst_def);
    }
    mlir::Value snd = op.getArgs()[1];
    while(true){
        auto snd_def = mlir::dyn_cast_or_null<ApplyGateOp>(snd.getDefiningOp());
        if(!snd_def) break;
        auto is_cz = isCZGate(snd_def);
        if(is_cz==NEITHER) break;
        if(is_cz==IS_CZ){
            if(ops.contains(snd_def)){
                // snd_def has an operand of both op's operands.
                // erase both.
                rewriter.replaceOp(op, op.getArgs());
                rewriter.replaceOp(snd_def, snd_def.getArgs());
                return mlir::success();
            }
        }
        snd = getCorrespondingArg(snd, snd_def);
    }
    return mlir::failure();
}

CancelRemoteCX::CancelRemoteCX(mlir::MLIRContext* ctx): mlir::OpRewritePattern<isq::ir::ApplyGateOp>(ctx){

}

mlir::LogicalResult CancelRemoteCX::matchAndRewrite(isq::ir::ApplyGateOp op, mlir::PatternRewriter &rewriter) const {
    bool need_extra_x = false;
    const int IS_CX = 0;
    const int IS_AD = 1;
    const int NEITHER = 2;
    auto is_cx = isCXorADGate(op);
    if(is_cx!=IS_CX) return mlir::failure();
    mlir::Value controller = op.getArgs()[0];
    mlir::SmallPtrSet<mlir::Operation*, 16> ops;
    while(true){
        auto fst_def = mlir::dyn_cast_or_null<ApplyGateOp>(controller.getDefiningOp());
        if(!fst_def) break;
        auto is_cx = isCXorADGate(fst_def);
        if(is_cx==NEITHER) break;
        if(is_cx==IS_CX){
            if(fst_def->getResult(0)==controller){
                ops.insert(fst_def);
            }else{
                break;
            }
            
        }
        if(is_cx==IS_AD){
            need_extra_x = !need_extra_x;
        }
        controller = getCorrespondingArg(controller, fst_def);
    }
    mlir::Value snd = op.getArgs()[1];
    while(true){
        auto snd_def = mlir::dyn_cast_or_null<ApplyGateOp>(snd.getDefiningOp());
        if(!snd_def) break;
        auto is_cx = isCXorADGate(snd_def);
        if(is_cx==NEITHER || is_cx==IS_AD) break;
        if(is_cx==IS_CX){
            if(ops.contains(snd_def)){
                // snd_def has an operand of both op's operands, and sharing the controller bit.
                // first, insert the X gate.
                if(need_extra_x){
                    rewriter.setInsertionPoint(snd_def);
                    mlir::Value controllee = snd_def.getArgs()[1];
                    emitBuiltinGate(rewriter, "x", {&controllee});
                    rewriter.startRootUpdate(snd_def);
                    snd_def.getArgsMutable()[1] = controllee;
                    rewriter.finalizeRootUpdate(snd_def);
                }
                
                // erase both.
                rewriter.replaceOp(op, op.getArgs());
                rewriter.replaceOp(snd_def, snd_def.getArgs());
                return mlir::success();
            }
        }
        snd = getCorrespondingArg(snd, snd_def);
    }
    return mlir::failure();
}

}
}
}
}
