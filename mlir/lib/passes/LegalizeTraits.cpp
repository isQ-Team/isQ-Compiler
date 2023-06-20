#include "isq/Operations.h"
#include "isq/QTypes.h"
#include <llvm/Support/Casting.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
namespace isq::ir::passes{
struct LegalizeUseGate : public mlir::OpRewritePattern<UseGateOp>{
    LegalizeUseGate(mlir::MLIRContext* ctx): mlir::OpRewritePattern<UseGateOp>(ctx, 1){
        
    }
    mlir::LogicalResult matchAndRewrite(UseGateOp op, mlir::PatternRewriter& rewriter) const override{
        auto used_op = llvm::dyn_cast_or_null<DefgateOp>(mlir::SymbolTable::lookupNearestSymbolFrom(op, op.getName()));
        assert(used_op);
        auto used_type = used_op.getTypeWhenUsed();
        if(used_type==op.getType()){
            return mlir::failure();
        }
        rewriter.updateRootInPlace(op, [&](){
            op.getResult().setType(used_type);
        });
        return mlir::success();
    }
};
struct LegalizeDecorateGate : public mlir::OpRewritePattern<DecorateOp>{
    LegalizeDecorateGate(mlir::MLIRContext* ctx): mlir::OpRewritePattern<DecorateOp>(ctx, 1){}
    mlir::LogicalResult matchAndRewrite(DecorateOp op, mlir::PatternRewriter& rewriter) const override{
        auto old_type = op.getArgs().getType().cast<GateType>();
        auto ctrls = op.getCtrl().getAsValueRange<mlir::BoolAttr>();
        auto all_one = std::all_of(ctrls.begin(), ctrls.end(), [](auto x){return x;});
        auto required_hint = DecorateOp::computePostDecorateTrait(old_type.getHints(), op.getCtrl().size(), op.getAdjoint(), all_one);
        auto found_hint = op.getType().getHints();
        if(required_hint==found_hint) return mlir::failure();
        rewriter.updateRootInPlace(op,[&](){
            op.getResult().setType(GateType::get(this->getContext(), op.getType().getSize(), required_hint));
        });
        return mlir::success();
    }
};

void addLegalizeTraitsRules(mlir::RewritePatternSet& patterns){
    patterns.add<LegalizeUseGate>(patterns.getContext());
    patterns.add<LegalizeDecorateGate>(patterns.getContext());
}

}