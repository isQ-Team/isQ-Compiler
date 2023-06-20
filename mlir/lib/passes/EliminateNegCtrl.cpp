#include "isq/Operations.h"
#include "isq/passes/Passes.h"
#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
namespace isq::ir::passes{
struct EliminateNegCtrlRule : public mlir::OpRewritePattern<ApplyGateOp>{
    EliminateNegCtrlRule(mlir::MLIRContext* ctx): mlir::OpRewritePattern<ApplyGateOp>(ctx, 1){}
    mlir::LogicalResult matchAndRewrite(ApplyGateOp op, mlir::PatternRewriter& rewriter) const override{
        auto ctx = op->getContext();
        auto decorate_op = llvm::dyn_cast<DecorateOp>(op.getGate().getDefiningOp());
        if(!decorate_op) return mlir::failure();
        mlir::SmallVector<bool> flags;
        for(auto flag: decorate_op.getCtrl().getAsValueRange<mlir::BoolAttr>()){
            flags.push_back(flag);
        }
        if(llvm::all_of(flags, [](bool x){return x;})){
            return mlir::failure();
        }
        
        mlir::PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(op);
        mlir::SmallVector<mlir::Value> new_operands;
        // create x before gate.
        for(auto i=0; i<flags.size(); i++){
            auto flag = flags[i];
            new_operands.push_back(op.getArgs()[i]);
            if(flag) continue;
            
            emitBuiltinGate(rewriter, "X",{&new_operands[i]});
        }
        for(auto i=flags.size(); i<op.getArgs().size(); i++){
            new_operands.push_back(op.getArgs()[i]);
        }
        // create x after gate.
        rewriter.setInsertionPointAfter(op);
        for(auto i=0; i<flags.size(); i++){
            auto flag = flags[i];
            if(!flag){
                mlir::Value u = op.getResult(i);
                mlir::Value v = u;
                emitBuiltinGate(rewriter, "X", {&v});
                u.replaceUsesWithIf(v, [&](mlir::OpOperand& operand){
                    return operand.getOwner()!=v.getDefiningOp();
                });
            }
        }

        // now update ctrl.
        mlir::SmallVector<mlir::Attribute> flags_attr;
        for(auto i=0; i<flags.size(); i++){
            flags_attr.push_back(mlir::BoolAttr::get(ctx, true));
        }
        auto new_ctrl = mlir::ArrayAttr::get(ctx, flags_attr);
        rewriter.setInsertionPoint(op);
        auto new_decorate = llvm::cast<DecorateOp>(rewriter.clone(*decorate_op));
        rewriter.updateRootInPlace(new_decorate, [&](){
            new_decorate.setCtrlAttr(new_ctrl);
        });
        rewriter.updateRootInPlace(op, [&](){
            op.getArgsMutable().assign(new_operands);
            op.getGateMutable().assign(new_decorate);
        });
        return mlir::success();
    }

};

struct EliminateNegCtrlPass : public mlir::PassWrapper<EliminateNegCtrlPass, mlir::OperationPass<mlir::ModuleOp>>{
    void runOnOperation() override {
        mlir::ModuleOp m = this->getOperation();
        auto ctx = m->getContext();
        do{
            mlir::RewritePatternSet rps(ctx);
            rps.add<EliminateNegCtrlRule>(ctx);
            mlir::FrozenRewritePatternSet frps(std::move(rps));
            (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
        }while(0);
    }
    mlir::StringRef getArgument() const final {
        return "isq-eliminate-neg-ctrl";
    }
    mlir::StringRef getDescription() const final {
        return  "Change neg ctrl into X.";
    }
};

void registerEliminateNegCtrl(){
    mlir::PassRegistration<EliminateNegCtrlPass>();
}



}
