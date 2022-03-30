#include "isq/GateDefTypes.h"
#include "isq/Operations.h"
#include "isq/QSynthesis.h"
#include "isq/passes/Passes.h"
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#define EPS (1e-6)
namespace isq::ir::passes{
struct RemoveTrivialConstantU3 : public mlir::OpRewritePattern<ApplyGateOp>{
    RemoveTrivialConstantU3(mlir::MLIRContext* ctx): mlir::OpRewritePattern<ApplyGateOp>(ctx, 1){}
    mlir::LogicalResult matchAndRewrite(ApplyGateOp op, mlir::PatternRewriter& rewriter) const override{
        auto ctx = op->getContext();
        auto decorate_op = llvm::dyn_cast<DecorateOp>(op.gate().getDefiningOp());
        UseGateOp usegate_op;
        if(decorate_op){
            usegate_op = llvm::dyn_cast<UseGateOp>(decorate_op.args().getDefiningOp());
        }else{
            usegate_op = llvm::dyn_cast<UseGateOp>(op.gate().getDefiningOp());
        } 
        if(!usegate_op) return mlir::failure();
        auto gatedef = llvm::dyn_cast_or_null<DefgateOp>(mlir::SymbolTable::lookupNearestSymbolFrom(usegate_op, usegate_op.name()));
        if(!gatedef) return mlir::failure();
        if(!isFamousGate(gatedef, "U3")) return mlir::failure();
        mlir::SmallVector<mlir::Value> operands;
        for(auto v: op.args()){
            operands.push_back(v);
        }
        auto theta = usegate_op.parameters()[0];
        auto phi = usegate_op.parameters()[1];
        auto lam = usegate_op.parameters()[2];
        double t, p, l;
        if(auto cop = mlir::dyn_cast_or_null<mlir::arith::ConstantFloatOp>(theta.getDefiningOp())){
            t=cop.value().convertToDouble();
        }else{
            return mlir::failure();
        }
        if(auto cop = mlir::dyn_cast_or_null<mlir::arith::ConstantFloatOp>(phi.getDefiningOp())){
            p=cop.value().convertToDouble();
        }else{
            return mlir::failure();
        }
        if(auto cop = mlir::dyn_cast_or_null<mlir::arith::ConstantFloatOp>(lam.getDefiningOp())){
            l=cop.value().convertToDouble();
        }else{
            return mlir::failure();
        }
        if(abs(t)>EPS || abs(p)>EPS || abs(l)>EPS){
            return mlir::failure();
        }
        // otherwise
        rewriter.replaceOp(op, op.args());
        return mlir::success();
    }
};

struct RemoveTrivialKnownGate : public mlir::OpRewritePattern<ApplyGateOp>{
    RemoveTrivialKnownGate(mlir::MLIRContext* ctx): mlir::OpRewritePattern<ApplyGateOp>(ctx, 1){}
    mlir::LogicalResult matchAndRewrite(ApplyGateOp op, mlir::PatternRewriter& rewriter) const override{
        auto ctx = op->getContext();
        auto decorate_op = llvm::dyn_cast<DecorateOp>(op.gate().getDefiningOp());
        if(!decorate_op) return mlir::failure();
        auto usegate_op = llvm::dyn_cast<UseGateOp>(decorate_op.args().getDefiningOp());
        if(!usegate_op) return mlir::failure();
        auto defgate = llvm::dyn_cast_or_null<DefgateOp>(mlir::SymbolTable::lookupNearestSymbolFrom(usegate_op, usegate_op.name()));
        if(!defgate) return mlir::failure();
        auto id=0;
        for(auto def: defgate.definition()->getAsRange<GateDefinition>()){
            auto d = AllGateDefs::parseGateDefinition(defgate, id, defgate.type(), def);
            if(d==std::nullopt) return mlir::failure();
            if(auto mat = llvm::dyn_cast_or_null<MatrixDefinition>(&**d)){
                auto m = mat->getMatrix();
                // TODO: what about converting nearly-e^{i\theta} SQ gates into gphase?
                for(auto i=0; i<m.size(); i++){
                    for(auto j=0; j<m.size(); j++){
                        auto expected = i==j?1.0:0.0;
                        if(std::norm(m[i][j]-expected)>EPS){
                            return mlir::failure();
                        }
                    }
                }
                // otherwise
                rewriter.replaceOp(op, op.args());
                return mlir::success();
            }
        }
        return mlir::failure();

    }
};
struct RemoveTrivialSQGatesPass : public mlir::PassWrapper<RemoveTrivialSQGatesPass, mlir::OperationPass<mlir::ModuleOp>>{
    void runOnOperation() override {
        mlir::ModuleOp m = this->getOperation();
        auto ctx = m->getContext();
        do{
            mlir::RewritePatternSet rps(ctx);
            rps.add<RemoveTrivialConstantU3>(ctx);
            rps.add<RemoveTrivialKnownGate>(ctx);
            mlir::FrozenRewritePatternSet frps(std::move(rps));
            (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
        }while(0);
        
    }
    mlir::StringRef getArgument() const final {
        return "isq-remove-trivial-sq-gates";
    }
    mlir::StringRef getDescription() const final {
        return  "Remove trivial single-qubit gates.";
    }
};

void registerRemoveTrivialSQGates(){
    mlir::PassRegistration<RemoveTrivialSQGatesPass>();
}


}
