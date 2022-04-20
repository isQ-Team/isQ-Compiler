#include "isq/Operations.h"
#include "isq/passes/Passes.h"
#include <llvm/Support/Casting.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace isq::ir::passes{

struct CX2HCZH : mlir::OpRewritePattern<ApplyGateOp>{
public:
    CX2HCZH(mlir::MLIRContext* ctx): mlir::OpRewritePattern<ApplyGateOp>(ctx, 1){}
    mlir::LogicalResult matchAndRewrite(ApplyGateOp apply, mlir::PatternRewriter& rewriter) const override{
        auto usegate = llvm::dyn_cast_or_null<UseGateOp>(apply.gate().getDefiningOp());
        if(!usegate) return mlir::failure();
        auto defgate = llvm::dyn_cast_or_null<DefgateOp>(mlir::SymbolTable::lookupNearestSymbolFrom(usegate, usegate.name()));
        if(!defgate) return mlir::failure();
        auto ctx = rewriter.getContext();
        if(isFamousGate(defgate, "cnot")){
            mlir::Value v1 = apply->getOperand(1);
            mlir::Value v2 = apply->getOperand(2);
            emitBuiltinGate(rewriter, "H", {&v2});
            emitBuiltinGate(rewriter, "CZ", {&v1, &v2});
            emitBuiltinGate(rewriter, "H", {&v2});
            rewriter.replaceOp(apply, mlir::ArrayRef<mlir::Value>{v1, v2});
            return mlir::success();
        }
        return mlir::failure();
    }
};

struct U3ToZYZ : mlir::OpRewritePattern<ApplyGateOp>{
public:
    U3ToZYZ(mlir::MLIRContext* ctx): mlir::OpRewritePattern<ApplyGateOp>(ctx, 1){}
    mlir::LogicalResult matchAndRewrite(ApplyGateOp apply, mlir::PatternRewriter& rewriter) const override{
        auto usegate = llvm::dyn_cast_or_null<UseGateOp>(apply.gate().getDefiningOp());
        if(!usegate) return mlir::failure();
        auto defgate = llvm::dyn_cast_or_null<DefgateOp>(mlir::SymbolTable::lookupNearestSymbolFrom(usegate, usegate.name()));
        if(!defgate) return mlir::failure();
        auto ctx = rewriter.getContext();
        if(isFamousGate(defgate, "u3")){
            auto lambda = usegate->getOperand(0);
            auto theta = usegate->getOperand(1);
            auto phi = usegate->getOperand(2);
            mlir::Value v1 = apply->getOperand(1);
            emitBuiltinGate(rewriter, "Rz", {&v1}, {lambda});
            emitBuiltinGate(rewriter, "Ry", {&v1}, {theta});
            emitBuiltinGate(rewriter, "Rz", {&v1}, {phi});
            rewriter.replaceOp(apply, mlir::ArrayRef<mlir::Value>{v1});
            return mlir::success();
        }
        return mlir::failure();
    }
};

struct InvTSRecog : mlir::OpRewritePattern<ApplyGateOp>{
public:
    InvTSRecog(mlir::MLIRContext* ctx): mlir::OpRewritePattern<ApplyGateOp>(ctx, 1){}
    mlir::LogicalResult matchAndRewrite(ApplyGateOp apply, mlir::PatternRewriter& rewriter) const override{
        auto decorate = llvm::dyn_cast_or_null<DecorateOp>(apply.gate().getDefiningOp());
        if(!decorate) return mlir::failure();
        if(!decorate.adjoint()) return mlir::failure();
        if(decorate.ctrl().size()) return mlir::failure();
        auto usegate = llvm::dyn_cast_or_null<UseGateOp>(decorate.args().getDefiningOp());
        if(!usegate) return mlir::failure();
        auto defgate = llvm::dyn_cast_or_null<DefgateOp>(mlir::SymbolTable::lookupNearestSymbolFrom(usegate, usegate.name()));
        if(!defgate) return mlir::failure();
        auto ctx = rewriter.getContext();
        if(isFamousGate(defgate, "T")){
            mlir::Value v1 = apply->getOperand(1);
            emitBuiltinGate(rewriter, "Tinv", {&v1});
            rewriter.replaceOp(apply, mlir::ArrayRef<mlir::Value>{v1});
            return mlir::success();
        }else if(isFamousGate(defgate, "S")){
            mlir::Value v1 = apply->getOperand(1);
            emitBuiltinGate(rewriter, "Sinv", {&v1});
            rewriter.replaceOp(apply, mlir::ArrayRef<mlir::Value>{v1});
            return mlir::success();
        }
        return mlir::failure();
    }
};

class TargetQCISSetPass : public mlir::PassWrapper<TargetQCISSetPass, mlir::OperationPass<mlir::ModuleOp>>{
    void runOnOperation() override{
        mlir::ModuleOp m = this->getOperation();
        auto ctx = m->getContext();
        do{
            mlir::RewritePatternSet rps(ctx);
            rps.add<CX2HCZH>(ctx);
            //rps.add<U3ToZYZ>(ctx);
            rps.add<InvTSRecog>(ctx);
            mlir::FrozenRewritePatternSet frps(std::move(rps));
            (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
        }while(0);
    }
    mlir::StringRef getArgument() const final {
        return "isq-target-qcis";
    }
    mlir::StringRef getDescription() const final {
        return  "Convert to QCIS instruction set. Convert U3 to rotation and CNOT to H-CZ-H.";
    }
};


void registerTargetQCISSet(){
    mlir::PassRegistration<TargetQCISSetPass>();
}


}
