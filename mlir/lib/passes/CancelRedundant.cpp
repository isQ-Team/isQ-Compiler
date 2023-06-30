#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "isq/Operations.h"

namespace isq::ir::passes{
struct CancelAssert : public mlir::OpRewritePattern<isq::ir::AssertOp>{
    CancelAssert(mlir::MLIRContext* ctx): mlir::OpRewritePattern<isq::ir::AssertOp>(ctx, 1){}
    mlir::LogicalResult matchAndRewrite(isq::ir::AssertOp op, mlir::PatternRewriter& rewriter) const override{
        rewriter.eraseOp(op);
        return mlir::success();
    }
};
struct CancelGPhase : public mlir::OpRewritePattern<isq::ir::AccumulateGPhase>{
    CancelGPhase(mlir::MLIRContext* ctx): mlir::OpRewritePattern<isq::ir::AccumulateGPhase>(ctx, 1){}
    mlir::LogicalResult matchAndRewrite(isq::ir::AccumulateGPhase op, mlir::PatternRewriter& rewriter) const override{
        rewriter.eraseOp(op);
        return mlir::success();
    }
};
struct CancelDealloc : public mlir::OpRewritePattern<mlir::memref::DeallocOp>{
    CancelDealloc(mlir::MLIRContext* ctx): mlir::OpRewritePattern<mlir::memref::DeallocOp>(ctx, 1){}
    mlir::LogicalResult matchAndRewrite(mlir::memref::DeallocOp op, mlir::PatternRewriter& rewriter) const override{
        rewriter.eraseOp(op);
        return mlir::success();
    }
};
struct CancelAlloc : public mlir::OpRewritePattern<mlir::memref::AllocOp>{
    CancelAlloc(mlir::MLIRContext* ctx): mlir::OpRewritePattern<mlir::memref::AllocOp>(ctx, 1){}
    mlir::LogicalResult matchAndRewrite(mlir::memref::AllocOp op, mlir::PatternRewriter& rewriter) const override{
        auto result = op.getResult();
        if (result.use_empty()){
            rewriter.eraseOp(op);
        }

        auto type = op.getResult().getType().dyn_cast<mlir::MemRefType>();
        if (type.getShape()[0] == 1 && type.getElementType().isa<mlir::IntegerType>()){
            auto ty = type.getElementType().dyn_cast<mlir::IntegerType>();
            if (ty && ty.getWidth() == 1){
                if (result.hasOneUse()){
                    mlir::Operation* first_use;
                    for (mlir::Operation *userOp : result.getUsers()){
                        first_use = userOp;
                    }
                    //os << "first user: " << first_use->getName().getStringRef().str() << endl;
                    if (auto first_store = llvm::dyn_cast<mlir::AffineStoreOp>(first_use)){
                        rewriter.eraseOp(first_store);
                    }
                }
            }
        }
        
        return mlir::success();
    }
};
struct RedundantPass : public mlir::PassWrapper<RedundantPass, mlir::OperationPass<mlir::ModuleOp>>{
    void runOnOperation() override {
        mlir::ModuleOp m = this->getOperation();
        auto ctx = m->getContext();
        do {
            mlir::RewritePatternSet rps(ctx);
            rps.add<CancelAssert>(ctx);
            rps.add<CancelDealloc>(ctx);
            rps.add<CancelGPhase>(ctx);
            mlir::FrozenRewritePatternSet frps(std::move(rps));
            (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
        } while(0);
        do {
            mlir::RewritePatternSet rps(ctx);
            rps.add<CancelAlloc>(ctx);
            mlir::FrozenRewritePatternSet frps(std::move(rps));
            (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
        } while(0);
        
    }
    mlir::StringRef getArgument() const final {
        return "isq-cancel-redundant";
    }
    mlir::StringRef getDescription() const final {
        return  "cancel all redundant operation before generated to OpenQASM";
    }
};

void registerRedundant(){
    mlir::PassRegistration<RedundantPass>();
}


}
