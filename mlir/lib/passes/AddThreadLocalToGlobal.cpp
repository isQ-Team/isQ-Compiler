#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace isq::ir::passes{
struct AddThreadLocalToGlobal : public mlir::OpRewritePattern<mlir::LLVM::GlobalOp>{
    AddThreadLocalToGlobal(mlir::MLIRContext* ctx): mlir::OpRewritePattern<mlir::LLVM::GlobalOp>(ctx, 1){}
    mlir::LogicalResult matchAndRewrite(mlir::LLVM::GlobalOp op, mlir::PatternRewriter& rewriter) const override{
        auto ctx = op->getContext();
        if (!op.getConstant()) {
            op.setThreadLocal_Attr(mlir::UnitAttr::get(ctx));
        }
        return mlir::success();
    }
};

/*
* Remove redundant StoreOp on Qubit.
*
* Example:
*    %q = llvm.load %addr : !llvm.ptr<ptr<struct<"Qubit", opaque>>>
*    llvm.call @__quantum__qis__x__body(%q) : (!llvm.ptr<struct<"Qubit", opaque>>) -> ()
*    llvm.store %q, %addr : !llvm.ptr<ptr<struct<"Qubit", opaque>>>
*
* This StoreOp would cause llc running extra long time.
*/
class RuleEliminateRefStore : public mlir::OpRewritePattern<mlir::LLVM::StoreOp> {
public:
    RuleEliminateRefStore(mlir::MLIRContext* ctx): mlir::OpRewritePattern<mlir::LLVM::StoreOp>(ctx, 1) {
    }
    mlir::LogicalResult matchAndRewrite(mlir::LLVM::StoreOp op,  mlir::PatternRewriter &rewriter) const override{
        mlir::Value val = op.getValue();
        mlir::LLVM::LLVMPointerType ptr = val.getType().dyn_cast_or_null<mlir::LLVM::LLVMPointerType>();
        if (!ptr) {
            return mlir::failure();
        }
        mlir::LLVM::LLVMStructType str = ptr.getElementType().dyn_cast_or_null<mlir::LLVM::LLVMStructType>();
        if (!str || !str.getName().equals(llvm::StringRef("Qubit"))) {
            return mlir::failure();
        }
        mlir::LLVM::LoadOp load = mlir::dyn_cast_or_null<mlir::LLVM::LoadOp>(val.getDefiningOp());
        if (!load || load.getAddr() != op.getAddr()) {
            return mlir::failure();
        }
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct GlobalThreadLocalPass : public mlir::PassWrapper<GlobalThreadLocalPass, mlir::OperationPass<mlir::ModuleOp>>{
    void runOnOperation() override {
        mlir::ModuleOp m = this->getOperation();
        auto ctx = m->getContext();
        do {
            mlir::RewritePatternSet rps(ctx);
            rps.add<RuleEliminateRefStore>(ctx);
            rps.add<AddThreadLocalToGlobal>(ctx);
            mlir::FrozenRewritePatternSet frps(std::move(rps));
            (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
        } while(0);
        
    }
    mlir::StringRef getArgument() const final {
        return "global-thread-local";
    }
    mlir::StringRef getDescription() const final {
        return  "Add a thread_local attribute to llvm.mlir.global";
    }
};

void registerGlobalThreadLocal(){
    mlir::PassRegistration<GlobalThreadLocalPass>();
}


}
