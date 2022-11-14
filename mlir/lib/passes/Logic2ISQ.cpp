#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "logic/IR.h"
#include "isq/IR.h"

namespace isq {
namespace ir {
namespace passes {
namespace {

class RuleReplaceLogicFunc : public mlir::OpRewritePattern<logic::ir::FuncOp> {
public:
    RuleReplaceLogicFunc(mlir::MLIRContext *ctx): mlir::OpRewritePattern<logic::ir::FuncOp>(ctx, 1) {}
    mlir::LogicalResult matchAndRewrite(logic::ir::FuncOp op, mlir::PatternRewriter &rewriter) const override {
        auto ctx = op.getContext();
        auto fntype = mlir::FunctionType::get(ctx, (mlir::TypeRange){::isq::ir::QStateType::get(ctx)}, (mlir::TypeRange){});
        auto funcop = rewriter.create<mlir::FuncOp>(mlir::UnknownLoc::get(ctx), op.sym_name(), fntype);
        auto entry_block = funcop.addEntryBlock();
        mlir::OpBuilder builder(entry_block, entry_block->begin());
        builder.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()));

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct LogicToISQPass : public mlir::PassWrapper<LogicToISQPass, mlir::OperationPass<mlir::ModuleOp>> {
    void runOnOperation() override {
        mlir::ModuleOp m = this->getOperation();
        auto ctx = m->getContext();

        mlir::RewritePatternSet rps(ctx);
        rps.add<RuleReplaceLogicFunc>(ctx);
        mlir::FrozenRewritePatternSet frps(std::move(rps));
        (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
    }
    mlir::StringRef getArgument() const final {
        return "logic-lower-to-isq";
    }
    mlir::StringRef getDescription() const final {
        return  "Generate iSQ gate based on logic oracle specification.";
    }
};

}

void registerLogicToISQ() {
    mlir::PassRegistration<LogicToISQPass>();
}

}
}
}