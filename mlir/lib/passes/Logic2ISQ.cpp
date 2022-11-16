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
        // construct func signature.
        ::mlir::SmallVector<::mlir::Type> argtypes;
        ::mlir::SmallVector<::mlir::Type> returntypes;
        mlir::AffineExpr d0, s0;
        mlir::bindDims(op.getContext(), d0);
        mlir::bindSymbols(op.getContext(), s0);
        auto affine_map = mlir::AffineMap::get(1, 1, d0+s0);
        auto memref_1_qstate = mlir::MemRefType::get(mlir::ArrayRef<int64_t>{1},::isq::ir::QStateType::get(op.getContext()), affine_map);
        for(auto i=0; i<1; i++){
            argtypes.push_back(memref_1_qstate);
        }
        auto functype = ::mlir::FunctionType::get(op->getContext(), argtypes, returntypes);
        auto funcop = rewriter.create<mlir::FuncOp>(mlir::UnknownLoc::get(ctx), op.sym_name(), functype);
        auto entry_block = funcop.addEntryBlock();
        mlir::OpBuilder builder(entry_block, entry_block->begin());
        builder.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()));

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

class RuleReplaceLogicCall : public mlir::OpRewritePattern<logic::ir::CallOp> {
public:
    RuleReplaceLogicCall(mlir::MLIRContext *ctx): mlir::OpRewritePattern<logic::ir::CallOp>(ctx, 1) {}
    mlir::LogicalResult matchAndRewrite(logic::ir::CallOp op, mlir::PatternRewriter &rewriter) const override {
        auto ctx = op.getContext();
        rewriter.create<mlir::CallOp>(mlir::UnknownLoc::get(ctx), op.callee(), (mlir::TypeRange){}, op.operands());

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

class RuleReplaceLogicReturn : public mlir::OpRewritePattern<logic::ir::ReturnOp> {
public:
    RuleReplaceLogicReturn(mlir::MLIRContext *ctx): mlir::OpRewritePattern<logic::ir::ReturnOp>(ctx, 1) {}
    mlir::LogicalResult matchAndRewrite(logic::ir::ReturnOp op, mlir::PatternRewriter &rewriter) const override {
        auto ctx = op.getContext();
        rewriter.create<mlir::ReturnOp>(mlir::UnknownLoc::get(ctx));

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

class RuleReplaceLogicApply : public mlir::OpRewritePattern<logic::ir::ApplyGateOp> {
public:
    RuleReplaceLogicApply(mlir::MLIRContext *ctx): mlir::OpRewritePattern<logic::ir::ApplyGateOp>(ctx, 1) {}
    mlir::LogicalResult matchAndRewrite(logic::ir::ApplyGateOp op, mlir::PatternRewriter &rewriter) const override {
        auto ctx = op.getContext();
        auto qst = QStateType::get(rewriter.getContext());
        mlir::SmallVector<mlir::Type> types;
        types.push_back(qst);
        rewriter.replaceOpWithNewOp<isq::ir::ApplyGateOp, mlir::ArrayRef<mlir::Type>, ::mlir::Value, ::mlir::ValueRange>(op, types, op.gate(), op.args());
        return mlir::success();
    }
};

class RuleReplaceLogicUse : public mlir::OpRewritePattern<logic::ir::UseGateOp> {
public:
    RuleReplaceLogicUse(mlir::MLIRContext *ctx): mlir::OpRewritePattern<logic::ir::UseGateOp>(ctx, 1) {}
    mlir::LogicalResult matchAndRewrite(logic::ir::UseGateOp op, mlir::PatternRewriter &rewriter) const override {
        auto ctx = op.getContext();
        //auto gate_type = ;
        //rewriter.create<isq::ir::UseGateOp>(mlir::UnknownLoc::get(ctx), op.result(), op.name(), op.parameters());
        rewriter.replaceOpWithNewOp<isq::ir::UseGateOp, isq::ir::GateType, ::mlir::SymbolRefAttr, ::mlir::ValueRange>(op, isq::ir::GateType::get(ctx, 1, GateTrait::General), op.name(), op.parameters());
        return mlir::success();
    }
};

struct LogicToISQPass : public mlir::PassWrapper<LogicToISQPass, mlir::OperationPass<mlir::ModuleOp>> {
    void runOnOperation() override {
        mlir::ModuleOp m = this->getOperation();
        auto ctx = m->getContext();

        mlir::RewritePatternSet rps(ctx);
        rps.add<RuleReplaceLogicReturn>(ctx);
        rps.add<RuleReplaceLogicFunc>(ctx);
        rps.add<RuleReplaceLogicApply>(ctx);
        rps.add<RuleReplaceLogicUse>(ctx);
        mlir::FrozenRewritePatternSet frps(std::move(rps));
        (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);

        mlir::RewritePatternSet rps2(ctx);
        rps2.add<RuleReplaceLogicCall>(ctx);
        mlir::FrozenRewritePatternSet frps2(std::move(rps2));
        (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps2);
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