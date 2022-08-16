#ifndef _ISQ_PASSES_CANONICALIZATION_CANONICALIZEDOWNGRADE_H
#define _ISQ_PASSES_CANONICALIZATION_CANONICALIZEDOWNGRADE_H
#include "isq/Operations.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
namespace isq{
namespace ir{
namespace passes{
namespace canonicalize{
struct MergeDowngrade : public mlir::OpRewritePattern<isq::ir::DowngradeGateOp>{
    MergeDowngrade(mlir::MLIRContext* ctx);
    mlir::LogicalResult matchAndRewrite(isq::ir::DowngradeGateOp op,  mlir::PatternRewriter &rewriter) const override;
};

struct EliminateUselessDowngrade : public mlir::OpRewritePattern<isq::ir::DowngradeGateOp>{
    EliminateUselessDowngrade(mlir::MLIRContext* ctx);
    mlir::LogicalResult matchAndRewrite(isq::ir::DowngradeGateOp op,  mlir::PatternRewriter &rewriter) const override;
};


}
}
}
}
#endif