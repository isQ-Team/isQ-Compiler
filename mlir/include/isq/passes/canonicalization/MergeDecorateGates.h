#ifndef _ISQ_PASSES_CANONICALIZATION_MERGEDECORATEGATES_H
#define _ISQ_PASSES_CANONICALIZATION_MERGEDECORATEGATES_H
#include "isq/Operations.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
namespace isq{
namespace ir{
namespace passes{
namespace canonicalize{
struct MergeDecorate : public mlir::OpRewritePattern<isq::ir::DecorateOp>{
    MergeDecorate(mlir::MLIRContext* ctx);
    mlir::LogicalResult matchAndRewrite(isq::ir::DecorateOp op,  mlir::PatternRewriter &rewriter) const override;
};
}
}
}
}
#endif