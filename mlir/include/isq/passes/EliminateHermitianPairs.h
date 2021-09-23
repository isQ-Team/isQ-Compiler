#include "../IR.h"
#include <mlir/IR/Dialect.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Types.h>
namespace isq {
namespace ir {
struct EliminateHermitianPairs : public mlir::OpRewritePattern<ApplyOp> {
    EliminateHermitianPairs(mlir::MLIRContext *context)
        : OpRewritePattern<ApplyOp>(context, /*benefit=*/1) {}

    mlir::LogicalResult matchAndRewrite(ApplyOp applyop,
                                        mlir::PatternRewriter &rewriter) const;
};
} // namespace ir
} // namespace isq
