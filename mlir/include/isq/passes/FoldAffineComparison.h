#include "../IR.h"
#include <mlir/IR/Dialect.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/IntegerSet.h>
namespace isq{
namespace ir{
namespace passes{

/**
    We merge arith.cmp between indices by replacing them with affine expression.
*/
struct FoldAffineComparison : public mlir::OpRewritePattern<mlir::AffineIfOp> {
    mlir::IntegerSet eq;
    // non-equal implemented by flipping if and else block.
    mlir::IntegerSet slt;
    mlir::IntegerSet sle;
    mlir::IntegerSet sgt;
    mlir::IntegerSet sge;
    FoldAffineComparison(mlir::MLIRContext *context);
    mlir::LogicalResult match(mlir::AffineIfOp op) const override;
    void rewrite(mlir::AffineIfOp op, mlir::PatternRewriter& rewriter) const override;
};
}
}
}
