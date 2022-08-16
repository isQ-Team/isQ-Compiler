#ifndef _ISQ_PASSES_CANONICALIZATION_CANONICALIZEAPPLYGATE_H
#define _ISQ_PASSES_CANONICALIZATION_CANONICALIZEAPPLYGATE_H
#include "isq/Operations.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
namespace isq{
namespace ir{
namespace passes{
namespace canonicalize{
struct NoDowngradeApply : public mlir::OpRewritePattern<isq::ir::ApplyGateOp>{
    NoDowngradeApply(mlir::MLIRContext* ctx);
    mlir::LogicalResult matchAndRewrite(isq::ir::ApplyGateOp op,  mlir::PatternRewriter &rewriter) const override;
};

struct CorrectSymmetryApplyOrder : public mlir::OpRewritePattern<isq::ir::ApplyGateOp>{
    CorrectSymmetryApplyOrder(mlir::MLIRContext* ctx);
    mlir::LogicalResult matchAndRewrite(isq::ir::ApplyGateOp op,  mlir::PatternRewriter &rewriter) const override;
};

struct CancelUV : public mlir::OpRewritePattern<isq::ir::ApplyGateOp>{
protected:
    virtual mlir::LogicalResult tryCancel(isq::ir::ApplyGateOp curr, isq::ir::ApplyGateOp prev, mlir::PatternRewriter& rewriter) const = 0;
public:
    CancelUV(mlir::MLIRContext* ctx);
    mlir::LogicalResult matchAndRewrite(isq::ir::ApplyGateOp op,  mlir::PatternRewriter &rewriter) const override;
};

struct CancelUUAdj : public CancelUV{
    mlir::LogicalResult tryCancel(isq::ir::ApplyGateOp curr, isq::ir::ApplyGateOp prev, mlir::PatternRewriter& rewriter) const override;
    CancelUUAdj(mlir::MLIRContext* ctx);
};
struct CancelHermitianUU : public CancelUV{
    mlir::LogicalResult tryCancel(isq::ir::ApplyGateOp curr, isq::ir::ApplyGateOp prev, mlir::PatternRewriter& rewriter) const override;
    CancelHermitianUU(mlir::MLIRContext* ctx);
};
}
}
}
}
#endif