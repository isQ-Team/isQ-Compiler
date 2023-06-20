#include "isq/Operations.h"
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Pass/Pass.h>
#include "isq/QTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// rewrite pass to generate onto openqasm3.

namespace isq::ir::passes{
class RuleRemoveDealloc : public mlir::OpRewritePattern<mlir::memref::DeallocOp>{
public:
    RuleRemoveDealloc(mlir::MLIRContext* ctx): mlir::OpRewritePattern<mlir::memref::DeallocOp>(ctx, 1){}
    mlir::LogicalResult matchAndRewrite(mlir::memref::DeallocOp op,  mlir::PatternRewriter &rewriter) const override{
        auto val_type = op.getMemref().getType().dyn_cast<mlir::MemRefType>();
        if(!val_type) return mlir::failure();
        if(!val_type.getElementType().isa<QIRQubitType>()) return mlir::failure();
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

}