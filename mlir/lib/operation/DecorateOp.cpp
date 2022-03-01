#include "isq/Enums.h"
#include "isq/passes/canonicalization/MergeDecorateGates.h"
#include "mlir/IR/PatternMatch.h"
#include <isq/Operations.h>
namespace isq {
namespace ir {
mlir::LogicalResult verify(DecorateOp op) {
    auto result = op.getResult().getType().cast<GateType>();
    auto operand = op.getOperand().getType().cast<GateType>();
    auto vr = result.getHints();
    auto vo = operand.getHints();
    auto expected_vo =  static_cast<uint32_t>(vr);
    auto ctrl = op.ctrl();
    auto adjoint = op.adjoint();
    if(ctrl.size()>0){
        expected_vo &= ~static_cast<uint32_t>(GateTrait::Symmetric);
    }
    auto size = result.getSize();
    auto expected_size = operand.getSize() + ctrl.size();
    if (expected_vo != static_cast<uint32_t>(vo)) {
        op.emitOpError("decorate trait mismatch");
        return mlir::failure();
    }
    if (size != expected_size){
        op.emitOpError("decorate size mismatch");
        return mlir::failure();
    }

    return mlir::success();
}

void DecorateOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                       mlir::MLIRContext *context) {
    patterns.add<passes::canonicalize::MergeDecorate>(context);
}


} // namespace ir
} // namespace isq