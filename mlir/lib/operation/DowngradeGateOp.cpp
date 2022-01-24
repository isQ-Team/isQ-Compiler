#include <isq/Operations.h>
namespace isq {
namespace ir {
mlir::LogicalResult verify(DowngradeGateOp op) {
    auto result = op.getResult().getType().cast<GateType>();
    auto operand = op.getOperand().getType().cast<GateType>();
    auto vr = result.getHints();
    auto vo = operand.getHints();
    uint32_t bigset = static_cast<uint32_t>(vo);
    uint32_t smallset = static_cast<uint32_t>(vr);
    if (smallset & (~bigset)) {
        op.emitOpError("downgraded gate contains new trait(s) compared with "
                       "original input.");
        return mlir::failure();
    }

    return mlir::success();
}

} // namespace ir
} // namespace isq