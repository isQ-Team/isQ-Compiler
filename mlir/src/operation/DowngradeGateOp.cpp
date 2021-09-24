#include <isq/Operations.h>
namespace isq {
namespace ir {
mlir::LogicalResult verify(DowngradeGateOp op) {
    auto result = op.getResult().getType().cast<GateType>().getGateInfo();
    auto operand = op.getOperand().getType().cast<GateType>().getGateInfo();
    auto vr = std::get<1>(result);
    auto vo = std::get<1>(operand);
    uint32_t bigset = static_cast<uint32_t>(vo.getValue());
    uint32_t smallset = static_cast<uint32_t>(vr.getValue());
    if (smallset & (~bigset)) {
        op.emitOpError("downgraded gate contains new trait(s) compared with "
                       "original input.");
        return mlir::failure();
    }

    return mlir::success();
}
} // namespace ir
} // namespace isq