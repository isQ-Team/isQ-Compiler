#include <isq/Operations.h>
namespace isq {
namespace ir {
mlir::LogicalResult verify(DowngradeGateOp op) {
    auto result = op.getResult().getType().cast<GateType>().getGateInfo();
    auto operand = op.getOperand().getType().cast<GateType>().getGateInfo();
    auto vr = std::get<1>(result);
    auto vo = std::get<1>(operand);
    if (!std::includes(vo.begin(), vo.end(), vr.begin(), vr.end())) {
        op.emitOpError("downgraded gate contains new trait(s) compared with "
                       "original input.");
        return mlir::failure();
    }
    return mlir::success();
}
} // namespace ir
} // namespace isq