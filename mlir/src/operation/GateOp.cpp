#include <isq/Operations.h>
namespace isq {
namespace ir {
mlir::LogicalResult verify(GateOp op) {
    if (op.gate_type() != op.getResult().getType()) {
        op.emitOpError("operation dimension mismatch.");
        return mlir::failure();
    }
    return mlir::success();
}
} // namespace ir
} // namespace isq