#include <isq/Operations.h>
namespace isq {
namespace ir {
mlir::LogicalResult verify(DeclareOp op) {
    if (op.op_type() != op.getResult().getType()) {
        op.emitOpError("operation signature mismatch.");
        return mlir::failure();
    }
    return mlir::success();
}
} // namespace ir
} // namespace isq