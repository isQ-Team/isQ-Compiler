#include <isq/Operations.h>
#include <mlir/IR/BuiltinTypes.h>
namespace isq {
namespace ir {
bool DefgateOp::isDeclaration() {
    auto defs = this->definition();
    return defs.hasValue();
}
bool DefgateOp::isGateArray() {
    auto shape = this->shape();
    return shape.hasValue();
}
::mlir::Type DefgateOp::getTypeWhenUsed() {
    auto shape = this->shape();
    if (shape.hasValue()) {
        auto arr = shape->getValue();
        ::mlir::SmallVector<int64_t> tmp;
        for (auto iter = arr.begin(); iter != arr.end(); iter++) {
            auto i = iter->dyn_cast<::mlir::IntegerAttr>();
            tmp.push_back(i.getInt());
        }
        return ::mlir::MemRefType::get(tmp, this->type());
    } else {
        return this->type();
    }
}
/*
mlir::LogicalResult verify(DeclareOp op) {
if (op.op_type() != op.getResult().getType()) {
    op.emitOpError("operation signature mismatch.");
    return mlir::failure();
}
return mlir::success();
}
*/
} // namespace ir
} // namespace isq