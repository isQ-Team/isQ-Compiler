#include "isq/QTypes.h"
#include <isq/Operations.h>
#include <mlir/IR/BuiltinTypes.h>
namespace isq {
namespace ir {
::mlir::Type DeclareQOpOp::getTypeWhenUsed() {
    ::mlir::SmallVector<::mlir::Type> inputs, outputs;
    ::mlir::SmallVector<::mlir::Type> tup_elements;
    for (auto i = 0; i < this->size(); i++) {
        auto q = QStateType::get(this->getContext());
        tup_elements.push_back(q);
    }
    auto tup = tup_elements;
    // auto tup = ::mlir::TupleType::get(this->getContext(), tup_elements);
    inputs.append(tup.begin(), tup.end());
    outputs.append(tup.begin(), tup.end());
    auto in = this->signature().getInputs();
    inputs.append(in.begin(), in.end());
    auto out = this->signature().getResults();
    outputs.append(out.begin(), out.end());
    return ::mlir::FunctionType::get(this->getContext(), inputs, outputs);
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