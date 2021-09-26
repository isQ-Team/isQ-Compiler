#include <isq/Operations.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/Casting.h>
#include <mlir/Support/LLVM.h>
namespace isq {
namespace ir {
::mlir::LogicalResult
UseGateOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
    auto symbol_def = symbolTable.lookupNearestSymbolFrom(*this, this->name());
    if (auto gatedef = llvm::dyn_cast_or_null<DefgateOp>(symbol_def)) {
        if (this->getResult().getType() == gatedef.getTypeWhenUsed()) {
            return mlir::success();
        } else {
            this->emitOpError()
                << "type mismatch, expected " << gatedef.getTypeWhenUsed();
            return mlir::failure();
        }
    }
    this->emitOpError() << "symbol `" << this->name()
                        << "` not found or has wrong type";
    return mlir::failure();
}
/*
mlir::LogicalRea        sult verify(GateOp op) {
if (op.gate_type() != op.getResult().getType()) {
    op.emitOpError("gate dimension or trait mismatch.");
    return mlir::failure();
}
return mlir::success();
}
*/
} // namespace ir
} // namespace isq