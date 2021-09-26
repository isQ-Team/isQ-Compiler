#include <isq/IR.h>
#include <mlir/Support/LogicalResult.h>
namespace isq {
namespace ir {

const char *QStateParsePrint::keyword() const { return "qstate"; }
bool QStateParsePrint::isa(::mlir::Type me) const {
    return me.isa<QStateType>();
}
::mlir::Type
QStateParsePrint::parseType(::mlir::DialectAsmParser &parser) const {
    return QStateType::get(parser.getBuilder().getContext());
}
void QStateParsePrint::printType(::mlir::Type type,
                                 ::mlir::DialectAsmPrinter &printer) const {
    printer << "qstate";
}
} // namespace ir
} // namespace isq