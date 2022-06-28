#include <isq/ParsePrint.h>
#include <isq/QTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/Support/LogicalResult.h>
namespace isq {
namespace ir {

const char *QIRQubitParsePrint::keyword() const { return "qir.qubit"; }
bool QIRQubitParsePrint::isa(::mlir::Type me) const {
    return me.isa<QIRQubitType>();
}
::mlir::Type
QIRQubitParsePrint::parseType(::mlir::AsmParser &parser) const {
    return QIRQubitType::get(parser.getBuilder().getContext());
}
void QIRQubitParsePrint::printType(::mlir::Type type,
                                 ::mlir::AsmPrinter &printer) const {
}


const char *QIRResultParsePrint::keyword() const { return "qir.result"; }
bool QIRResultParsePrint::isa(::mlir::Type me) const {
    return me.isa<QIRQubitType>();
}
::mlir::Type
QIRResultParsePrint::parseType(::mlir::AsmParser &parser) const {
    return QIRResultType::get(parser.getBuilder().getContext());
}
void QIRResultParsePrint::printType(::mlir::Type type,
                                 ::mlir::AsmPrinter &printer) const {
}

} // namespace ir
} // namespace isq