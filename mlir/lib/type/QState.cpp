#include <isq/IR.h>
#include <mlir/Support/LogicalResult.h>
namespace isq {
namespace ir {

const char *QStateParsePrint::keyword() const { return "qstate"; }
bool QStateParsePrint::isa(::mlir::Type me) const {
    return me.isa<QStateType>();
}
::mlir::Type
QStateParsePrint::parseType(::mlir::AsmParser &parser) const {
    /*
    if (parser.parseKeyword(this->keyword())){
        return nullptr;
    }*/
    return QStateType::get(parser.getBuilder().getContext());
}
void QStateParsePrint::printType(::mlir::Type type,
                                 ::mlir::AsmPrinter &printer) const {
    //printer << this->keyword();
    //printer << "qstate";
}
} // namespace ir
} // namespace isq