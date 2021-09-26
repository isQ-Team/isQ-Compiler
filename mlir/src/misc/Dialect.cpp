#include <isq/IR.h>
#include <llvm/ADT/StringExtras.h>
#include <isq/tblgen/ISQDialect.cpp.inc>
#define GET_OP_CLASSES
#include <isq/tblgen/ISQOPs.cpp.inc>
#include <isq/tblgen/ISQEnums.cpp.inc>

#define GET_TYPEDEF_CLASSES
#include <isq/tblgen/ISQTypes.cpp.inc>
namespace isq {
namespace ir {

mlir::Type ISQDialect::parseType(mlir::DialectAsmParser &parser) const {
    mlir::StringRef kw;
    auto kwLoc = parser.getCurrentLocation();
    if (parser.parseKeyword(&kw)) {
        return nullptr;
    }
    mlir::Type ty;
    generatedTypeParser(parser.getBuilder().getContext(), parser, kw, ty);
    if (!ty) {
        parser.emitError(kwLoc, "unrecognized type");
    }
    return ty;
}
void ISQDialect::printType(::mlir::Type type,
                           ::mlir::DialectAsmPrinter &printer) const {
    if (mlir::failed(generatedTypePrinter(type, printer))) {
        llvm_unreachable("bad ISQ type for printer.");
    }
}

void ISQDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <isq/tblgen/ISQTypes.cpp.inc>
        >();

    addOperations<
#define GET_OP_LIST
#include <isq/tblgen/ISQOPs.cpp.inc>
        >();
}
} // namespace ir
} // namespace isq
