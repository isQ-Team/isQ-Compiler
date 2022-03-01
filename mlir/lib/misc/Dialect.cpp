#include "isq/Dialect.h"
#include <isq/IR.h>
#include <llvm/ADT/StringExtras.h>
#include <isq/tblgen/ISQDialect.cpp.inc>
#include <mlir/IR/Attributes.h>
#define GET_OP_CLASSES
#include <isq/tblgen/ISQOPs.cpp.inc>
#include <isq/tblgen/ISQEnums.cpp.inc>

#define GET_TYPEDEF_CLASSES
#include <isq/tblgen/ISQTypes.cpp.inc>
#define GET_ATTRDEF_CLASSES
#include <isq/tblgen/ISQAttrs.cpp.inc>

#include <isq/tblgen/ISQStructAttrs.cpp.inc>
#include <isq/passes/Passes.h>
namespace isq {
namespace ir {

void ISQDialect::getCanonicalizationPatterns(mlir::RewritePatternSet &results) const {

}

mlir::Type ISQDialect::parseType(mlir::DialectAsmParser &parser) const {
    mlir::StringRef kw;
    auto kwLoc = parser.getCurrentLocation();
    if (parser.parseKeyword(&kw)) {
        parser.emitError(kwLoc, "unrecognized type");
        return nullptr;
    }
    mlir::Type ty;
    auto ret =
        generatedTypeParser(parser, kw, ty);
    if (!ret.hasValue()) {
        parser.emitError(kwLoc, "unrecognized type");
        return nullptr;
    }

    return ty;
}
void ISQDialect::printType(::mlir::Type type,
                           ::mlir::DialectAsmPrinter &printer) const {
    if (mlir::failed(generatedTypePrinter(type, printer))) {
        llvm_unreachable("bad ISQ type for printer.");
    }
}

mlir::Attribute ISQDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                           ::mlir::Type type) const {
    mlir::StringRef kw;
    auto kwLoc = parser.getCurrentLocation();
    if (parser.parseKeyword(&kw)) {
        parser.emitError(kwLoc, "unrecognized attribute");
        return nullptr;
    }
    mlir::Attribute attr;
    auto ret = generatedAttributeParser(parser, kw, type, attr);
    if (!ret.hasValue()) {
        parser.emitError(kwLoc, "unrecognized attribute");
        return nullptr;
    }
    return attr;
}
void ISQDialect::printAttribute(::mlir::Attribute attr,
                                ::mlir::DialectAsmPrinter &os) const {
    if (mlir::failed(generatedAttributePrinter(attr, os))) {
        llvm_unreachable("bas ISQ attribute for printer");
    }
}

void ISQDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <isq/tblgen/ISQTypes.cpp.inc>
        >();
    addAttributes<
#define GET_ATTRDEF_LIST
#include <isq/tblgen/ISQAttrs.cpp.inc>
        >();
    addOperations<
#define GET_OP_LIST
#include <isq/tblgen/ISQOPs.cpp.inc>
        >();

    
}

} // namespace ir
} // namespace isq
