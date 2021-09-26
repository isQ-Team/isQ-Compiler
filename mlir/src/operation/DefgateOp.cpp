#include <isq/Operations.h>
#include <mlir/IR/BuiltinTypes.h>
namespace isq {
namespace ir {
bool DefgateOp::isDeclaration() {
    auto defs = this->definition();
    return !defs.hasValue();
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

::mlir::LogicalResult DefgateOp::parseIR(::mlir::OpAsmParser &parser,
                                         ::mlir::OperationState &result) {
    ::mlir::StringAttr sym_nameAttr;
    ::mlir::TypeAttr typeAttr;

    if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
        return ::mlir::failure();
    ::mlir::NamedAttrList parsedAttributes;
    auto attributeDictLocation = parser.getCurrentLocation();
    if (parser.parseOptionalAttrDict(parsedAttributes)) {
        return ::mlir::failure();
    }
    for (::mlir::StringRef disallowed :
         {::mlir::SymbolTable::getVisibilityAttrName(),
          ::mlir::SymbolTable::getSymbolAttrName(),
          ::mlir::StringRef("type")}) {
        if (parsedAttributes.get(disallowed))
            return parser.emitError(attributeDictLocation, "'")
                   << disallowed
                   << "' is an inferred attribute and should not be specified "
                      "in the "
                      "explicit attribute dictionary";
    }
    auto ctx = parser.getBuilder().getContext();
    parsedAttributes.push_back(
        ::std::make_pair(::mlir::Identifier::get("sym_visibility", ctx),
                         ::mlir::StringAttr::get(ctx, "nested")));
    result.attributes.append(parsedAttributes);
    if (parser.parseColon())
        return ::mlir::failure();

    if (parser.parseAttribute(typeAttr,
                              parser.getBuilder().getType<::mlir::NoneType>(),
                              "type", result.attributes))
        return ::mlir::failure();
    return ::mlir::success();
}

void DefgateOp::printIR(::mlir::OpAsmPrinter &p) {
    p << "isq.defgate";
    p << ' ';
    p.printSymbolName(sym_nameAttr().getValue());
    p.printOptionalAttrDict(
        (*this)->getAttrs(),
        /*elidedAttrs=*/{"sym_name", "type", "sym_visibility"});
    p << ' ' << ":";
    p << ' ';
    p.printAttributeWithoutType(typeAttr());
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