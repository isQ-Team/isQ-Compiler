#include "isq/Math.h"
#include <isq/IR.h>
#include <llvm/ADT/StringSwitch.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LogicalResult.h>

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

mlir::LogicalResult verifyGateDefinition(DefgateOp op, int id,
                                         GateDefinition def, GateType ty) {
    if (!def) {
        llvm_unreachable("Null GateDefinition.");
    }
    auto name = def.type().getValue();
    if (name == "unitary") {
        // try to get as a matrix.
        std::vector<std::vector<std::complex<double>>> mat;
        auto arr = def.value().dyn_cast_or_null<::mlir::ArrayAttr>();
        if (!arr) {
            op->emitError()
                << "Definition #" << id << " should use a matrix as value.";
            return mlir::failure();
        }
        for (auto row : arr) {
            auto row_arr = row.dyn_cast_or_null<::mlir::ArrayAttr>();
            std::vector<std::complex<double>> row_vec;
            if (!row_arr) {
                op->emitError()
                    << "Definition #" << id << " should use a matrix as value.";
                return mlir::failure();
            }
            for (auto element : row_arr) {
                auto element_attr = element.dyn_cast_or_null<ComplexF64Attr>();
                if (!element_attr) {
                    op->emitError()
                        << "Definition #" << id
                        << " matrix entries should be complex numbers.";
                    return mlir::failure();
                }
                row_vec.push_back(element_attr.complexValue());
            }
            mat.push_back(std::move(row_vec));
        }
        math::InputMatrix wrapper;
        wrapper.body = std::make_unique<math::InputMatrix::Ty>(std::move(mat));
        auto dim = math::checkDimensionality(wrapper);
        if (!dim.hasValue()) {
            op->emitError()
                << "Definition #" << id << " input is not a matrix.";
            return mlir::failure();
        }
        auto dimension = dim.getValue();
        if (dimension != (1 << ty.getSize())) {
            op->emitError() << "Definition #" << id
                            << " matrix dimensionality and gate size mismatch.";
            return mlir::failure();
        }
        // check unitary.
        auto math_mat = math::toEigenMatrix(wrapper);
        if (!math_mat) {
            llvm_unreachable("nope");
        }
        if (!math::isUnitary(*math_mat)) {
            op->emitError()
                << "Definition #" << id << " matrix seems not unitary.";
            return mlir::failure();
        }
        auto hints = ty.getHints();
        if (bitEnumContains(hints, GateTrait::Hermitian)) {
            if (!math::isHermitian(*math_mat)) {
                op->emitError()
                    << "Definition #" << id << " matrix seems not hermitian.";
                return mlir::failure();
            }
        }
        if (bitEnumContains(hints, GateTrait::Diagonal)) {
            if (!math::isDiagonal(*math_mat)) {
                op->emitError()
                    << "Definition #" << id << " matrix seems not diagonal.";
                return mlir::failure();
            }
        }
        if (bitEnumContains(hints, GateTrait::Antidiagonal)) {
            if (!math::isAntiDiagonal(*math_mat)) {
                op->emitError() << "Definition #" << id
                                << " matrix seems not antidiagonal.";
                return mlir::failure();
            }
        }
    } else if (name == "decomposition") {

    } else {
        op->emitError() << "Definition #" << id << " type invalid.";
        return mlir::failure();
    }
    return mlir::success();
}
::mlir::LogicalResult DefgateOp::verifyIR() {
    if (this->definition().hasValue()) {
        auto defs = this->definition()->getValue();
        for (auto i = 0; i < defs.size(); i++) {
            auto def = defs[i];
            if (mlir::failed(verifyGateDefinition(
                    *this, i, def.dyn_cast<GateDefinition>(), this->type()))) {
                return mlir::failure();
            }
        }
    }
    return mlir::success();
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