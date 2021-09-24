#include "isq/Enums.h"
#include <isq/QTypes.h>
#include <isq/IR.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Support/LogicalResult.h>
namespace isq {
namespace ir {
llvm::hash_code GateTypeStorage::hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
}
GateTypeStorage *
GateTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                           const KeyTy &key) {
    return new (allocator.allocate<GateTypeStorage>())
        GateTypeStorage(key, allocator);
}
bool GateTypeStorage::operator==(const KeyTy &key) const {
    return key == KeyTy(size, hints);
}
GateType GateType::get(mlir::MLIRContext *ctx, GateInfo k) {
    return Base::get(ctx, k);
}
GateInfo GateType::getGateInfo() {
    return GateInfo(getImpl()->size, getImpl()->hints);
}

int64_t GateType::gateSize() { return getImpl()->size; }
GateTrait GateType::hints() { return getImpl()->hints.getValue(); }
bool GateType::hasHint(GateTrait t) {
    return bitEnumContains(this->hints(), t);
}

const char *GateOpParsePrint::keyword() const { return "gate"; }
bool GateOpParsePrint::isa(::mlir::Type me) const { return me.isa<GateType>(); }
::mlir::Type
GateOpParsePrint::parseType(::llvm::SMLoc kwLoc,
                            ::mlir::DialectAsmParser &parser) const {
    auto ctx = parser.getBuilder().getContext();
    if (parser.parseLess())
        return nullptr;
    auto typeLoc = parser.getCurrentLocation();
    int64_t gate_size = 0;
    if (parser.parseInteger(gate_size))
        return nullptr;
    GateTrait tr = GateTrait::General;
    while (mlir::succeeded(parser.parseOptionalComma())) {
        mlir::StringRef kw;
        if (parser.parseKeyword(&kw)) {
            return nullptr;
        }
        auto e = symbolizeEnum<GateTrait>(kw);
        if (!e.hasValue())
            return nullptr;
        if (e.getValue() == GateTrait::General)
            return nullptr;
        tr = tr | e.getValue();
    }
    if (parser.parseGreater()) {
        return nullptr;
    }
    if (gate_size <= 0) {
        parser.emitError(typeLoc, "gate size should be positive.");
        return nullptr;
    }
    return GateType::get(ctx, GateInfo(gate_size, GateTraitAttr::get(ctx, tr)));
}
void GateOpParsePrint::printType(::mlir::Type type,
                                 ::mlir::DialectAsmPrinter &printer) const {
    GateType t = type.cast<GateType>();
    auto info = t.getGateInfo();
    printer << "gate<" << std::get<0>(info);
    auto traits = std::get<1>(info);
    if (traits.getValue() != GateTrait::General) {
        auto t = stringifyGateTrait(traits.getValue());
        for (auto i = t.begin(); i != t.end(); i++) {
            if (*i == '|')
                *i = ',';
        }
        printer << ",";
        printer << t;
    }
    printer << ">";
}

} // namespace ir
} // namespace isq
