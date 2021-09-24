#include <isq/QTypes.h>
#include <isq/IR.h>
namespace isq {
namespace ir {
llvm::hash_code QOpTypeStorage::hashKey(const KeyTy &key) {
    return mlir::hash_value(key);
}
QOpTypeStorage *QOpTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                                          const KeyTy &key) {
    return new (allocator.allocate<QOpTypeStorage>()) QOpTypeStorage(key);
}
bool QOpTypeStorage::operator==(const KeyTy &key) const {
    return key == funcType;
}
QOpType QOpType::get(mlir::MLIRContext *ctx, mlir::FunctionType funcType) {
    return Base::get(ctx, funcType);
}
mlir::FunctionType QOpType::getFuncType() { return getImpl()->funcType; }
QOpType gateToQOp(GateType gate) {
    auto info = gate.getGateInfo();
    auto qubit_size = std::get<0>(info);
    auto tyinout = mlir::SmallVector<mlir::Type>();
    for (auto i = 0; i < qubit_size; i++) {
        tyinout.push_back(QStateType::get(gate.getContext()));
    }
    return QOpType::get(
        gate.getContext(),
        mlir::FunctionType::get(gate.getContext(), tyinout, tyinout));
}

const char *QOpParsePrint::keyword() const { return "qop"; }
bool QOpParsePrint::isa(::mlir::Type me) const { return me.isa<QOpType>(); }
::mlir::Type QOpParsePrint::parseType(::llvm::SMLoc kwLoc,
                                      ::mlir::DialectAsmParser &parser) const {
    auto ctx = parser.getBuilder().getContext();
    // qop<functiontype>
    mlir::Type f;
    if (parser.parseLess())
        return nullptr;
    auto typeLoc = parser.getCurrentLocation();
    if (parser.parseType(f) || parser.parseGreater())
        return nullptr;
    auto ft = f.cast<mlir::FunctionType>();
    if (!ft) {
        parser.emitError(typeLoc,
                         "QOp internal type should be a FunctionType.");
        return nullptr;
    }
    return QOpType::get(ctx, ft);
}
void QOpParsePrint::printType(::mlir::Type type,
                              ::mlir::DialectAsmPrinter &printer) const {
    printer << "qop<";
    printer << type.cast<QOpType>().getFuncType();
    printer << ">";
}
} // namespace ir
} // namespace isq