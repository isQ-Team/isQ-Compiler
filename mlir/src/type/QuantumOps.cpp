#include <isq/QTypes.h>
namespace isq{
    namespace ir {
        llvm::hash_code QOpTypeStorage::hashKey(const KeyTy &key) {
          return mlir::hash_value(key);
        }
        QOpTypeStorage *
        QOpTypeStorage::construct(mlir::TypeStorageAllocator &allocator,
                                  const KeyTy &key) {
          return new (allocator.allocate<QOpTypeStorage>()) QOpTypeStorage(key);
        }
        bool QOpTypeStorage::operator==(const KeyTy &key) const {
          return key == funcType;
        }
        QOpType QOpType::get(mlir::MLIRContext *ctx,
                             mlir::FunctionType funcType) {
          return Base::get(ctx, funcType);
        }
        mlir::FunctionType QOpType::getFuncType() {
          return getImpl()->funcType;
        }
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
    }
}