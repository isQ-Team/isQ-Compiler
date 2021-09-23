#include <isq/QTypes.h>
namespace isq{
    namespace ir{
        llvm::hash_code hash_value(const GateTypeHint &arg) {
            int val = -1;
            if (arg == GateTypeHint::Symmetric) {
            val = 0;
            } else if (arg == GateTypeHint::Diagonal) {
            val = 1;
            } else if (arg == GateTypeHint::AntiDiagonal) {
            val = 2;
            } else if (arg == GateTypeHint::Hermitian) {
            val = 3;
            }
            assert(val >= 0);
            return llvm::hash_code(val);
        }
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
          auto r = std::get<1>(k);
          assert(std::is_sorted(r.begin(), r.end()) &&
                 "hints should be sorted");
          assert(std::adjacent_find(r.begin(), r.end()) == r.end() &&
                 "no duplicate elements should exist");
          return Base::get(ctx, k);
        }
        GateInfo GateType::getGateInfo() {
          return GateInfo(getImpl()->size, getImpl()->hints);
        }
        } // namespace ir
}
