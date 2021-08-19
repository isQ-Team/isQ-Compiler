#include <tblgen/ISQDialect.h.inc>
#define GET_OP_CLASSES
#include <tblgen/ISQOps.h.inc>

namespace isq{
    namespace ir{
        class QubitType : public mlir::Type::TypeBase<QubitType, mlir::Type, mlir::TypeStorage>{
        public:
            using Base::Base;
        };
        class QStateType : public mlir::Type::TypeBase<QStateType, mlir::Type, mlir::TypeStorage>{
        public:
            using Base::Base;
        };
        
    }
}