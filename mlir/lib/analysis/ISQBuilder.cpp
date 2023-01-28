#include "isq/Enums.h"
#include "isq/Operations.h"
#include "isq/QTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
namespace isq {
namespace ir {
class ISQBuilder : public mlir::OpBuilder {
public:
    ISQBuilder(mlir::MLIRContext* context) : mlir::OpBuilder(context) {
    }
    QStateType getQStateType() {
        return QStateType::get(getContext());
    }
    GateType getGateType(size_t gate_size, GateTrait traits){
        return GateType::get(getContext(), gate_size, traits);
    }
    GateType getGateType(size_t gate_size){
        return GateType::get(getContext(), gate_size, GateTrait::General);
    }
    
};
} // namespace ir
} // namespace isq
