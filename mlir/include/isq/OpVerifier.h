#ifndef _ISQ_OPVERIFIER_H
#define _ISQ_OPVERIFIER_H
#include "./Operations.h"
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>
namespace isq {
namespace ir {
mlir::SmallVector<mlir::Type> getGateParameterType(GateType ty);
// mlir::LogicalResult verify(DeclareOp op);
mlir::LogicalResult verify(DowngradeGateOp op);
// mlir::LogicalResult verify(GateOp op);
} // namespace ir
} // namespace isq
#endif