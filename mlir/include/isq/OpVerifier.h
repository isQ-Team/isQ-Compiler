#ifndef _ISQ_OPVERIFIER_H
#define _ISQ_OPVERIFIER_H
#include "./Operations.h"
namespace isq {
namespace ir {
mlir::LogicalResult verify(DeclareOp op);
mlir::LogicalResult verify(DowngradeGateOp op);
mlir::LogicalResult verify(GateOp op);
} // namespace ir
} // namespace isq
#endif