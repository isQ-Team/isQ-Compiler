#ifndef _ISQ_DIALECT_H
#define _ISQ_DIALECT_H
#include <mlir/IR/Dialect.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <isq/tblgen/ISQDialect.h.inc>
namespace isq {
namespace ir {
void ISQToolsInitialize(mlir::DialectRegistry &registry);
}
} // namespace isq
#endif