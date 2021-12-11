#ifndef _ISQ_BACKENDS_H
#define _ISQ_BACKENDS_H
#include "mlir/IR/BuiltinOps.h"
#include <mlir/Support/LogicalResult.h>
#include <mlir/IR/MLIRContext.h>
#include <llvm/Support/raw_os_ostream.h>
namespace isq{
namespace ir{
mlir::LogicalResult generateOpenQASM3Logic(mlir::MLIRContext &context, mlir::ModuleOp &module, llvm::raw_fd_ostream &os);
}
}
#endif