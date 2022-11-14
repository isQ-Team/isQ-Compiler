#ifndef _LOGIC_OPERATIONS_H
#define _LOGIC_OPERATIONS_H
#include "llvm/ADT/TypeSwitch.h"
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
#define GET_OP_CLASSES
#include "logic/tblgen/LogicOps.h.inc"
#endif