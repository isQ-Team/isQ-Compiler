#ifndef _ISQ_OPERATIONS_H
#define _ISQ_OPERATIONS_H
#include "./QTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
<<<<<<< HEAD
#include <mlir/Parser.h>
#include <mlir/Parser/AsmParserState.h>
#include <mlir/Interfaces/DerivedAttributeOpInterface.h>
=======
#include <mlir/Parser/Parser.h>
#include <mlir/AsmParser/AsmParserState.h>
#include <mlir/Interfaces/DerivedAttributeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/IR/SymbolTable.h>
>>>>>>> merge
#define GET_OP_CLASSES
#include <isq/tblgen/ISQOps.h.inc>
#endif