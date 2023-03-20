#ifndef _ISQ_DIALECTS_EXTRA_H
#define _ISQ_DIALECTS_EXTRA_H
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/OpDefinition.h>
#include <isq/tblgen/ISQExtraDialect.h.inc>
#include <isq/tblgen/ISQExtraAttrs.h.inc>
#define GET_OP_FWD_DEFINES
#include <isq/tblgen/ISQExtraOps.h.inc>
#include <isq/tblgen/ISQExtraTypes.h.inc>

#define GET_OP_CLASSES
#include <isq/tblgen/ISQExtraOps.h.inc>
#endif