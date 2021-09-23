#include <isq/IR.h>
#include <isq/tblgen/ISQDialect.cpp.inc>
#define GET_OP_CLASSES
#include <isq/tblgen/ISQOPs.cpp.inc>
namespace isq {
namespace ir {
void ISQDialect::initialize() {
    addTypes<QStateType, GateType, QOpType>();
#define GET_OP_LIST
    addOperations<
#include <isq/tblgen/ISQOPs.cpp.inc>
        >();
}
} // namespace ir
} // namespace isq
