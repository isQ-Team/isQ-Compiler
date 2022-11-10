#include "logic/IR.h"
#include "logic/tblgen/LogicDialect.cpp.inc"
#define GET_OP_CLASSES
#include "logic/tblgen/LogicOPs.cpp.inc"

namespace logic {
    namespace ir {
        void LogicDialect::initialize() {
            addOperations<
        #define GET_OP_LIST
        #include <logic/tblgen/LogicOPs.cpp.inc>
                >();
        }
    }
}