#include <isq/dialects/Extra.h>
#include <isq/tblgen/ISQExtraAttrs.cpp.inc>
#include <isq/tblgen/ISQExtraDialect.cpp.inc>
#define GET_OP_CLASSES
#include <isq/tblgen/ISQExtraOPs.cpp.inc>
#include <isq/tblgen/ISQExtraTypes.cpp.inc>

void isq::extra::ISQExtraDialect::initialize(){
    addTypes<
#define GET_TYPEDEF_LIST
#include <isq/tblgen/ISQExtraTypes.cpp.inc>
        >();
    addAttributes<
#define GET_ATTRDEF_LIST
#include <isq/tblgen/ISQExtraAttrs.cpp.inc>
        >();
    addOperations<
#define GET_OP_LIST
#include <isq/tblgen/ISQExtraOPs.cpp.inc>
        >();

}