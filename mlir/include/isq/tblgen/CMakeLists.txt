set(LLVM_TARGET_DEFINITIONS IR.td)
mlir_tablegen(ISQOps.h.inc -gen-op-decls -dialect isq)
mlir_tablegen(ISQOPs.cpp.inc -gen-op-defs -dialect isq)
mlir_tablegen(ISQDialect.h.inc -gen-dialect-decls -dialect isq)
mlir_tablegen(ISQDialect.cpp.inc -gen-dialect-defs -dialect isq)
mlir_tablegen(Patterns.inc -gen-rewriters)
mlir_tablegen(ISQEnums.h.inc -gen-enum-decls -dialect isq)
mlir_tablegen(ISQEnums.cpp.inc -gen-enum-defs -dialect isq)
mlir_tablegen(__tmp_IR.md -gen-dialect-doc --dialect isq)
add_custom_command(OUTPUT IR.md 
    COMMAND ${PROJECT_SOURCE_DIR}/scripts/fix_dialect_doc.py __tmp_IR.md IR.md
    DEPENDS __tmp_IR.md)
add_custom_target(mlir-dialect-doc ALL DEPENDS IR.md)
add_public_tablegen_target(MLIRTablegen)