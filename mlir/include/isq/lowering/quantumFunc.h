#include "isq/IR.h"

using namespace mlir;

class LLVMQuantumFunc{

public:

    static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter, ModuleOp module);
    static FlatSymbolRefAttr getOrInsertAllocQubit(PatternRewriter &rewriter, ModuleOp module);
    static FlatSymbolRefAttr getOrInsertReleaseQubit(PatternRewriter &rewriter, ModuleOp module);
    static FlatSymbolRefAttr getOrInsertAllocQubitArray(PatternRewriter &rewriter, ModuleOp module);
    static FlatSymbolRefAttr getOrInsertMeasure(PatternRewriter &rewriter, ModuleOp module);
    static FlatSymbolRefAttr getOrInsertReset(PatternRewriter &rewriter, ModuleOp module);
    static FlatSymbolRefAttr getOrInsertGate(PatternRewriter &rewriter, ModuleOp module, isq::ir::DefgateOp op);
    static FlatSymbolRefAttr getGate(ModuleOp module, std::string gate_name);

private:
    
    inline static const std::string qir_printf = "printf";
    inline static const std::string qir_alloc_qubit_array = "__quantum__rt__qubit_allocate_array";
    inline static const std::string qir_alloc_qubit = "__quantum__rt__qubit_allocate";
    inline static const std::string qir_release_qubit = "__quantum__rt__qubit_release";
    inline static const std::string qir_measure = "__quantum__rt__measure";
    inline static const std::string qir_reset = "__quantum__rt__reset";
    inline static const std::string qir_gate_head = "__quantum__qir__";
};