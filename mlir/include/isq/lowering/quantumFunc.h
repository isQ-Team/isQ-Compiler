#include "isq/IR.h"
#include <map>
#include "isq/qsyn.h"

using namespace mlir;

class LLVMQuantumFunc{

public:

    static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter, ModuleOp module);
    static FlatSymbolRefAttr getOrInsertAllocQubit(PatternRewriter &rewriter, ModuleOp module);
    static FlatSymbolRefAttr getOrInsertReleaseQubit(PatternRewriter &rewriter, ModuleOp module);
    static FlatSymbolRefAttr getOrInsertAllocQubitArray(PatternRewriter &rewriter, ModuleOp module);
    static FlatSymbolRefAttr getOrInsertMeasure(PatternRewriter &rewriter, ModuleOp module);
    static FlatSymbolRefAttr getOrInsertResultGetOne(PatternRewriter &rewriter, ModuleOp module);
    static FlatSymbolRefAttr getOrInsertResultEqual(PatternRewriter &rewriter, ModuleOp module);
    static FlatSymbolRefAttr getOrInsertReset(PatternRewriter &rewriter, ModuleOp module);
    static FlatSymbolRefAttr getOrInsertGate(PatternRewriter &rewriter, ModuleOp module, isq::ir::DefgateOp op);
    static FlatSymbolRefAttr getGate(PatternRewriter &rewriter, ModuleOp module, std::string gate_name, llvm::ArrayRef<Attribute> ctrl, bool inv);
    static std::pair<int, synthesis::UnitaryVector> getInfo(std::string gate_name);
    static std::string getMainFuncName();
private:
    inline static const std::string main_func = "test_main";
    inline static std::map<std::string, std::pair<int, qsyn::UnitaryVector>> mat_def;
    inline static const std::string qir_printf = "printf";
    inline static const std::string qir_alloc_qubit_array = "__quantum__rt__qubit_allocate_array";
    inline static const std::string qir_alloc_qubit = "__quantum__rt__qubit_allocate";
    inline static const std::string qir_release_qubit = "__quantum__rt__qubit_release";
    inline static const std::string qir_measure = "__quantum__qis__measure";
    inline static const std::string qir_result_get_one = "__quantum__rt__result_get_one";
    inline static const std::string qir_result_equal = "__quantum__rt__result_equal";
    inline static const std::string qir_reset = "__quantum__qis__reset";
    inline static const std::string qir_gate_head = "__quantum__qis__";
};