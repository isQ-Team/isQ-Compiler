#include "isq/lowering/quantumFunc.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <set>
#include "isq/qsyn.h"

using namespace Eigen;
using qsyn::ComplexPair;

using namespace mlir;
using namespace std;

FlatSymbolRefAttr LLVMQuantumFunc::getOrInsertPrintf(PatternRewriter &rewriter, ModuleOp module){
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(qir_printf))
        return SymbolRefAttr::get(context, qir_printf);

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                  /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), qir_printf, llvmFnType);
    return SymbolRefAttr::get(context, qir_printf);
}

FlatSymbolRefAttr LLVMQuantumFunc::getOrInsertAllocQubitArray(PatternRewriter &rewriter, ModuleOp module){

    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(qir_alloc_qubit_array))
        return SymbolRefAttr::get(context, qir_alloc_qubit_array);
    
    auto qubit_type = IntegerType::get(context, 64);
    auto array_qbit_type =
        LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque("Array", context));
    auto qalloc_ftype =
        LLVM::LLVMFunctionType::get(array_qbit_type, qubit_type, false);

    // Insert the function declaration
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(),
                                      qir_alloc_qubit_array, qalloc_ftype);
    
    return SymbolRefAttr::get(context, qir_alloc_qubit_array);
}

FlatSymbolRefAttr LLVMQuantumFunc::getOrInsertAllocQubit(PatternRewriter &rewriter, ModuleOp module){

    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(qir_alloc_qubit))
        return SymbolRefAttr::get(context, qir_alloc_qubit);
    
    auto qbit_type =
        LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque("Qubit", context));
    auto qalloc_ftype =
        LLVM::LLVMFunctionType::get(qbit_type, llvm::None, false);

    // Insert the function declaration
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(),
                                      qir_alloc_qubit, qalloc_ftype);
    
    return SymbolRefAttr::get(context, qir_alloc_qubit);
}

FlatSymbolRefAttr LLVMQuantumFunc::getOrInsertReleaseQubit(PatternRewriter &rewriter, ModuleOp module){

    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(qir_release_qubit))
        return SymbolRefAttr::get(context, qir_release_qubit);
    
    auto qbit_type =
        LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque("Qubit", context));
    auto qalloc_ftype =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context), qbit_type, false);

    // Insert the function declaration
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(),
                                      qir_release_qubit, qalloc_ftype);
    
    return SymbolRefAttr::get(context, qir_release_qubit);
}

FlatSymbolRefAttr LLVMQuantumFunc::getOrInsertMeasure(PatternRewriter &rewriter, ModuleOp module){

    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(qir_measure))
        return SymbolRefAttr::get(context, qir_measure);
    
    auto res_type = LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque("Result", context));
    auto qbit_type =
        LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque("Qubit", context));
    auto measure_ftype =
        LLVM::LLVMFunctionType::get(res_type, qbit_type, false);

    // Insert the function declaration
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(),
                                      qir_measure, measure_ftype);
    
    return SymbolRefAttr::get(context, qir_measure);
}

FlatSymbolRefAttr LLVMQuantumFunc::getOrInsertReset(PatternRewriter &rewriter, ModuleOp module){

    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(qir_reset))
        return SymbolRefAttr::get(context, qir_reset);
    
    auto res_type = LLVM::LLVMVoidType::get(context);
    auto qbit_type =
        LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque("Qubit", context));
    auto reset_ftype =
        LLVM::LLVMFunctionType::get(res_type, qbit_type, false);

    // Insert the function declaration
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(),
                                      qir_reset, reset_ftype);
    
    return SymbolRefAttr::get(context, qir_reset);
}

FlatSymbolRefAttr getOrInsertBaseGate(PatternRewriter &rewriter, ModuleOp module, string gate_name, int gate_size){
    
    auto *context = module.getContext();
    string gate_name_low = gate_name;
    transform(gate_name.begin(), gate_name.end(), gate_name_low.begin(), [](unsigned char c) { return tolower(c); });

    string qir_gate_head = "__quantum__qir__";
    string qir_gate = qir_gate_head + gate_name_low;

    if (module.lookupSymbol<LLVM::LLVMFuncOp>(qir_gate))
        return SymbolRefAttr::get(context, qir_gate);
    
    auto res_type = LLVM::LLVMVoidType::get(context);
    
    llvm::SmallVector<Type> qbit_type;
    if (gate_name_low.compare("u3") == 0){
        for (int i = 0; i < 3; i++)
            qbit_type.push_back(mlir::FloatType::getF64(context));
    }else if (gate_name_low.compare("gphase") == 0){
        qbit_type.push_back(mlir::FloatType::getF64(context));
    }

    for (int i = 0; i < gate_size; i++)
        qbit_type.push_back(LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque("Qubit", context)));
    auto reset_ftype =
        LLVM::LLVMFunctionType::get(res_type, qbit_type, false);

    // Insert the function declaration and definition
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(),
                                      qir_gate, reset_ftype);

    return SymbolRefAttr::get(context, qir_gate);
}

bool decomposed(PatternRewriter &rewriter, ModuleOp module, mlir::ArrayAttr& definition, int shape, MutableArrayRef<mlir::BlockArgument> operands){
        
    double esp = 1e-6;
    qsyn::UnitaryVector Uvector;
    
    for (int i = 0; i < definition.size(); i++){
        auto line = definition[i].dyn_cast<mlir::ArrayAttr>();
        for (int j = 0; j < line.size(); j++){
            auto value = line[j].dyn_cast<isq::ir::ComplexF64Attr>();
            double real = value.getReal().convertToDouble();
            double imag = value.getImag().convertToDouble();
            Uvector.push_back(ComplexPair(real, imag));
        }
    }
    qsyn::qsyn A(shape, Uvector);
    qsyn::DecomposedGates sim_gates = qsyn::simplify(A.gates);
    
    if (!qsyn::verify(shape, Uvector, sim_gates, A.phase)){
        return false;
    }

    auto loc = module.getLoc();
    auto *context = module.getContext();

    if (abs(A.phase) > esp){
        auto func_name = getOrInsertBaseGate(rewriter, module, "gphase", 0);
        mlir::Value val = val = rewriter.create<LLVM::ConstantOp>(loc, FloatType::getF64(context), 
                                rewriter.getF64FloatAttr(A.phase));
        auto call = rewriter.create<CallOp>(loc, func_name, 
                        llvm::None, llvm::ArrayRef<mlir::Value>({val}));
    }

    for (int j=0; j< sim_gates.size(); j++) {
        auto type = get<0>(sim_gates[j]);
        auto pos = get<1>(sim_gates[j]);
        if (type == qsyn::GateType::CNOT){
            auto func_name = getOrInsertBaseGate(rewriter, module, "cnot", 2);
            auto call = rewriter.create<CallOp>(loc, func_name, 
                        llvm::None, llvm::ArrayRef<mlir::Value>({operands[pos[0]], operands[pos[1]]}));
        }else{

            llvm::SmallVector<mlir::Value> args;
            double theta[3] = {get<2>(sim_gates[j]), get<3>(sim_gates[j]), get<4>(sim_gates[j])};
            for (int k = 0; k < 3; k++){
                mlir::Value val = val = rewriter.create<LLVM::ConstantOp>(loc, FloatType::getF64(context), 
                                rewriter.getF64FloatAttr(theta[k]));
                args.push_back(val);
            }
            args.push_back(operands[pos[0]]);

            auto func_name = getOrInsertBaseGate(rewriter, module, "u3", 1);
            auto call = rewriter.create<CallOp>(loc, func_name, llvm::None, 
                                llvm::makeArrayRef(args));
        }
    }

    return true;

}

FlatSymbolRefAttr LLVMQuantumFunc::getOrInsertGate(PatternRewriter &rewriter, ModuleOp module, isq::ir::DefgateOp op){

    set<string> base_gate = {"h", "x", "y", "z", "cnot", "s", "t"};
    
    string gate_name = op.sym_name().str();
    int gate_size = op.type().getSize();

    auto *context = module.getContext();
    string gate_name_low = gate_name;
    transform(gate_name.begin(), gate_name.end(), gate_name_low.begin(), [](unsigned char c) { return tolower(c); });

    string qir_gate = qir_gate_head + gate_name_low;

    if (module.lookupSymbol<LLVM::LLVMFuncOp>(qir_gate))
        return SymbolRefAttr::get(context, qir_gate);
    
    auto res_type = LLVM::LLVMVoidType::get(context);
    llvm::SmallVector<Type> qbit_type;
    for (int i = 0; i < gate_size; i++)
        qbit_type.push_back(LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque("Qubit", context)));
    auto reset_ftype =
        LLVM::LLVMFunctionType::get(res_type, qbit_type, false);

    // Insert the function declaration and definition
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(),
                                      qir_gate, reset_ftype);

    if (op.definition() && base_gate.count(gate_name_low) == 0){
        auto &entryBlock = *funcOp.addEntryBlock();
        rewriter.setInsertionPointToStart(&entryBlock);

        for(auto& def_ : *op.definition()){
            auto def = def_.cast<isq::ir::GateDefinition>();
            if(def.type()=="unitary"){
                auto mat = def.value().cast<mlir::ArrayAttr>();
                decomposed(rewriter, module, mat, gate_size, entryBlock.getArguments());
            }
        }

        rewriter.create<LLVM::ReturnOp>(module.getLoc(), llvm::None);
    }

    return SymbolRefAttr::get(context, qir_gate);
}


FlatSymbolRefAttr LLVMQuantumFunc::getGate(ModuleOp module, string gate_name){

    auto *context = module.getContext();
    
    string gate_name_low = gate_name;
    transform(gate_name.begin(), gate_name.end(), gate_name_low.begin(), [](unsigned char c) { return tolower(c); });

    string qir_gate = qir_gate_head + gate_name_low;
    return SymbolRefAttr::get(context, qir_gate);

}