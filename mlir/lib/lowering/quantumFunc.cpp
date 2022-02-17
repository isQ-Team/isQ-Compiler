#include "isq/lowering/quantumFunc.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <set>
#include <math.h>
#include <iostream>

using namespace Eigen;
using qsyn::ComplexPair;

using namespace mlir;
using namespace std;

string LLVMQuantumFunc::getMainFuncName(){
    return main_func;
}

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

FlatSymbolRefAttr LLVMQuantumFunc::getOrInsertResultGetOne(PatternRewriter &rewriter, ModuleOp module){

    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(qir_result_get_one))
        return SymbolRefAttr::get(context, qir_result_get_one);
    
    auto res_type = LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque("Result", context));
    auto result_get_one_ftype =
        LLVM::LLVMFunctionType::get(res_type, llvm::None, false);

    // Insert the function declaration
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(),
                                      qir_result_get_one, result_get_one_ftype);
    
    return SymbolRefAttr::get(context, qir_result_get_one);
}

FlatSymbolRefAttr LLVMQuantumFunc::getOrInsertResultEqual(PatternRewriter &rewriter, ModuleOp module){

    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(qir_result_equal))
        return SymbolRefAttr::get(context, qir_result_equal);
    
    auto res_type = mlir::IntegerType::get(context, 1);
    llvm::SmallVector<Type> result_type;
    result_type.push_back(LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque("Result", context)));
    result_type.push_back(LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque("Result", context)));
    auto result_equal_ftype =
        LLVM::LLVMFunctionType::get(res_type, result_type, false);

    // Insert the function declaration
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(),
                                      qir_result_equal, result_equal_ftype);
    
    return SymbolRefAttr::get(context, qir_result_equal);
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

FlatSymbolRefAttr getOrInsertBaseGate(PatternRewriter &rewriter, ModuleOp module, string gate_name_low, string qir_gate, int gate_size){
    
    auto *context = module.getContext();
    
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

bool decomposed(PatternRewriter &rewriter, ModuleOp module, qsyn::UnitaryVector &Uvector, int shape, MutableArrayRef<mlir::BlockArgument> operands){
        
    double esp = 1e-6;
    
    qsyn::qsyn A(shape, Uvector);
    qsyn::DecomposedGates sim_gates = qsyn::simplify(A.gates);
    
    if (!qsyn::verify(shape, Uvector, sim_gates, A.phase)){
        return false;
    }

    auto loc = module.getLoc();
    auto *context = module.getContext();

    string qir_gate_head = "__quantum__qis__";

    if (abs(A.phase) > esp){
        auto func_name = getOrInsertBaseGate(rewriter, module, qir_gate_head + "gphase",  "gphase", 0);
        mlir::Value val = val = rewriter.create<LLVM::ConstantOp>(loc, FloatType::getF64(context), 
                                rewriter.getF64FloatAttr(A.phase));
        auto call = rewriter.create<CallOp>(loc, func_name, 
                        llvm::None, llvm::ArrayRef<mlir::Value>({val}));
    }

    for (int j=0; j< sim_gates.size(); j++) {
        auto type = get<0>(sim_gates[j]);
        auto pos = get<1>(sim_gates[j]);
        if (type == qsyn::GateType::CNOT){
            auto func_name = getOrInsertBaseGate(rewriter, module, qir_gate_head + "cnot", "cnot", 2);
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

            auto func_name = getOrInsertBaseGate(rewriter, module, qir_gate_head + "u3", "u3", 1);
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

    if (base_gate.count(gate_name_low) == 1 && gate_name_low != "cnot"){
        qir_gate += "__body";
    }

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
                qsyn::UnitaryVector Uvector;
                for (int i = 0; i < mat.size(); i++){
                    auto line = mat[i].dyn_cast<mlir::ArrayAttr>();
                    for (int j = 0; j < line.size(); j++){
                        auto value = line[j].dyn_cast<isq::ir::ComplexF64Attr>();
                        double real = value.getReal().convertToDouble();
                        double imag = value.getImag().convertToDouble();
                        Uvector.push_back(ComplexPair(real, imag));
                    }
                }
                mat_def.insert(make_pair(gate_name_low, make_pair(gate_size, Uvector)));
                decomposed(rewriter, module, Uvector, gate_size, entryBlock.getArguments());
            }
        }

        rewriter.create<LLVM::ReturnOp>(module.getLoc(), llvm::None);
    }

    return SymbolRefAttr::get(context, qir_gate);
}

// get u's pos
// if NCtrl, pos is at the upper left of mat
// if Ctrl pos is at the lower right of mat
int getPos(const string& ctrl, int idx, int size, int pos){
    size /= 2;
    if (ctrl.size() == idx){
        return pos;
    }

    if (ctrl[idx] == 'f'){
        return getPos(ctrl, idx+1, size, pos);
    }else{
        return getPos(ctrl, idx+1, size, pos+size);
    }
}

FlatSymbolRefAttr LLVMQuantumFunc::getGate(PatternRewriter &rewriter, ModuleOp module, string gate_name, llvm::ArrayRef<Attribute> ctrl, bool inv){

    auto *context = module.getContext();
    set<string> base_gate = {"h", "x", "y", "z", "cnot", "s", "t"};
    set<string> hermite = {"h", "x", "y", "z", "cnot"};
    string gate_name_low = gate_name;
    transform(gate_name.begin(), gate_name.end(), gate_name_low.begin(), [](unsigned char c) { return tolower(c); });

    if (ctrl.size() == 0 && (!inv || base_gate.count(gate_name_low) == 1)){
        string qir_gate = qir_gate_head + gate_name_low;
        if (base_gate.count(gate_name_low) == 1 && gate_name_low != "cnot"){
            if (inv && hermite.count(gate_name_low) == 0){
                qir_gate += "__adj";
            }else{
                qir_gate += "__body";
            }
            return getOrInsertBaseGate(rewriter, module, gate_name_low, qir_gate, 1);
        }
        return SymbolRefAttr::get(context, qir_gate);
    }else{
        auto size_mat = getInfo(gate_name_low);

        int ori_size = size_mat.first;
        int ori_shape = (1 << ori_size);
        qsyn::UnitaryVector ori_mat = size_mat.second;
        
        string inv_head = "";
        if (inv && hermite.count(gate_name_low) == 0){
            // if not base gate, get inverse mat
            inv_head = "inv_";
            qsyn::UnitaryVector ori_mat_inv;
            for (int i = 0; i < ori_shape; i++){
                for (int j = 0; j < ori_shape; j++){
                    ori_mat_inv.push_back(ComplexPair(ori_mat[j*ori_shape+i].first, -1*ori_mat[j*ori_shape+i].second));
                }
            }
            ori_mat = ori_mat_inv;
        }

        
        string qir_gate = qir_gate_head + inv_head + gate_name_low;
        string ctrl_head = "";
        if (ctrl.size() > 0){
            for (auto &attr: ctrl){
                auto c = attr.cast<mlir::BoolAttr>().getValue();
                if (c){
                    ctrl_head += "t";
                }else{
                    ctrl_head += "f";
                }
            }
            qir_gate = qir_gate_head + inv_head + ctrl_head + "_" + gate_name_low;
        }
        
        if (module.lookupSymbol<LLVM::LLVMFuncOp>(qir_gate))
            return SymbolRefAttr::get(context, qir_gate);
        
        int size = ori_size + ctrl.size();
        qsyn::UnitaryVector Uvector;
        
        //cout << qir_gate << ", size: " << size << endl;

        int shape = (1 << size);
        for (int i = 0; i < shape; i++){
            for (int j = 0; j < shape; j++){
                if (i == j){
                    Uvector.push_back(ComplexPair(1., 0.));
                }else{
                    Uvector.push_back(ComplexPair(0., 0.));
                }
            }
        }
        
        int pos = getPos(ctrl_head, 0, shape, 0);

        for (int i = 0; i < ori_shape; i++){
            for (int j = 0; j < ori_shape; j++){
                Uvector[(i+pos)*shape+(j+pos)] = ori_mat[i*ori_shape+j];
            }
        }
        /*
        cout << "size: " << size << endl;
        for (int i = 0; i < shape; i++){
            for (int j = 0; j < shape; j++){
                cout << '(' << Uvector[i*shape+j].first << ',' << Uvector[i*shape+j].second << "),";
            }
            cout << endl;
        }*/
        
        auto res_type = LLVM::LLVMVoidType::get(context);
        llvm::SmallVector<Type> qbit_type;
        for (int i = 0; i < size; i++)
            qbit_type.push_back(LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque("Qubit", context)));
        auto reset_ftype =
            LLVM::LLVMFunctionType::get(res_type, qbit_type, false);

        // Insert the function declaration and definition
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(),
                                        qir_gate, reset_ftype);
        
        auto &entryBlock = *funcOp.addEntryBlock();
        rewriter.setInsertionPointToStart(&entryBlock);
        if (!decomposed(rewriter, module, Uvector, size, entryBlock.getArguments())){
            cout << "decompose error" << endl;
        }
        rewriter.create<LLVM::ReturnOp>(module.getLoc(), llvm::None);

        return SymbolRefAttr::get(context, qir_gate);
    }
}

pair<int, qsyn::UnitaryVector> LLVMQuantumFunc::getInfo(string gate_name){
    
    set<string> base_gate = {"h", "x", "y", "z", "cnot", "s", "t"};

    auto iter = mat_def.find(gate_name);
    if (iter == mat_def.end()){
        if (base_gate.count(gate_name) == 1){
            if (gate_name == "h"){
                qsyn::UnitaryVector Uvector = {
                    ComplexPair(sqrt(0.5), 0.),ComplexPair(sqrt(0.5), 0.),
                    ComplexPair(sqrt(0.5), 0.),ComplexPair(-sqrt(0.5), 0.)
                };
                mat_def.insert(make_pair(gate_name, make_pair(1, Uvector)));
            }
            if (gate_name == "x"){
                qsyn::UnitaryVector Uvector = {
                    ComplexPair(0., 0.),ComplexPair(1., 0.),
                    ComplexPair(1., 0.),ComplexPair(0., 0.)
                };
                mat_def.insert(make_pair(gate_name, make_pair(1, Uvector)));
            }
            if (gate_name == "y"){
                qsyn::UnitaryVector Uvector = {
                    ComplexPair(0., 0.),ComplexPair(0., -1.),
                    ComplexPair(0., 1.),ComplexPair(0., 0.)
                };
                mat_def.insert(make_pair(gate_name, make_pair(1, Uvector)));
            }
            if (gate_name == "z"){
                qsyn::UnitaryVector Uvector = {
                    ComplexPair(1., 0.),ComplexPair(0., 0.),
                    ComplexPair(0., 0.),ComplexPair(-1., 0.)
                };
                mat_def.insert(make_pair(gate_name, make_pair(1, Uvector)));
            }
            if (gate_name == "s"){
                qsyn::UnitaryVector Uvector = {
                    ComplexPair(1., 0.),ComplexPair(0., 0.),
                    ComplexPair(0., 0.),ComplexPair(0., 1.)
                };
                mat_def.insert(make_pair(gate_name, make_pair(1, Uvector)));
            }
            if (gate_name == "t"){
                qsyn::UnitaryVector Uvector = {
                    ComplexPair(1., 0.),ComplexPair(0., 0.),
                    ComplexPair(0., 0.),ComplexPair(sqrt(0.5), sqrt(0.5))
                };
                mat_def.insert(make_pair(gate_name, make_pair(1, Uvector)));
            }
            if (gate_name == "cnot"){
                qsyn::UnitaryVector Uvector = {
                    ComplexPair(1,0.),ComplexPair(0.,0.),ComplexPair(0.,0.),ComplexPair(0.,0.),
                    ComplexPair(0.,0.),ComplexPair(1.,0.),ComplexPair(0.,0.),ComplexPair(0.,0.),
                    ComplexPair(0.,0.),ComplexPair(0.,0.),ComplexPair(0.,0.),ComplexPair(1.,0.),
                    ComplexPair(0.,0.),ComplexPair(0.,0.),ComplexPair(1.,0.),ComplexPair(0.,0.)
                };
                mat_def.insert(make_pair(gate_name, make_pair(2, Uvector)));
            }
        }else{
            cout << "not found: " << gate_name << endl;
            qsyn::UnitaryVector Uvector = {
                ComplexPair(1,0.)
            };
            return make_pair(0, Uvector);
        }
    }
    return mat_def.find(gate_name)->second;
}