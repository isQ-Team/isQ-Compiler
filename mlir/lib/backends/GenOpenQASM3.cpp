/*
#include <iostream>
#include <map> 
#include <set>
#include <vector>
#include <algorithm>

#include "isq/Dialect.h"
#include "isq/mlirGen.h"
#include <isq/IR.h>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"


#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

using namespace isq::ir;
using namespace isq;
using namespace std;

enum class OpType : uint32_t {
    EMPTY = 0,
    VAR = 1,
    ADD = 2,
    SUB = 3,
    MUL = 4,
    DIV = 5,
};

namespace {

//===----------------------------------------------------------------------===//
// Implementation of a simple MLIR pass from ModuleOp.

class MLIRPassImpl{

public:

    MLIRPassImpl(mlir::MLIRContext &context, mlir::ModuleOp &module, llvm::raw_fd_ostream &os) : context(&context), theModule(&module), os(os) {};
    
    void mlirPass(){

        indent = 0;
        argCnt = 1;
        tmpVarCnt = 1;
        qVarCnt = 1;
        tmpVarHead = "tmp_i_";
        qVarHead = "tmp_q_";
        nowBlockOp = "";
        funcName = "";
        
        //printOperation(theModule->getOperation());
        isqtool.initTools(context);
        initGate();
        openQasmHead();
        visitOperation(theModule->getOperation());
    }

private:

    mlir::MLIRContext* context;
    mlir::ModuleOp* theModule;
    llvm::raw_fd_ostream& os;

    int indent;
    int argCnt;
    int tmpVarCnt;
    int qVarCnt;
    bool hasRes;
    string nowBlockOp;
    string funcName;
    string tmpVarHead;
    string qVarHead;
    
    set<string> baseGate;
    map<string, string> gateMap;
    set<size_t> argSet;
    map<size_t, tuple<OpType, int, string>> symbolTable;
    map<size_t, tuple<bool, int, int>> ctrlGate;

    isqTools isqtool;

    void initGate(){
        baseGate = {"H", "X", "Y", "Z", "S", "T", "CZ", "CX", "CNOT"};
        for (auto &gate: baseGate){
            string newgate = gate;
            transform(gate.begin(), gate.end(), newgate.begin(), [](unsigned char c) { return tolower(c); });
            gateMap[gate] = newgate;
        }
        gateMap["CNOT"] = "cx";
    }
    
    //
    void printOperation(mlir::Operation *op){
        
        printIndent() << "visit op: " << op->getName() << " with " << op->getNumOperands() << " operands and "
            << op->getNumResults() << " results\n";
        
        if (!op->getAttrs().empty()){
            printIndent() << op->getAttrs().size() << " attributes: \n";
            for (mlir::NamedAttribute attr : op->getAttrs()){
                printIndent() << " - '" << attr.first << "' : " << attr.second << "'\n";
            }
        }

        for (auto indexOperand : llvm::enumerate(op->getOperands())){
            auto operand = indexOperand.value();
            
            printIndent() << " - Operand " << indexOperand.index() << " has hash code: " << mlir::hash_value(operand) << "\n";
            if (operand.getDefiningOp()){
                printIndent() << "    from " << operand.getDefiningOp()->getName() << "\n";    
            }
        }

        for (auto indexResult : llvm::enumerate(op->getResults())){
            auto result = indexResult.value();
            printIndent() << " - Result " << indexResult.index() << " type is: " << result.getType() << ", has hash code: " << mlir::hash_value(result) << "\n";
            if (result.use_empty()){
                printIndent() << "  has no use\n";
            }
            else{
                if (result.hasOneUse()){
                    printIndent() << "  has a single use:\n";
                }else{
                    printIndent() << "  has " << std::distance(result.getUses().begin(), result.getUses().end()) << " uses:\n";
                }
                for (mlir::Operation *userOp : result.getUsers()){
                    printIndent() << "    - " << userOp->getName() << "\n";
                }
            }
        }
        
        printIndent() << " " << op->getNumRegions() << " nested regions\n";
        
        auto indent = pushIndent();
        
        for (auto &r:op->getRegions()){
            printRegion(r);
        }
    }

    void printRegion(mlir::Region& region){
        printIndent() << "Region with " << region.getBlocks().size() << " blocks:\n";
        
        auto indent = pushIndent();

        for (auto &b : region.getBlocks()){
            printBlock(b);
        }
    }

    void printBlock(mlir::Block& block){
        
        printIndent() << "Block with " << block.getNumArguments() << " arguments, "
                    << block.getNumSuccessors() << " successors, and "
                    << block.getOperations().size() << " operations\n";

        for (auto &arg: block.getArguments()){
            printIndent() << " - Arg has code: " << mlir::hash_value(arg) << "\n";
            if (arg.use_empty()){
                printIndent() << "  has no use\n";
            }
            else{
                if (arg.hasOneUse()){
                    printIndent() << "  has a single use:\n";
                }else{
                    printIndent() << "  has " << std::distance(arg.getUses().begin(), arg.getUses().end()) << " uses:\n";
                }
                for (mlir::Operation *userOp : arg.getUsers()){
                    printIndent() << "    - " << userOp->getName() << "\n";
                }
            }
        }
        
        auto indent = pushIndent();

        for (auto &p : block.getOperations()){
            printOperation(&p);
        }

    }
    //

    void visitOperation(mlir::Operation *op){
        
        if (succeeded(updateSymbolUseOperation(op))){
            
            string preOp = nowBlockOp;

            nowBlockOp = op->getName().getStringRef().str();
            
            int cnt = 0;
            for (auto &r:op->getRegions()){
                if (nowBlockOp == "affine.if"){
                    os << " {\n";
                    indent += 1;
                }

                if (nowBlockOp == "scf.while" && cnt == 1){
                    os << " {\n";
                    indent += 1;
                }
                visitRegion(r);

                if (nowBlockOp == "scf.while" && cnt == 1){
                    indent -= 1;
                    openQasmNewLine();
                    os << "}\n";
                }
                
                if (nowBlockOp == "affine.if"){
                    indent -=1;
                    openQasmNewLine();
                    os << "}";
                    if (cnt == 0 && op->getNumRegions() == 2){
                        os << "else";
                    }else{
                        os << "\n";
                    }
                }

                cnt += 1;
            }

            nowBlockOp = preOp;
        }
    }

    void visitRegion(mlir::Region& region){
        

        for (auto &b : region.getBlocks()){
            visitBlock(b);
        }
    }

    void visitBlock(mlir::Block& block){
        
        updateSymbolUseBlock(block);

    }
    
    struct RttiIndent{
        int& indent;
        RttiIndent(int &indent): indent(indent){};
        ~RttiIndent() {indent -= 1;};
    };

    RttiIndent pushIndent(){
        return RttiIndent(++indent);
    }

    llvm::raw_ostream &printIndent() {
        for (int i = 0; i < indent; ++i)
            os << "    ";
        return os;
    }

    void openQasmHead(){
        os << "OpenQasm 3.0;\ninclude \"stdgates.inc\";\n\n";
    }

    void openQasmNewLine(){
        for (int i = 0; i < indent; ++i)
            os << "    ";
    }

    void openQasmVarDefine(string name, string type, int size){
        openQasmNewLine();
        os << type;
        if (size > 1){
            os << "[" << size << "]";
        }
        os << " " << name << ";\n";
    }

    void openQasmGateDefine(string name, int shape){
        openQasmNewLine();
        os << "gate "+ name;
        for (int i = 0; i < shape; i++){
            char tmp = 'a'+i;
            if (i > 0)
                os << ",";
            os << " " << tmp;
        }
        os << " {...};\n";
    }

    void openQasmFunc(string name, vector<tuple<string, string, int>> &arglist){
        
        openQasmNewLine();
        os << "def " << name << "(";
        for (int i = 0; i < arglist.size(); i++){
            if (i > 0)
                os << ", ";
            os << get<1>(arglist[i]);
            if (get<2>(arglist[i]) > 1){
                os << "[" << get<2>(arglist[i]) << "]";
            }
            os << " " << get<0>(arglist[i]);
        }
        os << ")";
        if (hasRes){
            os << " -> int";
        }
        os << " {\n";
    }

    void openQasmAssign(string lval, string rval){
        openQasmNewLine();
        os << lval << " = " << rval << ";\n";
    }

    string openQasmGate(string name){
        if (baseGate.count(name) == 0)
            return name;

        return gateMap[name];
    }

    void openQasmUnitary(string name, bool inv, int ctrl, int nctrl, vector<string>& qlist){
        openQasmNewLine();
        if (inv)
            os << "inv @ ";
        if (ctrl > 0){
            os << "ctrl";
            if (ctrl > 1){
                os << "(" << ctrl << ")";
            }
            os << " @ ";
        }
        if (nctrl > 0){
            os << "negctrl";
            if (nctrl > 1){
                os << "(" << nctrl << ")";
            }
            os << " @ ";
        }
        os << openQasmGate(name) << " ";
        for (int i = 0; i < qlist.size(); i++){
            if (i > 0)
                os << ", ";
            os << qlist[i];
        }
        os << ";\n";
    }

    void openQasmCall(string callstr){
        openQasmNewLine();
        os << callstr << ";\n";
    }

    void openQasmReturn(string res){
        openQasmNewLine();
        os << "return " << res << ";\n";
    }

    void openQasmIf(string lval, string rval, string op){
        openQasmNewLine();
        os << "if (" << lval << " " << op << " " << rval << ")";
    }

    void openQasmFor(string arg, string lval, string rval){
        openQasmNewLine();
        os << "for " << arg << " in [" << lval << ":" << rval << "] {\n";
    }

    void openQasmWhile(string cond){
        openQasmNewLine();
        os << "while (" << cond << ")";
    }

    mlir::LogicalResult updateSymbolUseBlock(mlir::Block &block){

        set<size_t> block_args;

        if (nowBlockOp == "builtin.func" && funcName != "main"){
            
            vector<tuple<string, string, int>> arglist;

            for (auto &arg: block.getArguments()){
                
                auto type = arg.getType();
                int shape = 1;
                string var_type = "int";
                if (type.isa<mlir::MemRefType>()){
                    shape = type.dyn_cast<mlir::MemRefType>().getShape()[0];
                    if (type.dyn_cast<mlir::MemRefType>().getElementType().isa<QStateType>()){
                        var_type = "qubit";
                    }
                }else if (type.isa<QStateType>()){
                    var_type = "qubit";
                }

                arglist.push_back(make_tuple("arg"+to_string(argCnt), var_type, shape));

                size_t code = size_t(mlir::hash_value(arg));
                argSet.insert(code);
                block_args.insert(code);
                symbolInsert(code, OpType::VAR, shape, "arg"+to_string(argCnt++));
            }

            openQasmFunc(funcName, arglist);
            indent += 1;
        }else if (nowBlockOp == "affine.for"){
    
            symbolInsert(size_t(mlir::hash_value(block.getArgument(0))), OpType::VAR, 1, tmpVarHead+to_string(tmpVarCnt++));
            indent += 1;
        }

        for (auto &p : block.getOperations()){
            visitOperation(&p);
        }

        if (nowBlockOp == "builtin.func" && funcName != "main"){
            indent -= 1;
            for (auto code: block_args){
                argSet.erase(code);
            }
            openQasmNewLine();
            os << "}\n\n";
        }else if (nowBlockOp == "affine.for"){
            indent -= 1;
            openQasmNewLine();
            os << "}\n";
        }

    }

    mlir::LogicalResult updateSymbolUseOperation(mlir::Operation *op){
        
        string op_name = op->getName().getStringRef().str();
        //os << op_name << "; ";
        if (op_name == "builtin.func"){

            tmpVarCnt = 1;
            qVarCnt = 1;
            argCnt = 1;
            hasRes = false;

            auto attr = op->getAttr(llvm::StringRef("sym_name")).dyn_cast<mlir::StringAttr>();
            
            funcName = attr.getValue().str();

            auto res_type = op->getAttr(llvm::StringRef("type")).dyn_cast<mlir::TypeAttr>().getValue().dyn_cast<mlir::FunctionType>();
            for (auto &type: res_type.getResults()){
                if (type.isa<mlir::IntegerType>()){
                    hasRes = true;
                    break;
                }
            }

        }else if (op_name == "memref.global"){

            auto id = op->getAttr(llvm::StringRef("sym_name")).dyn_cast<mlir::StringAttr>();
            auto type = op->getAttr(llvm::StringRef("type")).dyn_cast<mlir::TypeAttr>().getValue().dyn_cast<mlir::MemRefType>();
            
            string var_type = "int";
            int size = type.getShape()[0];
            if (type.getElementType().isa<QStateType>()){
                var_type = "qubit";
            }

            openQasmVarDefine(id.getValue().str(), var_type, size);


        }else if (op_name == "memref.get_global"){

            auto attr = op->getAttr(llvm::StringRef("name")).dyn_cast<mlir::FlatSymbolRefAttr>();
            //os << "global var: " << attr.getValue().str() << endl;
            auto type = op->getOpResult(0).getType().dyn_cast<mlir::MemRefType>();
            return symbolInsert(size_t(mlir::hash_value(op->getOpResult(0))), OpType::VAR, type.getShape()[0], attr.getValue().str());
        
        }else if (op_name == "std.constant"){
        
            auto attr = op->getAttr(llvm::StringRef("value")).dyn_cast<mlir::IntegerAttr>();
            //os << "value: " << attr.getInt() << endl;
            return symbolInsert(size_t(mlir::hash_value(op->getOpResult(0))), OpType::VAR, 1, to_string(attr.getInt()));
        
        }else if (op_name == "memref.alloc"){
            
            // no use, jump
            auto result = op->getOpResult(0);
            if (result.use_empty()){
                return mlir::success();
            }
            
            auto type = op->getOpResult(0).getType().dyn_cast<mlir::MemRefType>();
            if (type.getShape()[0] == 1){
                // jump this one: single int/qubit variable assign func args.
                mlir::Operation* first_use;
                for (mlir::Operation *userOp : result.getUsers()){
                    first_use = userOp;
                }
                //os << "first user: " << first_use->getName().getStringRef().str() << endl;
                if (first_use->getName().getStringRef().str() == "affine.store"){
                    auto operand = first_use->getOpOperand(0).get();
                    size_t code = size_t(mlir::hash_value(operand));
                    if (argSet.count(code) != 0){
                        auto res = getSymbol(code);
                        return symbolInsert(size_t(mlir::hash_value(op->getOpResult(0))), OpType::VAR, type.getShape()[0], get<2>(res));
                    } 
                }
            }

            if (type.getElementType().isa<QStateType>()){
                openQasmVarDefine(qVarHead+to_string(qVarCnt), "qubit", type.getShape()[0]);
                return symbolInsert(size_t(mlir::hash_value(op->getOpResult(0))), OpType::VAR, type.getShape()[0], qVarHead+to_string(qVarCnt++));
            }else{
                openQasmVarDefine(tmpVarHead+to_string(tmpVarCnt), "int", type.getShape()[0]);
                return symbolInsert(size_t(mlir::hash_value(op->getOpResult(0))), OpType::VAR, type.getShape()[0], tmpVarHead+to_string(tmpVarCnt++));   
            }
        
        }else if (op_name == "affine.load"){

            string loadstr = "";
            for (auto indexOperand : llvm::enumerate(op->getOperands())){
                auto operand = indexOperand.value();
                size_t code = size_t(mlir::hash_value(operand));
                auto res = getSymbol(code);
                //os << "operand " << indexOperand.index() << ": (" << get<0>(res) << ", " << get<1>(res) << ", " << get<2>(res) << ")";
                if (indexOperand.index() == 0){
                    loadstr += get<2>(res);
                    if (get<1>(res) == 1)
                        break;
                }else{
                    loadstr += "[" + get<2>(res) + "]";
                }
            }
            //os << "; loadstr: " << loadstr << endl;
            return symbolInsert(size_t(mlir::hash_value(op->getOpResult(0))), OpType::VAR, 1, loadstr);
        
        }else if (op_name == "affine.store"){

            string lval = "", rval = "";
            for (auto indexOperand : llvm::enumerate(op->getOperands())){
                auto operand = indexOperand.value();
                size_t code = size_t(mlir::hash_value(operand));
                auto res = getSymbol(code);
                //os << ", operand " << indexOperand.index() << ": (" << get<0>(res) << ", " << get<1>(res) << ", " << get<2>(res) << ")";
                
                if (get<0>(res) == OpType::EMPTY){
                    return mlir::failure();
                }
                
                if (indexOperand.index() == 0){                    
                    rval = get<2>(res);
                }else if (indexOperand.index() == 1){

                    if (operand.getType().dyn_cast<mlir::MemRefType>().getElementType().isa<QStateType>()){                         
                        return mlir::success();
                    }

                    lval = get<2>(res);
                    if (get<1>(res) == 1)
                        break;
                }else{
                    lval += "[" + get<2>(res) + "]";
                }
            }
            if (lval != rval)
                openQasmAssign(lval, rval);
    
        }else if (op_name == "std.index_cast"){
        
            size_t opcode = size_t(mlir::hash_value(op->getOperand(0)));
            auto res = getSymbol(opcode);
            return symbolInsert(size_t(mlir::hash_value(op->getOpResult(0))), OpType::VAR, 1, get<2>(res));
    
        }else if (op_name == "std.addi" || op_name == "std.subi" || op_name == "std.muli" || op_name == "std.divi"){
        
            string binastr = "";
            OpType op_t;
            for (auto indexOperand : llvm::enumerate(op->getOperands())){
                auto operand = indexOperand.value();
                size_t code = size_t(mlir::hash_value(operand));
                auto res = getSymbol(code);
                //os << "operand " << indexOperand.index() << ": (" << get<0>(res) << ", " << get<1>(res) << ", " << get<2>(res) << ")";
                if (indexOperand.index() == 0){
                    if ((op_name == "std.muli" || op_name == "std.divi") && (get<0>(res) == OpType::ADD || get<0>(res) == OpType::SUB)){
                        binastr += "(" + get<2>(res) + ")";
                    }else{
                        binastr += get<2>(res);
                    }

                    if (op_name == "std.addi"){
                        binastr += "+";
                        op_t = OpType::ADD;
                    }else if (op_name == "std.subi"){
                        binastr += "-";
                        op_t = OpType::SUB;
                    }else if (op_name == "std.muli"){
                        binastr += "*";
                        op_t = OpType::MUL;
                    }else{
                        binastr += "/";
                        op_t = OpType::DIV;
                    }

                }else{
                    if ((op_name == "std.muli" || op_name == "std.divi") && (get<0>(res) == OpType::ADD || get<0>(res) == OpType::SUB)){
                        binastr += "(" + get<2>(res) + ")";
                    }else{
                        binastr += get<2>(res);
                    }
                }
            }
            //os << "; binastr: " << binastr << endl;
            return symbolInsert(size_t(mlir::hash_value(op->getOpResult(0))), op_t, 1, binastr);
        
        }else if(op_name == "isq.use"){
        
            auto attr = op->getAttr(llvm::StringRef("name")).dyn_cast<mlir::SymbolRefAttr>();
            //os << "use gate: " << attr.getLeafReference().str() << endl;
            return symbolInsert(size_t(mlir::hash_value(op->getOpResult(0))), OpType::VAR, 0, attr.getLeafReference().str());
        
        }else if (op_name == "isq.use_ctrl"){
            
            auto attr = op->getAttr(llvm::StringRef("name")).dyn_cast<mlir::SymbolRefAttr>();
            bool inv = false;
            int ctrl = 0, nctrl = 0;
            for (mlir::NamedAttribute attr : op->getAttrs()){
                if (attr.first.str() == "inv"){
                    inv = true;
                }else if (attr.first.str() == "pos"){
                    ctrl = attr.second.dyn_cast<mlir::IntegerAttr>().getInt();
                }else if (attr.first.str() == "neg"){
                    nctrl = attr.second.dyn_cast<mlir::IntegerAttr>().getInt();
                }
            }

            size_t code = size_t(mlir::hash_value(op->getOpResult(0)));
            if (failed(ctrlGateInsert(code, inv, ctrl, nctrl))){
                emitError(op->getLoc(), "use gate error");
                return mlir::failure();
            }
            //os << "use ctrl gate: " << attr.getLeafReference().str() << ", (" << inv << "," << ctrl << "," << nctrl << ")" << endl;
            return symbolInsert(code, OpType::VAR, 1, attr.getLeafReference().str());
        
        }else if (op_name == "isq.apply"){
        
            vector<string> qlist;
            bool inv = false;
            int ctrl = 0, nctrl = 0;
            string gate_name = "";

            for (auto indexOperand : llvm::enumerate(op->getOperands())){
                auto operand = indexOperand.value();
                size_t code = size_t(mlir::hash_value(operand));
                auto res = getSymbol(code);
                //os << "operand " << indexOperand.index() << ": (" << get<0>(res) << ", " << get<1>(res) << ", " << get<2>(res) << ")";
                if (indexOperand.index() == 0){
                    if (get<1>(res) == 1){
                        auto ctrl_type = getCtrl(code);
                        inv = get<0>(ctrl_type);
                        ctrl = get<1>(ctrl_type);
                        nctrl = get<2>(ctrl_type);
                    }
                    gate_name = get<2>(res);
                    continue;
                }

                qlist.push_back(get<2>(res));
            }
            openQasmUnitary(gate_name, inv, ctrl, nctrl, qlist);
        
        }else if (op_name == "std.call"){

            auto func_name = op->getAttr(llvm::StringRef("callee")).dyn_cast<mlir::FlatSymbolRefAttr>();
            string call_str = func_name.getValue().str() + "(";
            
            for (auto indexOperand : llvm::enumerate(op->getOperands())){
                auto operand = indexOperand.value();
                size_t code = size_t(mlir::hash_value(operand));
                auto res = getSymbol(code);
                if (indexOperand.index() > 0){
                    call_str += ", ";
                }
                call_str += get<2>(res);
            }
            call_str += ")";
            for (auto indexResult : llvm::enumerate(op->getResults())){
                auto result = indexResult.value();
                if (result.getType().isa<mlir::IntegerType>()){
                    if (result.use_empty())
                        break;
                    return symbolInsert(size_t(mlir::hash_value(result)), OpType::VAR, 1, call_str);
                }
            }
            openQasmCall(call_str);

        }else if (op_name == "isq.call_qop"){
            
            string qop_name = op->getAttr(llvm::StringRef("callee")).dyn_cast<mlir::SymbolRefAttr>().getLeafReference().str();
            string call_qop_str = qop_name + " ";

            for (auto indexOperand : llvm::enumerate(op->getOperands())){
                auto operand = indexOperand.value();
                size_t code = size_t(mlir::hash_value(operand));
                auto res = getSymbol(code);
                if (indexOperand.index() > 0){
                    call_qop_str += ", ";
                }
                call_qop_str += get<2>(res);
            }
            
            for (auto indexResult : llvm::enumerate(op->getResults())){
                auto result = indexResult.value();
                if (result.getType().isa<mlir::IntegerType>()){
                    if (result.use_empty())
                        break;
                    return symbolInsert(size_t(mlir::hash_value(result)), OpType::VAR, 1, call_qop_str);
                }
            }

            openQasmCall(call_qop_str);
            
        }else if (op_name == "std.return"){
            
            if (hasRes){
                size_t opcode = size_t(mlir::hash_value(op->getOperand(0)));
                auto res = getSymbol(opcode);
                openQasmReturn(get<2>(res));
            }

        }else if (op_name == "isq.defgate"){

            string gate_name = op->getAttr(llvm::StringRef("sym_name")).dyn_cast<mlir::StringAttr>().getValue().str();
            int shape = op->getAttr(llvm::StringRef("type")).dyn_cast<mlir::TypeAttr>().getValue().dyn_cast<GateType>().getSize();
            
            if (op->hasAttr(llvm::StringRef("definition"))){
                openQasmGateDefine(gate_name, shape);
            }
        }else if (op_name == "affine.if"){

            auto condition = op->getAttr(llvm::StringRef("condition")).dyn_cast<mlir::IntegerSetAttr>().getValue();

            string asso, lval, rval;

            if (condition == isqtool.greateSet){
                asso = ">";
            }else if (condition == isqtool.greateEqualSet){
                asso = ">=";
            }else if (condition == isqtool.equalSet){
                asso = "==";
            }else if (condition == isqtool.lessEqualSet){
                asso = "<=";
            }else{
                asso = "<";
            }

            for (auto indexOperand : llvm::enumerate(op->getOperands())){
                auto operand = indexOperand.value();
                size_t code = size_t(mlir::hash_value(operand));
                auto res = getSymbol(code);
                if (indexOperand.index() == 0){
                    lval = get<2>(res);
                }else{
                    rval = get<2>(res);
                }
            }
            openQasmIf(lval, rval, asso);

        }else if (op_name == "affine.for"){

            string lval, rval;
            int idx = 0;

            auto lmap = op->getAttr(llvm::StringRef("lower_bound")).dyn_cast<mlir::AffineMapAttr>().getValue();
            if (lmap != isqtool.singleSymbol){
                lval = to_string(lmap.getSingleConstantResult());
            }else{
                size_t opcode = size_t(mlir::hash_value(op->getOperand(idx)));
                auto res = getSymbol(opcode);
                lval = get<2>(res);
                idx += 1;
            }

            auto rmap = op->getAttr(llvm::StringRef("upper_bound")).dyn_cast<mlir::AffineMapAttr>().getValue();
            if (rmap != isqtool.singleSymbol){
                rval = to_string(rmap.getSingleConstantResult());
            }else{
                size_t opcode = size_t(mlir::hash_value(op->getOperand(idx)));
                auto res = getSymbol(opcode);
                rval = get<2>(res);
                idx += 1;
            }
            
            openQasmFor(tmpVarHead+to_string(tmpVarCnt), lval, rval);
            

        }else if (op_name == "scf.while"){

            return mlir::success();
        }else if (op_name == "std.cmpi"){

            int pred = op->getAttr(llvm::StringRef("predicate")).dyn_cast<mlir::IntegerAttr>().getInt();
            string asso = "";
            switch (pred)
            {
            case 0:
                asso = " == ";
                break;
            case 1:
                asso = " != ";
                break;
            case 2:
                asso = " < ";
                break;
            case 3:
                asso = " <= ";
                break;
            case 4:
                asso = " > ";
                break;
            case 5:
                asso = " >= ";
                break;
            default:
                break;
            }
            
            string cmp_str = "";
            for (auto indexOperand : llvm::enumerate(op->getOperands())){
                auto operand = indexOperand.value();
                size_t code = size_t(mlir::hash_value(operand));
                auto res = getSymbol(code);
                cmp_str += get<2>(res);
                if (indexOperand.index() == 0){
                    cmp_str += asso;
                }
            }

            return symbolInsert(size_t(mlir::hash_value(op->getOpResult(0))), OpType::VAR, 1, cmp_str);

        }else if (op_name == "scf.condition"){
            
            auto operand = op->getOpOperand(0).get();
            auto res = getSymbol(size_t(mlir::hash_value(operand)));
            openQasmWhile(get<2>(res));
        }

        return mlir::success();
    }

    mlir::LogicalResult symbolInsert(size_t code, OpType type, int shape, string str){
        
        auto iter = symbolTable.find(code);
        if (iter != symbolTable.end()){
            return mlir::failure();
        }
        
        symbolTable.insert(make_pair(code, make_tuple(type, shape, str)));
        return mlir::success();
    }

    tuple<OpType, int, string> getSymbol(size_t code){
        auto iter = symbolTable.find(code);
        if (iter != symbolTable.end()){
            return iter->second;
        }

        return make_tuple(OpType::EMPTY, 0, "");
    }

    mlir::LogicalResult ctrlGateInsert(size_t code, bool inv, int ctrl, int nctrl){
        
        auto iter = ctrlGate.find(code);
        if (iter != ctrlGate.end()){
            return mlir::failure();
        }

        ctrlGate.insert(make_pair(code, make_tuple(inv, ctrl, nctrl)));
        return mlir::success();
    }

    tuple<bool, int, int> getCtrl(size_t code){
        auto iter = ctrlGate.find(code);
        if (iter != ctrlGate.end()){
            return iter->second;
        }

        return make_tuple(false, -1, -1);
    }
}; 

}


namespace isq {

    void mlirPass(mlir::MLIRContext &context, mlir::ModuleOp &module, llvm::raw_fd_ostream &os) {
        MLIRPassImpl(context, module, os).mlirPass();
    }
}
*/