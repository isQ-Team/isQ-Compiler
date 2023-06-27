#include <iostream>
#include <map> 
#include <set>
#include <vector>
#include <algorithm>

#include "isq/Dialect.h"
#include <isq/utils/DispatchOperation.h>
#include <isq/IR.h>

#include "isq/Operations.h"
#include "isq/QAttrs.h"
#include "isq/utils/Decomposition.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"


#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"

#include "llvm/Support/raw_ostream.h"
#include "isq/Backends.h"
#include "mlir/AsmParser/AsmParser.h"

#define TRY(x) if(mlir::failed(x)) return mlir::failure();

using namespace isq::ir;
using namespace isq;
using std::vector;
using std::string;
using std::map;
using std::set;
using std::to_string;
using std::tuple;
using std::get;
using std::make_tuple;
using std::pair;
using std::make_pair;
//using namespace std;

enum class OpType : uint32_t {
    EMPTY = 0,
    VAR = 1,
    ADD = 2,
    SUB = 3,
    MUL = 4,
    DIV = 5,
};


namespace isq{
namespace ir{

namespace details{
using namespace std;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::memref;
using namespace mlir::scf;
using CodegenOpVisitor = OpVisitor<
    mlir::func::FuncOp, scf::IfOp, scf::ForOp, scf::ExecuteRegionOp,
    GetGlobalOp, GlobalOp,
    mlir::arith::ConstantOp, mlir::arith::ExtUIOp, mlir::arith::IndexCastOp, mlir::arith::SIToFPOp,
    AllocOp, DeallocOp, memref::LoadOp, memref::StoreOp, SubViewOp, memref::CastOp, CmpIOp,
    AddIOp, SubIOp, MulIOp, DivSIOp, RemSIOp, AddFOp, SubFOp, MulFOp, DivFOp, NegFOp, AndIOp,
    UseGateOp, DecorateOp, ApplyGateOp, CallQOpOp, AccumulateGPhase, DeclareQOpOp, AssertOp,
    mlir::func::CallOp, mlir::func::ReturnOp, DefgateOp, scf::WhileOp, scf::ConditionOp,
    ModuleOp, PassOp, AffineYieldOp, scf::YieldOp,
    mlir::cf::CondBranchOp, mlir::cf::BranchOp, AffineLoadOp, AffineStoreOp
    >;
}



/// Code generator from isQ MLIR Dialect to logical OpenQASM 3.0.
/// https://arxiv.org/pdf/2104.14722.pdf
class MLIRPassImpl: public details::CodegenOpVisitor{
public:
    MLIRPassImpl(mlir::MLIRContext &context, mlir::ModuleOp &module, llvm::raw_ostream &os) : context(&context), theModule(&module), os(os) {};
    

    mlir::LogicalResult mlirPass(){

        indent = 0;
        argCnt = 1;
        tmpVarCnt = 1;
        qVarCnt = 1;
        tmpVarHead = "tmp_c_";
        qVarHead = "tmp_q_";
        nowBlockOp = "";
        funcName = "";
        
        //printOperation(theModule->getOperation());
        //isqtool.initTools(context);
        initGate();
        initializeIntegerSets();

        openQasmHead();

        TRY(traverseOperation(theModule->getOperation()));
        buildCallDic();
        

        for (auto f: callfunc){
            nowfunc = f;
            if (nowfunc == "__isq__main") continue;
            TRY(visitOp(funcMap[nowfunc].second));
        }
        
        nowfunc = "__isq__main";
        TRY(visitOp(funcMap[nowfunc].second));
        return mlir::success();
    }

private:
    mlir::IntegerSet eq, slt, sle, sgt, sge;
    mlir::AffineMap singleSymbol;
    void initializeIntegerSets(){
        eq = mlir::parseIntegerSet("()[s0, s1]: (s0-s1 == 0)", context);
        // s0-s1<0 s0-s1+1<=0
        slt = mlir::parseIntegerSet("()[s0, s1]: (s1-s0-1 >= 0)", context);
        // s0-s1<=0
        sle = mlir::parseIntegerSet("()[s0, s1]: (s1-s0 >= 0)", context);
        // s0-s1>0 s0-s1-1>=0
        sgt = mlir::parseIntegerSet("()[s0, s1]: (s0-s1-1 >= 0)", context);
        // s0-s1>=0
        sge = mlir::parseIntegerSet("()[s0, s1]: (s0-s1 >= 0)", context);
        auto s0 = getAffineSymbolExpr(0, context);
        singleSymbol = mlir::AffineMap::get(0, 1, {s0}, context);
    }
    mlir::MLIRContext* context;
    mlir::ModuleOp* theModule;
    llvm::raw_ostream& os;

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


    map<string, pair<set<string>, mlir::func::FuncOp>> funcMap;
    map<string, int> globalSymbol;
    set<string> callfunc;
    string nowfunc;

    size_t yieldRes;
    string whileCond;

    //isqTools isqtool;

    // Initialize predefined gates in OpenQASM 3.0.
    void initGate(){
        baseGate = {"H", "X", "Y", "Z", "S", "T", "CZ", "CX", "CNOT"};
        for (auto &gate: baseGate){
            string newgate = gate;
            transform(gate.begin(), gate.end(), newgate.begin(), [](unsigned char c) { return tolower(c); });
            gateMap[gate] = newgate;
        }
        gateMap["CNOT"] = "cx";
    }
    
    // Printing operation info to output stream. For debugging.
    void printOperation(mlir::Operation *op){
        
        printIndent() << "visit op: " << op->getName() << " with " << op->getNumOperands() << " operands and "
            << op->getNumResults() << " results\n";
        
        if (!op->getAttrs().empty()){
            printIndent() << op->getAttrs().size() << " attributes: \n";
            for (mlir::NamedAttribute attr : op->getAttrs()){
                printIndent() << " - '" << attr.getName() << "' : " << attr.getValue() << "'\n";
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

    // Printing region info to output stream. For debugging.
    void printRegion(mlir::Region& region){
        printIndent() << "Region with " << region.getBlocks().size() << " blocks:\n";
        
        auto indent = pushIndent();

        for (auto &b : region.getBlocks()){
            printBlock(b);
        }
    }

    // Printing block info to output stream. For debugging.
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

    mlir::LogicalResult traverseRegion(mlir::Region& region){

        for (auto &b : region.getBlocks()){
            TRY(traverseBlock(b));
        }
        return mlir::success();
    }

    mlir::LogicalResult traverseBlock(mlir::Block& block){
        for (auto &p : block.getOperations()){
            auto fop = mlir::dyn_cast_or_null<mlir::func::FuncOp>(&p);
            if (fop != nullptr){
                set<string> funcs;
                nowfunc = fop.getSymName().str();
                funcMap.insert(make_pair(nowfunc, make_pair(funcs, fop)));
            }
            TRY(traverseOperation(&p));
        }
        return mlir::success();
    }

    mlir::LogicalResult getVarType(mlir::Type ty, string& var_type, mlir::Location loc){

        var_type = "int";
        if (ty.isa<QStateType>()){
            var_type = "qubit";
        }else if (ty.isa<mlir::IndexType>() || ty.isa<mlir::IntegerType>()){
            auto ity = ty.dyn_cast_or_null<mlir::IntegerType>();
            if (ity != nullptr){
                auto width = ity.getWidth();
                if (width == 1) var_type = "bool";
            }
        }else if (ty.isa<mlir::Float64Type>()){
            var_type = "float";
        }else{
            return error(loc, "invalid type in OpenQASM3");
        }
        return mlir::success();
    }

    mlir::LogicalResult traverseOperation(mlir::Operation* op){
        auto callop = mlir::dyn_cast_or_null<mlir::func::CallOp>(op);
        if (callop != nullptr){
            auto name = callop.getCallee().str();
            funcMap[nowfunc].first.insert(name);
        }

        auto globalop = mlir::dyn_cast_or_null<mlir::memref::GlobalOp>(op);
        if (globalop != nullptr){
            auto name = globalop.getSymName().str();
            if (name != ".__qmpi_rank") {
                auto var_name = getQasmName(name);
                int size = globalop.getType().getShape()[0];
                string var_type;
                TRY(getVarType(globalop.getType().getElementType(), var_type, op->getLoc()))
                openQasmVarDefine(var_name, var_type, size);
            }
        }

        for (auto &r:op->getRegions()){
            TRY(traverseRegion(r));
        }
        return mlir::success();
    }

    void buildCallDic(){
        callfunc.insert("__isq__main");
        set<string> now = {"__isq__main"};
        while (!now.empty()){
            set<string> tmp;
            for (auto fn: now){
                for (auto nfn: funcMap[fn].first){
                    if (callfunc.count(nfn) == 0){
                        tmp.insert(nfn);
                        callfunc.insert(nfn);
                    }
                }
            }
            now = tmp;
        }
    }

    mlir::LogicalResult error(mlir::Location loc, string msg){
        emitError(loc, msg);
        return mlir::failure();
    }

    string getGateName(string gate){
        int n = gate.size();
        int pos = 0;
        for(int i = n-1; i>=0; i--){
            if (gate[i] == '_'){
                pos = i+1;
                break;
            }
        }
        string gn = gate.substr(pos, n-pos);
        string gn_up = gn;
        transform(gn.begin(), gn.end(), gn_up.begin(), ::tolower);
        return gn_up;
    }

    string getQasmName(string func){
        int n = func.size();
        int pos = 0;
        for(int i = n-1; i>=0; i--){
            if (func[i] == '.'){
                pos = i+1;
                break;
            }
        }
        if (func[pos] == '$') pos += 1;
        return func.substr(pos, n-pos);
    }

    // First we handle blockful ops, i.e. operations with blocks.
    // Modules, If, While, For, Function.
    mlir::LogicalResult visitBlock(mlir::Block* block){
        if(!block) return mlir::success();
        for(auto& child: block->getOperations()){
            TRY(visitOperation(&child));
        }
        return mlir::success();
    }
    mlir::LogicalResult visitOp(mlir::ModuleOp curr_module) override{
        auto mod_name = curr_module.getSymName();
        /*
        if(mod_name){
            if(*mod_name == "isq_builtin"){
                // Just ignore it.
                return mlir::success();
            }
        }
        */
        TRY(visitBlock(curr_module.getBody()));
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::scf::IfOp if_stmt) override{
        // update_symbol_use_operation
        
        auto condition = if_stmt.getCondition();

        openQasmIf(get<2>(getSymbol(condition)));
        os << "{\n";
        indent += 1;
        TRY(visitBlock(if_stmt.thenBlock()));
        // update_symbol_use_operation
        indent-=1;
        printIndent()<<"}";
        if(if_stmt.elseBlock()){
            printIndent()<<"else{\n";
            indent++;
            TRY(visitBlock(if_stmt.elseBlock()));
            indent--;
            printIndent()<<"}";
        }
        os << '\n';
        
        return mlir::success();
    }
    
    mlir::LogicalResult visitOp(mlir::scf::WhileOp while_stmt) override{
        // Generate while-statement using manual guard.
        auto cond_block = while_stmt.getBefore().getBlocks().begin();
        TRY(visitBlock(&*cond_block));
        openQasmWhile(whileCond);
        os << "{\n";
        indent++;
        auto while_body = while_stmt.getAfter().getBlocks().begin();
        TRY(visitBlock(&*while_body));
        indent--;
        printIndent()<<"}\n";
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::scf::ExecuteRegionOp op) override{
        
        int cnt = 0;
        for (auto pb = op.getRegion().getBlocks().begin(); pb != op.getRegion().getBlocks().end(); ++pb){
            if (cnt == 1){
                TRY(visitBlock(&*pb));
                auto var = getSymbol(yieldRes);
                TRY(symbolInsert(mlir::hash_value(op->getOpResult(0)), OpType::VAR, 1, get<2>(var)));
            }
            cnt += 1;
        }
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::scf::ConditionOp op) override{
        auto var = getSymbol(op.getCondition());
        if (get<0>(var) != OpType::VAR){
            op->emitError("get condition failed");
        }
        whileCond = get<2>(var);
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::func::FuncOp func_op) override{
        // update_symbol_use_operation
        os << "\n";
        tmpVarCnt = 1;
        qVarCnt = 1;
        argCnt = 1;
        hasRes = false;
        string rty;

        auto attr = func_op.getSymNameAttr();
        
        funcName = getQasmName(attr.getValue().str());

        auto res_type = func_op.getFunctionType();

        for (auto &type: res_type.getResults()){
            if (mlir::succeeded(getVarType(type, rty, func_op.getLoc()))){
                if (rty != "qubit"){
                    hasRes = true;
                    break;
                }
            }
        }
        // update_region
        auto not_main = func_op.getSymName() != "__isq__main";
        set<size_t> block_args;
        auto& func_body = func_op.getBody();
        /*
        if(!func_body.hasOneBlock()) {
            func_op.emitError("OpenQASM codegen only supports functions with exactly one basic block.");
            return mlir::failure();
        }*/
        auto func_block = func_body.getBlocks().begin();
        if(not_main){
            vector<tuple<string, string, int>> arglist;
            for (auto &arg: func_block->getArguments()){
                
                auto type = arg.getType();
                int shape = 1;
                string var_type = "int";
                if (type.isa<mlir::MemRefType>()){
                    shape = type.dyn_cast<mlir::MemRefType>().getShape()[0];
                    if (shape == -1){
                        func_op.emitError("OpenQASM3 codegen only supports parameter with exactly size.");
                        return mlir::failure();
                    }
                    type = type.dyn_cast<mlir::MemRefType>().getElementType();
                }
                TRY(getVarType(type, var_type, func_op.getLoc()));
                arglist.push_back(make_tuple("arg"+to_string(argCnt), var_type, shape));

                size_t code = size_t(mlir::hash_value(arg));
                argSet.insert(code);
                block_args.insert(code);
                TRY(symbolInsert(code, OpType::VAR, shape, "arg"+to_string(argCnt++)));
            }

            openQasmFunc(funcName, arglist, rty);
            
            indent += 1;
        }

        for (auto &block: func_op.getBlocks()){
            TRY(visitBlock(&block));
        }

        if(not_main){
            indent -= 1;
            for (auto code: block_args){
                argSet.erase(code);
            }
            openQasmNewLine();
            os << "}\n";
        }
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::cf::CondBranchOp) override{
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::scf::ForOp for_stmt) override{
        
        string lval = get<2>(getSymbol(for_stmt.getLowerBound()));
        string rval = get<2>(getSymbol(for_stmt.getUpperBound()));
        string step = get<2>(getSymbol(for_stmt.getStep()));
        
        openQasmFor("arg"+to_string(argCnt), lval, rval, step);

        // update_symbol_use_block
        auto block = for_stmt.getLoopBody().getBlocks().begin();
        if((*block).getNumArguments()!=1){
            for_stmt->emitError("for-loop with more than 1 arguments (a.k.a. for-body yielding) is not supported");
            return mlir::failure();
        }
        TRY(symbolInsert(size_t(mlir::hash_value((*block).getArgument(0))), OpType::VAR, 1, "arg"+to_string(argCnt++)));
        indent += 1;
        TRY(visitBlock(&*block));
        indent -= 1;
        printIndent() << "}\n";
        
        return mlir::success();
    }
    // Then we handle blockless operations.

    std::string next_tempInt(){
        auto id = tmpVarCnt++;
        auto name = tmpVarHead + to_string(id);
        openQasmVarDefine(name, "int", 1);
        return name;
    }
    std::string next_tempBit(){
        auto id = tmpVarCnt++;
        auto name = tmpVarHead + to_string(id);
        openQasmVarDefine(name, "bit", 1);
        return name;
    }
    std::string next_tempName(){
        auto id = tmpVarCnt++;
        auto name = tmpVarHead + to_string(id);
        return name;
    }
    
    /*
    mlir::LogicalResult visitOp(mlir::memref::GlobalOp op) override{
        auto id = op.getSymName();
        auto type = op.getType();
        string var_type = "int";
        int size = type.getShape()[0];
        if (type.getElementType().isa<QStateType>()){
            var_type = "qubit";
        }
        openQasmVarDefine(id.str(), var_type, size);
        return mlir::success();
    }*/

    mlir::LogicalResult visitOp(mlir::memref::GetGlobalOp op) override{
        auto attr = op.getNameAttr();
        //os << "global var: " << attr.getValue().str() << endl;
        auto type = op.getResult().getType().dyn_cast<mlir::MemRefType>();
        return symbolInsert(size_t(mlir::hash_value(op.getResult())), OpType::VAR, type.getShape()[0], getQasmName(attr.getValue().str()));
    }
    mlir::LogicalResult visitOp(mlir::arith::ConstantOp op) override{
        auto ci_attr = op.getValueAttr().dyn_cast<mlir::IntegerAttr>();
        if (ci_attr != nullptr){
            return symbolInsert(size_t(mlir::hash_value(op->getOpResult(0))), OpType::VAR, 1, to_string(ci_attr.getInt()));
        }
        auto fi_attr = op.getValueAttr().dyn_cast<mlir::FloatAttr>();
        if (fi_attr != nullptr){
            return symbolInsert(size_t(mlir::hash_value(op->getOpResult(0))), OpType::VAR, 1, to_string(fi_attr.getValueAsDouble()));
        }
        op->emitError("invalid value type in OpenQASM3");
    }
    mlir::LogicalResult visitOp(mlir::memref::AllocOp op) override{
        // no use, jump
        auto result = op.getResult();
        if (result.use_empty()){
            return mlir::success();
        }
        
        auto type = op.getResult().getType().dyn_cast<mlir::MemRefType>();
        if (type.getShape()[0] == 1){
            // jump this one: single int/qubit variable assign func args.
            mlir::Operation* first_use;
            for (mlir::Operation *userOp : result.getUsers()){
                first_use = userOp;
            }
            //os << "first user: " << first_use->getName().getStringRef().str() << endl;
            if (auto first_store = llvm::dyn_cast<mlir::AffineStoreOp>(first_use)){
                auto operand = first_use->getOperand(0);
                size_t code = size_t(mlir::hash_value(operand));
                if (argSet.count(code) != 0){
                    auto res = getSymbol(code);
                    return symbolInsert(size_t(mlir::hash_value(op->getOpResult(0))), OpType::VAR, type.getShape()[0], get<2>(res));
                } 
            }
        }

        string var_type;
        string var_name;
        TRY(getVarType(type.getElementType(), var_type, op->getLoc()));
        
        if (type.getElementType().isa<QStateType>()){
            var_name = qVarHead+to_string(qVarCnt++);
        }else{
            var_name = tmpVarHead+to_string(tmpVarCnt++);      
        }
        openQasmVarDefine(var_name, var_type, type.getShape()[0]);
        return symbolInsert(size_t(mlir::hash_value(op->getOpResult(0))), OpType::VAR, type.getShape()[0], var_name);
    }

    mlir::LogicalResult visitOp(mlir::AffineLoadOp op) override{
        string loadstr = "";
        for (auto indexOperand : llvm::enumerate(op->getOperands())){
            auto operand = indexOperand.value();
            size_t code = size_t(mlir::hash_value(operand));
            auto res = getSymbol(code);
            //std::cout << "operand " << indexOperand.index() << ": (" << get<1>(res) << ", " << get<2>(res) << ")\n";
            if (indexOperand.index() == 0){
                loadstr += get<2>(res);
                //if (get<1>(res) == 1)
                //    break;
            }
        }
        /*
        if(op->getOperands().size()==2){
            if(op.getAffineMap().isConstant()){
                loadstr += "[" + to_string(op.getAffineMap().getConstantResults()[0]) + "]";
            }
        }*/
        //std::cout << "loadstr: " << loadstr << "\n";
        return symbolInsert(size_t(mlir::hash_value(op.getResult())), OpType::VAR, 1, loadstr);
    }
    mlir::LogicalResult visitOp(mlir::AffineStoreOp op) override{
        string lval = "", rval = "";
        for (auto indexOperand : llvm::enumerate(op->getOperands())){
            auto operand = indexOperand.value();
            
            auto type = operand.getType().dyn_cast<mlir::MemRefType>();
            if (type != nullptr) if (type.getElementType().isa<QStateType>()) return mlir::success();

            size_t code = size_t(mlir::hash_value(operand));
            auto res = getSymbol(code);
            //os << ", operand " << indexOperand.index() << ": (" << get<1>(res) << ", " << get<2>(res) << ")";

            /*
            if (get<0>(res) == OpType::EMPTY){
                op->emitError("Can't backtrace operand ")<<operand;
                return mlir::failure();
            }*/
            
            if (indexOperand.index() == 0){                    
                rval = get<2>(res);
            }else if (indexOperand.index() == 1){
                lval = get<2>(res);
                //if (get<1>(res) == 1)
                //    break;
            }else{
                if (get<1>(res) > 1){
                    lval += "[" + get<2>(res) + "]";
                }
            }
        }

        /*
        if(op->getOperands().size()==2){
            if(op.getAffineMap().isConstant()){
                lval += "[" + to_string(op.getAffineMap().getConstantResults()[0]) + "]";
            }
        }*/

        if (lval != rval)
            openQasmAssign(lval, rval);
        return mlir::success();
    }
    mlir::LogicalResult visitOp(mlir::arith::IndexCastOp op) override{
        auto res = getSymbol(op.getIn());
        return symbolInsert(size_t(mlir::hash_value(op.getOut())), OpType::VAR, 1, get<2>(res));
    }
    mlir::LogicalResult visitOp(mlir::arith::ExtUIOp op) override{
        auto in_var = getSymbol(op.getIn());
        return symbolInsert(size_t(mlir::hash_value(op.getOut())), OpType::VAR, 1, get<2>(in_var));
    }
    mlir::LogicalResult visitOp(mlir::arith::SIToFPOp op) override{
        auto in_var = getSymbol(op.getIn());
        return symbolInsert(size_t(mlir::hash_value(op.getOut())), OpType::VAR, 1, get<2>(in_var));
    }
    mlir::LogicalResult visitBinaryOp(OpType op_t, char op, mlir::Value lhs, mlir::Value rhs, mlir::Value ret){
        auto lhs_s = get<2>(getSymbol(mlir::hash_value(lhs)));
        auto rhs_s = get<2>(getSymbol(mlir::hash_value(rhs)));
        //auto temp = next_tempInt();
        //openQasmAssign(temp, lhs_s+op+rhs_s);
        return symbolInsert(ret, op_t, 1, "(" + lhs_s + op + rhs_s + ")");
    }
    mlir::LogicalResult visitOp(mlir::arith::AddIOp op) override{
        return visitBinaryOp(OpType::ADD, '+', op.getLhs(), op.getRhs(), op.getResult());
    }
    mlir::LogicalResult visitOp(mlir::arith::AddFOp op) override{
        return visitBinaryOp(OpType::ADD, '+', op.getLhs(), op.getRhs(), op.getResult());
    }
    mlir::LogicalResult visitOp(mlir::arith::SubIOp op) override{
        return visitBinaryOp(OpType::SUB, '-', op.getLhs(), op.getRhs(), op.getResult());
    }
    mlir::LogicalResult visitOp(mlir::arith::SubFOp op) override{
        return visitBinaryOp(OpType::SUB, '-', op.getLhs(), op.getRhs(), op.getResult());
    }
    mlir::LogicalResult visitOp(mlir::arith::MulIOp op) override{
        return visitBinaryOp(OpType::MUL, '*', op.getLhs(), op.getRhs(), op.getResult());
    }
    mlir::LogicalResult visitOp(mlir::arith::MulFOp op) override{
        return visitBinaryOp(OpType::MUL, '*', op.getLhs(), op.getRhs(), op.getResult());
    }
    mlir::LogicalResult visitOp(mlir::arith::DivSIOp op) override{
        return visitBinaryOp(OpType::DIV, '/', op.getLhs(), op.getRhs(), op.getResult());
    }
    mlir::LogicalResult visitOp(mlir::arith::DivFOp op) override{
        return visitBinaryOp(OpType::DIV, '/', op.getLhs(), op.getRhs(), op.getResult());
    }
    mlir::LogicalResult visitOp(UseGateOp op) override{
        auto attr = op.getNameAttr();
        //os << "use gate: " << attr.getLeafReference().str() << endl;
        auto leaf = attr.getLeafReference().str();
        auto gate = getGateName(leaf);
        if (op.getParameters().size() > 0){
            gate += "(";
            for (auto param : ::llvm::enumerate(op.getParameters())){
                auto index = param.index();
                auto operand = param.value();
                if (index > 0){
                    gate += ",";
                }
                gate += get<2>(getSymbol(operand));
            }
            gate += ")";
        }
        return symbolInsert(op.getResult(), OpType::VAR, 0, gate);
    }
    mlir::LogicalResult visitOp(DecorateOp op) override{
        auto modifiers = std::string();
        if(op.getAdjoint()) modifiers += "inv @ ";
        for(auto flag: op.getCtrl()){
            if(flag){
                modifiers += "ctrl @ ";
            }else{
                modifiers += "nctrl @ ";
            }
        }
        auto prevSymbol = getSymbol(op.getOperand());
        return symbolInsert(op.getResult(), OpType::VAR, 1, modifiers + get<2>(prevSymbol));
    }
    mlir::LogicalResult visitOp(ApplyGateOp op) override{
        vector<string> qlist;
        bool inv = false;
        int ctrl = 0, nctrl = 0;
        string gate_name_with_modifier = "";
        gate_name_with_modifier = get<2>(getSymbol(op.getGate()));
        for (auto indexed_operand : ::llvm::enumerate(op.getArgs())){
            auto index = indexed_operand.index();
            auto operand = indexed_operand.value();
            auto res = getSymbol(operand);
            TRY(symbolInsert(op.getResult(index), OpType::VAR, 1, get<2>(getSymbol(operand))));
            qlist.push_back(get<2>(res));
        }
        openQasmUnitary(gate_name_with_modifier, qlist);
        return mlir::success();
    }
    mlir::LogicalResult visitOp(mlir::func::CallOp op) override{
        auto func_name = op.getCalleeAttr().getValue().str();
        string call_str = getQasmName(func_name) + "(";
        for (auto indexOperand : llvm::enumerate(op->getOperands())){
            auto operand = indexOperand.value();
            size_t code = size_t(mlir::hash_value(operand));
            auto res = getSymbol(code);
            if (indexOperand.index() > 0){
                call_str += ", ";
            }
            call_str += get<2>(res);
            //call_str += "[0]";
        }
        call_str += ")";
        for (auto indexResult : llvm::enumerate(op->getResults())){
            auto result = indexResult.value();            
            string res_type;
            TRY(getVarType(result.getType(), res_type, op->getLoc()));
            if (res_type != "qubit"){
                if (result.use_empty()){
                    openQasmCall(call_str);
                    return mlir::success();
                }else{
                    return symbolInsert(size_t(mlir::hash_value(result)), OpType::VAR, 1, call_str);
                }
            }
            /*
            if (result.getType().isa<mlir::IndexType>()){
                auto tmp = next_tempInt();
                openQasmAssign(tmp+"[0]", call_str);
                return symbolInsert(size_t(mlir::hash_value(result)), OpType::VAR, 1, tmp);
            }*/
        }

        openQasmCall(call_str);
        
        return mlir::success();
    }

    mlir::LogicalResult visitOp(CallQOpOp op) override {
        string qop_name = op->getAttr(llvm::StringRef("callee")).dyn_cast<mlir::SymbolRefAttr>().getLeafReference().str();

        if (qop_name == "__isq__builtin__measure" || qop_name == "__isq__builtin__reset"){
            
            string call_qop_str = qop_name.substr(16) + " ";

            for (auto indexOperand : llvm::enumerate(op->getOperands())){
                auto operand = indexOperand.value();
                size_t code = size_t(mlir::hash_value(operand));
                auto res = getSymbol(code);
                if (indexOperand.index() > 0){
                    call_qop_str += ", ";
                }
                call_qop_str += get<2>(res);
                //call_qop_str += "[0]";
            }
            auto result_range = op->getResults();

            // First N operands: build qubit mapping for results
            for (auto indexed_operand: llvm::enumerate(op.getOperands().take_front(op.getSize()))){
                auto index = indexed_operand.index();
                auto operand = indexed_operand.value();
                auto res = getSymbol(operand);
                TRY(symbolInsert(op.getResult(index), OpType::VAR, 1, get<2>(getSymbol(operand))));
            }
            // Rest results: assign integer.
            for (auto indexResult: result_range.drop_front(result_range.size()-op.getSize())){
                auto result = indexResult;
                // Only consider measure.
                if (result.getType().isa<mlir::IntegerType>()){
                    auto temp_bool = tmpVarHead+to_string(tmpVarCnt++);
                    //auto temp_int = next_tempInt();
                    openQasmAssign("bit "+temp_bool, call_qop_str);
                    //openQasmAssign(temp_int+"[0]", (string("(int)"))+temp_bool+"[0]");
                    return symbolInsert(size_t(mlir::hash_value(result)), OpType::VAR, 1, temp_bool);
                }
            }
            // consider reset
            openQasmCall(call_qop_str);
        }else{
            op->emitOpError("invalid qop in OpenQASM3");
            return mlir::failure();
        }

        return mlir::success();
    }
    mlir::LogicalResult visitOp(mlir::func::ReturnOp op) override{
        if (hasRes){
            size_t opcode = size_t(mlir::hash_value(op->getOperand(0)));
            auto res = getSymbol(opcode);
            openQasmReturn(get<2>(res));
        }
        
        return mlir::success();
    }
    mlir::LogicalResult visitOp(DefgateOp op) override{
        string gate_name = op.getSymName().str();
        int shape = op.getType().getSize();
        if(shape!=1){
            op->emitOpError("with shape > 1 not supported in this codegen. Decompose first.");
            return mlir::failure();
        }
        if (!op.getDefinition()){
            op.emitOpError("without definition not supported.");
            return mlir::failure();
        }
        auto def = op.getDefinition();
        for(auto& def_ : *def){
            auto def = def_.cast<GateDefinition>();
            if(def.getType()=="unitary"){
                auto mat = def.getValue().cast<DenseComplexF64MatrixAttr>();
                auto matval = mat.toMatrixVal();
                /*
                auto mat = def.getValue().cast<mlir::ArrayAttr>();
                std::complex<double> matv[2][2];
                for(int i=0; i<2; i++){
                    for(int j=0; j<2; j++){
                        matv[i][j]=mat[i].cast<mlir::ArrayAttr>()[j].cast<ComplexF64Attr>().complexValue();
                    }
                }
                */
                std::complex<double> matv[2][2];
                for(int i=0; i<2; i++){
                    for(int j=0; j<2; j++){
                        matv[i][j]=matval[i][j];
                    }
                }
                openQasmGateDefine(gate_name, shape, matv);
                return mlir::success();
            }
        }
        op->emitOpError("without unitary definition not supported.");
        return mlir::failure();
    }
    mlir::LogicalResult visitOp(mlir::arith::CmpIOp op) override{
        int pred = static_cast<int>(op.getPredicate());
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
        //auto temp_bool = next_tempBit();
        //openQasmAssign(temp_bool, cmp_str);
        return symbolInsert(size_t(mlir::hash_value(op->getOpResult(0))), OpType::VAR, 1, cmp_str);
    }
    string memrefObtainArg(mlir::Operation::operand_range args, mlir::ArrayRef<int64_t> static_args, size_t index){
        if(index>=args.size()){
            auto cval = static_args[index-args.size()];
            auto temp_constant = next_tempInt();
            openQasmAssign(temp_constant+"[0]", to_string(cval));
            return temp_constant+"[0]";
        }else{
            return get<2>(getSymbol(args[0]));
        }

    }
    mlir::LogicalResult visitOp(mlir::memref::SubViewOp op) override{
        auto arr_in = op.getSource();
        auto arr_out = op.getResult();
        auto arr_in_ty = op.getSourceType();
        auto arr_out_ty = arr_out.getType().cast<mlir::MemRefType>();
        if(arr_in_ty.getRank()!=1 || arr_out_ty.getRank()!=1){
            op.emitOpError("with rank!=1 is not supported");
            return mlir::failure();
        }
        
        auto source_var = getSymbol(arr_in);
        if (get<1>(source_var) == 1){
            return symbolInsert(size_t(mlir::hash_value(op->getOpResult(0))), OpType::VAR, 1, get<2>(source_var));
        }else{
            string offset;
            if (op.offsets().size() > 0){
                offset = get<2>(getSymbol(op.offsets()[0]));
            }else{
                auto offset_val = op.static_offsets()[0];
                offset = to_string(offset_val);
            }
            return symbolInsert(size_t(mlir::hash_value(op->getOpResult(0))), OpType::VAR, 1, get<2>(source_var)+"["+offset+"]");
        }
        /*
        auto offset = memrefObtainArg(op.offsets(), op.static_offsets(), 0);
        auto size = memrefObtainArg(op.sizes(), op.static_sizes(), 0);
        auto stride = memrefObtainArg(op.strides(), op.static_strides(), 0);
        auto temp_val = next_tempName();
        openQasmSlice(temp_val, get<2>(getSymbol(arr_in)), offset, string(offset)+"+"+size, stride);
        return symbolInsert(size_t(mlir::hash_value(op->getOpResult(0))), OpType::VAR, -1, temp_val);
        */
    }
    mlir::LogicalResult visitOp(PassOp op) override{
        return mlir::success();
    }
    mlir::LogicalResult visitOp(mlir::AffineYieldOp) override{
        return mlir::success();
    }
    mlir::LogicalResult visitOp(mlir::scf::YieldOp op) override{
        for (auto operand: op.getOperands()){
            yieldRes = mlir::hash_value(operand);
        }
        return mlir::success();
    }
    mlir::LogicalResult visitOp(mlir::memref::CastOp op) override{
        // todo: we hope it is identical cast.
        auto symbol = getSymbol(op.getSource());
        return symbolInsert(size_t(mlir::hash_value(op->getOpResult(0))), get<0>(symbol), get<1>(symbol), get<2>(symbol));
    }
    mlir::LogicalResult visitOp(mlir::Operation* op) override{
        op->emitOpError("is unsupported in code generation");
        return mlir::failure();
    }
    
    struct RaiiIndent{
        int& indent;
        RaiiIndent(int &indent): indent(indent){};
        ~RaiiIndent() {indent -= 1;};
    };

    RaiiIndent pushIndent(){
        return RaiiIndent(++indent);
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

    void openQasmGateDefine(string name, int shape, std::complex<double> matrix[2][2]){
        openQasmNewLine();
        auto zyz = zyzDecomposition(matrix);
        os << "gate "+ name;
        for (int i = 0; i < shape; i++){
            char tmp = 'a'+i;
            if (i > 0)
                os << ",";
            os << " " << tmp;
        }
        os << " {\n";
        os << "    U("<<zyz.theta<<","<<zyz.phi<<","<<zyz.lam<<") a;\n";
        os << "};\n";
    }

    void openQasmFunc(string name, vector<tuple<string, string, int>> &arglist, string rty){
        
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
            os << " -> " << rty;
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

    void openQasmUnitary(string name, vector<string>& qlist){
        openQasmNewLine();
        os << name << " ";
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

    void openQasmIf(string cond){
        openQasmNewLine();
        os << "if (" << cond << ")";
    }

    void openQasmFor(string arg, string lval, string rval, string step){
        openQasmNewLine();
        os << "for " << arg << " in [" << lval << ":" << rval;
        if (step != "1") os << ':' << step;
        os << "] {\n";
    }

    void openQasmWhile(string cond){
        openQasmNewLine();
        os << "while (" << cond << ")";
    }
    void openQasmSlice(string ret, string arr, string lo, string hi,string stride){
        openQasmNewLine();
        os << "let "<<ret<<" = "<<arr<<"["<<lo<<":"<<stride<<":"<<hi<<"]\n";
    }

    mlir::LogicalResult symbolInsert(::mlir::Value v, OpType type, int shape, string str){
        return symbolInsert(mlir::hash_value(v), type, shape, str);
    }
    mlir::LogicalResult symbolInsert(size_t code, OpType type, int shape, string str){
        
        auto iter = symbolTable.find(code);
        if (iter != symbolTable.end()){
            os << "Error: symbol is already defined\n";
            return mlir::failure();
        }
        
        symbolTable.insert(make_pair(code, make_tuple(type, shape, str)));
        return mlir::success();
    }
    tuple<OpType, int, string> getSymbol(mlir::Value val){
        return getSymbol(mlir::hash_value(val));
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
}


namespace isq {
namespace ir{
mlir::LogicalResult generateOpenQASM3Logic(mlir::MLIRContext &context, mlir::ModuleOp &module, llvm::raw_fd_ostream &os) {
    return MLIRPassImpl(context, module, os).mlirPass();
}
}
}