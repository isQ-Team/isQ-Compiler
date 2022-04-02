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
#include "isq/QStructs.h"
#include "isq/utils/Decomposition.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
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
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"


#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"

#include "llvm/Support/raw_ostream.h"
#include "isq/Backends.h"

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

namespace isq{
namespace ir{

namespace details{
using namespace std;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::memref;
using namespace mlir::scf;
using CodegenOpVisitor = OpVisitor<
    FuncOp, scf::IfOp, AffineForOp, scf::ExecuteRegionOp,
    GetGlobalOp, GlobalOp,
    mlir::arith::ConstantOp, mlir::arith::ExtUIOp, mlir::arith::IndexCastOp,
    AllocOp, DeallocOp, memref::LoadOp, memref::StoreOp, SubViewOp, memref::CastOp, CmpIOp,
    AddIOp, SubIOp, MulIOp, DivSIOp, RemSIOp,
    UseGateOp, DecorateOp, ApplyGateOp, CallQOpOp, AccumulateGPhase, DeclareQOpOp,
    CallOp, ReturnOp, DefgateOp, scf::WhileOp, ConditionOp,
    ModuleOp, PassOp, AffineYieldOp, scf::YieldOp,
    CondBranchOp, BranchOp
    >;
}


/// Code generator from isQ MLIR Dialect to QCIS.
class MLIRPassImpl: public details::CodegenOpVisitor{
public:
    MLIRPassImpl(mlir::MLIRContext &context, mlir::ModuleOp &module, llvm::raw_ostream &os, bool printast) : context(&context), theModule(&module), os(os), printast(printast) {};
    

    mlir::LogicalResult mlirPass(){
        
        indent = 0;
        qbitSize = 0;
        tmp_cnt = 0;
        tmp_head = "tmp_";
        res = "";
        if (printast) printOperation(theModule->getOperation());
        initGate();
        initializeIntegerSets();
        visitOperation(theModule->getOperation());
        // find main function, and visit it's body
        auto iter = funcMap.find("__isq__main");
        if (iter != funcMap.end()){
            return visitOp(funcMap["__isq__main"]);
        }
        return mlir::failure();
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
    bool printast;
    
    int indent;

    set<string> baseGate;
    map<string, string> gateMap; //decomposition_raw gate
    map<string, map<int, int>> symbolTable;
    set<int> measured;
    map<string, mlir::FuncOp> funcMap;
    set<string> visitFunc;
    map<string, int> funcArgs;

    map<size_t, int> intValueTable;
    map<size_t, pair<string, int>> indexTable;
    map<size_t, string> gateTable;
    int qbitSize;
    string tmp_head;
    int tmp_cnt;
    string res;
    //isqTools isqtool;

    // Initialize predefined gates in OpenQASM 3.0.
    void initGate(){
        baseGate = {"H", "X", "Y", "Z", "S", "T", "CZ", "X2P", "X2M", "Y2P", "Y2M", "CNOT"};
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

    mlir::LogicalResult visitBlock(mlir::Block* block){
        if(!block) return mlir::success();
        for(auto& child: block->getOperations()){
            // save all functions in funcMap
            auto fop = mlir::dyn_cast_or_null<mlir::FuncOp>(&child);
            if (fop != nullptr){
                funcMap.insert(make_pair(fop.sym_name().str(), fop));
                continue;
            }
            //child.getName().dump();
            TRY(visitOperation(&child));
        }
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::ModuleOp curr_module) override{
        auto mod_name = curr_module.sym_name();
        if(mod_name){
            if(*mod_name == "isq_builtin"){
                // Just ignore it.
                return mlir::success();
            }
        }
        TRY(visitBlock(curr_module.getBody()));
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::CondBranchOp op) override{
        
        auto code = size_t(mlir::hash_value(op.getCondition()));
        auto b_i = getValue(code);
        if (b_i.first == false) return error(op.getLoc(), "jump condition must be a determined boolean");
        if (b_i.second == 0){
            return visitBlock(op.getFalseDest());
        }else{
            return visitBlock(op.getTrueDest());
        }
    }

    mlir::LogicalResult visitOp(mlir::BranchOp op) override{
        return visitBlock(op.getDest());
    }


    mlir::LogicalResult visitOp(mlir::memref::GlobalOp op) override{
        // store global var to symbolTable
        auto id = op.sym_name();
        auto type = op.type();
        int size = type.getShape()[0];
        if (type.getElementType().isa<QStateType>()){
            map<int, int> qmap;
            for (int i = 0; i < size; i++) qmap[i] = qbitSize+1+i;
            symbolTable.insert(make_pair(id.str(), qmap));
            qbitSize += size;
        }else if (type.getElementType().isa<mlir::IndexType>()){
            map<int, int> imap;
            for (int i = 0; i < size; i++) imap[i] = 0;
            symbolTable.insert(make_pair(id.str(), imap));
        }else{
            return error(op.getLoc(), "qcis can only use 'qbit' or 'int' type.");
        }
        
        return mlir::success();
    }
    
    mlir::LogicalResult visitOp(DefgateOp op) override{
        // user can not define gate in qcis
        // deriving gate is allowed, save func name in gateMap
        auto gate_name = op.sym_name().str();
        if (op.definition()){
            for(auto& def_ : *op.definition()){
                auto def = def_.cast<GateDefinition>();
                if(def.type()=="unitary") return error(op.getLoc(), "sorry, qcis can not define gate.");
                if (def.type() == "decomposition_raw"){
                    gateMap[gate_name] = def.value().dyn_cast<mlir::SymbolRefAttr>().getLeafReference().str();
                }
            }
        }
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::FuncOp func_op) override{
        // visit func body
        auto func_name = func_op.sym_name().str();
        auto visibility = func_op.sym_visibility();
        if (visibility.getValueOr<llvm::StringRef>("public").str() == "private") return mlir::success();
        auto func_block = func_op.getBody().getBlocks().begin();
        return visitBlock(&*func_block);
    }

    mlir::LogicalResult visitOp(mlir::memref::GetGlobalOp op) override{
        auto attr = op.nameAttr();
        auto type = op.result().getType().dyn_cast<mlir::MemRefType>();
        auto code = size_t(mlir::hash_value(op.result()));
        indexTable[code] = make_pair(attr.getValue().str(), 0);
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::arith::ConstantOp op) override{
        auto attr = op.getValueAttr().dyn_cast<mlir::IntegerAttr>();
        auto code = size_t(mlir::hash_value(op->getOpResult(0)));
        intValueTable[code] = attr.getInt();
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::arith::ExtUIOp op) override{
        auto icode = size_t(mlir::hash_value(op.getIn()));
        auto ocode = size_t(mlir::hash_value(op.getOut()));
        intValueTable[ocode] = intValueTable[icode];
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::arith::IndexCastOp op) override{
        auto icode = size_t(mlir::hash_value(op.getIn()));
        auto ocode = size_t(mlir::hash_value(op.getOut()));
        intValueTable[ocode] = intValueTable[icode];
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::memref::AllocOp op) override{
        // create new var name
        // store to symbolTable and indexTable
        auto result = op.getResult();
        if (result.use_empty()){
            return mlir::success();
        }

        auto type = op.getResult().getType().dyn_cast<mlir::MemRefType>();
        if (type.getElementType().isa<QStateType>()) return error(op.getLoc(), "sorry, qbit var can only define at global area.");

        string var_name = getTmpName();
        int size = type.getShape()[0];
        map<int, int> imap;
        for (int i = 0; i < size; i++) imap[i] = 0;
        symbolTable.insert(make_pair(var_name, imap));
        
        auto code = size_t(mlir::hash_value(op->getOpResult(0)));
        indexTable[code] = make_pair(var_name, 0);

        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::memref::DeallocOp op) override{
        // every local var has unique var_name in symbolTable, just jump it.
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::memref::SubViewOp op) override{
        // update index in indexTable
        auto scode = mlir::hash_value(op.source());
        auto fcode = mlir::hash_value(op.offsets()[0]);
        auto rcode = mlir::hash_value(op.result());
        auto s_i = getIndex(scode);
        if (s_i.second == -1) return error(op.getLoc(), "get var error");
        auto f_v = getValue(fcode);
        if (f_v.first == false) return error(op.getLoc(), "index must a determined number");
        int index = s_i.second + f_v.second;
        if (outOfBorder(s_i.first, index)) return error(op.getLoc(), "index out of border");
        indexTable[rcode] = make_pair(s_i.first, index);
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::memref::LoadOp op) override{
        string var_name;
        int index = 0;
        // load value from symbolTable using var_name and index
        for (auto indexOperand : llvm::enumerate(op->getOperands())){
            auto operand = indexOperand.value();
            size_t code = size_t(mlir::hash_value(operand));
            if (indexOperand.index() == 0){
                auto s_i = getIndex(code);
                if (s_i.second < 0) return error(op.getLoc(), "load value error");
                var_name = s_i.first;
                index = s_i.second;
            }else{
                auto b_i = getValue(code);
                if (b_i.first == false) return error(op.getLoc(), "index must a determined number");
                index += b_i.second;
            }
        }
        //std::cout << var_name << ": " << index << std::endl;
        if (outOfBorder(var_name, index)) return error(op.getLoc(), "index out of border");

        auto code = size_t(mlir::hash_value(op->getOpResult(0)));
        intValueTable[code] = getVarValue(var_name, index);
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::memref::StoreOp op) override{
        int val, index;
        string var_name;
        // store index value to symtable, qbit ignore
        for (auto indexOperand : llvm::enumerate(op->getOperands())){
            auto operand = indexOperand.value();
            size_t code = size_t(mlir::hash_value(operand));
            if (indexOperand.index() == 0){
                if(operand.getType().isa<QStateType>()) return mlir::success();
                auto b_i = getValue(code);
                if (b_i.first == false) return error(op.getLoc(), "must assign a determined value");
                val = b_i.second;
            }else if (indexOperand.index() == 1){
                auto s_i = getIndex(code);
                if (s_i.second < 0) return error(op.getLoc(), "get left var error");
                var_name = s_i.first;
                index = s_i.second;
            }else{
                auto b_i = getValue(code);
                if (b_i.first == false) return error(op.getLoc(), "index must be a determined value");
                index += b_i.second;
            }
        }
        
        if (outOfBorder(var_name, index)) return error(op.getLoc(), "index out of border");
        setVarValue(var_name, index, val);

        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::memref::CastOp op) override{
        return error(op.getLoc(), "qcis don't support dynamic array");
    }

    mlir::LogicalResult visitBinaryOp(char op, mlir::Value lhs, mlir::Value rhs, mlir::Value ret){
        
        // get lhs's and rhs's value from intValueTable
        auto lhs_b_i = getValue(mlir::hash_value(lhs));
        auto rhs_b_i = getValue(mlir::hash_value(rhs));
        int val = 0;
        switch (op)
        {
        case '+':
            val = lhs_b_i.second + rhs_b_i.second;
            break;
        case '-':
            val = lhs_b_i.second - rhs_b_i.second;
            break;
        case '*':
            val = lhs_b_i.second * rhs_b_i.second;
            break;
        case '/':
            val = lhs_b_i.second / rhs_b_i.second;
            break;
        case '%':
            val = lhs_b_i.second % rhs_b_i.second;
            break;
        default:
            return mlir::failure();
            break;
        }
        intValueTable[mlir::hash_value(ret)] = val;
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::arith::AddIOp op) override{
        return visitBinaryOp('+', op.getLhs(), op.getRhs(), op.getResult());
    }
    mlir::LogicalResult visitOp(mlir::arith::SubIOp op) override{
        return visitBinaryOp('-', op.getLhs(), op.getRhs(), op.getResult());
    }
    mlir::LogicalResult visitOp(mlir::arith::MulIOp op) override{
        return visitBinaryOp('*', op.getLhs(), op.getRhs(), op.getResult());
    }
    mlir::LogicalResult visitOp(mlir::arith::DivSIOp op) override{
        return visitBinaryOp('/', op.getLhs(), op.getRhs(), op.getResult());
    }
    mlir::LogicalResult visitOp(mlir::arith::RemSIOp op) override{
        return visitBinaryOp('%', op.getLhs(), op.getRhs(), op.getResult());
    }

    mlir::LogicalResult visitOp(UseGateOp op) override{
        // save gate name in gateTable
        auto attr = op.nameAttr();
        auto gate = attr.getLeafReference().str();
        auto rcode = mlir::hash_value(op.getResult());
        gateTable[rcode] = gate;
        return mlir::success();
    }

    mlir::LogicalResult visitOp(DecorateOp op) override{
        return error(op.getLoc(), "qcis don't support ctrl or nctrl or inv decorate");
    }

    mlir::LogicalResult visitOp(ApplyGateOp op) override{

        auto gate = gateTable[mlir::hash_value(op.gate())];
        if (baseGate.count(gate) != 1 && !derivingGate(gate)) return error(op.getLoc(), "gate "+gate+" is not support in qcis.");
        // get args
        // qbit <memref> : get value from symbolTable
        // if qbit has already measured, error
        vector<int> args;
        for (auto indexed_operand : ::llvm::enumerate(op.args())){
            auto index = indexed_operand.index();
            auto operand = indexed_operand.value();
            auto b_i = getValue(mlir::hash_value(operand));
            if (b_i.first == false) return error(op.getLoc(), "gate operate need use determined qbit");
            if (measured.count(b_i.second) == 1) return error(op.getLoc(), "qbit has already measured, can not use again.");
            args.push_back(b_i.second);
        }
        if (baseGate.count(gate) == 1){
            // baseGate direct print qcis
            QcisPrint(gate, args);
            return mlir::success();
        }else{
            // deriving gate visit func body
            auto func_name = gateMap[gate];
            if (visitFunc.count(func_name) == 1) return error(op.getLoc(), "qcis don't support two function call each other");
            if (mlir::failed(updateArgs(funcMap[func_name], args))) return mlir::failure();
            visitFunc.insert(func_name);
            if (mlir::failed(visitOp(funcMap[func_name]))) return mlir::failure();
            visitFunc.erase(func_name);
            return mlir::success();
        }
    }

    mlir::LogicalResult updateArgs(mlir::FuncOp op, vector<int>& args){
        // store args
        // index: store in intValueTable
        // qbit: store in symbolTable and indexTable
        auto func_block = op.getBody().getBlocks().begin();
        int idx = 0;
        for (auto &arg: func_block->getArguments()){
            if (arg.getType().isa<mlir::IndexType>()){
                if (!arg.use_empty()){
                    auto code = mlir::hash_value(arg);
                    intValueTable[code] = args[idx];
                }
            }else{
                if (arg.getType().dyn_cast<mlir::MemRefType>().getShape()[0] != 1) return error(op->getLoc(), "qcis func don't support array as arg");
                if (!arg.use_empty()){
                    auto code = mlir::hash_value(arg);
                    string arg_name = getTmpName();
                    symbolTable[arg_name] = {{0, args[idx]}};
                    indexTable[code] = make_pair(arg_name, 0);
                }
            }
            idx += 1;
        }

        return mlir::success();
    }

    mlir::LogicalResult visitOp(CallQOpOp op) override {
        string qop_name = op->getAttr(llvm::StringRef("callee")).dyn_cast<mlir::SymbolRefAttr>().getLeafReference().str();
        // only measure support
        if (qop_name != "__isq__builtin__measure") return error(op.getLoc(), "sorry, qcis can only do measure qop");
        // get measure qbit
        // if qbit has already measured, error
        for (auto indexOperand : llvm::enumerate(op->getOperands())){
            auto operand = indexOperand.value();
            size_t code = size_t(mlir::hash_value(operand));
            auto b_i  = getValue(code);
            if (b_i.first == false) return error(op.getLoc(), "measure a determined qbit");
            if (measured.count(b_i.second) == 1) return error(op.getLoc(), "qbit has already measured, can not use again.");
            QcisPrint("M", {b_i.second});
            measured.insert(b_i.second);
        }
        auto rcode = mlir::hash_value(op.getResult(1));
        intValueTable[rcode] = 0;
        return mlir::success();
    }

    mlir::LogicalResult visitOp(AccumulateGPhase op) override {
        // qcis don't has gphase, just jump it
        return mlir::success();
    }

    mlir::LogicalResult visitOp(DeclareQOpOp op) override {
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::CallOp op) override{
        auto func_name = op.getCalleeAttr().getValue().str();
        // get args
        // index : get value from intValueTable
        // qbit <memref> : get value from symbolTable
        vector<int> args;
        for (auto indexOperand : ::llvm::enumerate(op.getOperands())){
            auto operand = indexOperand.value();
            if (operand.getType().isa<mlir::IndexType>()){
                auto b_i = getValue(mlir::hash_value(operand));
                if (b_i.first == false) return error(op.getLoc(), "call func args need use determined value");
                args.push_back(b_i.second);
            }else{
                auto s_i = getIndex(mlir::hash_value(operand));
                if (s_i.second < 0) return error(op.getLoc(), "call func args need use determined qbit");
                args.push_back(getVarValue(s_i.first, s_i.second));
            }
        }
        
        // visit func body
        if (visitFunc.count(func_name) == 1) return error(op.getLoc(), "qcis don't support two function call each other");
        if (mlir::failed(updateArgs(funcMap[func_name], args))) return mlir::failure();
        visitFunc.insert(func_name);
        if (mlir::failed(visitOp(funcMap[func_name]))) return mlir::failure();
        visitFunc.erase(func_name);
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::ReturnOp op) override{
        if (op.getOperands().size() > 0) return error(op.getLoc(), "qcis don't support function with return value");
        return mlir::success();
    }
    mlir::LogicalResult visitOp(PassOp op) override{
        return mlir::success();
    }
    mlir::LogicalResult visitOp(mlir::AffineYieldOp op) override{
        return mlir::success();
    }
    mlir::LogicalResult visitOp(mlir::scf::YieldOp op) override{
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::scf::ExecuteRegionOp op) override{

        auto exec_block = op.getRegion().getBlocks().begin();
        return visitBlock(&*exec_block);
    }

    mlir::LogicalResult visitOp(mlir::AffineForOp for_stmt) override{
        
        int lval, rval, step;
        // get start, end, step value
        auto lmap = for_stmt.getLowerBoundMap();
        if (lmap != singleSymbol){
            lval = lmap.getSingleConstantResult();
        }else{
            size_t opcode = size_t(mlir::hash_value(for_stmt.getLowerBound().getOperand(0)));
            auto b_i = getValue(opcode);
            if (b_i.first == false) return error(for_stmt.getLoc(), "for start need a determined value");
            lval = b_i.second;
        }

        auto rmap = for_stmt.getUpperBoundMap();
        if (rmap != singleSymbol){
            rval = rmap.getSingleConstantResult();
        }else{
            size_t opcode = size_t(mlir::hash_value(for_stmt.getUpperBound().getOperand(0)));
            auto b_i = getValue(opcode);
            if (b_i.first == false) return error(for_stmt.getLoc(), "for end need a determined value");
            rval = b_i.second;
        }

        step = for_stmt.getStep();
        // use for loop to visit for-loop-body
        auto block = for_stmt.getBody();
        for (int i = lval; i < rval; i += step){
            auto code = mlir::hash_value(block->getArgument(0));
            intValueTable[code] = i;
            if (mlir::failed(visitBlock(block))) return mlir::failure();
        }
        
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::arith::CmpIOp op) override{
        
        int lval, rval;
        // get cmp's lhs and rhs value
        for (auto indexOperand : llvm::enumerate(op->getOperands())){
            auto operand = indexOperand.value();
            size_t code = size_t(mlir::hash_value(operand));
            auto b_i = getValue(code);
            if (b_i.first == false) return error(op.getLoc(), "condition need determined value");
            if (indexOperand.index() == 0){
                lval = b_i.second;
            }else{
                rval = b_i.second;
            }
        }
        bool ans;
        int pred = static_cast<int>(op.getPredicate());
        switch (pred)
        {
        case 0:
            ans = (lval == rval);
            break;
        case 1:
            ans = (lval != rval);
            break;
        case 2:
            ans = (lval < rval);
            break;
        case 3:
            ans = (lval <= rval);
            break;
        case 4:
            ans = (lval > rval);
            break;
        case 5:
            ans = (lval >= rval);
            break;
        default:
            return error(op.getLoc(), "condition is not support in qcis");
            break;
        }
        // set condition value
        auto code = mlir::hash_value(op->getOpResult(0));
        intValueTable[code] = ans;
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::scf::IfOp if_stmt) override{
        // get condition value
        auto condition_code = mlir::hash_value(if_stmt.getCondition());
        auto b_i = getValue(condition_code);
        if (b_i.first == false) return error(if_stmt.getLoc(), "if need a determined condition");
        
        if (b_i.second == 1){
            return visitBlock(if_stmt.thenBlock());
        }else{
            return visitBlock(if_stmt.elseBlock());
        }
    }

    mlir::LogicalResult visitOp(mlir::scf::WhileOp while_stmt) override{
        // while stmt may in dead cycle, user can use for stmt instead. 
        return error(while_stmt.getLoc(), "qcis don't support while stmt yet, you can use for stmt instead");
    }

    mlir::LogicalResult error(mlir::Location loc, string msg){
        emitError(loc, msg);
        return mlir::failure();
    }

    string getTmpName(){
        auto name = tmp_head + to_string(tmp_cnt);
        tmp_cnt += 1;
        return name;
    }

    void QcisPrint(string gate, vector<int> qbit){
        os << gate;
        for (auto idx: qbit) os << " Q" << idx;
        os << "\n";
    }

    bool derivingGate(string gate){
        auto iter = gateMap.find(gate);
        if (iter != gateMap.end()) return true;
        return false;
    }

    pair<string, int>getIndex(size_t code){
        
        auto iter = indexTable.find(code);
        if (iter != indexTable.end()){
            return iter->second;
        }
        return make_pair("", -1);
    }

    pair<bool, int> getValue(size_t code){

        auto iter = intValueTable.find(code);
        if (iter != intValueTable.end()){
            return make_pair(true, iter->second);
        }
        return make_pair(false, -1);
    }

    bool outOfBorder(string name, int index){
        auto iter = symbolTable.find(name);
        auto iiter = iter->second.find(index);
        if (iiter == iter->second.end()) return true;
        return false;
    }

    int getVarValue(string name, int index){
        auto iter = symbolTable.find(name);
        auto iiter = iter->second.find(index);
        return iiter->second;
    }

    void setVarValue(string name, int index, int val){
        auto iter = symbolTable.find(name);
        iter->second[index] = val;
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
    
}; 
}
}


namespace isq {
namespace ir{
mlir::LogicalResult generateQCIS(mlir::MLIRContext &context, mlir::ModuleOp &module, llvm::raw_fd_ostream &os, bool printast) {
    return MLIRPassImpl(context, module, os, printast).mlirPass();
}
}
}