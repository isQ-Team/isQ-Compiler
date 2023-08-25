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
#define EPS (1e-6)

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
    mlir::func::FuncOp, scf::IfOp, scf::ForOp, scf::ExecuteRegionOp,
    GetGlobalOp, GlobalOp,
    mlir::arith::ConstantOp, mlir::arith::ExtUIOp, mlir::arith::IndexCastOp, mlir::arith::SIToFPOp,
    AllocOp, DeallocOp, memref::LoadOp, memref::StoreOp, SubViewOp, memref::CastOp, CmpIOp,
    AddIOp, SubIOp, MulIOp, DivSIOp, RemSIOp, AddFOp, SubFOp, MulFOp, DivFOp, NegFOp, AndIOp,
    UseGateOp, DecorateOp, ApplyGateOp, CallQOpOp, AccumulateGPhase, DeclareQOpOp, AssertOp,
    mlir::func::CallOp, mlir::func::ReturnOp, DefgateOp, scf::WhileOp, ConditionOp,
    ModuleOp, PassOp, AffineYieldOp, scf::YieldOp,
    mlir::cf::CondBranchOp, mlir::cf::BranchOp, AffineLoadOp, AffineStoreOp
    >;
}

#define MOV "LDI"
#define LOAD "LD"
#define STORE "ST"
#define ADD "ADD"
#define SUB "SUB"
#define AND "AND"
#define MUL "MUL"
#define CMP "CMP"
#define FMR "FMR"
#define BR "BR"

#define MEASE "MEASE"

#define SPEC1 "R1"
#define SPEC2 "R2"
#define CSTACK "RC"
#define QSTACK "RQ"
#define QNUM "RN"
#define CMEMSTART 0
#define QMEMSTART 1000

struct varValue{
    
    int ival;
    double dval;

    varValue() : ival(0), dval(0.0) {}
    varValue(int x) : ival(x), dval(0.0) {}
    varValue(double x): ival(), dval(x) {}

    varValue operator+(const varValue &r){
        varValue res;
        res.ival = this->ival + r.ival;
        res.dval = this->dval + r.dval;
        return res;
    }

    varValue operator-(const varValue &r){
        varValue res;
        res.ival = this->ival - r.ival;
        res.dval = this->dval - r.dval;
        return res;
    }

    varValue operator*(const varValue &r){
        varValue res;
        res.ival = this->ival * r.ival;
        res.dval = this->dval * r.dval;
        return res;
    }

    varValue operator/(const varValue &r){
        varValue res;
        if (r.ival != 0) res.ival = this->ival / r.ival;
        if (r.dval != 0) res.dval = this->dval / r.dval;
        return res;
    }

    varValue operator%(const varValue &r){
        varValue res;
        res.ival = this->ival % r.ival;
        res.dval = 0.0;
        return res;
    }
};


struct tempValue{
    
    int index;
    bool is_addr;

    tempValue(): index(-1), is_addr(false) {}
    tempValue(int i) : index(i), is_addr(false) {}
    tempValue(int i, bool b): index(i), is_addr(b) {}
};



/// Code generator from isQ MLIR Dialect to EQASM.
class MLIRPassImplEQASM: public details::CodegenOpVisitor{
public:
    MLIRPassImplEQASM(mlir::MLIRContext &context, mlir::ModuleOp &module, llvm::raw_ostream &os, bool printast) : context(&context), theModule(&module), os(os), printast(printast) {};
    

    mlir::LogicalResult mlirPass(){
        
        indent = cstack = qstack = 0;
        if (printast) printOperation(theModule->getOperation());
        TRY(traverseOperation(theModule->getOperation()));
        buildCallDic();
        //printinfo();
        label_cnt = 0;
        init();
        nowfunc = "__isq__main";
        TRY(visitOp(callDict["__isq__main"].second));
        for (auto f: callDict){
            nowfunc = f.first;
            if (nowfunc == "__isq__main") continue;
            TRY(visitOp(f.second.second));
        }
        print_label("process_end");
        return mlir::success();
    }

private:
    
    mlir::MLIRContext* context;
    mlir::ModuleOp* theModule;
    llvm::raw_ostream& os;
    bool printast;
    
    int indent;

    set<string> baseGate;
    int cstack, qstack, label_cnt;
    string nowfunc;
    map<string, pair<map<string, int>, mlir::func::FuncOp>> funcMap;
    map<string, pair<int, mlir::func::FuncOp>> callDict;
    map<string, int> callCnt;
    map<string, pair<int, int>> funcStack;
    map<string, int> globalSymbol;
    map<size_t, tempValue> tempSymbol;
    map<size_t, varValue> valueTable;
    map<size_t, pair<string, vector<double>>> gateTable;

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
                map<string, int> callfunc;
                nowfunc = fop.getSymName().str();
                //if (nowfunc[0] == '$' && nowfunc != "$__isq__builtin__toffoli__decomposition__famous") continue;
                //if (nowfunc[0] == '_' && nowfunc[1] == '_' && nowfunc != "__isq__main") continue;
                funcMap.insert(make_pair(nowfunc, make_pair(callfunc, fop)));
            }
            TRY(traverseOperation(&p));
        }
        return mlir::success();
    }

    mlir::LogicalResult traverseOperation(mlir::Operation* op){
        auto callop = mlir::dyn_cast_or_null<mlir::func::CallOp>(op);
        if (callop != nullptr){
            auto name = callop.getCallee().str();
            //if (name == "$__isq__builtin__toffoli__decomposition__famous" || (name[0] != '$' && (name[0] != '_' && name[1] != '_'))){
                if (funcMap[nowfunc].first.count(name) == 0){
                    funcMap[nowfunc].first.insert(make_pair(name, 0));
                }
                funcMap[nowfunc].first[name] += 1;
            //}
        }

        auto globalop = mlir::dyn_cast_or_null<mlir::memref::GlobalOp>(op);
        if (globalop != nullptr){
            auto name = globalop.getSymName().str();
            int size = globalop.getType().getShape()[0];
            if (globalop.getType().getElementType().isa<QStateType>()){
                globalSymbol.insert(make_pair(name, qstack));
                qstack += size;
            }else if (globalop.getType().getElementType().isa<mlir::IndexType>())
            {
                globalSymbol.insert(make_pair(name, cstack));
                cstack += size;
            }else{
                return error(op->getLoc(), "eqasm can only define 'qbit'/'int' var.");
            }
            
        }

        for (auto &r:op->getRegions()){
            TRY(traverseRegion(r));
        }
        return mlir::success();
    }

    void buildCallDic(){
        callDict.insert(make_pair("__isq__main", make_pair(0, funcMap["__isq__main"].second)));
        funcStack.insert(make_pair("__isq__main", make_pair(0, 0)));
        set<string> now = {"__isq__main"};
        while (!now.empty()){
            set<string> tmp;
            for (auto fn: now){
                for (auto nfn: funcMap[fn].first){
                    if (callDict.count(nfn.first) == 0){
                        callDict.insert(make_pair(nfn.first, make_pair(0, funcMap[nfn.first].second)));
                        funcStack.insert(make_pair(nfn.first, make_pair(0, 0)));
                        callCnt.insert(make_pair(nfn.first, 0));
                        tmp.insert(nfn.first);
                    }
                    callDict[nfn.first].first += nfn.second;
                }
            }
            now = tmp;
        }
    }

    void init(){
        // init CSTACK
        print_op(MOV, {CSTACK, to_string(CMEMSTART + cstack)});
    
        // init QSTACK
        print_op(MOV, {QSTACK, to_string(QMEMSTART)});
        print_op(MOV, {QNUM, "0"});
        if (qstack > 0){
            print_op(MOV, {SPEC2, "0"});
            print_label("alloc_start_"+to_string(label_cnt));
            print_op(MOV, {SPEC1, to_string(qstack)});
            print_op(CMP, {SPEC2, SPEC1});
            print_op(BR, {"GEQ", "alloc_end_"+to_string(label_cnt)});
            print_op(ADD, {SPEC1, SPEC2, QSTACK});
            print_op(STORE, {QNUM, SPEC1});
            print_op(MOV, {SPEC1, "1"});
            print_op(ADD, {QNUM, QNUM, SPEC1});
            print_op(ADD, {SPEC2, SPEC2, SPEC1});
            print_op(BR, {"ALWAYS", "alloc_start_"+to_string(label_cnt)});
            print_label("alloc_end_"+to_string(label_cnt));
            label_cnt += 1;
        }
        print_op(MOV, {QSTACK, to_string(QMEMSTART + qstack)});
    }

    mlir::LogicalResult visitBlock(mlir::Block* block){
        if(!block) return mlir::success();
        for(auto& child: block->getOperations()){
            //os << "\n===========\n" << child.getName().getIdentifier().str() << "\n===========\n";
            TRY(visitOperation(&child));
        }
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::func::FuncOp func_op) override{
        // visit func body
        auto func_name = func_op.getSymName().str();
        print_label(func_name);
        int cnt = 0;
        for (auto &block: func_op.getBlocks()){
            for (auto &arg: block.getArguments()){
                auto code = size_t(mlir::hash_value(arg));
                if (arg.use_empty()){
                    continue;
                }
                else{
                    auto type = arg.getType();
                    if (arg.getType().isa<mlir::MemRefType>()){
                        type = arg.getType().dyn_cast<mlir::MemRefType>().getElementType();
                    }
                    if (type.isa<mlir::IndexType>() || arg.getType().isa<mlir::IntegerType>()){
                        tempSymbol.insert(make_pair(code, funcStack[nowfunc].first));
                        funcStack[nowfunc].first += 1;
                    }else if (type.isa<QStateType>()){
                        tempSymbol.insert(make_pair(code, funcStack[nowfunc].second));
                        funcStack[nowfunc].second += 1;
                    }else{
                        return error(func_op.getLoc(), "eqasm only support 'qbit'/'int' paramters.");
                    }
                }
            }

            print_label(func_name + "_b" + to_string(cnt));
            TRY(visitBlock(&block));
            cnt += 1;
        }
        //go back
        print_op(MOV, {SPEC1, "1"});
        print_op(SUB, {CSTACK, CSTACK, SPEC1});
        get_load(SPEC1, CSTACK, 0);
        int i = 0;
        while (i < callDict[func_name].first){
            print_op(MOV, {SPEC2, to_string(i)});
            print_op(CMP, {SPEC1, SPEC2});
            print_op(BR, {"EQ", func_name + "_back_" + to_string(i)});
            i += 1;
        }
        print_op(BR, {"ALWAYS", "process_end"});
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::arith::ConstantOp op) override{

        auto code = size_t(mlir::hash_value(op->getOpResult(0)));
        auto attr = op.getValueAttr().dyn_cast_or_null<mlir::IntegerAttr>();
        if (attr != nullptr){
            valueTable[code] = varValue(int(attr.getInt()));
        }else{
            auto attr = op.getValueAttr().dyn_cast_or_null<mlir::FloatAttr>();
            if (attr != nullptr){
                valueTable[code] = varValue(attr.getValue().convertToDouble());
            }else{
                return error(op.getLoc(), "error type");
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::memref::AllocOp op) override{
        // create new var name
        // store to symbolTable and indexTable
        auto code = size_t(mlir::hash_value(op->getOpResult(0)));
        auto type = op.getResult().getType().dyn_cast<mlir::MemRefType>();
        int size = type.getShape()[0];

        if (type.getElementType().isa<QStateType>()){
            tempSymbol.insert(make_pair(code, funcStack[nowfunc].second));
            // alloc qbit id in QMEM
            print_op(MOV, {SPEC2, "0"});
            print_label("alloc_start_"+to_string(label_cnt));
            print_op(MOV, {SPEC1, to_string(size)});
            print_op(CMP, {SPEC2, SPEC1});
            print_op(BR, {"GEQ", "alloc_end_"+to_string(label_cnt)});
            print_op(MOV, {SPEC1, to_string(funcStack[nowfunc].second)});
            print_op(ADD, {SPEC1, SPEC1, SPEC2});
            print_op(ADD, {SPEC1, QSTACK, SPEC1});
            print_op(STORE, {QNUM, SPEC1});
            print_op(MOV, {SPEC1, "1"});
            print_op(ADD, {QNUM, QNUM, SPEC1});
            print_op(ADD, {SPEC2, SPEC2, SPEC1});
            print_op(BR, {"ALWAYS", "alloc_start_"+to_string(label_cnt)});
            print_label("alloc_end_"+to_string(label_cnt));
            label_cnt += 1;
            funcStack[nowfunc].second += size;

        }else if(type.getElementType().isa<mlir::IndexType>() || type.getElementType().isa<mlir::IntegerType>()){
            tempSymbol.insert(make_pair(code, funcStack[nowfunc].first));
            funcStack[nowfunc].first += size;
        }else{
            return error(op.getLoc(), "'double' type is not allowed in eqasm.");
        }

        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::memref::DeallocOp op) override{
        // create new var name
        // store to symbolTable and indexTable
        auto code = size_t(mlir::hash_value(op.getMemref()));
        auto type = op.getMemref().getType().dyn_cast<mlir::MemRefType>();
        int size = type.getShape()[0];

        // release qubit
        /*
        if (type.getElementType().isa<QStateType>()){
            int index = tempSymbol[code];
            print_op(MOV, {SPEC1, to_string(index)});
            print_op(ADD, {SPEC1, SPEC1, QSTACK});
            print_op(LOAD, {"qr0", SPEC1});
            print_free_qubit("qr0");
            funcStack[nowfunc].second -= size;
            print_op(MOV, {SPEC1, "1"});
            print_op(SUB, {QNUM, QNUM, SPEC1});
        }*/

        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::memref::CastOp op) override{
        auto icode = size_t(mlir::hash_value(op.getSource()));
        auto ocode = size_t(mlir::hash_value(op.getDest()));
        tempSymbol.insert(make_pair(ocode, tempSymbol[icode]));
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::arith::IndexCastOp op) override{
        auto icode = size_t(mlir::hash_value(op.getIn()));
        auto ocode = size_t(mlir::hash_value(op.getOut()));
        auto v = getValue(icode);
        if (v.first){
            valueTable[ocode] = valueTable[icode].ival;
        }else{
            tempSymbol.insert(make_pair(ocode, tempSymbol[icode]));
        }
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::arith::ExtUIOp op) override{
        auto icode = size_t(mlir::hash_value(op.getIn()));
        auto ocode = size_t(mlir::hash_value(op.getOut()));
        auto v = getValue(icode);
        if (v.first){
            valueTable[ocode] = valueTable[icode].ival;
        }else{
            tempSymbol.insert(make_pair(ocode, tempSymbol[icode]));
        }
        return mlir::success();
    }

    mlir::LogicalResult visitBinaryOp(char op, size_t lcode, size_t rcode, size_t res){
        
        // get lhs's and rhs's value from intValueTable
        TRY(print_load_value(SPEC1, CSTACK, lcode));
        TRY(print_load_value(SPEC2, CSTACK, rcode));
        switch (op)
        {
        case '+':
            print_op(ADD, {SPEC1, SPEC1, SPEC2});
            break;
        case '-':
            print_op(SUB, {SPEC1, SPEC1, SPEC2});
            break;
        case '&':
            print_op(AND, {SPEC1, SPEC1, SPEC2});
            break;
        case '*':
            print_op(MUL, {SPEC1, SPEC1, SPEC2});
            break;
        default:
            return mlir::failure();
            break;
        }
        get_store(SPEC1, SPEC2, CSTACK, funcStack[nowfunc].first);
        tempSymbol.insert(make_pair(res, funcStack[nowfunc].first));
        funcStack[nowfunc].first += 1;

        return mlir::success();
    }
    
    mlir::LogicalResult visitOp(mlir::arith::AddIOp op) override{
        auto lcode = size_t(mlir::hash_value(op.getLhs()));
        auto rcode = size_t(mlir::hash_value(op.getRhs()));
        auto res = size_t(mlir::hash_value(op.getResult()));
        return visitBinaryOp('+', lcode, rcode, res);
    }

    mlir::LogicalResult visitOp(mlir::arith::SubIOp op) override{
        auto lcode = size_t(mlir::hash_value(op.getLhs()));
        auto rcode = size_t(mlir::hash_value(op.getRhs()));
        auto res = size_t(mlir::hash_value(op.getResult()));
        return visitBinaryOp('-', lcode, rcode, res);
    }

    mlir::LogicalResult visitOp(mlir::arith::AndIOp op) override{
        auto lcode = size_t(mlir::hash_value(op.getLhs()));
        auto rcode = size_t(mlir::hash_value(op.getRhs()));
        auto res = size_t(mlir::hash_value(op.getResult()));
        return visitBinaryOp('&', lcode, rcode, res);
    }

    mlir::LogicalResult visitOp(mlir::arith::MulIOp op) override{
        auto lcode = size_t(mlir::hash_value(op.getLhs()));
        auto rcode = size_t(mlir::hash_value(op.getRhs()));
        auto res = size_t(mlir::hash_value(op.getResult()));
        return visitBinaryOp('*', lcode, rcode, res);
    }

    mlir::LogicalResult visitOp(mlir::arith::CmpIOp op) override{
        
        vector<string> reg = {SPEC1, SPEC2};
        // get cmp's lhs and rhs value
        for (auto indexOperand : llvm::enumerate(op->getOperands())){
            auto operand = indexOperand.value();
            size_t code = size_t(mlir::hash_value(operand));
            auto r = reg[indexOperand.index()];
            TRY(print_load_value(r, CSTACK, code));
        }

        print_op(CMP, {SPEC1, SPEC2});
        print_op(MOV, {SPEC1, "1"});
        string br;
        int pred = static_cast<int>(op.getPredicate());
        switch (pred)
        {
        case 0:
            br = "EQ";
            break;
        case 1:
            br = "NEQ";
            break;
        case 2:
            br = "LT";
            break;
        case 3:
            br = "LEQ";
            break;
        case 4:
            br = "GT";
            break;
        case 5:
            br =  "GEQ";
            break;
        default:
            return error(op.getLoc(), "condition is not support in eqasm");
            break;
        }
        print_op(BR, {br, "cmp_value_"+to_string(label_cnt)});
        print_op(MOV, {SPEC1, "0"});
        print_label("cmp_value_"+to_string(label_cnt));
        get_store(SPEC1, SPEC2, CSTACK, funcStack[nowfunc].first);
        auto code = mlir::hash_value(op->getOpResult(0));
        tempSymbol.insert(make_pair(code, funcStack[nowfunc].first));
        funcStack[nowfunc].first += 1;
        label_cnt += 1;

        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::scf::IfOp if_stmt) override{
        // get condition value
        auto condition_code = mlir::hash_value(if_stmt.getCondition());
        auto b_i = getSymbolIndex(condition_code).index;
        if (b_i < 0) return error(if_stmt.getLoc(), "if need a determined condition");
        int now_label_cnt = label_cnt;
        label_cnt += 1;
        get_load(SPEC1, CSTACK, b_i);
        print_op(MOV, {SPEC2, "1"});
        print_op(CMP, {SPEC1, SPEC2});
        print_op(BR, {"NEQ", "else_block_"+to_string(now_label_cnt)});
        TRY(visitBlock(if_stmt.thenBlock()));
        print_op(BR, {"ALWAYS", "end_if_"+to_string(now_label_cnt)});
        print_label("else_block_"+to_string(now_label_cnt));
        TRY(visitBlock(if_stmt.elseBlock()));
        print_label("end_if_"+to_string(now_label_cnt));

        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::scf::ForOp for_stmt) override{
        
        auto block = for_stmt.getBody();

        // get left value
        auto lcode = size_t(mlir::hash_value(for_stmt.getLowerBound()));
        TRY(print_load_value(SPEC1, CSTACK, lcode));
        // store left value to for's arg
        auto arg_code = size_t(mlir::hash_value(block->getArgument(0)));
        get_store(SPEC1, SPEC2, CSTACK, funcStack[nowfunc].first);
        tempSymbol.insert(make_pair(arg_code, funcStack[nowfunc].first));
        funcStack[nowfunc].first += 1;
        int now_label_cnt = label_cnt;
        label_cnt += 1;
        // begin for loop
        print_label("forloop_start_"+to_string(now_label_cnt));
        // get right value, then cmp
        auto rcode = size_t(mlir::hash_value(for_stmt.getUpperBound()));
        TRY(print_load_value(SPEC2, CSTACK, rcode));
        print_op(CMP, {SPEC1, SPEC2});
        print_op(BR, {"GEQ", "forloop_end_"+to_string(now_label_cnt)});
        // visit loop body
        TRY(visitBlock(block));
        // update arg value and goto start
        get_load(SPEC1, CSTACK, getSymbolIndex(arg_code).index);
        auto scode = size_t(mlir::hash_value(for_stmt.getStep()));
        TRY(print_load_value(SPEC2, CSTACK, scode));
        print_op(ADD, {SPEC1, SPEC1, SPEC2});
        get_store(SPEC1, SPEC2, CSTACK, getSymbolIndex(arg_code).index);
        print_op(BR, {"ALWAYS", "forloop_start_"+to_string(now_label_cnt)});
        print_label("forloop_end_"+to_string(now_label_cnt));

        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::AffineLoadOp op) override{
        
        auto result = op.getResult();
        if (result.use_empty()){
            return mlir::success();
        }

        string var_name;
        int index = 0;
        bool is_addr = false;
        // load value from symbolTable using var_name and index
        for (auto indexOperand : llvm::enumerate(op->getOperands())){
            auto operand = indexOperand.value();
            size_t code = size_t(mlir::hash_value(operand));
            if (indexOperand.index() == 0){
                auto tv = getSymbolIndex(code);
                index = tv.index;
                is_addr = tv.is_addr;
                if (index < 0) return error(op.getLoc(), "load value error");
            }else{
                auto v = getValue(code);
                if (v.first == false) return error(op.getLoc(), "index must a determined number");
                index += v.second.ival;
            }
        }

        auto type = op.getMemRefType().getElementType();
        auto code = size_t(mlir::hash_value(op->getOpResult(0)));
        
        if (type.isa<QStateType>()){
            get_load(SPEC1, QSTACK, index);
            if (is_addr) print_op(LOAD, {SPEC1, SPEC1});
            get_store(SPEC1, SPEC2, QSTACK, funcStack[nowfunc].second);
            tempSymbol.insert(make_pair(code, funcStack[nowfunc].second));
            funcStack[nowfunc].second += 1;
        }else{
            get_load(SPEC1, CSTACK, index);
            if (is_addr) print_op(LOAD, {SPEC1, SPEC1});
            get_store(SPEC1, SPEC2, CSTACK, funcStack[nowfunc].first);
            tempSymbol.insert(make_pair(code, funcStack[nowfunc].first));
            funcStack[nowfunc].first += 1;
        }

        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::AffineStoreOp op) override{
        int index;
        bool is_addr = false;
        varValue val;
        string var_name;
        auto type = op.getMemRefType().getElementType();
        // store index value to symtable, qbit ignore
        for (auto indexOperand : llvm::enumerate(op->getOperands())){
            auto operand = indexOperand.value();
            size_t code = size_t(mlir::hash_value(operand));
            if (indexOperand.index() == 0){
                if(operand.getType().isa<QStateType>()) return mlir::success();
                TRY(print_load_value(SPEC2, CSTACK, code));
            }else if (indexOperand.index() == 1){
                auto tv = getSymbolIndex(code);
                index = tv.index;
                is_addr = tv.is_addr;
                if (index < 0) return error(op.getLoc(), "get store left var error");
            }else{
                auto b_i = getValue(code);
                if (b_i.first == false) return error(op.getLoc(), "index must be a determined value");
                index += b_i.second.ival;
            }
        }

        if (is_addr){
            if (index > 0){
                print_op(MOV, {SPEC1, to_string(index)});
                print_op(ADD, {SPEC1, SPEC1, CSTACK});
                print_op(LOAD, {SPEC1, SPEC1});
            }else{
                print_op(LOAD, {SPEC1, CSTACK});
            }
            print_op(STORE, {SPEC2, SPEC1});
        }else{
            get_store(SPEC2, SPEC1, CSTACK, index);
        }
        return mlir::success();
    }

    
    mlir::LogicalResult visitOp(mlir::memref::GetGlobalOp op) override{
        auto name = op.getNameAttr().getValue().str();
        auto type = op.getResult().getType().dyn_cast<mlir::MemRefType>();
        auto code = size_t(mlir::hash_value(op.getResult()));
        // save address to the mem
        if (type.getElementType().isa<QStateType>()){
            print_op(MOV, {SPEC1, to_string(globalSymbol[name] + QMEMSTART)});
            get_store(SPEC1, SPEC2, QSTACK, funcStack[nowfunc].second);
            tempSymbol.insert(make_pair(code, tempValue(funcStack[nowfunc].second, true)));
            funcStack[nowfunc].second += 1;
        }else{
            print_op(MOV, {SPEC1, to_string(globalSymbol[name] + CMEMSTART)});
            get_store(SPEC1, SPEC2, CSTACK, funcStack[nowfunc].first);
            tempSymbol.insert(make_pair(code, tempValue(funcStack[nowfunc].first, true)));
            funcStack[nowfunc].first += 1;
        }

        return mlir::success();
    }
    
    mlir::LogicalResult visitOp(mlir::memref::SubViewOp op) override{
        
        auto type = op.getSource().getType().dyn_cast<mlir::MemRefType>();
        
        auto stack = CSTACK;
        if (type.getElementType().isa<QStateType>()) stack = QSTACK;
        
        // get offset
        if (op.offsets().size() > 0){
            auto fcode = mlir::hash_value(op.offsets()[0]);
            TRY(print_load_value(SPEC1, CSTACK, fcode));
        }else{
            auto offset = op.static_offsets()[0];
            print_op(MOV, {SPEC1, to_string(offset)});
        }
        // get address
        auto scode = mlir::hash_value(op.getSource());
        auto tv = getSymbolIndex(scode);
        if (tv.is_addr){
            get_load(SPEC2, stack, tv.index);   
        }else{
            print_op(MOV, {SPEC2, to_string(tv.index)});
        }
        print_op(ADD, {SPEC1, SPEC2, SPEC1});

        // save address to the mem
        auto rcode = mlir::hash_value(op.getResult());
        if (type.getElementType().isa<QStateType>()){
            if (!tv.is_addr) {
                print_op(MOV, {SPEC2, to_string(QMEMSTART)});
                print_op(ADD, {SPEC1, SPEC1, SPEC2});
            }
            get_store(SPEC1, SPEC2, QSTACK, funcStack[nowfunc].second);
            tempSymbol.insert(make_pair(rcode, tempValue(funcStack[nowfunc].second, true)));
            funcStack[nowfunc].second += 1;
        }else{
            get_store(SPEC1, SPEC2, CSTACK, funcStack[nowfunc].first);
            tempSymbol.insert(make_pair(rcode, tempValue(funcStack[nowfunc].first, true)));
            funcStack[nowfunc].first += 1;
        }
        
        return mlir::success();
    }

    mlir::LogicalResult visitOp(mlir::func::CallOp op) override{

        auto func_name = op.getCalleeAttr().getValue().str();
    
        // push return addresss
        print_op(MOV, {SPEC2, to_string(callCnt[func_name])});
        get_store(SPEC2, SPEC1, CSTACK, funcStack[nowfunc].first);
        
        vector<int> args;
        int ci = funcStack[nowfunc].first + 1;
        int qi = funcStack[nowfunc].second;
        for (auto indexOperand : ::llvm::enumerate(op.getOperands())){
            auto operand = indexOperand.value();
            auto code = mlir::hash_value(operand);
            auto type = operand.getType();
            if (type.isa<mlir::IndexType>() || type.isa<mlir::IntegerType>()){
                TRY(print_load_value(SPEC2, CSTACK, code));
                get_store(SPEC2, SPEC1, CSTACK, ci);
                ci += 1;
            }else if (type.isa<mlir::MemRefType>() || type.isa<QStateType>()){
                if (type.isa<mlir::MemRefType>()){
                    if (type.dyn_cast<mlir::MemRefType>().getShape()[0] > 1) return error(op.getLoc(), "eqasm not support use 'arr' in parameters\n");
                    type = operand.getType().dyn_cast<mlir::MemRefType>().getElementType();
                }
                if (type.isa<QStateType>()){
                    auto tv = getSymbolIndex(code);
                    if (tv.index < 0) return error(op.getLoc(), "call func args need use determined qbit");
                    get_load(SPEC2, QSTACK, tv.index);
                    if (tv.is_addr) print_op(LOAD, {SPEC2, SPEC2});
                    get_store(SPEC2, SPEC1, QSTACK, qi);
                    qi += 1;
                    args.push_back(tv.index);
                }else{
                    return error(op.getLoc(), "eqasm func args only support 'int' and 'qbit' type");
                }
            }else{
                return error(op.getLoc(), "eqasm func args only support 'int' and 'qbit' type");
            }
        }
    
        // push stacks start address
        print_op(MOV, {SPEC1, to_string(funcStack[nowfunc].first+1)});
        print_op(ADD, {CSTACK, CSTACK, SPEC1});
        
        if (funcStack[nowfunc].second > 0){
            print_op(MOV, {SPEC1, to_string(funcStack[nowfunc].second)});
            print_op(ADD, {QSTACK, QSTACK, SPEC1});
        }

        // jump to call fun
        print_op(BR, {"ALWAYS", func_name});

        // return addresss
        print_label(func_name+"_back_"+to_string(callCnt[func_name]));
        callCnt[func_name] += 1;
        
        // pop stack
        if (funcStack[nowfunc].first > 0){
            print_op(MOV, {SPEC1, to_string(funcStack[nowfunc].first)});
            print_op(SUB, {CSTACK, CSTACK, SPEC1});
        }
        if (funcStack[nowfunc].second > 0){
            print_op(MOV, {SPEC1, to_string(funcStack[nowfunc].second)});
            print_op(SUB, {QSTACK, QSTACK, SPEC1});
        }

        for (auto indexResult : llvm::enumerate(op->getResults())){
            auto index = indexResult.index();
            auto result = indexResult.value();
            auto code = mlir::hash_value(result);
            tempSymbol.insert(make_pair(code, args[index]));
        }
        return mlir::success();
    }

    mlir::LogicalResult visitOp(UseGateOp op) override{
        // save gate name in gateTable
        auto attr = op.getNameAttr();
        auto gate = attr.getLeafReference().str();
        auto rcode = mlir::hash_value(op.getResult());
        vector<double> angle;
        for (auto par : op.getParameters()){
            size_t code = size_t(mlir::hash_value(par));
            auto b_f = getValue(code);
            if (b_f.first == false) return error(op.getLoc(), "gate need a determined angle");
            angle.push_back(b_f.second.dval);
        }
        auto name = getGateName(gate);
        //std::cout << "\nGate: " << name << '\n';
        gateTable[rcode] = make_pair(name, angle);
        return mlir::success();
    }

    mlir::LogicalResult visitOp(ApplyGateOp op) override{

        auto code = mlir::hash_value(op.getGate());
        auto gate = gateTable[code].first;
        auto param = gateTable[code].second;

        vector<int> args;
        for (auto indexed_operand : ::llvm::enumerate(op.getArgs())){
            auto index = indexed_operand.index();
            auto operand = indexed_operand.value();
            auto b_i = getSymbolIndex(mlir::hash_value(operand)).index;
            if (b_i < 0) return error(op.getLoc(), "gate operate need use determined qbit");
            args.push_back(b_i);
        }

        get_load(SPEC1, QSTACK, args[0]);
        if (args.size() == 1){
            print_gate(gate, {SPEC1}, param);
        }else{
            get_load(SPEC2, QSTACK, args[1]);
            print_gate(gate, {SPEC1, SPEC2}, param);
        }
        
        for (auto indexed_result : ::llvm::enumerate(op.getR())){
            auto index = indexed_result.index();
            auto result = indexed_result.value();
            auto code = mlir::hash_value(result);
            tempSymbol.insert(make_pair(code, args[index]));
        }

        return mlir::success();
    }

    mlir::LogicalResult visitOp(CallQOpOp op) override {
        string qop_name = op->getAttr(llvm::StringRef("callee")).dyn_cast<mlir::SymbolRefAttr>().getLeafReference().str();
        // only measure support
        if (qop_name != "__isq__builtin__measure") return error(op.getLoc(), "sorry, eqasm can only do measure qop");
        // get measure qbit
        // if qbit has already measured, error
        for (auto indexOperand : llvm::enumerate(op->getOperands())){
            auto operand = indexOperand.value();
            size_t code = size_t(mlir::hash_value(operand));
            TRY(print_load_value(SPEC1, QSTACK, code));
            print_op(MEASE, {SPEC1});
        }
        auto rcode = mlir::hash_value(op.getResult(1));
        get_store(SPEC1, SPEC2, CSTACK, funcStack[nowfunc].first);
        tempSymbol.insert(make_pair(rcode, funcStack[nowfunc].first));
        funcStack[nowfunc].first += 1;
        return mlir::success();
    }

    mlir::LogicalResult visitOp(AccumulateGPhase op) override {
        // qcis don't has gphase, just jump it
        return mlir::success();
    }
    
    mlir::LogicalResult visitOp(AssertOp op) override {
        // qcis don't has gphase, just jump it
        return mlir::success();
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
        transform(gn.begin(), gn.end(), gn_up.begin(), ::toupper);
        return gn_up;
    }

    tempValue getSymbolIndex(size_t code){
        
        auto iter = tempSymbol.find(code);
        if (iter != tempSymbol.end()){
            return iter->second;
        }
        return -1;
    }

    pair<bool, varValue> getValue(size_t code){

        auto iter = valueTable.find(code);
        if (iter != valueTable.end()){
            return make_pair(true, iter->second);
        }
        return make_pair(false, -1);
    }


    void print_label(string lable){
        os << lable << ":\n"; 
    }

    void print_op(string op, vector<string> operand){
        os << "  " << op;
        for(auto od: operand){
            os << ' ' << od; 
        }
        os << '\n';
    }

    void print_gate(string gate, vector<string> operand, vector<double> param){
        os << "  " << gate;
        if (param.size() > 0){
            os << " (";
            for (int i = 0; i < param.size(); i++){
                if (i > 0) os << ',';
                os << param[i];
            }
            os << ")";
        }
        for (auto od: operand){
            os << ' ' << od;
        }
        os << '\n';
    }

    mlir::LogicalResult print_load_value(string creg, string stack, size_t code){
        auto v = getValue(code);
        if (v.first){
            print_op(MOV, {creg, to_string(v.second.ival)});
        }else{
            auto offset = getSymbolIndex(code).index;
            if (offset < 0) return mlir::failure();
            get_load(creg, stack, offset);
        }
        return mlir::success();
    }

    void get_load(string creg, string stack, int offset){
        if (offset > 0){
            print_op(MOV, {creg, to_string(offset)});
            print_op(ADD, {creg, creg, stack});
            print_op(LOAD, {creg, creg});
        }else{
            print_op(LOAD, {creg, stack});
        }

    }

    void get_store(string vreg, string creg, string stack, int offset){
        if (offset > 0){
            print_op(MOV, {creg, to_string(offset)});
            print_op(ADD, {creg, creg, stack});
            print_op(STORE, {vreg, creg});
        }else{
            print_op(STORE, {vreg, stack});
        }
    }

    void print_free_qubit(string qreg){
        
        print_op(MEASE, {qreg});
        print_op(MOV, {SPEC1, "1"});
        print_op(FMR, {SPEC2, qreg});
        print_op(CMP, {SPEC1, SPEC2});
        print_op(BR, {"NQ", "free_end_"+to_string(label_cnt)});
        print_op("X", {qreg});
        print_label("free_end_"+to_string(label_cnt));
        label_cnt += 1;
    }

    void printinfo(){
        std::cout << "-------\nglobal paramter and its offset:\n--------\n";
        for (auto v: globalSymbol){
            std::cout << v.first << ' ' << v.second << std::endl;
        }
        std::cout << "-------\nfunc info:\n-------\n";
        for (auto f: callDict){
            std::cout << f.first << ' ' << f.second.first << std::endl;
        }
    }

    mlir::LogicalResult error(mlir::Location loc, string msg){
        emitError(loc, msg);
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
            os << "  ";
        return os;
    }
    
}; 
}
}


namespace isq {
namespace ir{
mlir::LogicalResult generateEQASM(mlir::MLIRContext &context, mlir::ModuleOp &module, llvm::raw_string_ostream &os, bool printast) {
    return MLIRPassImplEQASM(context, module, os, printast).mlirPass();
}
}
}