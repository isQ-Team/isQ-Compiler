#include "ANTLRInputStream.h"
#include "CommonTokenStream.h"
#include "ISQLexer.h"
#include "ISQParser.h"
#include "ISQParserBaseVisitor.h"

#include <iostream>
#include <map>
#include <set>
#include <vector>
#include "isq/Dialect.h"
#include <isq/IR.h>

#include "isq/QAttrs.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/IntegerSet.h"


#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"
#include <mlir/InitAllDialects.h>
#include "mlir/Support/MlirOptMain.h"
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/IR/OperationSupport.h"


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <complex>

using namespace isq::ir;
using namespace isq;
using namespace std;
using namespace antlr4;

namespace {


//===----------------------------------------------------------------------===//
// Implementation of a simple MLIR emission from the isq file.
class MLIRGenImpl: public ISQParserBaseVisitor{

public:
    MLIRGenImpl(mlir::MLIRContext &context) : builder(&context), context(&context) {}

    mlir::ModuleOp mlirGen(llvm::StringRef inputfile){
        
        filename = std::make_shared<std::string>(std::move(inputfile.str()));

        theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

        intergerSetInit();
                
        ifstream ifs;
        ifs.open(string(inputfile).c_str());
        ANTLRInputStream input(ifs);
        ISQLexer lexer(&input);
        CommonTokenStream tokens(&lexer);
        ISQParser parser(&tokens);
        ISQParser::ProgramContext* program = parser.program();

        if (parser.getNumberOfSyntaxErrors() > 0){
            //cout << "isq file syntax error" << endl;
            return nullptr;
        }

        nowBlock = "";
        this->visit(program);

        ifs.close();
        return theModule;
    }


private:

    mlir::ModuleOp theModule;
    mlir::ModuleOp globalModule;
    mlir::OpBuilder builder;
    mlir::MLIRContext* context;
    std::shared_ptr<std::string> filename;
    
    string builtinName;
    map<string, bool> producer;

    llvm::SmallVector<llvm::SmallVector<mlir::Attribute>> gate;
    
    mlir::Type nowType;
    map<string, pair<int, mlir::Type>>globalSymbolMap;
    map<string, pair<int, mlir::Type>>::iterator globaliter;

    map<string, pair<int, mlir::Type>>globalGateMap;
    map<string, pair<int, mlir::Type>>::iterator globalgateiter;

    map<string, map<string, mlir::Value>> localGateMap;
    map<string, map<string, mlir::Value>>::iterator localgatemapiter;
    map<string, mlir::Value>::iterator localgateiter;
    
    map<string, map<string, mlir::Value>> localSymbalMap;
    map<string, map<string, mlir::Value>>::iterator localmapiter;
    map<string, mlir::Value>::iterator localiter;

    map<string, map<string, tuple<int, mlir::Type, bool, bool>>> localTypeMap;
    map<string, map<string, tuple<int, mlir::Type, bool, bool>>>::iterator localtypemapiter;
    map<string, tuple<int, mlir::Type, bool, bool>>::iterator localtypeiter;

    map<string, map<int, mlir::Value>> localConstant;
    map<string, map<int, mlir::Value>> localIndex;

    string nowBlock;
    
    map<string, llvm::SmallVector<mlir::Type>> parArray;
    map<string, vector<tuple<string, int, mlir::Type>>> protoArgs;
    
    llvm::SmallVector<tuple<string, mlir::Value, mlir::Value>> varList;

    llvm::SmallVector<mlir::Value> returnval;
    mlir::Value callres;

    mlir::IntegerSet greateSet;
    mlir::IntegerSet greateEqualSet;
    mlir::IntegerSet equalSet;
    mlir::IntegerSet lessEqualSet;
    mlir::IntegerSet lessSet;
    mlir::AffineMap singleSymbol;

    void intergerSetInit(){
        mlir::AffineExpr d0, d1;
        bindDims(context, d0, d1);
        greateSet = mlir::IntegerSet::get(2, 0, {d0 - d1 - 1}, {false});
        //auto greateSetAttr = mlir::IntegerSetAttr::get(greateSet);
        greateEqualSet = mlir::IntegerSet::get(2, 0, {d0 - d1}, {false});
        equalSet = mlir::IntegerSet::get(2, 0, {d0 - d1}, {true});
        lessEqualSet = mlir::IntegerSet::get(2, 0, {d1 - d0 - 1}, {false});
        lessSet = mlir::IntegerSet::get(2, 0, {d1 - d0}, {false});

        auto s0 = getAffineSymbolExpr(0, context); (void)s0;
        singleSymbol = mlir::AffineMap::get(0, 1, {s0}, context);
    }

    mlir::IntegerSet getSetFromAsso(string asso){
        if (asso.compare("==") == 0){
            return equalSet;
        }else if (asso.compare("<=") == 0)
        {
            return lessEqualSet;
        }else if (asso.compare("<") == 0)
        {
            return lessSet;
        }else if (asso.compare(">=") == 0)
        {
            return greateEqualSet;
        }else{
            return greateSet;
        }
    }

    mlir::CmpIPredicate getCmpFromAsso(string asso){
        if (asso.compare("==") == 0){
            return mlir::CmpIPredicate::eq;
        }else if (asso.compare("<=") == 0)
        {
            return mlir::CmpIPredicate::sle;
        }else if (asso.compare("<") == 0)
        {
            return mlir::CmpIPredicate::slt;
        }else if (asso.compare(">=") == 0)
        {
            return mlir::CmpIPredicate::sge;
        }else{
            return mlir::CmpIPredicate::sgt;
        }
    }

    mlir::Location loc(antlr4::Token* token) {
        return mlir::FileLineColLoc::get(builder.getIdentifier(*filename), token->getLine(), token->getCharPositionInLine());
    }

    void createSymbolTable(string id){
        map<string, mlir::Value> localmap;
        map<string, tuple<int, mlir::Type, bool, bool>>localtypemap;
        map<string, mlir::Value> localgatemap;
        map<int, mlir::Value> localconstant;
        map<int, mlir::Value> localindex;
        
        localSymbalMap.insert(pair<string, map<string, mlir::Value>>(id, localmap));
        localTypeMap.insert(pair<string, map<string, tuple<int, mlir::Type, bool, bool>>>(id, localtypemap));
        localGateMap.insert(make_pair(id, localgatemap));
        localConstant.insert(make_pair(id, localconstant));
        localIndex.insert(make_pair(id, localindex));
    }

    void deleteSymbolTable(string id){
        localSymbalMap.erase(id);
        localTypeMap.erase(id);
        localGateMap.erase(id);
        localConstant.erase(id);
        localIndex.erase(id);
    }

    vector<string> getBlockList(string block){
        vector<string> res;
        string s = "";
        for (int i = 0; i < block.size(); i++){
            if (block[i] == '.'){
                //cout << s << endl;
                res.push_back(s);
            }
            s += block[i];
        }
        res.push_back(s);
        return res;
    }

    mlir::LogicalResult defineQop(string id, uint64_t size, mlir::TypeRange input, mlir::TypeRange output, antlr4::Token* token){
        
        mlir::TypeAttr funcAttr = mlir::TypeAttr::get(mlir::FunctionType::get(context, input, output));
        llvm::StringRef sym_name = llvm::StringRef(id);
        llvm::StringRef sym_visibility = llvm::StringRef("nested");

        auto qOp = builder.create<DeclareQOpOp>(loc(token), sym_name, sym_visibility, size, funcAttr);
        //globalModule.push_back(qOp);
        return mlir::success();
    }

    mlir::ArrayAttr arrayToDef(mlir::ArrayAttr arr){
        return builder.getArrayAttr({builder.getDictionaryAttr({builder.getNamedAttr("type", builder.getStringAttr("unitary")), builder.getNamedAttr("value", arr)})});
        
    }
    mlir::LogicalResult defineGate(string id, int shape, bool def, antlr4::Token* token){

        mlir::TypeAttr gateAttr = mlir::TypeAttr::get(isq::ir::GateType::get(context, shape, GateTrait::General));
        llvm::StringRef sym_name = llvm::StringRef(id);
        llvm::StringRef sym_visibility = llvm::StringRef("nested");
        
        mlir::ArrayAttr arr = nullptr;
        if (def){
            llvm::SmallVector<mlir::Attribute> definition;
            for (int i = 0; i < gate.size(); i++){
                definition.push_back(builder.getArrayAttr(gate[i]));
            }
            arr = builder.getArrayAttr(definition);
        }
        
        if (failed(gateInsert(id, shape, isq::ir::GateType::get(context, shape, GateTrait::General)))){
            emitError(loc(token), "gate '") << id << "' already defined";
            return mlir::failure();
        }
        
        auto gateOp = builder.create<DefgateOp>(loc(token), gateAttr, sym_name, sym_visibility, nullptr, arr?arrayToDef(arr):nullptr);

        //globalModule.push_back(gateOp);
        return mlir::success();

    }

    mlir::LogicalResult gateInsert(string id, int shape, mlir::Type type){

        globalgateiter = globalGateMap.find(id);
        if (globalgateiter != globalGateMap.end()){
            return mlir::failure();
        }
        globalGateMap.insert(make_pair(id, make_pair(shape, type)));
        return mlir::success();
    }

    mlir::LogicalResult findGlobalGate(string id, int idx, antlr4::Token* token){
        globalgateiter = globalGateMap.find(id);
        if (globalgateiter != globalGateMap.end()){
            return mlir::success();
        }

        mlir::Block* nowblock = builder.getInsertionBlock();
        builder.setInsertionPointToEnd(&globalModule.body().front());
        // define common gate
        bool flag = true;
        switch (idx)
        {
            case 1:
                if (failed(defineGate("H", 1, false, token)))
                    flag = false;
                break;
            case 2:
                if (failed(defineGate("X", 1, false, token)))
                    flag = false;
                break;
            case 3:
                if (failed(defineGate("Y", 1, false, token)))
                    flag = false;
                break;
            case 4:
                if (failed(defineGate("Z", 1, false, token)))
                    flag = false;
                break;
            case 5:
                if (failed(defineGate("T", 1, false, token)))
                    flag = false;
                break;
            case 6:
                if (failed(defineGate("S", 1, false, token)))
                    flag = false;
                break;
            case 7:
                if (failed(defineGate("CX", 2, false, token)))
                    flag = false;
                break;
            case 8:
                if (failed(defineGate("XZ", 2, false, token)))
                    flag = false;
                break;
            case 9:
                if (failed(defineGate("CNOT", 2, false, token)))
                    flag = false;
                break;
            default:
                flag = false;
                break;
        }
        builder.setInsertionPointToEnd(nowblock);
        
        if (flag)
            return mlir::success();
        return mlir::failure();
    }

    mlir::Value getLocalGate(string id){

        auto blocklist = getBlockList(nowBlock);
        for (int i = blocklist.size()-1; i >= 0; i--){
            localgatemapiter = localGateMap.find(blocklist[i]);
            if (localgatemapiter != localGateMap.end()){
                localgateiter = localgatemapiter->second.find(id);
                if (localgateiter != localgatemapiter->second.end()){
                    return localgateiter->second;
                }
            }
        }
        return nullptr;
    }

    mlir::LogicalResult setLocalGate(string id, int idx, antlr4::Token* token){

        if (failed(findGlobalGate(id, idx, token))){
            emitError(loc(token), "gate '") << id << "' not defined";
            return mlir::failure();
        }
        
        if (getLocalGate(id) != nullptr)
            return mlir::success();
        
        localgatemapiter = localGateMap.find(nowBlock);
        if (localgatemapiter != localGateMap.end()){
            localgateiter = localgatemapiter->second.find(id);
            if (localgateiter == localgatemapiter->second.end()){
                globalgateiter = globalGateMap.find(id);
                mlir::Type type = globalgateiter->second.second;
                //mlir::SymbolRefAttr sym_name = mlir::FlatSymbolRefAttr::get(context, llvm::StringRef(id));
                llvm::SmallVector<mlir::FlatSymbolRefAttr> nestedReferences;
                nestedReferences.push_back(mlir::FlatSymbolRefAttr::get(context, llvm::StringRef(id)));
                mlir::SymbolRefAttr sym_name = mlir::SymbolRefAttr::get(context, llvm::StringRef(builtinName), nestedReferences);
                auto gate = builder.create<UseGateOp>(loc(token), type, sym_name);
                localgatemapiter->second.insert(make_pair(id, gate));
                return mlir::success();
            }
        }
        emitError(loc(token), "gate '") << id << "' set error";
        return mlir::failure();
    }

    mlir::LogicalResult globalVarDefined(string id){
        globaliter = globalSymbolMap.find(id);
        if (globaliter != globalSymbolMap.end()){
            return mlir::success();
        }
        return mlir::failure();
    }

    mlir::LogicalResult localVarDefined(string id){

        auto blocklist = getBlockList(nowBlock);
        localmapiter = localSymbalMap.find(blocklist[blocklist.size()-1]);
        if (localmapiter != localSymbalMap.end()){
            localiter = localmapiter->second.find(id);
            if (localiter != localmapiter->second.end()){
                return mlir::success();
            }
        }

        return mlir::failure();
    }

    void setType(ISQParser::VarTypeContext* varType){
        
        if(varType->Int() != nullptr){
            nowType = builder.getI32Type();
        }else{
            nowType = QStateType::get(context);
        }
    }

    mlir::LogicalResult varInsert(string id, mlir::Value value){
        auto blocklist = getBlockList(nowBlock);
        for (int i = blocklist.size()-1; i >= 0; i--){
            localmapiter = localSymbalMap.find(blocklist[i]);
            if (localmapiter != localSymbalMap.end()){
                localiter = localmapiter->second.find(id);
                if (localiter != localmapiter->second.end()){
                    localmapiter->second.erase(localiter);
                    localmapiter->second.insert(make_pair(id, value));
                    return mlir::success();
                }
            }
        }

        localmapiter = localSymbalMap.find(blocklist[blocklist.size()-1]);
        if (localmapiter != localSymbalMap.end()){
            localmapiter->second.insert(make_pair(id, value));
            return mlir::success();
        }
        return mlir::failure();
    }

    mlir::LogicalResult varLocalAdd(string id, mlir::Value value){
        auto blocklist = getBlockList(nowBlock);
        localmapiter = localSymbalMap.find(blocklist[blocklist.size()-1]);
        if (localmapiter != localSymbalMap.end()){
            localmapiter->second.insert(make_pair(id, value));
            return mlir::success();
        }
        return mlir::failure();
    }

    mlir::LogicalResult typeInsert(string id, int num, mlir::Type type, bool isGlobal, bool isPara){
        auto blocklist = getBlockList(nowBlock);
        for (int i = blocklist.size()-1; i >= 0; i--){
            localtypemapiter = localTypeMap.find(blocklist[i]);
            if (localtypemapiter != localTypeMap.end()){
                localtypeiter = localtypemapiter->second.find(id);
                if (localtypeiter != localtypemapiter->second.end()){
                    localtypemapiter->second.erase(localtypeiter);
                    localtypemapiter->second.insert(make_pair(id, make_tuple(num, type, isGlobal, isPara)));
                    return mlir::success();
                }
            }
        }

        localtypemapiter = localTypeMap.find(blocklist[blocklist.size()-1]);
        if (localtypemapiter != localTypeMap.end()){
            localtypemapiter->second.insert(make_pair(id, make_tuple(num, type, isGlobal, isPara)));
            return mlir::success();
        }
        return mlir::failure();
    }

    mlir::LogicalResult typeLocalAdd(string id, int num, mlir::Type type, bool isGlobal, bool isPara){
        auto blocklist = getBlockList(nowBlock);
        localtypemapiter = localTypeMap.find(blocklist[blocklist.size()-1]);
        if (localtypemapiter != localTypeMap.end()){
            localtypemapiter->second.insert(make_pair(id, make_tuple(num, type, isGlobal, isPara)));
            return mlir::success();
        }
        return mlir::failure();
    }

    mlir::LogicalResult defineVar(antlr4::Token* token, string id, int num, bool isArray){
        if (nowBlock == ""){
            if (succeeded(globalVarDefined(id))){
                emitError(loc(token), "global var '") << id << "' has already defined";
                return mlir::failure();
            }

            llvm::SmallVector<int64_t> shape = {num};
            auto globalOp = builder.create<mlir::memref::GlobalOp>(loc(token), llvm::StringRef(id), nullptr, mlir::MemRefType::get(shape, nowType), nullptr, false);
            //globalOp.dump();
            //globalModule.push_back(globalOp);

            if (!isArray){
                num = -1;
            }
            globalSymbolMap.insert(pair<string, pair<int, mlir::Type>>(id, pair<int, mlir::Type>(num, nowType)));

        }else{

            if (succeeded(localVarDefined(id))){
                emitError(loc(token), "local var '") << id << "' has already defined";
                return mlir::failure();
            }
            
            mlir::Value val = nullptr;
            llvm::SmallVector<int64_t> shape = {num};
            val = builder.create<mlir::memref::AllocOp>(loc(token), mlir::MemRefType::get(shape, nowType));
            
            if (!isArray){
                num = -1;
            }
            if (failed(varLocalAdd(id, val)) | failed(typeLocalAdd(id, num, nowType, false, false))){
                emitError(loc(token), "local var '") << id << "' alloc error";
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::Value getLocalVar(string id){
        auto blocklist = getBlockList(nowBlock);
        for (int i = blocklist.size()-1; i >= 0; i--){
            localmapiter = localSymbalMap.find(blocklist[i]);
            if (localmapiter != localSymbalMap.end()){
                localiter = localmapiter->second.find(id);
                if (localiter != localmapiter->second.end()){
                    return localiter->second;
                }
            }
        }
        return nullptr;
    }

    tuple<int, mlir::Type, bool, bool> getTypeById(string id){
        auto blocklist = getBlockList(nowBlock);
        for (int i = blocklist.size()-1; i >= 0; i--){

            localtypemapiter = localTypeMap.find(blocklist[i]);
            if (localtypemapiter != localTypeMap.end()){
                localtypeiter = localtypemapiter->second.find(id);
                if (localtypeiter != localtypemapiter->second.end()){
                    return localtypeiter->second;
                }
            }
        }
        return make_tuple(-1, nullptr, false, false);
    }

    mlir::Value setLocalVar(antlr4::Token* token, string id){
        
        if (failed(globalVarDefined(id))){
            emitError(loc(token), "var '") << id << "' has not defined";
            return nullptr;
        }

        int shape = globaliter->second.first;
        int num = shape > 0 ? shape: 1;
        mlir::Type type = globaliter->second.second;
        
        auto val = builder.create<mlir::memref::GetGlobalOp>(loc(token), mlir::MemRefType::get(num, type), mlir::FlatSymbolRefAttr::get(context, llvm::StringRef(id)));
        
        if (failed(declare(id, val, shape, type, true, false))){
            emitError(loc(token), "gloabl var '") << id << "' get error";
            return nullptr;
        }

        return val;
    }

    mlir::LogicalResult declare(string id, mlir::Value value, int num, mlir::Type type, bool isGlobal, bool isPara) {
        
        mlir::Value val = getLocalVar(id);
        if (!val){
            if (failed(varLocalAdd(id, value)) | failed(typeLocalAdd(id, num, type, isGlobal, isPara)))
                return mlir::failure();
            return mlir::success();
        }
        return mlir::failure();
    }

    mlir::Value getConstant(antlr4::Token* token, int id){
        auto blocklist = getBlockList(nowBlock);
        for (int i = blocklist.size()-1; i >= 0; i--){
            auto iter = localConstant.find(blocklist[i]);
            if (iter != localConstant.end()){
                auto iter2 = iter->second.find(id);
                if (iter2 != iter->second.end()){
                    return iter2->second;
                }
            }
        }
        auto iter = localConstant.find(blocklist[blocklist.size()-1]);
        if (iter != localConstant.end()){
            mlir::Value val = builder.create<mlir::ConstantOp>(loc(token), builder.getI32IntegerAttr(id));
            iter->second.insert(make_pair(id, val));
            return val;
        }
        return nullptr;
    }

    mlir::Value getIndex(antlr4::Token* token, int id){
        auto blocklist = getBlockList(nowBlock);
        for (int i = blocklist.size()-1; i >= 0; i--){

            auto iter = localIndex.find(blocklist[i]);
            if (iter != localIndex.end()){
                auto iter2 = iter->second.find(id);
                if (iter2 != iter->second.end()){
                    return iter2->second;
                }
            }
        }

        auto iter = localIndex.find(blocklist[blocklist.size()-1]);
        if (iter != localIndex.end()){
            mlir::Value val = builder.create<mlir::ConstantOp>(loc(token), builder.getIndexAttr(id));
            iter->second.insert(make_pair(id, val));
            return val;
        }
        return nullptr;
    }
    
    antlrcpp::Any visitProgram(ISQParser::ProgramContext *ctx){

        builtinName = "isq.buildin";
        globalModule = mlir::ModuleOp::create(loc(ctx->getStart()), llvm::StringRef(builtinName));
        builder.setInsertionPointToStart(&globalModule.body().front());
        // declare Qop
        // M
        mlir::TypeRange input(llvm::None);
        llvm::SmallVector<mlir::Type> outputArr;
        outputArr.push_back(builder.getI32Type());
        mlir::TypeRange output(outputArr);
        if (failed(defineQop("measure", 1, input, output, ctx->getStart()))){
            emitError(loc(ctx->getStart()), "Measurement define error");
            theModule = nullptr;
            return -1;
        }
        //reset
        output = llvm::None;
        if (failed(defineQop("reset", 1, input, output, ctx->getStart()))){
            emitError(loc(ctx->getStart()), "reset define error");
            theModule = nullptr;
            return -1;
        }

        // user-defined gate
        int defnum = ctx->gateDefclause().size();
        //cout << "def model number: " << defnum << endl;
        for (int i = 0; i < defnum; i++){
            int shape = visit(ctx->gateDefclause(i)).as<int>();
            if( shape == -1){
                theModule = nullptr;
                return -1;
            }
         
            string id = ctx->gateDefclause(i)->Identifier()->getText();
            
            if (failed(defineGate(id, shape, true, ctx->gateDefclause(i)->getStart()))){
                theModule = nullptr;
                return -1;
            }
        }
        builder.setInsertionPointToStart(&theModule.body().front());
        
        for (int i = 0; i < ctx->defclause().size(); i++){
            int res = visit(ctx->defclause(i)).as<int>();
            if (res == -1){
                theModule = nullptr;
                return -1;
            }
        }

        theModule.push_back(globalModule);

        int res = visit(ctx->programBody()).as<int>();
        if (res == -1){
            theModule = nullptr;
            return -1;
        }
        return 0;
    }

    antlrcpp::Any visitGateDefclause(ISQParser::GateDefclauseContext *ctx){
        //cout << "new gate: " << ctx->Identifier()->getText() << endl;
        string id = ctx->Identifier()->getText();

        for (int i = 0; i < gate.size(); i++){
            gate[i].clear();
        }
        gate.clear();
        
        llvm::SmallVector<mlir::Attribute> new_line;
        gate.push_back(new_line);
        visit(ctx->matrixContents());
        
        int d1 = gate.size();
        int d2 = gate[0].size();
        if (d1 != d2){
            emitError(loc(ctx->Identifier()->getSymbol()), "gate '") << id << "' dim error";
            return -1;
        }
        for (int i = 0; i < d1; i++){
            if (gate[i].size() != d2){
                emitError(loc(ctx->Identifier()->getSymbol()), "gate '") << id << "' dim error";
                return -1;
            }
        }

        int d = gate.size();
        int qnum = 0;
        while (d > 1){
            if (d % 2 == 1){
                emitError(loc(ctx->Identifier()->getSymbol()), "gate '") << id << "' dim error";
                return -1;
            }
            qnum += 1;
            d /= 2;
        }
        //cout << endl;
        return qnum;
    }

    antlrcpp::Any visitMatrixvaldef(ISQParser::MatrixvaldefContext *ctx){
        string cnum_str = visit(ctx->cNumber()).as<string>();
        gate[gate.size()-1].push_back(isq::ir::ComplexF64Attr::get(this->context, ::llvm::APFloat(last_number.real()), ::llvm::APFloat(last_number.imag())));
        //gate[gate.size()-1].push_back(builder.getStringAttr(cnum_str));
        return 0;
    }

    antlrcpp::Any visitMatrixdef(ISQParser::MatrixdefContext *ctx){

        string cnum_str = visit(ctx->cNumber()).as<string>();
        gate[gate.size()-1].push_back(isq::ir::ComplexF64Attr::get(this->context, ::llvm::APFloat(last_number.real()), ::llvm::APFloat(last_number.imag())));

        if (ctx->Semi() != nullptr){
            llvm::SmallVector<mlir::Attribute> new_line;
            gate.push_back(new_line);
        }
        visit(ctx->matrixContents());
        return 0;
    }
    
    std::complex<double> last_number;
    antlrcpp::Any visitCNumber(ISQParser::CNumberContext *ctx){
        
        string cnum_str = "";
        if (ctx->Minus() != nullptr){
            cnum_str += '-';
        }
        cnum_str += visit(ctx->numberExpr()).as<string>();
        if (ctx->Minus() != nullptr){
            last_number.real(-last_number.real());
        }
        return cnum_str;
    }

    antlrcpp::Any visitNumberExpr(ISQParser::NumberExprContext *ctx){
        //cout << ctx->Number(0)->getText();
        string num_str = ctx->Number(0)->getText();
        last_number = std::complex<double>();
        last_number.real(atof(num_str.c_str()));
        if (ctx->Number().size() > 1){
            if (ctx->Plus() != nullptr){
                num_str += '+';
            }else{
                num_str += '-';
            }
            last_number.imag(atof(ctx->Number(1)->getText().c_str()));
            if (ctx->Minus()) last_number.imag(-last_number.imag());
            num_str += ctx->Number(1)->getText();
        }
        return num_str;
    }

    antlrcpp::Any visitDefclause(ISQParser::DefclauseContext *ctx){
        visit(ctx->varType());
        int res = visit(ctx->idlist()).as<int>();
        if (res == -1)
            return res;
        //cout << endl;
        return 0;
    }

    antlrcpp::Any visitVarType(ISQParser::VarTypeContext *ctx){
        setType(ctx);
        return 0;
    }

    antlrcpp::Any visitIdlistdef(ISQParser::IdlistdefContext *ctx){
        int res = visit(ctx->idlist(0)).as<int>();
        if (res == -1)
            return res;
        return visit(ctx->idlist(1));
    }

    antlrcpp::Any visitArraydef(ISQParser::ArraydefContext *ctx){

        string id = ctx->Identifier()->getText();
        int num = atoi(ctx->Number()->getText().c_str());

        if (failed(defineVar(ctx->Identifier()->getSymbol(), id, num, true))){
            return -1;
        }

        return 0;
    }

    antlrcpp::Any visitSingleiddef(ISQParser::SingleiddefContext *ctx){
        //cout << (ctx->Identifier()->getText());
        
        string id = ctx->Identifier()->getText();
        
        if (failed(defineVar(ctx->Identifier()->getSymbol(), id, 1, false))){
            return -1;
        }
        return 0;
    }

    antlrcpp::Any visitProgramBody(ISQParser::ProgramBodyContext *ctx){
        for (int i = 0; i < ctx->procedureBlock().size(); i++){
            int res = visit(ctx->procedureBlock(i)).as<int>();
            if (res == -1)
                return res;
        }
        return visit(ctx->procedureMain());
    }

    antlrcpp::Any visitProcedureBlock(ISQParser::ProcedureBlockContext *ctx){
        //cout << "def producer: " << ctx->Identifier()->getText() << endl;

        string id = ctx->Identifier()->getText();
        auto iter = producer.find(id);
        if (iter != producer.end()){
            emitError(loc(ctx->Identifier()->getSymbol()), "producer '") << id << "' already defined";
            return -1;
        }

        bool has_return = false;
        if (ctx->Int() != nullptr){
            has_return = true;
        }

        if (!has_return && ctx->procedureBody()->returnStatement() != nullptr){
            emitError(loc(ctx->Identifier()->getSymbol()), "producer '") << id << "' should'n has return";
            return -1;
        }

        if (has_return && ctx->procedureBody()->returnStatement() == nullptr){
            emitError(loc(ctx->Identifier()->getSymbol()), "producer '") << id << "' need has return";
            return -1;
        }
        producer.insert(make_pair(id, has_return));

        returnval.clear();

        nowBlock = id;
        createSymbolTable(id);

        llvm::SmallVector<mlir::Type> returntype;
        if (has_return)
            returntype.push_back(builder.getI32Type());
        auto func_type = builder.getFunctionType(llvm::None, returntype);
        
        vector<tuple<string, int, mlir::Type>> args;
        llvm::SmallVector<mlir::Type> argsType;
        protoArgs.insert(make_pair(id, args));
        parArray.insert(make_pair(id, argsType));
        
        if (ctx->callParas() != nullptr){
            //cout << "params: ";
            visit(ctx->callParas());
            // get return type which depend the single qbit number in parameters
            for (auto name_type : protoArgs.find(id)->second){
                if (get<1>(name_type) == -1 && get<2>(name_type).isa<QStateType>()){
                    returntype.push_back(QStateType::get(context));
                }
            }
            func_type = builder.getFunctionType(parArray.find(id)->second, returntype);
            //cout << endl;
        }
        auto funcOp = mlir::FuncOp::create(loc(ctx->Identifier()->getSymbol()), llvm::StringRef(id), func_type);
        auto &entryBlock = *funcOp.addEntryBlock();

        builder.setInsertionPointToStart(&entryBlock);
        theModule.push_back(funcOp);

        for (const auto nameValue : llvm::zip(protoArgs.find(id)->second, entryBlock.getArguments())){
            mlir::Value val = get<1>(nameValue);
            if (get<1>(get<0>(nameValue)) == -1){
                llvm::SmallVector<int64_t> shape = {1};
                auto allocOp = builder.create<mlir::memref::AllocOp>(loc(ctx->Identifier()->getSymbol()), mlir::MemRefType::get(shape, get<2>(get<0>(nameValue))));
                auto index = getIndex(ctx->Identifier()->getSymbol(), 0);
                builder.create<mlir::AffineStoreOp>(loc(ctx->Identifier()->getSymbol()), val, allocOp, mlir::ValueRange(index));
                val = allocOp;
            }
            if (failed(declare(get<0>(get<0>(nameValue)), val, get<1>(get<0>(nameValue)), get<2>(get<0>(nameValue)), false, true))){
                emitError(loc(ctx->Identifier()->getSymbol()), "producer '") << id << "' define param error";
                return -1;
            }
        }

        int res = visit(ctx->procedureBody()).as<int>();
        if (res < 0)
            return res;
        
        // return int and single qbit value
        for (auto name_type : protoArgs.find(id)->second){
            if (get<1>(name_type) == -1 && get<2>(name_type).isa<QStateType>()){
                auto val = getLocalVar(get<0>(name_type));
                auto index = getIndex(ctx->Identifier()->getSymbol(), 0);
                auto res = builder.create<mlir::AffineLoadOp>(loc(ctx->Identifier()->getSymbol()), val, mlir::ValueRange(index));
                returnval.push_back(res);
            }
        }
        builder.create<mlir::ReturnOp>(loc(ctx->Identifier()->getSymbol()), mlir::ValueRange(returnval));

        return 0;
    }

    antlrcpp::Any visitCallParas(ISQParser::CallParasContext *ctx){
        
        if (ctx->callParas().size() > 0){
            for (int i = 0; i < ctx->callParas().size(); i++){
                visit(ctx->callParas(i));
                //cout << ", ";
            }
        }else{
            bool isArray = false;
            visit(ctx->varType());
            //cout << ctx->Identifier()->getText();
            if (ctx->LeftBracket() != nullptr){
                isArray = true;
                //cout << ctx->LeftBracket()->getText() << ctx->RightBracket()->getText();
            }
            mlir::Type type = nowType;
            int num = -1;
            if (isArray){
                parArray.find(nowBlock)->second.push_back(mlir::MemRefType::get(0, type));
                num = 0;
            }else{
                parArray.find(nowBlock)->second.push_back(type);
            }
            protoArgs.find(nowBlock)->second.push_back(make_tuple(ctx->Identifier()->getText(), num, type));
        }
        return 0;
    }

    antlrcpp::Any visitProcedureMain(ISQParser::ProcedureMainContext *ctx){
        //cout << "def producer: " << ctx->Main()->getText() << endl;
        string id = ctx->Main()->getText();
        
        if (ctx->procedureBody()->returnStatement() != nullptr){
            emitError(loc(ctx->Main()->getSymbol()), "producer '") << id << "' should'n has return";
            return -1;
        }
        
        nowBlock = id;
        createSymbolTable(id);
        
        auto func_type = builder.getFunctionType(llvm::None, llvm::None);
        auto funcOp = mlir::FuncOp::create(loc(ctx->Main()->getSymbol()), llvm::StringRef(ctx->Main()->getText()), func_type);
        auto &entryBlock = *funcOp.addEntryBlock();
        builder.setInsertionPointToStart(&entryBlock);
        
        theModule.push_back(funcOp);
        
        int res = visit(ctx->procedureBody()).as<int>();
        if (res < 0)
            return res;
        
        builder.create<mlir::ReturnOp>(loc(ctx->Main()->getSymbol()), mlir::ValueRange(llvm::None));

        return 0;
    }

    antlrcpp::Any visitProcedureBody(ISQParser::ProcedureBodyContext *ctx){
        int res = visit(ctx->statementBlock()).as<int>();
        if (res < 0)
            return res;
        if (ctx->returnStatement() != nullptr)
            return visit(ctx->returnStatement());
        return 0;
    }

    antlrcpp::Any visitStatementBlock(ISQParser::StatementBlockContext *ctx){  
        for (int i = 0; i < ctx->statement().size(); i++){
            //cout << "statement " << i << endl;
            int res = visit(ctx->statement(i)).as<int>();
            if (res < 0)
                return res;
        }
        return 0;
    }

    antlrcpp::Any visitQbitinitdef(ISQParser::QbitinitdefContext *ctx){
        return visit(ctx->qbitInitStatement());
    }

    antlrcpp::Any visitCinassigndef(ISQParser::CinassigndefContext *ctx){   
        return visit(ctx->cintAssign());
    }

    antlrcpp::Any visitFordef(ISQParser::FordefContext *ctx){   
        return visit(ctx->forStatement());
    }

    antlrcpp::Any visitPrintdef(ISQParser::PrintdefContext *ctx){   
        return visit(ctx->printStatement());
    }

    antlrcpp::Any visitCalldef(ISQParser::CalldefContext *ctx){   
        return visit(ctx->callStatement());
    }

    antlrcpp::Any visitQbitunitarydef(ISQParser::QbitunitarydefContext *ctx){   
        return visit(ctx->qbitUnitaryStatement());
    }

    antlrcpp::Any visitWhiledef(ISQParser::WhiledefContext *ctx){   
        return visit(ctx->whileStatement());
    }

    antlrcpp::Any visitIfdef(ISQParser::IfdefContext *ctx){ 
        return visit(ctx->ifStatement());
    }

    antlrcpp::Any visitPassdef(ISQParser::PassdefContext *ctx){
        return 0;
    }

    antlrcpp::Any visitVardef(ISQParser::VardefContext *ctx){
        return visit(ctx->defclause());
    }

    antlrcpp::Any visitUGate(ISQParser::UGateContext *ctx){
        if (ctx->H() != nullptr){
            if (failed(setLocalGate(ctx->H()->getText(), 1,  ctx->H()->getSymbol())))
                return nullptr;
            return getLocalGate(ctx->H()->getText());
        }else if (ctx->X() != nullptr){
            if (failed(setLocalGate(ctx->X()->getText(), 2, ctx->X()->getSymbol())))
                return nullptr;
            return getLocalGate(ctx->X()->getText());
        }else if (ctx->Y() != nullptr){
            if (failed(setLocalGate(ctx->Y()->getText(), 3, ctx->Y()->getSymbol())))
                return nullptr;
            return getLocalGate(ctx->Y()->getText());
        }else if (ctx->Z() != nullptr){
            if (failed(setLocalGate(ctx->Z()->getText(), 4, ctx->Z()->getSymbol())))
                return nullptr;
            return getLocalGate(ctx->Z()->getText());
        }else if (ctx->T() != nullptr){
            if (failed(setLocalGate(ctx->T()->getText(), 5, ctx->T()->getSymbol())))
                return nullptr;
            return getLocalGate(ctx->T()->getText());
        }else if (ctx->S() != nullptr){
            if (failed(setLocalGate(ctx->S()->getText(), 6, ctx->S()->getSymbol())))
                return nullptr;
            return getLocalGate(ctx->S()->getText());
        }else if (ctx->CX() != nullptr){
            if (failed(setLocalGate(ctx->CX()->getText(), 7, ctx->CX()->getSymbol())))
                return nullptr;
            return getLocalGate(ctx->CX()->getText());
        }else if (ctx->CZ() != nullptr){
            if (failed(setLocalGate(ctx->CZ()->getText(), 8, ctx->CZ()->getSymbol())))
                return nullptr;
            return getLocalGate(ctx->CZ()->getText());
        }else if (ctx->CNOT() != nullptr){
            if (failed(setLocalGate(ctx->CNOT()->getText(), 9, ctx->CNOT()->getSymbol())))
                return nullptr;
            return getLocalGate(ctx->CNOT()->getText());
        }else{
            if (failed(setLocalGate(ctx->Identifier()->getText(), 10, ctx->Identifier()->getSymbol())))
                return nullptr;
            return getLocalGate(ctx->Identifier()->getText());
        }
        return nullptr;
    }

    antlrcpp::Any visitVariable(ISQParser::VariableContext *ctx){
        mlir::Value val;
        
        if (ctx->Number() != nullptr){
            int id = atoi(ctx->Number()->getText().c_str());
            val = getConstant(ctx->Number()->getSymbol(), id);
        }else{
            string id = ctx->Identifier()->getText();
            
            val = getLocalVar(id);
            if (!val){
                val = setLocalVar(ctx->Identifier()->getSymbol(), id);
                if (!val)
                    return nullptr;
            }
            
            auto num_type = getTypeById(id);
            mlir::Value index = nullptr;
            //cout << ctx->Identifier()->getText();
            if (ctx->LeftBracket() != nullptr){
                //cout << ctx->LeftBracket()->getText();
                if (get<0>(num_type) < 0){
                    emitError(loc(ctx->Identifier()->getSymbol()), "var '") << id << "' is not array";
                    return nullptr;
                }    

                if (ctx->variable()->Number() != nullptr){
                    index = getIndex(ctx->Identifier()->getSymbol(), atoi(ctx->variable()->Number()->getText().c_str()));
                }else{
                    auto res = visit(ctx->variable());
                    if (res.isNull())
                        return nullptr;
                    index = res.as<mlir::Value>();
                }
                //cout << ctx->RightBracket()->getText();
            }else{
                if (get<0>(num_type) >= 0){
                    emitError(loc(ctx->Identifier()->getSymbol()), "var '") << id << "' is array";
                    return nullptr;
                }
                index = getIndex(ctx->Identifier()->getSymbol(), 0);
            }
            
            if (index){
                
                if (!index.getType().isa<mlir::IntegerType>() && !index.getType().isa<mlir::IndexType>()){
                    emitError(loc(ctx->Identifier()->getSymbol()), "index is not int type");
                    return -1;
                }

                if (!index.getType().isa<mlir::IndexType>()){
                    index = builder.create<mlir::IndexCastOp>(loc(ctx->Identifier()->getSymbol()), builder.getIndexType(), index);
                }
                val = builder.create<mlir::AffineLoadOp>(loc(ctx->Identifier()->getSymbol()), val, mlir::ValueRange(index));
            }
        }
        return val;
    }
        
    antlrcpp::Any visitVariableList(ISQParser::VariableListContext *ctx){
        
        if (ctx->variableList().size() > 0){
            int res = visit(ctx->variableList(0)).as<int>();
            if (res < 0)
                return -1;
            res = visit(ctx->variableList(1)).as<int>();
            if (res < 0)
                return -1;
            return 0;
        }else{
            string id = ctx->variable()->Identifier()->getText();
            mlir::Value val = getLocalVar(id);
            if (!val){
                val = setLocalVar(ctx->variable()->Identifier()->getSymbol(), id);
                if (!val)
                    return -1;
            }

            auto num_type = getTypeById(id);
            mlir::Value index = nullptr;
            if (ctx->variable()->LeftBracket() != nullptr){
                if (get<0>(num_type) < 0){
                    emitError(loc(ctx->variable()->Identifier()->getSymbol()), "var '") << id << "' is not array";
                    return -1;
                }

                if (ctx->variable()->variable()->Number() != nullptr){
                    index = getIndex(ctx->variable()->Identifier()->getSymbol(), atoi(ctx->variable()->variable()->Number()->getText().c_str()));
                }else{
                    auto res = visit(ctx->variable()->variable());
                    if (res.isNull())
                        return -1;
                    index = res.as<mlir::Value>();
                }
                if (!index.getType().isa<mlir::IntegerType>() && !index.getType().isa<mlir::IndexType>()){
                    emitError(loc(ctx->variable()->Identifier()->getSymbol()), "index is not int type");
                    return -1;
                }
                if (!index.getType().isa<mlir::IndexType>())
                    index = builder.create<mlir::IndexCastOp>(loc(ctx->variable()->Identifier()->getSymbol()), builder.getIndexType(), index);
                
            }else{
                if (get<0>(num_type) < 0){
                    index = getIndex(ctx->variable()->Identifier()->getSymbol(), 0);
                }
            }
            
            varList.push_back(make_tuple(id, val, index));
            return 0;
        }
    }

    antlrcpp::Any visitMExpression(ISQParser::MExpressionContext *ctx){
                
        string id = ctx->variable()->Identifier()->getText();
        
        mlir::Value val = getLocalVar(id);
        if (!val){
            val = setLocalVar(ctx->variable()->Identifier()->getSymbol(), id);
            if (!val)
                return nullptr;
        }

        mlir::Value index = nullptr;
        auto num_type = getTypeById(id);

        if (!get<1>(num_type).isa<QStateType>()){
            emitError(loc(ctx->variable()->Identifier()->getSymbol()), "var '") << id << "' is not qbit type";
            return nullptr;
        }

        // get index
        if (ctx->variable()->LeftBracket() != nullptr){
            if (get<0>(num_type) < 0){
                emitError(loc(ctx->variable()->Identifier()->getSymbol()), "var '") << id << "' is not array";
                return nullptr;
            }

            if (ctx->variable()->variable()->Number() != nullptr){
                index = getIndex(ctx->variable()->Identifier()->getSymbol(), atoi(ctx->variable()->variable()->Number()->getText().c_str()));
            }else{
                auto res = visit(ctx->variable()->variable());
                if (res.isNull())
                    return nullptr;
                index = res.as<mlir::Value>();
            }
        }else{
            if (get<0>(num_type) >= 0){
                emitError(loc(ctx->variable()->Identifier()->getSymbol()), "var '") << id << "' is array";
                return nullptr;
            }

            index = getIndex(ctx->variable()->Identifier()->getSymbol(), 0);
        }

        mlir::Value qval = val;
        if (index){
            if (!index.getType().isa<mlir::IntegerType>() && !index.getType().isa<mlir::IndexType>()){
                emitError(loc(ctx->variable()->Identifier()->getSymbol()), "index is not int type");
                return nullptr;
            }
            if (!index.getType().isa<mlir::IndexType>())
                index = builder.create<mlir::IndexCastOp>(loc(ctx->variable()->Identifier()->getSymbol()), builder.getIndexType(), index);
            qval = builder.create<mlir::AffineLoadOp>(loc(ctx->variable()->Identifier()->getSymbol()), val, mlir::ValueRange(index));
        }

        llvm::SmallVector<mlir::Value> arg;
        arg.push_back(qval);
        mlir::ValueRange input(arg);
                
        //mlir::SymbolRefAttr sym_name = mlir::FlatSymbolRefAttr::get(context, llvm::StringRef(ctx->M()->getText()));
        llvm::SmallVector<mlir::FlatSymbolRefAttr> nestedReferences;
        nestedReferences.push_back(mlir::FlatSymbolRefAttr::get(context, llvm::StringRef("measure")));
        mlir::SymbolRefAttr sym_name = mlir::SymbolRefAttr::get(context, llvm::StringRef(builtinName), nestedReferences);
                
        llvm::SmallVector<mlir::Type> outputArr;
        outputArr.push_back(QStateType::get(context));
        outputArr.push_back(builder.getI32Type());
        mlir::TypeRange output(outputArr);
        auto callOp = builder.create<CallQOpOp>(loc(ctx->M()->getSymbol()), output, sym_name, input, builder.getIntegerAttr(builder.getIntegerType(64, false), 1), mlir::TypeAttr::get(builder.getFunctionType({}, {builder.getI1Type()})));

        if (index){
            builder.create<mlir::AffineStoreOp>(loc(ctx->variable()->Identifier()->getSymbol()), callOp.getResult(0), val, mlir::ValueRange(index));
        }else{
            if (failed(varInsert(id, callOp.getResult(0)))){
                emitError(loc(ctx->variable()->Identifier()->getSymbol()), "insert result error");
                return nullptr;
            }
        }
        return callOp.getResult(1);
    }

    antlrcpp::Any visitExpression(ISQParser::ExpressionContext *ctx){
        
        auto res = visit(ctx->multexp(0));
        if (res.isNull())
            return res;
        
        mlir::Value lval = res.as<mlir::Value>();
        for (int i = 1; i < ctx->multexp().size(); i++){
            res = visit(ctx->multexp(i));
            if (res.isNull())
                return res;
            
            auto rval = res.as<mlir::Value>();
            if (ctx->binopPlus(i-1)->Plus() != nullptr){
                lval = builder.create<mlir::AddIOp>(loc(ctx->getStart()), lval, rval);
            }else{
                lval = builder.create<mlir::SubIOp>(loc(ctx->getStart()), lval, rval);
            }
        }

        return lval;
    }

    antlrcpp::Any visitMultexp(ISQParser::MultexpContext *ctx){

        auto res = visit(ctx->atomexp(0));
        if (res.isNull())
            return res;
        
        mlir::Value lval = res.as<mlir::Value>();
        for (int i = 1; i < ctx->atomexp().size(); i++){
            res = visit(ctx->atomexp(i));
            if (res.isNull())
                return res;
            
            auto rval = res.as<mlir::Value>();
            if (ctx->binopMult(i-1)->Mult() != nullptr){
                lval = builder.create<mlir::MulIOp>(loc(ctx->getStart()), lval, rval);
            }else{
                lval = builder.create<mlir::SignedDivIOp>(loc(ctx->getStart()), lval, rval);
            }
        }

        return lval;
        
    }

    antlrcpp::Any visitAtomexp(ISQParser::AtomexpContext *ctx){

        if (ctx->variable() != nullptr){
            auto res = visit(ctx->variable());
            if (res.isNull())
                return res;
            auto val = res.as<mlir::Value>();
            if (!val.getType().isa<mlir::IntegerType>() && !val.getType().isa<mlir::IndexType>()){
                emitError(loc(ctx->variable()->getStart()), "is not int type");
                return nullptr;
            }
            return val;
        }else{
            return visit(ctx->expression());
        }
    }

    antlrcpp::Any visitAssociation(ISQParser::AssociationContext *ctx){
        if (ctx->Equal() != nullptr){
            return ctx->Equal()->getText();
        }else if (ctx->GreaterEqual() != nullptr){
            return ctx->GreaterEqual()->getText();
        }else if (ctx->LessEqual() != nullptr){
            return ctx->LessEqual()->getText();
        }else if (ctx->Greater() != nullptr){
            return ctx->Greater()->getText();
        }else{
            return ctx->Less()->getText();
        }
    }

    antlrcpp::Any visitQbitInitStatement(ISQParser::QbitInitStatementContext *ctx){
        
        string id = ctx->variable()->Identifier()->getText();
        
        mlir::Value val = getLocalVar(id);
        if (!val){
            val = setLocalVar(ctx->variable()->Identifier()->getSymbol(), id);
            if (!val)
                return -1;
        }

        mlir::Value index = nullptr;
        auto num_type = getTypeById(id);

        if (!get<1>(num_type).isa<QStateType>()){
            emitError(loc(ctx->variable()->Identifier()->getSymbol()), "var '") << id << "' is not qbit type";
            return -1;
        }

        // get index
        if (ctx->variable()->LeftBracket() != nullptr){
            if (get<0>(num_type) < 0){
                emitError(loc(ctx->variable()->Identifier()->getSymbol()), "var '") << id << "' is not array";
                return -1;
            }

            if (ctx->variable()->variable()->Number() != nullptr){
                index = getIndex(ctx->variable()->Identifier()->getSymbol(), atoi(ctx->variable()->variable()->Number()->getText().c_str()));
            }else{
                auto res = visit(ctx->variable()->variable());
                if (res.isNull())
                    return -1;
                index = res.as<mlir::Value>();
            }
        }else{
            if (get<0>(num_type) >= 0){
                emitError(loc(ctx->variable()->Identifier()->getSymbol()), "var '") << id << "' is array";
                return -1;
            }

            index = getIndex(ctx->variable()->Identifier()->getSymbol(), 0);
        }

        mlir::Value qval = val;
        if (index){
            if (!index.getType().isa<mlir::IntegerType>() && !index.getType().isa<mlir::IndexType>()){
                emitError(loc(ctx->variable()->Identifier()->getSymbol()), "index is not int type");
                return -1;
            }
            if (!index.getType().isa<mlir::IndexType>())
                index = builder.create<mlir::IndexCastOp>(loc(ctx->variable()->Identifier()->getSymbol()), builder.getIndexType(), index);
            qval = builder.create<mlir::AffineLoadOp>(loc(ctx->variable()->Identifier()->getSymbol()), val, mlir::ValueRange(index));
        }

        llvm::SmallVector<mlir::Value> arg;
        arg.push_back(qval);
        mlir::ValueRange input(arg);
                
        llvm::SmallVector<mlir::FlatSymbolRefAttr> nestedReferences;
        nestedReferences.push_back(mlir::FlatSymbolRefAttr::get(context, llvm::StringRef("reset")));
        mlir::SymbolRefAttr sym_name = mlir::SymbolRefAttr::get(context, llvm::StringRef(builtinName), nestedReferences);
                
        llvm::SmallVector<mlir::Type> outputArr;
        outputArr.push_back(QStateType::get(context));
        mlir::TypeRange output(outputArr);
        auto callOp = builder.create<CallQOpOp>(loc(ctx->variable()->Identifier()->getSymbol()), output, sym_name, input,builder.getIntegerAttr(builder.getIntegerType(64, false), 1), mlir::TypeAttr::get(builder.getFunctionType({}, {})));

        builder.create<mlir::AffineStoreOp>(loc(ctx->variable()->Identifier()->getSymbol()), callOp.getResult(0), val, mlir::ValueRange(index));
        return 0;
    }

    antlrcpp::Any visitQbitUnitaryStatement(ISQParser::QbitUnitaryStatementContext *ctx){
        
        varList.clear();

        auto gateres = visit(ctx->uGate());
        if (gateres.isNull())
            return -1;
        
        mlir::Value gate = gateres.as<mlir::Value>();
        
        int res = visit(ctx->variableList()).as<int>();
        if (res < 0){
            emitError(loc(ctx->variableList()->getStart()), "get variable list error");
            return -1;
        }

        llvm::SmallVector<mlir::Value> input;
        llvm::SmallVector<mlir::Type> output;
        for (int i = 0; i < varList.size(); i++){
            
            if (!get<2>(varList[i])){
                emitError(loc(ctx->variableList()->getStart()), "need single qbit");
                return -1;
            }
            auto val = builder.create<mlir::AffineLoadOp>(loc(ctx->getStart()), get<1>(varList[i]), mlir::ValueRange(get<2>(varList[i])));
            input.push_back(val);
            /*
            if (get<2>(varList[i])){
                auto val = builder.create<mlir::AffineLoadOp>(loc(ctx->getStart()), get<1>(varList[i]), mlir::ValueRange(get<2>(varList[i])));
                input.push_back(val);
            }else{
                if (get<2>(getTypeById(get<0>(varList[i])))){
                    auto index = getIndex(ctx->getStart(), 0);
                    auto val = builder.create<mlir::AffineLoadOp>(loc(ctx->getStart()), get<1>(varList[i]), mlir::ValueRange(index));
                    input.push_back(val);
                }else
                    input.push_back(get<1>(varList[i]));
            }*/
            output.push_back(QStateType::get(context));
        }
        
        auto applyGate = builder.create<ApplyGateOp>(loc(ctx->getStart()), mlir::TypeRange(output), gate, mlir::ValueRange(input));
        
        for (int i = 0; i < varList.size(); i++){
            
            builder.create<mlir::AffineStoreOp>(loc(ctx->getStart()), applyGate.getResult(i), get<1>(varList[i]), mlir::ValueRange(get<2>(varList[i])));
            /*
            if (get<2>(varList[i])){
                builder.create<mlir::AffineStoreOp>(loc(ctx->getStart()), applyGate.getResult(i), get<1>(varList[i]), mlir::ValueRange(get<2>(varList[i])));
            }else{
                if (get<2>(getTypeById(get<0>(varList[i])))){
                    auto index = getIndex(ctx->getStart(), 0);
                    builder.create<mlir::AffineStoreOp>(loc(ctx->getStart()), applyGate.getResult(i), get<1>(varList[i]), mlir::ValueRange(index));
                }else
                    if (failed(varInsert(get<0>(varList[i]), applyGate.getResult(i)))){
                        emitError(loc(ctx->getStart()), "update variable val error");
                        return -1;
                    }
            }*/
        }
        return 0;
    }

    antlrcpp::Any visitCintAssign(ISQParser::CintAssignContext *ctx){
        
        string id = ctx->variable()->Identifier()->getText();
        
        // get right val
        mlir::Value expval;
        if (ctx->expression() != nullptr){
            auto express = visit(ctx->expression());
            if (express.isNull()){
                return -1;
            }
            expval = express.as<mlir::Value>();
        }else if(ctx->mExpression()!= nullptr){
            auto express = visit(ctx->mExpression());
            if (express.isNull()){
                return -1;
            }
            expval = express.as<mlir::Value>();
        }else{
            int res = visit(ctx->callStatement()).as<int>();
            if (res < 0)
                return -1;
            if (callres == nullptr){
                emitError(loc(ctx->variable()->Identifier()->getSymbol()), "procedure has no return");
                return -1;
            }
            expval = callres;
        }
        
        // get left varible
        mlir::Value val = getLocalVar(id);
        if (!val){
            val = setLocalVar(ctx->variable()->Identifier()->getSymbol(), id);
            if (!val)
                return -1;
        }

        mlir::Value index = nullptr;
        auto num_type = getTypeById(id);

        if (!get<1>(num_type).isa<mlir::IntegerType>()){
            emitError(loc(ctx->variable()->Identifier()->getSymbol()), "var '") << id << "' is not int type";
            return -1;
        }

        // get index
        if (ctx->variable()->LeftBracket() != nullptr){
            if (get<0>(num_type) < 0){
                emitError(loc(ctx->variable()->Identifier()->getSymbol()), "var '") << id << "' is not array";
                return -1;
            }

            if (ctx->variable()->variable()->Number() != nullptr){
                index = getIndex(ctx->variable()->Identifier()->getSymbol(), atoi(ctx->variable()->variable()->Number()->getText().c_str()));
            }else{
                auto res = visit(ctx->variable()->variable());
                if (res.isNull())
                    return -1;
                index = res.as<mlir::Value>();
            }
        }else{
            if (get<0>(num_type) >= 0){
                emitError(loc(ctx->variable()->Identifier()->getSymbol()), "var '") << id << "' is array";
                return -1;
            }
            index = getIndex(ctx->variable()->Identifier()->getSymbol(), 0);
        }
        if (index){

            if (!index.getType().isa<mlir::IntegerType>() && !index.getType().isa<mlir::IndexType>()){
                emitError(loc(ctx->variable()->Identifier()->getSymbol()), "index is not int type");
                return -1;
            }
            
            if (!index.getType().isa<mlir::IndexType>())
                index = builder.create<mlir::IndexCastOp>(loc(ctx->variable()->Identifier()->getSymbol()), builder.getIndexType(), index);
            
            if (expval.getType().isa<mlir::IndexType>()){
                expval = builder.create<mlir::IndexCastOp>(loc(ctx->variable()->Identifier()->getSymbol()), builder.getI32Type(), expval);
            }

            builder.create<mlir::AffineStoreOp>(loc(ctx->variable()->Identifier()->getSymbol()), expval, val, mlir::ValueRange(index));
        }else
            if (failed(varInsert(id, expval))){
                emitError(loc(ctx->variable()->Identifier()->getSymbol()), "var '") << id << "' update error";
                return -1;
            }
        return 0;
    }

    antlrcpp::Any visitRegionBody(ISQParser::RegionBodyContext *ctx) {
        if (ctx->LeftBrace() != nullptr){
            return visit(ctx->statementBlock());
        }

        return visit(ctx->statement());
    }

    antlrcpp::Any visitIfStatement(ISQParser::IfStatementContext *ctx){
        
        llvm::SmallVector<mlir::Value> if_arg;

        // get left exp val
        auto lexpress = visit(ctx->expression(0));
        if (lexpress.isNull()){
            return -1;
        }
        auto lval = lexpress.as<mlir::Value>();
        if (!lval.getType().isa<mlir::IndexType>()){
            lval = builder.create<mlir::IndexCastOp>(loc(ctx->getStart()), builder.getIndexType(), lval);
        }
        if_arg.push_back(lval);

        // get right exp val
        auto rexpress = visit(ctx->expression(1));
        if (rexpress.isNull()){
            return -1;
        }
        auto rval = rexpress.as<mlir::Value>();
        if (!rval.getType().isa<mlir::IndexType>()){
            rval = builder.create<mlir::IndexCastOp>(loc(ctx->getStart()), builder.getIndexType(), rval);
        }
        if_arg.push_back(rval);

        // get association

        mlir::IntegerSet ass = getSetFromAsso(visit(ctx->association()).as<string>());

        bool has_else = false;
        if (ctx->Else() != nullptr){
            has_else = true;
        }

        // define ifOp
        auto ifOp = builder.create<mlir::AffineIfOp>(loc(ctx->getStart()), ass, mlir::ValueRange(if_arg), has_else);
        
        // save origin block;
        string originBlock = nowBlock;
        mlir::Block* nowblock = builder.getInsertionBlock();

        // move point to then block
        nowBlock = originBlock + ".if";
        createSymbolTable(nowBlock);
        builder.setInsertionPointToStart(ifOp.getThenBlock());
        int res = visit(ctx->regionBody(0)).as<int>();
        if (res < 0)
            return res;
        deleteSymbolTable(nowBlock);

        if (has_else){
            // move point to else block
            nowBlock = originBlock + ".else";
            createSymbolTable(nowBlock);
            builder.setInsertionPointToStart(ifOp.getElseBlock());
            res = visit(ctx->regionBody(1)).as<int>();
            if (res < 0)
                return res;
            deleteSymbolTable(nowBlock);
        }
        // move point to the end of origin block;
        builder.setInsertionPointToEnd(nowblock);
        nowBlock = originBlock;
        return 0;
    }

    antlrcpp::Any visitForStatement(ISQParser::ForStatementContext *ctx){
        //cout << "start for ------- " << endl;
        string id = ctx->Identifier()->getText();

        mlir::Value val = getLocalVar(id);
        if (!val){
            val = setLocalVar(ctx->Identifier()->getSymbol(), id);
            if (!val)
                return -1;
        }

        auto num_type = getTypeById(id);
        if (!get<1>(num_type).isa<mlir::IntegerType>()){
            emitError(loc(ctx->Identifier()->getSymbol()), "var '") << id << "' type error, need int but qbit";
            return -1;
        }
        if (get<0>(num_type) != -1){
            emitError(loc(ctx->Identifier()->getSymbol()), "var '") << id << "' type error, need int but array";
            return -1;
        }
        
        mlir::Value index = getIndex(ctx->Identifier()->getSymbol(), 0);
        
        // lower bound
        llvm::SmallVector<mlir::Value> lbval;
        mlir::AffineMap lmap = singleSymbol;
        if (ctx->variable(0)->Number() != nullptr){
            lmap = mlir::AffineMap::getConstantMap(atoi(ctx->variable(0)->Number()->getText().c_str()), context);
        }else{
            auto res = visit(ctx->variable(0));
            if (res.isNull())
                return -1;
                
            mlir::Value lhs = res.as<mlir::Value>();
            if (!lhs.getType().isa<mlir::IntegerType>() && !lhs.getType().isa<mlir::IndexType>()){
                emitError(loc(ctx->variable(0)->getStart()), "is not int type");
                return -1;
            }
            if (lhs.getType().isa<mlir::IntegerType>()){
                lhs = builder.create<mlir::IndexCastOp>(loc(ctx->variable(0)->getStart()), builder.getIndexType(), lhs);
            }
            lbval.push_back(lhs);
        }
        
        // upper bound
        llvm::SmallVector<mlir::Value> rbval;
        mlir::AffineMap rmap = singleSymbol;
        if (ctx->variable(1)->Number() != nullptr){
            rmap = mlir::AffineMap::getConstantMap(atoi(ctx->variable(1)->Number()->getText().c_str()), context);
        }else{
            auto res = visit(ctx->variable(1));
            if (res.isNull())
                return -1;

            mlir::Value rhs = res.as<mlir::Value>();
            if (!rhs.getType().isa<mlir::IntegerType>() && !rhs.getType().isa<mlir::IndexType>()){
                emitError(loc(ctx->variable(1)->getStart()), "is not int type");
                return -1;
            }
            if (rhs.getType().isa<mlir::IntegerType>()){
                rhs = builder.create<mlir::IndexCastOp>(loc(ctx->variable(1)->getStart()), builder.getIndexType(), rhs);
            }
            rbval.push_back(rhs);
        }
        
        auto forOp = builder.create<mlir::AffineForOp>(loc(ctx->getStart()), lbval, lmap, rbval, rmap);

        // save origin block;
        string originBlock = nowBlock;
        mlir::Block* nowblock = builder.getInsertionBlock();

        // move point to for body
        nowBlock = originBlock + ".for";
        createSymbolTable(nowBlock);
        builder.setInsertionPointToStart(&forOp.getLoopBody().front());
        // save arg val to the identify
        auto arg_val = builder.create<mlir::IndexCastOp>(loc(ctx->regionBody()->getStart()), builder.getI32Type(), forOp.getInductionVar());
        builder.create<mlir::AffineStoreOp>(loc(ctx->regionBody()->getStart()), arg_val, val, mlir::ValueRange(index));
        int res = visit(ctx->regionBody()).as<int>();
        if (res < 0)
            return res;
        deleteSymbolTable(nowBlock);
        
        // move point to the end of origin block;
        builder.setInsertionPointToEnd(nowblock);
        nowBlock = originBlock;

        return 0;
    }

    antlrcpp::Any visitWhileStatement(ISQParser::WhileStatementContext *ctx){


        string originBlock = nowBlock;
        mlir::Block* nowblock = builder.getInsertionBlock();
        // creat while Op
        auto whileOp = builder.create<mlir::scf::WhileOp>(loc(ctx->getStart()), mlir::TypeRange(llvm::None), mlir::ValueRange(llvm::None));
        
        // create before Block
        mlir::Block *before = builder.createBlock(&whileOp.before(), {}, llvm::None);
        
        // in before Block, judge the condition

        // get left exp val
        auto lexpress = visit(ctx->expression(0));
        if (lexpress.isNull()){
            return -1;
        }
        auto lval = lexpress.as<mlir::Value>();

        // get right exp val
        auto rexpress = visit(ctx->expression(1));
        if (rexpress.isNull()){
            return -1;
        }
        auto rval = rexpress.as<mlir::Value>();
        
        // get association
        auto asso = getCmpFromAsso(visit(ctx->association()).as<string>());

        mlir::Value condition = builder.create<mlir::CmpIOp>(loc(ctx->getStart()), asso, lval, rval);
        
        // set condition
        builder.create<mlir::scf::ConditionOp>(loc(ctx->getStart()), condition, llvm::None);
        

        // create after block
        mlir::Block *after = builder.createBlock(&whileOp.after(), {}, llvm::None);
        nowBlock = originBlock + ".while";
        createSymbolTable(nowBlock);
        int res = visit(ctx->regionBody()).as<int>();
        if (res < 0)
            return res;
        builder.create<mlir::scf::YieldOp>(loc(ctx->getStart()));
        deleteSymbolTable(nowBlock);

        // move point to the end of origin block;
        builder.setInsertionPointToEnd(nowblock);
        nowBlock = originBlock;
        return 0;
    }

    antlrcpp::Any visitCallStatement(ISQParser::CallStatementContext *ctx){
        
        //cout << "call: " << ctx->Identifier()->getText() << endl;
        string id = ctx->Identifier()->getText();

        auto iter = producer.find(id);
        if (iter == producer.end()){
            emitError(loc(ctx->Identifier()->getSymbol()), "function '") << id << "' not defined";
            return -1;
        }

        bool has_return = iter->second;

        varList.clear();

        if (ctx->variableList() != nullptr){
            int res = visit(ctx->variableList()).as<int>();
            if (res < 0){
                emitError(loc(ctx->variableList()->getStart()), "get variable list error");
                return -1;
            }
        }

        // assert param number
        auto args = protoArgs.find(id)->second;
        if (args.size() != varList.size()){
            emitError(loc(ctx->Identifier()->getSymbol()), "param number error");
            return -1;
        }

        llvm::SmallVector<mlir::Value> input;
        llvm::SmallVector<mlir::Type> output;
        if (has_return)
            output.push_back(builder.getI32Type());

        vector<int> single_qbit;
        for (int i = 0; i < varList.size(); i++){
            // assert param type
            auto in_num_type = getTypeById(get<0>(varList[i]));
            //cout << get<0>(varList[i]) << endl << get<0>(args[i]) << endl;

            if (get<2>(args[i]).isa<QStateType>()){
                if (!get<1>(in_num_type).isa<QStateType>()){
                    emitError(loc(ctx->Identifier()->getSymbol()), "param '") << get<0>(varList[i]) << "' type error, need qbit";
                    return -1;
                }
            }else{
                if (get<1>(in_num_type).isa<QStateType>()){
                    emitError(loc(ctx->Identifier()->getSymbol()), "param '") << get<0>(varList[i]) << "' type error, need int";
                    return -1;
                }
            }

            if (get<1>(args[i]) == -1){
                if (get<0>(in_num_type) != -1 && !get<2>(varList[i])){
                    emitError(loc(ctx->Identifier()->getSymbol()), "param '") << get<0>(varList[i]) << "' type error, need single value!";
                    return -1;
                }
            }else{
                if (get<0>(in_num_type) == -1 || get<2>(varList[i])){
                    emitError(loc(ctx->Identifier()->getSymbol()), "param '") << get<0>(varList[i]) << "' type error, need array!";
                    return -1;
                }
            }
            
            // prepare input and output
        
            if (get<2>(varList[i])){
                auto val = builder.create<mlir::AffineLoadOp>(loc(ctx->getStart()), get<1>(varList[i]), mlir::ValueRange(get<2>(varList[i])));
                input.push_back(val);
            }else{
                input.push_back(get<1>(varList[i]));
            }

            if (get<2>(args[i]).isa<QStateType>() && get<1>(args[i]) == -1 ){
                output.push_back(QStateType::get(context));
                single_qbit.push_back(i);
            }
        }
        auto callOp = builder.create<mlir::CallOp>(loc(ctx->getStart()), llvm::StringRef(id), mlir::TypeRange(output), mlir::ValueRange(input));
        
        // store the global single qbit
        callres = has_return ? callOp.getResult(0) : nullptr;
        int offset = has_return ? 1: 0;
        for (int i = 0; i < single_qbit.size(); i++){
            builder.create<mlir::AffineStoreOp>(loc(ctx->getStart()), callOp.getResult(i+offset), get<1>(varList[single_qbit[i]]), mlir::ValueRange(get<2>(varList[single_qbit[i]])));
            /*
            auto in_num_type = getTypeById(get<0>(varList[single_qbit[i]]));
            if (get<2>(in_num_type)){
                auto index = getIndex(ctx->getStart(), 0);
                builder.create<mlir::AffineStoreOp>(loc(ctx->getStart()), callOp.getResult(i), get<1>(varList[single_qbit[i]]), mlir::ValueRange(index));
            }else{
                if (failed(varInsert(get<0>(varList[single_qbit[i]]), callOp.getResult(i)))){
                    emitError(loc(ctx->Identifier()->getSymbol()), "return error");
                    return -1;
                }
            }*/
        }
        return 0;
    }

    antlrcpp::Any visitPrintStatement(ISQParser::PrintStatementContext *ctx){
        //cout << ctx->Print()->getText() << ' ';
        auto res = visit(ctx->variable());
        if (res.isNull())
            return -1;
        
        auto val = res.as<mlir::Value>();
        if (!val.getType().isa<mlir::IntegerType>()){
            emitError(loc(ctx->getStart()), "can only print int type");
            return -1;
        }

        builder.create<PrintOp>(loc(ctx->getStart()), val);
        return 0;
    }

    antlrcpp::Any visitReturnStatement(ISQParser::ReturnStatementContext *ctx){

        auto res = visit(ctx->variable());
        if (res.isNull())
            return -1;
        
        auto val = res.as<mlir::Value>();
        if (!val.getType().isa<mlir::IntegerType>()){
            emitError(loc(ctx->getStart()), "can only return int type");
            return -1;
        }

        returnval.push_back(val);
        return 0;
    }
};

}

namespace isq {

    mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context,
                      llvm::StringRef inputfile) {
        return MLIRGenImpl(context).mlirGen(inputfile);
    }
}
using namespace llvm;
int main(int argc, char** argv){
    static cl::opt<string> inputFilename(cl::Positional,
                                        cl::desc("<input file>"),
                                        cl::init("-"),
                                        cl::value_desc("filename"));

    static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"),
                                             cl::init("-"));
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
    // Load our Dialect in this MLIR Context.
    mlir::DialectRegistry registry;
    isq::ir::ISQToolsInitialize(registry);

    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects(); 
    mlir::OwningModuleRef module = mlirGen(context, inputFilename);
    if (!module)
        return 1;
    
    std::string errorMessage;
    auto output = mlir::openOutputFile(outputFilename, &errorMessage);
    if (!output) {
        llvm::errs() << errorMessage << "\n";
        return 1;
    }

    module->print(output->os(), mlir::OpPrintingFlags().printGenericOpForm().enableDebugInfo(false));
    output->keep();
    return 0;
}