#include "isq/GateDefTypes.h"
#include "isq/Operations.h"
#include "isq/QAttrs.h"
#include "isq/QStructs.h"
#include "isq/QTypes.h"
#include "isq/passes/Mem2Reg.h"
#include "isq/passes/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Casting.h"
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/TypeRange.h>
#include <optional>
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace isq{
namespace ir{
namespace passes{
namespace{

std::vector<std::vector<std::complex<double>>> appendMatrix(const std::vector<std::vector<std::complex<double>>>& mat, ::mlir::ArrayRef<bool> ctrl, bool adj){
    auto mat_qubit_num = (int)std::log2(mat.size());
    auto new_mat_size = ((1<<ctrl.size()) * mat.size());
    std::vector<std::vector<std::complex<double>>> new_matrix;
    new_matrix.resize(new_mat_size);
    for(auto i=0; i<new_mat_size; i++){
        new_matrix[i].resize(new_mat_size);
        new_matrix[i][i]=1.0;
    }
    uint64_t mat_mask = 0;
    for(auto i=0; i<ctrl.size(); i++){
        mat_mask = (mat_mask<<1) | (ctrl[i]?1:0);
    }
    mat_mask = mat_mask << mat_qubit_num;
    for(auto i=0; i<(1<<mat_qubit_num); i++){
        for(auto j=0; j<(1<<mat_qubit_num); j++){
            if(adj){
                new_matrix[i|mat_mask][j|mat_mask] = std::conj(mat[j][i]);
            }else{
                new_matrix[i|mat_mask][j|mat_mask] = mat[i][j];
            }
            
        }
    }
    return new_matrix;
}

const char* ISQ_DECORATE_FOLDING_PROPAGATE_CTRL_START_INDEX = "ISQ_DECORATE_FOLDING_PROPAGATE_CTRL_START_INDEX";
const char* ISQ_DECORATE_FOLDING_PROPAGATE_CTRL_BITS = "ISQ_DECORATE_FOLDING_PROPAGATE_CTRL_BITS";
const char* ISQ_DECORATE_FOLDING_PROPAGATE_TO_PROCESS = "ISQ_DECORATE_FOLDING_PROPAGATE_TO_PROCESS";
const char* ISQ_FAKELOAD = "isq.intermediate.fakeload";
const char* ISQ_FAKESTORE = "isq.intermediate.fakestore";
const char* ISQ_FAKELOADSTORE_ID = "ISQ_FAKELOADSTORE_ID";
struct DecorateFoldRewriteRule : public mlir::OpRewritePattern<isq::ir::ApplyGateOp>{
    mlir::ModuleOp rootModule;
    DecorateFoldRewriteRule(mlir::MLIRContext* ctx, mlir::ModuleOp module): mlir::OpRewritePattern<isq::ir::ApplyGateOp>(ctx, 1), rootModule(module){

    }

    mlir::LogicalResult createControlledDefgate(isq::ir::DefgateOp defgate, mlir::ArrayRef<bool> ctrl, bool adj, mlir::FlatSymbolRefAttr sym, mlir::PatternRewriter &rewriter) const{
        auto ctx = rewriter.getContext();
        mlir::SmallVector<mlir::Attribute> usefulGatedefs;
        auto id=0;
        auto new_qubit_num = (int)defgate.type().getSize() + ctrl.size();
        for(auto def: defgate.definition()->getAsRange<GateDefinition>()){
            auto d = AllGateDefs::parseGateDefinition(defgate, id, defgate.type(), def);
            if(d==std::nullopt) return mlir::failure();
            
            if(auto mat = llvm::dyn_cast_or_null<MatrixDefinition>(&**d)){
                auto& old_matrix = mat->getMatrix();

                // construct new matrix.
                
                auto new_matrix = appendMatrix(old_matrix, ctrl, adj);
                mlir::SmallVector<mlir::Attribute> matrix_attr;
                for(auto& row: new_matrix){
                    mlir::SmallVector<mlir::Attribute> row_attr;
                    for(auto column: row){
                        auto c = ComplexF64Attr::get(ctx, ::llvm::APFloat(column.real()), ::llvm::APFloat(column.imag()));
                        row_attr.push_back(c);
                    }
                    matrix_attr.push_back(::mlir::ArrayAttr::get(ctx, row_attr));
                }
                usefulGatedefs.push_back(GateDefinition::get(mlir::StringAttr::get(ctx, "unitary"), mlir::ArrayAttr::get(ctx, matrix_attr), ctx));
            }else if(auto decomp = llvm::dyn_cast_or_null<DecompositionDefinition>(&**d)){
                auto ip = rewriter.saveInsertionPoint();
                auto fn = decomp->getDecomposedFunc();
                // TODO: adjoint op support.
                // Do we need to "revert" all steps?
                if(adj) return mlir::failure();
                auto new_fn = fn.clone();
                mlir::ModuleOp rootModule = this->rootModule;
                rewriter.setInsertionPointToStart(rootModule.getBody());
                rewriter.insert(new_fn);
                rewriter.startRootUpdate(new_fn);
                auto new_fn_name = "$__isq__decomposition__"+sym.getValue();
                new_fn.sym_nameAttr(mlir::StringAttr::get(ctx, new_fn_name));
                
                mlir::SmallVector<mlir::Attribute> ctrl_attr;
                for(auto b: ctrl){
                    ctrl_attr.push_back(mlir::BoolAttr::get(ctx, b));
                } 

                // insert control qubits.
                auto old_size = new_fn.getType().getNumInputs();
                mlir::SmallVector<mlir::Type> args;
                mlir::SmallVector<mlir::Type> results;
                for(auto input: new_fn.getType().getInputs()){
                    args.push_back(input);
                }
                for(auto output: new_fn.getType().getResults()){
                    results.push_back(output);
                }
                for(auto i=0; i<ctrl.size(); i++){
                    args.push_back(QStateType::get(ctx));
                    results.push_back(QStateType::get(ctx));
                }
                new_fn.setType(mlir::FunctionType::get(ctx, args, results));
                mlir::SmallVector<mlir::Value> controlQubits;
                auto insert_index = old_size - defgate.type().getSize();
                for(auto i=0; i<ctrl.size(); i++){
                    controlQubits.push_back(new_fn.body().insertArgument(new_fn.getArguments().begin()+insert_index+i, QStateType::get(ctx), mlir::UnknownLoc::get(ctx)));
                }
                new_fn->setAttr(ISQ_DECORATE_FOLDING_PROPAGATE_CTRL_START_INDEX, mlir::IntegerAttr::get(rewriter.getI64Type(), insert_index));
                new_fn->setAttr(ISQ_DECORATE_FOLDING_PROPAGATE_CTRL_BITS, mlir::ArrayAttr::get(ctx, ctrl_attr));
                new_fn.walk([=](ApplyGateOp op){
                    op->setAttr(ISQ_DECORATE_FOLDING_PROPAGATE_TO_PROCESS, mlir::UnitAttr::get(ctx));
                });
                rewriter.finalizeRootUpdate(new_fn);
                rewriter.restoreInsertionPoint(ip);
                usefulGatedefs.push_back(GateDefinition::get(mlir::StringAttr::get(ctx, "decomposition"), mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(ctx, new_fn_name)), ctx));
            }
            id++;
        }
        if(usefulGatedefs.size()==0){
            return mlir::failure();
        }
        auto ip = rewriter.saveInsertionPoint();
        mlir::ModuleOp rootModule = this->rootModule;
        rewriter.setInsertionPointToStart(rootModule.getBody());
        rewriter.create<DefgateOp>(::mlir::UnknownLoc::get(ctx), mlir::TypeAttr::get(GateType::get(ctx, new_qubit_num, GateTrait::General)), sym.getAttr(), mlir::StringAttr::get(ctx, "nested"), ::mlir::ArrayAttr{}, ::mlir::ArrayAttr::get(ctx, usefulGatedefs), ::mlir::ArrayAttr::get(ctx, ::llvm::ArrayRef<::mlir::Attribute>{}));
        rewriter.restoreInsertionPoint(ip);
        return mlir::success();
    }
    mlir::LogicalResult matchAndRewrite(isq::ir::ApplyGateOp op,  mlir::PatternRewriter &rewriter) const override{
        // Check if it is a use-decorate-apply pattern.
        auto decorate_op = mlir::dyn_cast_or_null<DecorateOp>(op.gate().getDefiningOp());
        if(!decorate_op) return mlir::failure();
        auto use_op = mlir::dyn_cast_or_null<UseGateOp>(decorate_op.args().getDefiningOp());
        if(!use_op) return mlir::failure();
        if(use_op.parameters().size()>0){
            return mlir::failure(); // Only matrix-gates are supported.
        }
        auto defgate = mlir::SymbolTable::lookupNearestSymbolFrom<DefgateOp>(use_op.getOperation(), use_op.name());
        assert(defgate);
        if(!defgate.definition()) return mlir::failure();

        // cx is cnot.
        if(isFamousGate(defgate, "X") && decorate_op.ctrl().size()==1){
            auto ctx = getContext();
            rewriter.replaceOpWithNewOp<UseGateOp>(decorate_op, mlir::TypeRange{GateType::get(ctx, 2, GateTrait::General)}, mlir::FlatSymbolRefAttr::get(ctx, getFamousName("CNOT")), mlir::ValueRange{});
            return mlir::success();
        }

        // construct new matrix name.
        // TODO: resolve decorate-and-decorate problem.
        auto new_defgate_name = std::string(defgate.sym_name());
        if(decorate_op.ctrl().size()>0){
            new_defgate_name+="_ctrl_";
            for(auto c: decorate_op.ctrl().getAsValueRange<mlir::BoolAttr>()){
                new_defgate_name+= c?"1":"0";
            }
        }
        if(decorate_op.adjoint()){
            new_defgate_name += "_adj";
        }
        auto new_defgate_sym = mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(rewriter.getContext(), new_defgate_name));
        auto new_defgate = mlir::SymbolTable::lookupNearestSymbolFrom<DefgateOp>(op, new_defgate_sym);
            
        auto ctrl_array = mlir::SmallVector<bool>();
        for(auto attr: decorate_op.ctrl().getAsValueRange<mlir::BoolAttr>()){
            ctrl_array.push_back(attr);
        }
        if(!new_defgate){
            if(mlir::failed(createControlledDefgate(defgate, ctrl_array, decorate_op.adjoint(), new_defgate_sym, rewriter))){
                return mlir::failure();
            }
        }
        new_defgate = mlir::SymbolTable::lookupNearestSymbolFrom<DefgateOp>(op, new_defgate_sym);
        auto ctx = getContext();
        auto new_qubit_num = (int)defgate.type().getSize() + ctrl_array.size();
        auto ip = rewriter.saveInsertionPoint();
        rewriter.setInsertionPoint(op);
        auto new_use_gate = rewriter.create<UseGateOp>(op->getLoc(), GateType::get(ctx, new_qubit_num, GateTrait::General), new_defgate_sym, ::mlir::ValueRange{});
        rewriter.restoreInsertionPoint(ip);
        rewriter.replaceOpWithNewOp<ApplyGateOp>(op.getOperation(), op->getResultTypes(), new_use_gate.result(), op.args());
        return mlir::success();
    }
};

}

// Insert controller bits and insert fake load/store pairs.
struct InsertControllerBits : public mlir::OpRewritePattern<ApplyGateOp>{
    ::mlir::ArrayAttr controls;
    int controlStartIndex;
    mlir::FuncOp currentFunc;
    InsertControllerBits(mlir::MLIRContext* ctx, ::mlir::ArrayAttr controls, int control_start_index, mlir::FuncOp fn): mlir::OpRewritePattern<ApplyGateOp>(ctx, 1), currentFunc(fn),controls(controls),controlStartIndex(control_start_index){}
    // First, decorate every apply gate with an additional decorate.
    // Second, insert a fake load-store pair and inject the qubits into apply op. These fake ops will be eliminated by mem2reg pass.
    mlir::LogicalResult matchAndRewrite(ApplyGateOp op, mlir::PatternRewriter& rewriter) const override{
        auto ctx = rewriter.getContext();
        if(!op->hasAttr(ISQ_DECORATE_FOLDING_PROPAGATE_TO_PROCESS)){
            return mlir::failure();
        }
        op->removeAttr(ISQ_DECORATE_FOLDING_PROPAGATE_TO_PROCESS);
        mlir::PatternRewriter::InsertionGuard guard(rewriter);
        auto old_gate_type = op.gate().getType().cast<GateType>();
        rewriter.setInsertionPoint(op);
        auto new_decorate = rewriter.create<DecorateOp>(mlir::UnknownLoc::get(ctx), GateType::get(ctx, controls.size()+old_gate_type.getSize(), old_gate_type.getHints()), op.gate(), false, controls);
        
        rewriter.setInsertionPoint(op);
        mlir::SmallVector<mlir::Value> ctrl_loaded;
        auto fn = currentFunc;
        for(auto i=0; i<controls.size(); i++){
            auto ctrl_argument = fn.body().getArgument(i+controlStartIndex);
            mlir::OperationState state(mlir::UnknownLoc::get(ctx), ISQ_FAKELOAD, mlir::ValueRange{ctrl_argument}, mlir::TypeRange{QStateType::get(ctx)}, mlir::ArrayRef<mlir::NamedAttribute>{mlir::NamedAttribute(mlir::StringAttr::get(ctx, ISQ_FAKELOADSTORE_ID), mlir::IntegerAttr::get(rewriter.getI64Type(), i))});
            auto fake_load = rewriter.createOperation(state);
            ctrl_loaded.push_back(fake_load->getResult(0));
        }
        // Now get ready to recreate the applygate op...
        mlir::SmallVector<mlir::Value> args;
        mlir::SmallVector<mlir::Type> results;
        rewriter.setInsertionPointAfter(op);
        for(auto i=0; i<controls.size(); i++){
            args.push_back(ctrl_loaded[i]);
            results.push_back(QStateType::get(ctx));
        }
        args.append(op.args().begin(), op.args().end());
        results.append(op->result_type_begin(), op.result_type_end());
        auto new_op = rewriter.create<ApplyGateOp>(op->getLoc(), results, new_decorate.getResult(), args);
        new_op->setAttrs(op->getAttrs());
        // For the first results, fake-store back.
        rewriter.setInsertionPointAfter(new_op);
        for(auto i=0; i<controls.size(); i++){
            auto ctrl_argument = fn.body().getArgument(i+controlStartIndex);
            mlir::OperationState state(mlir::UnknownLoc::get(ctx), ISQ_FAKESTORE, mlir::ValueRange{new_op.getResult(i), ctrl_argument}, mlir::TypeRange{},mlir::ArrayRef<mlir::NamedAttribute>{mlir::NamedAttribute(mlir::StringAttr::get(ctx, ISQ_FAKELOADSTORE_ID), mlir::IntegerAttr::get(rewriter.getI64Type(), i))});
            
            auto fake_store = rewriter.createOperation(state);
        }
        // For the rest results, replace original op.
        rewriter.replaceOp(op, new_op->getResults().drop_front(controls.size()));
        return mlir::success();
    }
};


struct FakeMem2RegRewrite : public Mem2RegRewrite{
    bool isLoad(mlir::Operation* op) const {
        return op->getName().getStringRef()==ISQ_FAKELOAD;
    }
    int loadId(mlir::Operation* op) const {
        return op->getAttrOfType<mlir::IntegerAttr>(ISQ_FAKELOADSTORE_ID).getInt();
    }
    bool isStore(mlir::Operation* op) const {
        return op->getName().getStringRef()==ISQ_FAKESTORE;
    }
    int storeId(mlir::Operation* op) const {
        return op->getAttrOfType<mlir::IntegerAttr>(ISQ_FAKELOADSTORE_ID).getInt();
    }
    int storeValue(mlir::Operation* op) const {
        return 0;
    }
};

// Insert controller bits and insert fake load/store pairs.
struct FakeMem2Reg : public mlir::OpRewritePattern<mlir::FuncOp>{
    FakeMem2Reg(mlir::MLIRContext* ctx): mlir::OpRewritePattern<mlir::FuncOp>(ctx, 1){}

    mlir::LogicalResult matchAndRewrite(mlir::FuncOp op, mlir::PatternRewriter& rewriter) const override{
        auto ctx = rewriter.getContext();
        if(!op->hasAttr(ISQ_DECORATE_FOLDING_PROPAGATE_CTRL_BITS)){
            return mlir::failure();
        }
        int controlSize=op->getAttrOfType<mlir::ArrayAttr>(ISQ_DECORATE_FOLDING_PROPAGATE_CTRL_BITS).size();
        int controlStartIndex=op->getAttrOfType<mlir::IntegerAttr>(ISQ_DECORATE_FOLDING_PROPAGATE_CTRL_START_INDEX).getInt();
        int originalGateSize = op.getNumArguments() - controlStartIndex - controlSize;
        mlir::SmallVector<mlir::Value> args;
        mlir::SmallVector<mlir::Type> argTypes;
        for(auto i=0; i<controlSize; i++){
            args.push_back(op.body().getArgument(i+controlStartIndex));
            argTypes.push_back(QStateType::get(ctx));
        }
        rewriter.startRootUpdate(op);
        op->removeAttr(ISQ_DECORATE_FOLDING_PROPAGATE_CTRL_BITS);
        op->removeAttr(ISQ_DECORATE_FOLDING_PROPAGATE_CTRL_START_INDEX);
        FakeMem2RegRewrite mem2reg;
        for(auto& block: op.body()){
            if(block.isEntryBlock()){
                mem2reg.mem2regKeepBlockParam(&block, rewriter, args);
            }else{
                mem2reg.mem2regAlterBlockParam(argTypes, &block, rewriter);
                if(auto last = llvm::dyn_cast<mlir::ReturnOp>(block.getTerminator())){
                    // twist back.
                    mlir::SmallVector<mlir::Value> twistedReturnOrder;
                    for(auto i=0; i<controlSize; i++){
                        twistedReturnOrder.push_back(last.getOperand(originalGateSize+i));
                    }
                    for(auto i=0; i<originalGateSize; i++){
                        twistedReturnOrder.push_back(last.getOperand(i));
                    }
                    last.operandsMutable().assign(twistedReturnOrder);
                }
            }
        }
        rewriter.finalizeRootUpdate(op);
        return mlir::success();
    }
};

struct DecorateFoldingPass : public mlir::PassWrapper<DecorateFoldingPass, mlir::OperationPass<mlir::ModuleOp>>{
    void runOnOperation() override {
        mlir::ModuleOp m = this->getOperation();
        auto ctx = m->getContext();
        do{
            mlir::RewritePatternSet rps(ctx);
            rps.add<DecorateFoldRewriteRule>(ctx, m);
            mlir::FrozenRewritePatternSet frps(std::move(rps));
            (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
        }while(0);
        do{
            m->walk([=](mlir::FuncOp fn){
                if(!fn->hasAttr(ISQ_DECORATE_FOLDING_PROPAGATE_CTRL_BITS)){
                    return;
                }
                auto controls=fn->getAttrOfType<mlir::ArrayAttr>(ISQ_DECORATE_FOLDING_PROPAGATE_CTRL_BITS);
                int controlStartIndex=fn->getAttrOfType<mlir::IntegerAttr>(ISQ_DECORATE_FOLDING_PROPAGATE_CTRL_START_INDEX).getInt();
                mlir::RewritePatternSet rps(ctx);
                rps.add<InsertControllerBits>(ctx, controls, controlStartIndex, fn);
                mlir::FrozenRewritePatternSet frps(std::move(rps));
                (void)mlir::applyPatternsAndFoldGreedily(fn, frps);
            });
        }while(0);
        do{
            mlir::RewritePatternSet rps(ctx);
            rps.add<FakeMem2Reg>(ctx);
            mlir::FrozenRewritePatternSet frps(std::move(rps));
            (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
        }while(0);
    }
  mlir::StringRef getArgument() const final {
    return "isq-fold-decorated-gates";
  }
  mlir::StringRef getDescription() const final {
    return  "Folding for known/decomposed decorated gates.";
  }
};

void registerDecorateFolding(){
    mlir::PassRegistration<DecorateFoldingPass>();
}

}
}
}