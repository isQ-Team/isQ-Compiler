#include "isq/GateDefTypes.h"
#include "isq/Operations.h"
#include "isq/QSynthesis.h"
#include "isq/passes/Passes.h"
#include "isq/Lower.h"
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace isq::ir::passes{

void insertGate(mlir::PatternRewriter& rewriter, mlir::ModuleOp rootModule, mlir::Operation* op, std::string gate_name, mlir::ArrayRef<mlir::Value*> qubits){
    
    auto ctx = rewriter.getContext();
    int qbit_num = qubits.size();
    auto new_defgate_sym = mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(ctx, gate_name));
    auto new_defgate = mlir::SymbolTable::lookupNearestSymbolFrom<DefgateOp>(op, new_defgate_sym);
    
    if (!new_defgate){
        auto ip = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointToStart(rootModule.getBody());
        mlir::SmallVector<mlir::Attribute> new_mat_defs;
        std::string func_name = "__quantum__qis__tmp";
        new_mat_defs.push_back(GateDefinition::get(mlir::StringAttr::get(ctx, "qir"), mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(ctx, func_name)), ctx));
        rewriter.create<DefgateOp>(::mlir::UnknownLoc::get(ctx), mlir::TypeAttr::get(GateType::get(ctx, qbit_num, GateTrait::General)), mlir::StringAttr::get(ctx, gate_name), mlir::StringAttr::get(ctx, "nested"), ::mlir::ArrayAttr{}, ::mlir::ArrayAttr::get(ctx, new_mat_defs), ::mlir::ArrayAttr::get(ctx, ::llvm::ArrayRef<::mlir::Attribute>{}));
        rewriter.restoreInsertionPoint(ip);    
    }  
    auto gate_type = GateType::get(ctx, qbit_num, GateTrait::General);
    auto use_gate = rewriter.create<UseGateOp>(op->getLoc(), gate_type, mlir::FlatSymbolRefAttr::get(ctx, gate_name), mlir::ArrayRef<mlir::Value>({}));
    mlir::SmallVector<mlir::Type> qubitTypes;
    for(auto i=0; i<qbit_num; i++) qubitTypes.push_back(QStateType::get(ctx));
    mlir::SmallVector<mlir::Value> qubitValues;
    for(auto i=0; i<qbit_num; i++){
        qubitValues.push_back(*qubits[i]);
    }
    auto apply_op = rewriter.create<ApplyGateOp>(op->getLoc(), qubitTypes, use_gate.result(), qubitValues);
    for(auto i=0; i<qbit_num; i++){
        *qubits[i] = apply_op->getResult(i);
    }

}


struct ReplaceCnot : public mlir::OpRewritePattern<ApplyGateOp>{
    mlir::ModuleOp rootModule;
    ReplaceCnot(mlir::MLIRContext* ctx, mlir::ModuleOp module): mlir::OpRewritePattern<ApplyGateOp>(ctx, 1), rootModule(module){}
    mlir::LogicalResult matchAndRewrite(ApplyGateOp op, mlir::PatternRewriter& rewriter) const override{
        auto ctx = op->getContext();
        auto decorate_op = llvm::dyn_cast<DecorateOp>(op.gate().getDefiningOp());
        if (decorate_op) return mlir::failure();
        auto usegate_op = llvm::dyn_cast<UseGateOp>(op.gate().getDefiningOp());
        if(!usegate_op) return mlir::failure();
        auto gatedef = llvm::dyn_cast_or_null<DefgateOp>(mlir::SymbolTable::lookupNearestSymbolFrom(usegate_op, usegate_op.name()));
        if(!gatedef) return mlir::failure();
        auto gate_name = gatedef.sym_name().str();
        if(gate_name != "CNOT") return mlir::failure();
        mlir::SmallVector<mlir::Value> operands;
        for(auto v: op.args()){
            operands.push_back(v);
        }
        insertGate(rewriter, rootModule, op, "H", {&operands[1]});
        insertGate(rewriter, rootModule, op, "CZ", {&operands[0], &operands[1]});
        insertGate(rewriter, rootModule, op, "H", {&operands[1]});
        
        rewriter.replaceOp(op, operands);
        return mlir::success();
    }
};

struct ReplaceInvTS : public mlir::OpRewritePattern<ApplyGateOp>{
    mlir::ModuleOp rootModule;
    ReplaceInvTS(mlir::MLIRContext* ctx, mlir::ModuleOp module): mlir::OpRewritePattern<ApplyGateOp>(ctx, 1), rootModule(module){}
    mlir::LogicalResult matchAndRewrite(ApplyGateOp op, mlir::PatternRewriter& rewriter) const override{
        auto ctx = op->getContext();
        auto decorate_op = llvm::dyn_cast<DecorateOp>(op.gate().getDefiningOp());
        if(!decorate_op) return mlir::failure();
        if(decorate_op.ctrl().size() > 0) return mlir::failure();
        if (!decorate_op.adjoint()) return mlir::failure();
        auto usegate_op = llvm::dyn_cast<UseGateOp>(decorate_op.args().getDefiningOp());
        if(!usegate_op) return mlir::failure();
        auto defgate = llvm::dyn_cast_or_null<DefgateOp>(mlir::SymbolTable::lookupNearestSymbolFrom(usegate_op, usegate_op.name()));
        if(!defgate) return mlir::failure();
        auto gate_name = defgate.sym_name().str();
        if(gate_name != "T" && gate_name != "S") return mlir::failure();
        gate_name = gate_name + "D";
        mlir::SmallVector<mlir::Value> operands;
        for(auto v: op.args()){
            operands.push_back(v);
        }
        insertGate(rewriter, rootModule, op, gate_name, {&operands[0]});
        rewriter.replaceOp(op, operands);
        return mlir::success();

    }
};
struct LowerToQCISPass : public mlir::PassWrapper<LowerToQCISPass, mlir::OperationPass<mlir::ModuleOp>>{
    void runOnOperation() override {
        mlir::ModuleOp m = this->getOperation();
        auto ctx = m->getContext();
        do{
            mlir::RewritePatternSet rps(ctx);
            rps.add<ReplaceCnot>(ctx, m);
            rps.add<ReplaceInvTS>(ctx, m);
            mlir::FrozenRewritePatternSet frps(std::move(rps));
            (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
        }while(0);
        
    }
    mlir::StringRef getArgument() const final {
        return "isq-convert-to-qcis";
    }
    mlir::StringRef getDescription() const final {
        return  "Remove trivial single-qubit gates.";
    }
};


void registerConvertToQCIS(){
    mlir::PassRegistration<LowerToQCISPass>();
}

}
