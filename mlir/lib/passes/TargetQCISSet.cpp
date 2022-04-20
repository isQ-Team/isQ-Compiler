#include "isq/Operations.h"
#include "isq/passes/Passes.h"
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace isq::ir::passes{

struct CX2HCZH : mlir::OpRewritePattern<ApplyGateOp>{
public:
    CX2HCZH(mlir::MLIRContext* ctx): mlir::OpRewritePattern<ApplyGateOp>(ctx, 1){}
    mlir::LogicalResult matchAndRewrite(ApplyGateOp apply, mlir::PatternRewriter& rewriter) const override{
        auto usegate = llvm::dyn_cast_or_null<UseGateOp>(apply.gate().getDefiningOp());
        if(!usegate) return mlir::failure();
        auto defgate = llvm::dyn_cast_or_null<DefgateOp>(mlir::SymbolTable::lookupNearestSymbolFrom(usegate, usegate.name()));
        if(!defgate) return mlir::failure();
        auto ctx = rewriter.getContext();
        if(isFamousGate(defgate, "cnot")){
            mlir::Value v1 = apply->getOperand(1);
            mlir::Value v2 = apply->getOperand(2);
            emitBuiltinGate(rewriter, "H", {&v2});
            emitBuiltinGate(rewriter, "CZ", {&v1, &v2});
            emitBuiltinGate(rewriter, "H", {&v2});
            rewriter.replaceOp(apply, mlir::ArrayRef<mlir::Value>{v1, v2});
            return mlir::success();
        }
        return mlir::failure();
    }
};

struct U3ToZYZ : mlir::OpRewritePattern<ApplyGateOp>{
public:
    U3ToZYZ(mlir::MLIRContext* ctx): mlir::OpRewritePattern<ApplyGateOp>(ctx, 1){}
    mlir::LogicalResult matchAndRewrite(ApplyGateOp apply, mlir::PatternRewriter& rewriter) const override{
        auto usegate = llvm::dyn_cast_or_null<UseGateOp>(apply.gate().getDefiningOp());
        if(!usegate) return mlir::failure();
        auto defgate = llvm::dyn_cast_or_null<DefgateOp>(mlir::SymbolTable::lookupNearestSymbolFrom(usegate, usegate.name()));
        if(!defgate) return mlir::failure();
        auto ctx = rewriter.getContext();
        if(isFamousGate(defgate, "u3")){
            auto lambda = usegate->getOperand(0);
            auto theta = usegate->getOperand(1);
            auto phi = usegate->getOperand(2);
            mlir::Value v1 = apply->getOperand(1);
            emitBuiltinGate(rewriter, "Rz", {&v1}, {lambda});
            emitBuiltinGate(rewriter, "Ry", {&v1}, {theta});
            emitBuiltinGate(rewriter, "Rz", {&v1}, {phi});
            rewriter.replaceOp(apply, mlir::ArrayRef<mlir::Value>{v1});
            return mlir::success();
        }
        return mlir::failure();
    }
};

struct InvTSRecog : mlir::OpRewritePattern<ApplyGateOp>{
public:
    InvTSRecog(mlir::MLIRContext* ctx): mlir::OpRewritePattern<ApplyGateOp>(ctx, 1){}
    mlir::LogicalResult matchAndRewrite(ApplyGateOp apply, mlir::PatternRewriter& rewriter) const override{
        auto decorate = llvm::dyn_cast_or_null<DecorateOp>(apply.gate().getDefiningOp());
        if(!decorate) return mlir::failure();
        if(!decorate.adjoint()) return mlir::failure();
        if(decorate.ctrl().size()) return mlir::failure();
        auto usegate = llvm::dyn_cast_or_null<UseGateOp>(decorate.args().getDefiningOp());
        if(!usegate) return mlir::failure();
        auto defgate = llvm::dyn_cast_or_null<DefgateOp>(mlir::SymbolTable::lookupNearestSymbolFrom(usegate, usegate.name()));
        if(!defgate) return mlir::failure();
        auto ctx = rewriter.getContext();
        if(isFamousGate(defgate, "T")){
            mlir::Value v1 = apply->getOperand(1);
            emitBuiltinGate(rewriter, "Tinv", {&v1});
            rewriter.replaceOp(apply, mlir::ArrayRef<mlir::Value>{v1});
            return mlir::success();
        }else if(isFamousGate(defgate, "S")){
            mlir::Value v1 = apply->getOperand(1);
            emitBuiltinGate(rewriter, "Sinv", {&v1});
            rewriter.replaceOp(apply, mlir::ArrayRef<mlir::Value>{v1});
            return mlir::success();
        }
        return mlir::failure();
    }
};

class TargetQCISSetPass : public mlir::PassWrapper<TargetQCISSetPass, mlir::OperationPass<mlir::ModuleOp>>{
    void runOnOperation() override{
        mlir::ModuleOp m = this->getOperation();
        auto ctx = m->getContext();
        // first check measurements.
        bool failed = false;
        m->walk([&](CallQOpOp op){
            if(op.callee().getLeafReference().getValue()=="__isq__builtin__measure"){
                auto uses = op->getResult(1).getUses();
                if(!uses.empty()){
                    op->emitOpError("result should not be used for feedback control.");
                    failed=true;
                }
            }else if(op.callee().getLeafReference().getValue()=="__isq__builtin__reset"){
                op->emitOpError("is not supported for QCIS.");
                failed = true;
            }
        });
        if(failed){
            return signalPassFailure();
        }
        do{
            mlir::RewritePatternSet rps(ctx);
            rps.add<CX2HCZH>(ctx);
            //rps.add<U3ToZYZ>(ctx);
            rps.add<InvTSRecog>(ctx);
            mlir::FrozenRewritePatternSet frps(std::move(rps));
            (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
        }while(0);
        // append finalize
        const char* finalize_qir_name = "__quantum__qis__qcis__finalize";
        const char* isq_entry_name = "__isq__entry";
        mlir::OpBuilder builder(ctx);
        builder.setInsertionPointToStart(&*m.body().begin());
        auto builtin_loc = mlir::NameLoc::get(builder.getStringAttr("<builtin>"));
        if(!mlir::SymbolTable::lookupSymbolIn(m, finalize_qir_name)){
            auto funcType = mlir::FunctionType::get(ctx, (mlir::TypeRange){}, (mlir::TypeRange){});
            builder.create<mlir::FuncOp>(builtin_loc, finalize_qir_name, funcType, builder.getStringAttr("private"));
        }
        auto isq_entry = llvm::dyn_cast_or_null<mlir::FuncOp>(mlir::SymbolTable::lookupSymbolIn(m, isq_entry_name));
        if(isq_entry){
            auto first_block= &*isq_entry.body().begin();
            builder.setInsertionPoint(first_block->getTerminator());
            builder.create<mlir::CallOp>(builtin_loc, ::mlir::FlatSymbolRefAttr::get(ctx, finalize_qir_name), ::mlir::TypeRange{}, ::mlir::ValueRange{});
        }
    }
    mlir::StringRef getArgument() const final {
        return "isq-target-qcis";
    }
    mlir::StringRef getDescription() const final {
        return  "Convert to QCIS instruction set. Convert U3 to rotation and CNOT to H-CZ-H.";
    }
};


void registerTargetQCISSet(){
    mlir::PassRegistration<TargetQCISSetPass>();
}


}
