#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Support/LLVM.h>
#include "isq/Operations.h"
#include "isq/QStructs.h"
#include "isq/QTypes.h"
#include "isq/passes/Passes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include <unordered_set>
#include "isq/GateDefTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
namespace isq{
namespace ir{
namespace passes{

const char* ISQ_PURE_GATE = "isq_pure_gate";
const char* ISQ_PURE_LOAD = "isq_load";
const char* ISQ_PURE_STORE = "isq_store";
const char* ISQ_PURE_STORE_OPERAND = "isq_store_operand";
using StringSmallPtrSet = llvm::SmallPtrSet<::mlir::StringAttr, 16>;
class DetectPureGates : public mlir::OpRewritePattern<DefgateOp>{
    mlir::ModuleOp rootModule;
    StringSmallPtrSet& pureSymbols;
public:
    DetectPureGates (mlir::MLIRContext* ctx, mlir::ModuleOp module, StringSmallPtrSet& pureSymbols): mlir::OpRewritePattern<DefgateOp>(ctx, 1), rootModule(module), pureSymbols(pureSymbols){
        
    }
    mlir::LogicalResult matchAndRewrite(DefgateOp op, mlir::PatternRewriter& rewriter) const override{
        auto ctx = op->getContext();
        if(op->hasAttr(ISQ_PURE_GATE)){
            return mlir::failure();
        }
        rewriter.startRootUpdate(op);
        op->setAttr(mlir::StringAttr::get(ctx, ISQ_PURE_GATE), mlir::UnitAttr::get(ctx));
        rewriter.finalizeRootUpdate(op);
        int id = 0;
        for(auto def: op.definition()->getAsRange<GateDefinition>()){
            auto d = AllGateDefs::parseGateDefinition(op, id, op.type(), def);
            if(d==std::nullopt) return mlir::failure();
            if(auto def = llvm::dyn_cast_or_null<DecompositionRawDefinition>(&**d)){
                pureSymbols.insert(def->getDecomposedFunc().sym_nameAttr());
            }
            id++;
        }
        
        return mlir::success();
    }
};

class TagPureGates : public mlir::OpRewritePattern<mlir::FuncOp>{
    mlir::ModuleOp rootModule;
    StringSmallPtrSet& pureSymbols;
public:
    TagPureGates (mlir::MLIRContext* ctx, mlir::ModuleOp module, StringSmallPtrSet& pureSymbols): mlir::OpRewritePattern<mlir::FuncOp>(ctx, 1), rootModule(module), pureSymbols(pureSymbols){

    }
    mlir::LogicalResult matchAndRewrite(mlir::FuncOp op, mlir::PatternRewriter& rewriter) const override{
        auto ctx = op->getContext();
        if(op->hasAttr(ISQ_PURE_GATE)){
            return mlir::failure();
        }
        if(pureSymbols.find(op.sym_nameAttr())!=pureSymbols.end()){
            rewriter.startRootUpdate(op);
            op->setAttr(mlir::StringAttr::get(ctx, ISQ_PURE_GATE), mlir::UnitAttr::get(ctx));
            rewriter.finalizeRootUpdate(op);
        }

        return mlir::success();
    }
};

class PureGatedefCleanup : public mlir::OpRewritePattern<DefgateOp>{
    mlir::ModuleOp rootModule;
public:
    PureGatedefCleanup(mlir::MLIRContext* ctx, mlir::ModuleOp module): mlir::OpRewritePattern<DefgateOp>(ctx, 1), rootModule(module){

    }
    mlir::LogicalResult matchAndRewrite(DefgateOp op, mlir::PatternRewriter& rewriter) const override{
        auto ctx = op->getContext();
        // Check for pure-gate notation.
        if(!op->hasAttr(ISQ_PURE_GATE)){
            return mlir::failure();
        }
        rewriter.startRootUpdate(op);
        if(op.definition()){
            mlir::SmallVector<mlir::Attribute> attrs; 
            for(auto def: *op.definition()){
                
                auto gatedef = def.cast<GateDefinition>();
                if(gatedef.type()=="decomposition_raw"){
                    attrs.push_back(GateDefinition::get(mlir::StringAttr::get(ctx, "decomposition"), gatedef.value(), ctx));
                }else{
                    attrs.push_back(gatedef);
                }
            }
            op.definitionAttr(mlir::ArrayAttr::get(ctx, attrs));
        }
        op->removeAttr(ISQ_PURE_GATE);
        rewriter.finalizeRootUpdate(op);
        return mlir::success();
    }
};

class PureGateRewrite : public mlir::OpRewritePattern<mlir::FuncOp>{
    mlir::ModuleOp rootModule;
public:
    PureGateRewrite(mlir::MLIRContext* ctx, mlir::ModuleOp module): mlir::OpRewritePattern<mlir::FuncOp>(ctx, 1), rootModule(module){

    }

    void mem2regOp(mlir::SmallVector<mlir::Value>& values, mlir::PatternRewriter& rewriter, mlir::scf::IfOp op) const{
        mlir::SmallVector<mlir::Type> ifReturn;
        for(auto ty: op->getResultTypes()){
            ifReturn.push_back(ty);
        }
        for(auto v: values){
            ifReturn.push_back(v.getType());
        }
        mlir::PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(op);
        auto new_if = rewriter.create<mlir::scf::IfOp>(op->getLoc(), op.getCondition(), true);
        op.getThenRegion().takeBody(new_if.getThenRegion());
        if(op.elseBlock()){
            op.getElseRegion().takeBody(new_if.getElseRegion());
        }
        mem2regBlock({}, new_if.thenBlock(), rewriter, values);
        mem2regBlock({}, new_if.elseBlock(), rewriter, values);
        for(auto i=0; i<values.size(); i++){
            values[i]=new_if->getResult(i+op->getNumResults());
        }
        rewriter.replaceOp(op, new_if->getOpResults().take_front(op.getNumResults()));
    }

    // TODO: take this part out as a standalone mem2reg rewriter. 
    void mem2regBlock(mlir::TypeRange types, mlir::Block* block, mlir::PatternRewriter& rewriter, mlir::SmallVector<mlir::Value> blockArgs) const {
        if(blockArgs.size()==0){
            auto original_id = block->getNumArguments();
            for(auto ty: types){
                blockArgs.push_back(block->addArgument(ty, mlir::UnknownLoc::get(rewriter.getContext())));
            }
        }

        // go forward.
        mlir::SmallVector<mlir::Operation*> all_ops;
        for(auto& op: block->getOperations()){
            all_ops.push_back(&op);
        }
        for(auto pop: all_ops){
            auto& op = *pop;
            if(auto attr = op.getAttr(ISQ_PURE_LOAD).dyn_cast_or_null<mlir::IntegerAttr>()){
                auto load_id = attr.getInt();
                rewriter.replaceOp(&op, blockArgs[load_id]);
            }else if(auto attr = op.getAttr(ISQ_PURE_STORE).dyn_cast_or_null<mlir::IntegerAttr>()){
                auto store_id = attr.getInt();
                auto operand_id = op.getAttrOfType<mlir::IntegerAttr>(ISQ_PURE_STORE_OPERAND).getInt();
                auto stored_value = op.getOperand(operand_id);
                blockArgs[store_id] = stored_value;
                rewriter.eraseOp(&op);
            }else if (op.mightHaveTrait<mlir::OpTrait::IsTerminator>()){
                do{
                    auto last = &op;
                    mlir::PatternRewriter::InsertionGuard guard(rewriter);
                    mlir::SmallVector<mlir::Value> allArgs;
                    for(auto arg: last->getOperands()){
                        allArgs.push_back(arg);
                    }
                    for(auto blockArg: blockArgs){
                        allArgs.push_back(blockArg);
                    }
                    mlir::OperationState state(last->getLoc(), last->getName(), allArgs, last->getResultTypes(), last->getAttrs(), last->getSuccessors(), {});
                    rewriter.setInsertionPointAfter(last);
                    rewriter.createOperation(state);
                    rewriter.eraseOp(last);
                }while(0);
            }else if(auto new_op = llvm::dyn_cast<mlir::scf::IfOp>(op)){
                mem2regOp(blockArgs, rewriter, new_op);
            }
        }
    }
    
    mlir::LogicalResult matchAndRewrite(mlir::FuncOp op, mlir::PatternRewriter& rewriter) const override{
        auto ctx = op->getContext();
        // Check for pure-gate notation.
        if(!op->hasAttr(ISQ_PURE_GATE)){
            return mlir::failure();
        }
        
        
        rewriter.startRootUpdate(op);
        op->removeAttr(ISQ_PURE_GATE);
        // Transform all parameters.
        auto func_type = op.getType();
        ::mlir::SmallVector<::mlir::Type> args;
        ::mlir::SmallVector<::mlir::Type> results;
        ::mlir::SmallVector<::mlir::Type> extra_args;
        assert(func_type.getResults().size()==0);
        ::mlir::SmallVector<int> arg_rewrite;
        for(auto argi=0; argi<func_type.getNumInputs(); argi++){
            auto arg = func_type.getInput(argi);
            auto memref = arg.dyn_cast<mlir::MemRefType>();
            if(memref && memref.getElementType().isa<QStateType>()){
                // assert they are single qubits.
                // Thus load/store/casting/subviewing can be seen as nop.
                assert(memref.getShape().size()==1);
                assert(memref.getDimSize(0)==1);
                args.push_back(QStateType::get(ctx));
                results.push_back(QStateType::get(ctx));
                extra_args.push_back(QStateType::get(ctx));
                arg_rewrite.push_back(argi);
            }else{
                args.push_back(arg);
            }
        }
        op.setType(mlir::FunctionType::get(ctx, args, results));
        ::mlir::SmallVector<::mlir::Value> arg_rewrite_args;
        for(auto index: arg_rewrite){
            auto arg = op.getArgument(index);
            arg.setType(QStateType::get(ctx));
            arg_rewrite_args.push_back(arg);
        }
        for(auto index: arg_rewrite){
            auto arg = op.getArgument(index);
            for(auto op: arg.getUsers()){
                if(llvm::isa<mlir::memref::LoadOp>(op) || llvm::isa<mlir::AffineLoadOp>(op)  || llvm::isa<mlir::memref::CastOp>(op) || llvm::isa<mlir::memref::SubViewOp>(op)){
                    op->setAttr(ISQ_PURE_LOAD, mlir::IntegerAttr::get(rewriter.getI64Type(), index));
                }else if(llvm::isa<mlir::memref::StoreOp>(op)){
                    op->setAttr(ISQ_PURE_STORE, mlir::IntegerAttr::get(rewriter.getI64Type(), index));
                    op->setAttr(ISQ_PURE_STORE_OPERAND, mlir::IntegerAttr::get(rewriter.getI64Type(), 0));
                }else if(llvm::isa<mlir::AffineStoreOp>(op)){
                    op->setAttr(ISQ_PURE_STORE, mlir::IntegerAttr::get(rewriter.getI64Type(), index));
                    op->setAttr(ISQ_PURE_STORE_OPERAND, mlir::IntegerAttr::get(rewriter.getI64Type(), 0));
                }
                
            }
        }
        for(auto& block: op.getBlocks()){
            if(block.isEntryBlock()){
                mem2regBlock({}, &block, rewriter, arg_rewrite_args);
            }else{
                mem2regBlock(extra_args, &block, rewriter, mlir::SmallVector<mlir::Value>{});
            }
        }

        rewriter.finalizeRootUpdate(op);
        return mlir::success();
    }
};

struct PureGateDetectPass : public mlir::PassWrapper<PureGateDetectPass, mlir::OperationPass<mlir::ModuleOp>>{
    void runOnOperation() override{
        StringSmallPtrSet pureFuncs;
        mlir::ModuleOp m = this->getOperation();
        auto ctx = m->getContext();
        // Find all pure gate functions.
        do{
            mlir::RewritePatternSet rps(ctx);
            rps.add<DetectPureGates>(ctx, m, pureFuncs);
            mlir::FrozenRewritePatternSet frps(std::move(rps));
            (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
        }while(0);
        // Tag the pure gate functions.
        do{
            mlir::RewritePatternSet rps(ctx);
            rps.add<TagPureGates>(ctx, m, pureFuncs);
            mlir::FrozenRewritePatternSet frps(std::move(rps));
            (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
        }while(0);
        // Perform rewrite on functions and rewrite gatedefs.
        do{
            mlir::RewritePatternSet rps(ctx);
            rps.add<PureGateRewrite>(ctx, m);
            rps.add<PureGatedefCleanup>(ctx, m);
            mlir::FrozenRewritePatternSet frps(std::move(rps));
            (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
        }while(0);

    }
    mlir::StringRef getArgument() const final{
        return "isq-pure-gate-detection";
    }
    mlir::StringRef getDescription() const final{
        return "Transform qubits in pure gates into SSA form.";
    }
};

void registerPureGateDetect(){
    mlir::PassRegistration<PureGateDetectPass>();
}
bool isFamousGate(DefgateOp op, const char* famous_gate){
    return op->hasAttr(ISQ_FAMOUS_GATE) && op->getAttrOfType<mlir::StringAttr>(ISQ_FAMOUS_GATE).strref()==famous_gate;
}

}
}
}