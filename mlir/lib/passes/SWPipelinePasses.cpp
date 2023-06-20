#include "isq/Operations.h"
#include "isq/QTypes.h"
#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineExprVisitor.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mutex>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <sstream>
#include <string>
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "nlohmann/json.hpp"
namespace isq::ir::passes{
    using nlohmann::json;
    const char* ISQ_ALLOC_ID = "isq_alloc_id";
    const char* ISQ_LOOP_ID = "isq_loop_id";
    const char* ISQ_AFFINE_LOADED = "isq_affine_loaded";
    const char* ISQ_AFFINE_SRC = "isq_affine_src";
    struct SWPState{
        int counter;
        int loop_counter;
        std::vector<mlir::memref::AllocOp> alloc_ops;
        SWPState(){
            this->counter = 0;
            this->loop_counter = 0;
        }
        int next_alloc(mlir::memref::AllocOp op){
            alloc_ops.push_back(op);
            return counter++;
        }
        int next_loop(){
            return loop_counter++;
        }
    };
    struct MarkLoop : public mlir::OpRewritePattern<mlir::AffineForOp>{
        SWPState* state;
        MarkLoop(mlir::MLIRContext* ctx, SWPState* state):
        mlir::OpRewritePattern<mlir::AffineForOp>(ctx, 10), state(state){

        }
        mlir::LogicalResult matchAndRewrite(mlir::AffineForOp op, mlir::PatternRewriter& rewriter) const override{
            if(op->hasAttr(ISQ_LOOP_ID)) return mlir::failure();
            rewriter.startRootUpdate(op);
            op->setAttr(ISQ_LOOP_ID,rewriter.getIndexAttr(state->next_loop()));
            rewriter.finalizeRootUpdate(op);
            return mlir::success();
        }
        
    };
    struct MarkAlloc : public mlir::OpRewritePattern<mlir::memref::AllocOp>{
        SWPState* state;
        MarkAlloc(mlir::MLIRContext* ctx, SWPState* state):
        mlir::OpRewritePattern<mlir::memref::AllocOp>(ctx, 10), state(state){

        }
        mlir::LogicalResult matchAndRewrite(mlir::memref::AllocOp op, mlir::PatternRewriter& rewriter) const override{
            if(op->hasAttr(ISQ_ALLOC_ID)) return mlir::failure();
            if(!llvm::isa<mlir::func::FuncOp>(op->getParentOp())) return mlir::failure();
            if(op.getMemref().getType().cast<mlir::MemRefType>().getElementType() == QStateType::get(this->getContext())){
                rewriter.startRootUpdate(op);
                op->setAttr(ISQ_ALLOC_ID,rewriter.getIndexAttr(state->next_alloc(op)));
                rewriter.finalizeRootUpdate(op);
                return mlir::success();
            }
            return mlir::failure();
        }
        
    };
    struct PropagateLoad : public mlir::OpRewritePattern<mlir::AffineLoadOp>{
        PropagateLoad(mlir::MLIRContext* ctx): mlir::OpRewritePattern<mlir::AffineLoadOp>(ctx){}
        std::optional<mlir::Attribute> try_get_loop_id(mlir::Value val) const{
            auto block = val.getParentBlock();
            if(!block) return std::nullopt;
            if(block->getNumArguments()!=1) return std::nullopt;
            if(block->getArgument(0)!=val) return std::nullopt;
            // check if from affine loop
            auto op = val.getParentRegion()->getParentOp();
            auto loop = llvm::dyn_cast_or_null<mlir::AffineForOp>(op);
            if(!loop) return std::nullopt;
            if(!loop->hasAttr(ISQ_LOOP_ID)) return std::nullopt;
            return loop->getAttr(ISQ_LOOP_ID);
        }
        mlir::LogicalResult matchAndRewrite(mlir::AffineLoadOp op, mlir::PatternRewriter& rewriter) const override{
            if(op->hasAttr(ISQ_AFFINE_LOADED)) return mlir::failure();
            auto src = llvm::dyn_cast_or_null<mlir::memref::AllocOp>(op.getMemRef().getDefiningOp());
            if(!src) return mlir::failure();
            if(!src->hasAttr(ISQ_ALLOC_ID)) return mlir::failure();
            auto alloc_id = src->getAttr(ISQ_ALLOC_ID).cast<mlir::IntegerAttr>().getInt();
            mlir::SmallVector<mlir::Attribute> indices;
            for(auto arg : op.getIndices()){
                auto maybe_loop_id = try_get_loop_id(arg);
                if(maybe_loop_id==std::nullopt){
                    return mlir::failure();
                }
                indices.push_back(*maybe_loop_id);
            }
            indices.push_back(op.getAffineMapAttr());
            indices.push_back(rewriter.getIndexAttr(alloc_id));
            mlir::ArrayAttr attr = rewriter.getArrayAttr(indices);
            
            rewriter.startRootUpdate(op);
            op->setAttr(ISQ_AFFINE_LOADED,attr);
            rewriter.finalizeRootUpdate(op);
            return mlir::success();
        }
    };
    struct PropagateApply : public mlir::OpRewritePattern<ApplyGateOp>{
        SWPState* state;
        PropagateApply(mlir::MLIRContext* ctx): mlir::OpRewritePattern<ApplyGateOp>(ctx, 1){

        } 
        mlir::LogicalResult matchAndRewrite(ApplyGateOp op, mlir::PatternRewriter& rewriter) const override{
            if(op->hasAttr(ISQ_AFFINE_SRC)) return mlir::failure();
            mlir::SmallVector<mlir::Attribute> affine_exprs;
            auto gate = op.getGate();
            auto gate_use = mlir::dyn_cast_or_null<UseGateOp>(gate.getDefiningOp());
            if(!gate_use) return mlir::failure();
            affine_exprs.push_back(gate_use.getName());
            //auto id = 0;
            for(auto q : op.getArgs()){
                auto val = q;
                auto src = val.getDefiningOp();
                if(!src) return mlir::failure();
                auto ret = llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(src)
                .Case<ApplyGateOp>([&](ApplyGateOp op){
                    if(!op->hasAttr(ISQ_AFFINE_SRC)) return mlir::failure();
                    auto tup = op->getAttr(ISQ_AFFINE_SRC).cast<mlir::ArrayAttr>();
                    for(auto i=0; i<op->getNumResults(); i++){
                        if(op->getResults()[i] == val){
                            affine_exprs.push_back(tup[i+1]);
                            break;
                        }
                    }
                    return mlir::success();
                })
                .Case<mlir::AffineLoadOp>([&](mlir::AffineLoadOp op){
                    if(!op->hasAttr(ISQ_AFFINE_LOADED)) return mlir::failure();
                    auto list = op->getAttr(ISQ_AFFINE_LOADED);
                    affine_exprs.push_back(list);
                    return mlir::success();
                }).Default([&](auto _){
                    return mlir::failure();
                });
                if(mlir::failed(ret)) return mlir::failure();
                //id++;
            }
            rewriter.startRootUpdate(op);
            op->setAttr(ISQ_AFFINE_SRC, rewriter.getArrayAttr(affine_exprs));
            rewriter.finalizeRootUpdate(op);
            return mlir::success();
        }
    };
    struct SimpleAffineExpr{
        unsigned dimpos;
        int64_t factor;
        int64_t bias;
        SimpleAffineExpr(): dimpos(-1), factor(0), bias(0){

        }
        SimpleAffineExpr(unsigned dimpos, int64_t factor, int64_t bias): dimpos(dimpos), factor(factor), bias(bias){

        }
        nlohmann::json toJson() const{
            json obj;
            obj["dimpos"] = dimpos;
            obj["factor"] = factor;
            obj["bias"] = bias;
            return obj;
        }
        /*
        std::string toString() const{
            return "("+std::to_string(dimpos) + "," + std::to_string(factor)+","+std::to_string(bias)+")";
        }
        */
        static std::optional<SimpleAffineExpr> fromAffineExpr(mlir::AffineExpr expr){
            return llvm::TypeSwitch<mlir::AffineExpr, std::optional<SimpleAffineExpr>>(expr)
            .Case<mlir::AffineDimExpr>([](mlir::AffineDimExpr expr)->std::optional<SimpleAffineExpr>{
                return SimpleAffineExpr(expr.getPosition(), 1, 0);
            })
            .Case<mlir::AffineConstantExpr>([](mlir::AffineConstantExpr expr)->std::optional<SimpleAffineExpr>{
                return SimpleAffineExpr(-1, 0, expr.getValue());
            })
            .Case<mlir::AffineBinaryOpExpr>([](mlir::AffineBinaryOpExpr expr)->std::optional<SimpleAffineExpr>{
                auto maybe_lhs = fromAffineExpr(expr.getLHS());
                auto maybe_rhs = fromAffineExpr(expr.getRHS());
                if(!maybe_lhs) return std::nullopt;
                if(!maybe_rhs) return std::nullopt;
                auto lhs = *maybe_lhs;
                auto rhs = *maybe_rhs;
                switch(expr.getKind()){
                    case mlir::AffineExprKind::Add:
                    {
                        if(lhs.dimpos == rhs.dimpos){
                            return SimpleAffineExpr(lhs.dimpos, lhs.factor + rhs.factor, lhs.bias+rhs.bias);
                        }
                        if(lhs.factor==0){
                            return SimpleAffineExpr(rhs.dimpos, rhs.factor, lhs.bias + rhs.bias);
                        }
                        if(rhs.factor==0){
                            return SimpleAffineExpr(lhs.dimpos, lhs.factor, lhs.bias + rhs.bias);
                        }
                        return std::nullopt;
                    }
                    break;
                    case mlir::AffineExprKind::Mul:
                    {
                        if(lhs.factor==0){
                            return SimpleAffineExpr(rhs.dimpos, rhs.factor * lhs.bias, lhs.bias * rhs.bias);
                        }
                        // k1x+b1 * b2
                        if(rhs.factor==0){
                            return SimpleAffineExpr(lhs.dimpos, lhs.factor * rhs.bias, lhs.bias * rhs.bias);
                        }
                        return std::nullopt;
                    }
                    break;
                    default:
                    return std::nullopt;
                }
            })
            .Default([](auto _)->std::optional<SimpleAffineExpr>{
                return std::nullopt;
            });
        }
    };
    struct QubitArrayRef{
        int alloc_id;
        mlir::SmallVector<int> used_loop_vars;
        mlir::SmallVector<SimpleAffineExpr> array_subscripts;
    };
    struct ExportedGate{
        std::string exported_name;
        mlir::SmallVector<QubitArrayRef> args;
        
        ExportedGate(llvm::StringRef name, mlir::SmallVector<QubitArrayRef>&& args) : exported_name(name), args(args){
            
        }
        json toJson() const{
            json obj;
            obj["gate"] = exported_name;
            llvm::SmallVector<json> json_args;
            for(auto& arg: args){
                json term;
                term["alloc_id"] = arg.alloc_id;
                term["loop_vars"] = json(arg.used_loop_vars);
                mlir::SmallVector<json> subscripts;
                for(auto& expr : arg.array_subscripts){
                    subscripts.push_back(std::move(expr.toJson()));
                }
                term["array_subscripts"] = json(subscripts);
                json_args.push_back(std::move(term));
            }
            obj["args"] = json(json_args);
            return obj;
        }
    };
    struct SWPipelinePass : public mlir::PassWrapper<SWPipelinePass, mlir::OperationPass<mlir::func::FuncOp>>{
        std::shared_ptr<std::mutex> mutex;
        SWPipelinePass(): mutex(std::make_shared<std::mutex>()){

        }
        void runOnOperation() override{
            auto func = this->getOperation();
            auto ctx = func->getContext();
            SWPState state;
            do{
                mlir::RewritePatternSet rps(ctx);
                rps.add<MarkLoop>(ctx, &state);
                rps.add<MarkAlloc>(ctx, &state);
                rps.add<PropagateLoad>(ctx);
                rps.add<PropagateApply>(ctx);
                mlir::FrozenRewritePatternSet frps(std::move(rps));
                mlir::GreedyRewriteConfig config;
                config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;
                (void)mlir::applyPatternsAndFoldGreedily(func, frps, config);
            }while(0);
            // collect all allocate ops.
            // find and extract all loops
            
            this->getOperation()->walk([&](mlir::AffineForOp op){
                bool all_apply_marked = true;
                
                op->walk([&](ApplyGateOp apply){
                    if(!all_apply_marked) return;
                    if(!apply->hasAttr(ISQ_AFFINE_SRC)){
                        all_apply_marked = false;
                    }
                });
                if(!all_apply_marked) return ;
                all_apply_marked = true;
                
                mlir::SmallVector<json> exported_gates;
                op->walk([&](ApplyGateOp apply){
                    if(!all_apply_marked) return;
                    auto attr = apply->getAttrOfType<mlir::ArrayAttr>(ISQ_AFFINE_SRC);
                    if(!attr){
                        all_apply_marked=false; return;
                    }
                    auto name = attr.getValue()[0].cast<mlir::SymbolRefAttr>();
                    llvm::SmallVector<QubitArrayRef> args;
                    for(auto i=0; i<apply->getNumResults(); i++){
                        auto arg = attr.getValue()[i+1].cast<mlir::ArrayAttr>();
                        auto dims_and_affinemap_and_allocid = arg.getValue();
                        llvm::SmallVector<int> dims;
                        for(auto i=0; i<dims_and_affinemap_and_allocid.size()-2; i++){
                            dims.push_back(dims_and_affinemap_and_allocid[i].cast<mlir::IntegerAttr>().getInt());
                        }
                        auto alloc_id = dims_and_affinemap_and_allocid.back().cast<mlir::IntegerAttr>().getInt();
                        auto affinemap = (*(dims_and_affinemap_and_allocid.end()-2)).cast<mlir::AffineMapAttr>().getAffineMap();
                        mlir::SmallVector<SimpleAffineExpr> subscripts;
                        for(auto subscript : affinemap.getResults()){
                            auto simple_expr = SimpleAffineExpr::fromAffineExpr(subscript);
                            if(!simple_expr){
                                all_apply_marked = false;
                                return;
                            }
                            subscripts.push_back(*simple_expr);
                        }
                        QubitArrayRef ref = {
                            .alloc_id = static_cast<int>(alloc_id),
                            .used_loop_vars = dims,
                            .array_subscripts = subscripts,
                        };
                        args.push_back(std::move(ref));
                    }
                    auto exported = ExportedGate(name.getLeafReference().str(), std::move(args));
                    exported_gates.push_back(std::move(exported.toJson()));
                });
                if(!all_apply_marked) return ;
                auto s = json(exported_gates);
                std::string str = s.dump();
                mutex->lock();
                llvm::outs()<<"json:\n"<<str<<"\n";
                mutex->unlock();
            });
            // export all loops to qswp and get the schedule back.

            // erase attributes.

        }
        mlir::StringRef getArgument() const final override{
            return "isq-affine-swp";
        }
        mlir::StringRef getDescription() const final override{
            return "(Experimental) perform software pipelining on affine loops";
        }
    };
    void registerAffineSWP(){
        mlir::PassRegistration<SWPipelinePass>();
    }
}