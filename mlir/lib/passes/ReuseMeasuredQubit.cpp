#include "isq/Operations.h"
#include "isq/QTypes.h"
#include "isq/passes/Passes.h"
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/AsmParser/AsmParser.h>
namespace isq::ir::passes{
    using mlir::func::FuncOp;
    using mlir::Value;
    struct ReusableQubit{
        Value memref;
        Value measured_result;
        void restore_state(mlir::RewriterBase& rewriter){
            auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 0);
            auto loaded = rewriter.create<mlir::AffineLoadOp>(rewriter.getUnknownLoc(), memref, mlir::ValueRange{zero->getResult(0)});
            auto resetted = rewriter.create<CallQOpOp>(rewriter.getUnknownLoc(), mlir::TypeRange{QStateType::get(rewriter.getContext())},mlir::FlatSymbolRefAttr::get(rewriter.getStringAttr("__isq__builtin__reset")), mlir::ValueRange{loaded.getResult()}, 1, mlir::TypeAttr::get(rewriter.getFunctionType({}, {})));
            rewriter.create<mlir::AffineStoreOp>(rewriter.getUnknownLoc(), resetted.getResult(0), memref, mlir::ValueRange{zero->getResult(0)});
            auto true_ = rewriter.create<mlir::arith::ConstantIntOp>(rewriter.getUnknownLoc(), 1, 1);
            auto test_at = rewriter.create<mlir::arith::CmpIOp>(rewriter.getUnknownLoc(), mlir::arith::CmpIPredicate::eq, measured_result, true_.getResult());
            
            auto branch_reset =rewriter.create<mlir::scf::IfOp>(rewriter.getUnknownLoc(), test_at.getResult(), false);
            mlir::RewriterBase::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(branch_reset.thenBlock());
            do{
                auto loaded = rewriter.create<mlir::AffineLoadOp>(rewriter.getUnknownLoc(), memref, mlir::ValueRange{zero->getResult(0)});
                auto loaded_qubit = loaded.getResult();
                emitBuiltinGate(rewriter, "x", {&loaded_qubit});
                rewriter.create<mlir::AffineStoreOp>(rewriter.getUnknownLoc(), loaded_qubit, memref, mlir::ValueRange{zero->getResult(0)});
            }while(0);

        }
    };
    struct ReuseQubitInterpreter{
        FuncOp func;
        llvm::DenseMap<mlir::Value, mlir::Value> cast_mapping;
        ReuseQubitInterpreter(FuncOp func): func(func){

        }
        // Finds the corresponding dealloc op.
        // If there are more than 1 dealloc, give up.
        // TODO: introducing lifetime analysis.
        mlir::memref::DeallocOp findDealloc(Value memref){
            mlir::memref::DeallocOp dealloc;
            for(auto user: memref.getUsers()){
                auto candidate = llvm::dyn_cast_or_null<mlir::memref::DeallocOp>(user);
                if(candidate){
                    if(dealloc) return nullptr;
                    dealloc = candidate;
                }
            }
            return dealloc;
        }

        mlir::LogicalResult visit(mlir::IRRewriter& rewriter, mlir::Operation* op, mlir::SmallVector<ReusableQubit>& measured_qubits){
            return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
            .Case<FuncOp>([&](auto cop){ return visit(rewriter, cop, measured_qubits);})
            .Case<mlir::AffineIfOp>([&](auto cop){ return visit(rewriter, cop, measured_qubits);})
            .Case<mlir::memref::AllocOp>([&](auto cop){ return visit(rewriter, cop, measured_qubits);})
            .Case<mlir::memref::CastOp>([&](auto cop){ return visit(rewriter, cop, measured_qubits);})
            .Case<CallQOpOp>([&](auto cop){ return visit(rewriter, cop, measured_qubits);})
            .Case<ApplyGateOp>([&](auto cop){ return visit(rewriter, cop, measured_qubits);})
            .Case<mlir::scf::IfOp>([&](auto cop){ return visit(rewriter, cop, measured_qubits);})
            .Default([&](auto op){ return mlir::success();});
        }
        mlir::LogicalResult visit(mlir::IRRewriter& rewriter, FuncOp func, mlir::SmallVector<ReusableQubit>& measured_qubits){
            if(func.getBody().hasOneBlock()){
                (void)visitBunch(rewriter, &*func.getBody().begin(), measured_qubits);
            }
            return mlir::success();
        }
        mlir::LogicalResult visitBunch(mlir::IRRewriter& rewriter, mlir::Block* block, mlir::SmallVector<ReusableQubit>& measured_qubits){
            if(!block) return mlir::success();
            
            mlir::SmallVector<mlir::Operation*> ops;
            for(auto& op: block->getOperations()){
                ops.push_back(&op);
            }
            llvm::outs()<<"bunch"<<ops.size()<<"\n";
            for(auto op: ops){
                (void)visit(rewriter, op, measured_qubits);
            }
            return mlir::success();
        }
        mlir::LogicalResult visit(mlir::IRRewriter& rewriter, mlir::AffineIfOp op, mlir::SmallVector<ReusableQubit>& measured_qubits){
            return visitTwoBranches(rewriter, op.getThenBlock(), op.getElseBlock(), measured_qubits);
        }
        mlir::LogicalResult visit(mlir::IRRewriter& rewriter, mlir::scf::IfOp op, mlir::SmallVector<ReusableQubit>& measured_qubits){
            return visitTwoBranches(rewriter, op.thenBlock(), op.elseBlock(), measured_qubits);
        }

        mlir::LogicalResult visitTwoBranches(mlir::IRRewriter& rewriter, mlir::Block* thenBlock, mlir::Block* elseBlock, mlir::SmallVector<ReusableQubit>& measured_qubits){
            auto branch_1 = mlir::SmallVector(measured_qubits);
            auto branch_2 = mlir::SmallVector(measured_qubits);
            auto original = mlir::SmallVector(measured_qubits);
            (void)visitBunch(rewriter, thenBlock, branch_1);
            (void)visitBunch(rewriter, elseBlock, branch_2);



            measured_qubits.clear();
            for(auto originally_cleared : original){
                bool still_clear = false;
                for(auto also_cleared: branch_1){
                    if(also_cleared.measured_result == originally_cleared.measured_result && also_cleared.memref == originally_cleared.memref){
                        still_clear=true; break;
                    }
                }
                if(!still_clear) break;
                still_clear = false;
                for(auto also_cleared: branch_2){
                    if(also_cleared.measured_result == originally_cleared.measured_result && also_cleared.memref == originally_cleared.memref){
                        still_clear=true; break;
                    }
                }
                if(!still_clear) break;
                measured_qubits.push_back(originally_cleared);
            }
            return mlir::success();
        }

        mlir::Value traceMemref(mlir::Value qstate){
            auto first_loaded_value = traceBackwardQState(qstate);
            auto first_load = qstate.getDefiningOp<mlir::AffineLoadOp>();
            if(!first_load) return nullptr;
            auto memref = first_load.getMemRef();
            if(memref){
                if(cast_mapping.count(memref)){
                    return cast_mapping[memref];
                }
            }
            return memref;
        }

        void taintQState(mlir::Value qstate, mlir::SmallVector<ReusableQubit>& measured_qubits){
            if(!qstate.getType().isa<QStateType>()) return;
            auto possible_memref = traceMemref(qstate);
            if(possible_memref){
                mlir::SmallVector<ReusableQubit> qubits;
                for(auto x : measured_qubits){
                    if(x.memref!=possible_memref){
                        qubits.push_back(x);
                    }
                }
                measured_qubits = qubits;
            }else{
                // conservatively assume all qubits are tainted.
                measured_qubits.clear();
            }
        }
        mlir::LogicalResult visit(mlir::IRRewriter& rewriter, mlir::memref::CastOp cast_op, mlir::SmallVector<ReusableQubit>& measured_qubits){
            cast_mapping.insert(std::make_pair(cast_op.getResult(), cast_op.getSource()));
            return mlir::success();
        }
        mlir::LogicalResult visit(mlir::IRRewriter& rewriter, CallQOpOp call_qop, mlir::SmallVector<ReusableQubit>& measured_qubits){
            llvm::outs()<<"visit callqop\n";
            for(auto arg: call_qop.getArgs()){
                taintQState(arg, measured_qubits);
            }
            // check if it is measurement.
            if(call_qop.getCallee().getLeafReference().getValue()=="__isq__builtin__measure"){
                auto memref = traceMemref(call_qop.getArgs()[0]);
                if(memref && memref.isa<mlir::BlockArgument>()){
                    measured_qubits.push_back(ReusableQubit{
                        .memref = memref,
                        .measured_result = call_qop->getResult(1)
                    });
                }
            }
            return mlir::success();
        }
        mlir::LogicalResult visit(mlir::IRRewriter& rewriter, ApplyGateOp call_qop, mlir::SmallVector<ReusableQubit>& measured_qubits){
            for(auto arg: call_qop.getArgs()){
                taintQState(arg, measured_qubits);
            }
            return mlir::success();
        }
        mlir::LogicalResult visit(mlir::IRRewriter& rewriter, mlir::memref::AllocOp alloc, mlir::SmallVector<ReusableQubit>& measured_qubits){
            if(measured_qubits.size()){
                // try reuse
                auto memref = alloc.getMemref().getType().cast<mlir::MemRefType>();
                if(memref == mlir::MemRefType::get({1}, QStateType::get(rewriter.getContext()))){
                    auto dealloc = findDealloc(alloc);
                    if(dealloc){
                        // reuse.
                        auto reused = measured_qubits.back();
                        measured_qubits.pop_back();
                        //rewriter.setInsertionPoint(alloc);
                        rewriter.replaceOp(alloc, {reused.memref});
                        rewriter.setInsertionPoint(dealloc);
                        reused.restore_state(rewriter);
                        rewriter.eraseOp(dealloc);
                        /*rewriter.create<mlir::memref::SubViewOp>(mlir::UnknownLoc::get(rewriter.getContext()), reused.memref, mlir::ArrayRef<int64_t>{0}, mlir::ArrayRef<int64_t>{1}, mlir::ArrayRef<int64_t>{0});
                        rewriter.replaceOp(alloc, );*/
                    }
                }
            }

            return mlir::success();
        }

        void run(){
            mlir::IRRewriter rewriter(func.getContext());
            mlir::SmallVector<ReusableQubit> measured_qubits;
            (void)visit(rewriter, func, measured_qubits);
        }
        
    };
    struct ReuseQubitPass : public mlir::PassWrapper<ReuseQubitPass, mlir::OperationPass<mlir::ModuleOp>>{
        void runOnOperation() override{
            getOperation()->walk([](FuncOp func){
                //FuncOp func = getOperation();
                ReuseQubitInterpreter interp(func);
                interp.run();
            });
            
        }
    mlir::StringRef getArgument() const final {
        return "isq-reuse-qubit";
    }
    mlir::StringRef getDescription() const final {
        return  "Try reuse measured qubits.";
    }
    };
    void registerReuseQubit(){
        mlir::PassRegistration<ReuseQubitPass>();
    }
}