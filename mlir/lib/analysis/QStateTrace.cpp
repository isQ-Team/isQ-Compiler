#include "isq/Operations.h"
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Operation.h>
#include <llvm/ADT/TypeSwitch.h>
#include <isq/IR.h>
#include <mlir/IR/Value.h>
namespace isq {
namespace ir {
mlir::Value traceForwardQState(mlir::Value val){
    auto is_drain = false;
    while(!is_drain){
        if(!val.hasOneUse()) {break;}
        auto* usage = &*val.getUses().begin();
        mlir::TypeSwitch<mlir::Operation*, void>(usage->getOwner()).Case<ApplyGateOp>([&](ApplyGateOp op){
            auto id = usage->getOperandNumber();
            val = op->getResult(id);
        }).Case<CallQOpOp>([&](CallQOpOp op){
            auto id = usage->getOperandNumber();
            val = op->getResult(id);
        }).Default([&](auto op){
            is_drain = true;
        });
    }
    return val;
}
mlir::OpResult asResult(mlir::Value val){
    auto def = val.getDefiningOp();
    if(!def) return nullptr;
    for(auto result: def->getResults()){
        if(result==val){
            return result;
        }
    }
    llvm_unreachable("impossible!");
}
mlir::Value traceBackwardQState(mlir::Value val){
    auto is_drain = false;
    while(!is_drain){
        auto val_result = asResult(val);
        if(!val_result) break;
        mlir::TypeSwitch<mlir::Operation*, void>(val.getDefiningOp()).Case<ApplyGateOp>([&](ApplyGateOp op){
            auto id = val_result.getResultNumber();
            val = op.args()[id];
        }).Case<CallQOpOp>([&](CallQOpOp op){
            auto id = val_result.getResultNumber();
            val = op.args()[id];
        }).Default([&](auto op){
            is_drain = true;
        });
    }
    return val;
}
} // namespace ir
} // namespace isq