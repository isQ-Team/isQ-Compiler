#include "isq/Operations.h"
#include <mlir/IR/Operation.h>
#include <llvm/ADT/TypeSwitch.h>
#include <isq/IR.h>
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
} // namespace ir
} // namespace isq