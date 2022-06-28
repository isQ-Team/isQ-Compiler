#ifndef _ISQ_UTILS_DISPATCHOPERATION_H
#define _ISQ_UTILS_DISPATCHOPERATION_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
namespace isq{
namespace ir{
template<typename ... OpTypes> class OpVisitor{};
template<> class OpVisitor<>{
protected:
    mlir::LogicalResult visitOperation(mlir::Operation* op){
        return visitOp(op);
    }
    virtual mlir::LogicalResult visitOp(mlir::Operation* op){
        return mlir::failure();
    }
};
template<typename CurrOp, typename ...OpTypes>
class OpVisitor<CurrOp, OpTypes...>: protected OpVisitor<OpTypes...>{
private:
    using Parent = OpVisitor<OpTypes...>;
protected:
    using Parent::visitOp;
    virtual mlir::LogicalResult visitOp(CurrOp op){
        return mlir::success();
    }
    mlir::LogicalResult visitOperation(mlir::Operation* op){
        CurrOp wrapped = mlir::dyn_cast<CurrOp>(op);
        if(wrapped){
            return visitOp(wrapped);
        }else{
            return Parent::visitOperation(op);
        }
    }
};
}
}

#endif