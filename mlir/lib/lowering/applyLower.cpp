#include "isq/lowering/applyLower.h"
#include "isq/lowering/quantumFunc.h"
#include "isq/IR.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <iostream>
using namespace std;
using namespace mlir;


LogicalResult applyGateOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const{
    

    auto loc = op->getLoc();

    MLIRContext *context = rewriter.getContext();

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    auto applyOp = cast<isq::ir::ApplyGateOp>(op);
    auto decorateOp = applyOp.gate().getDefiningOp<isq::ir::DecorateOp>();

    //isq::ir::UseGateOp gateOp;

    if (decorateOp == nullptr){
        cout << "decorate null" << endl;
        return mlir::failure();
        //gateOp = applyOp.gate().getDefiningOp<isq::ir::UseGateOp>();
    }/*
    else{
        gateOp = decorateOp.args().getDefiningOp<isq::ir::UseGateOp>();
    }*/

    auto gateOp = decorateOp.args().getDefiningOp<isq::ir::UseGateOp>();
    
    auto ctrl = decorateOp.ctrl();
    auto inv = decorateOp.adjoint();
    auto gate_name = gateOp.name().getRootReference().getValue().str();
    auto func_name = LLVMQuantumFunc::getGate(rewriter, parentModule, gate_name, ctrl.getValue(), inv);
    rewriter.create<CallOp>(loc, func_name, llvm::None, operands.drop_front());

    for (int i = 1; i < operands.size(); i++){
        auto res = applyOp.getResult(i-1);
        for (auto user : res.getUsers()){
            user->replaceUsesOfWith(res, operands[i]);
        }
    }
    
    rewriter.eraseOp(op);

    return mlir::success();
}