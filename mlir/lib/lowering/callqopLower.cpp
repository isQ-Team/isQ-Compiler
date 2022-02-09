#include "isq/lowering/callqopLower.h"
#include "isq/lowering/quantumFunc.h"
#include "isq/IR.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <iostream>
using namespace std;
using namespace mlir;


LogicalResult callQOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const{
    

    auto loc = op->getLoc();

    MLIRContext *context = rewriter.getContext();

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    auto callqop = cast<isq::ir::CallQOpOp>(op);
    auto callee = callqop.callee().getNestedReferences().front().getValue();
    
    auto qres = callqop.getResult(0);
    for (auto user : qres.getUsers()) {
        user->replaceUsesOfWith(qres, operands[0]);
    }
    //cout << callee.str() << endl;
    if (callee.equals(StringRef("measure"))){
        auto func_name = LLVMQuantumFunc::getOrInsertMeasure(rewriter, parentModule);
        
        auto call = rewriter.create<CallOp>(loc, func_name, 
                        LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque("Result", context)),
                        operands);
        auto bitcast = rewriter.create<LLVM::BitcastOp>(
            loc, LLVM::LLVMPointerType::get(rewriter.getIntegerType(1)),
            call.getResult(0));
        
        auto o = rewriter.create<LLVM::LoadOp>(loc, rewriter.getIntegerType(1), bitcast.getRes());
        
        auto ires = callqop.getResult(1);
        //cout << ires.getType().isInteger(1) << endl;
        //cout << o.getRes().getType().isInteger(1) << endl;
        
        for (auto user : ires.getUsers()){
            user->replaceUsesOfWith(ires, o.getRes());
        }

    }else if (callee.equals(StringRef("reset"))){
        auto func_name = LLVMQuantumFunc::getOrInsertReset(rewriter, parentModule);
    
        auto call = rewriter.create<CallOp>(loc, func_name, 
                        llvm::None,
                        operands);
    }
    rewriter.eraseOp(op);
    return mlir::success();
}