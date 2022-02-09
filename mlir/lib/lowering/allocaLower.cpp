#include "isq/lowering/allocaLower.h"
#include "isq/lowering/quantumFunc.h"
#include "isq/IR.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include <iostream>

using namespace std;
using namespace mlir;

std::tuple<Value, Value> AllocaOpLowering::allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const {

    // With alloca, one gets a pointer to the element type right away.
    // For stack allocations.

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    MLIRContext *context = rewriter.getContext();

    auto allocaOp = cast<memref::AllocaOp>(op);

    auto elementPtrType = this->getElementPtrType(allocaOp.getType());

    auto allocatedElementPtr = rewriter.create<LLVM::AllocaOp>(
        loc, elementPtrType, sizeBytes,
        allocaOp.alignment() ? *allocaOp.alignment() : 0);

    // init: classical var set 0, qubit alloc;
    auto size = allocaOp.getType().getShape().front();
    auto ele_type = LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque("Qubit", context));
    
    
    auto start = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto end = rewriter.create<arith::ConstantIndexOp>(loc, size);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    
    
    auto allocloop = rewriter.create<scf::ForOp>(loc, start, end, step);
    PatternRewriter::InsertionGuard insertGuard(rewriter);

    for (Operation &nested : *allocloop.getBody())
        rewriter.eraseOp(&nested);

    
    rewriter.setInsertionPointToEnd(allocloop.getBody());
    
    mlir::Value val;
    if (allocaOp.getType().getElementType().isa<isq::ir::QStateType>()){  
        auto func_name = LLVMQuantumFunc::getOrInsertAllocQubit(rewriter, parentModule);
        auto call = rewriter.create<CallOp>(loc, func_name, ele_type, llvm::None);
        val = call.getResult(0);
    }else{
        val = rewriter.create<LLVM::ConstantOp>(loc, 
                  IntegerType::get(rewriter.getContext(), 64),
                  rewriter.getIntegerAttr(rewriter.getI64Type(), 0));
    }
    auto arg = allocloop.getInductionVar();
    auto idx = rewriter.create<mlir::arith::IndexCastOp>(loc, IntegerType::get(rewriter.getContext(), 64), arg);
    auto ele_ptr = rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(ele_type), 
                        allocatedElementPtr, llvm::ArrayRef<mlir::Value>({idx}));
    rewriter.create<LLVM::StoreOp>(loc, val, ele_ptr);

    rewriter.create<scf::YieldOp>(loc);

    /*
    cout << "parent op: " << op->getParentRegion()->back().getParentOp()->getName().getStringRef().str() << endl;
    
    
    cout << "-------------" << endl;
        for (auto &sop : op->getParentRegion()->back().getOperations()){
            cout << sop.getName().getStringRef().str() << endl;
        }
    cout << "-------------" << endl;
    */
      
    // qubit release
    
    if (allocaOp.getType().getElementType().isa<isq::ir::QStateType>()){
        
        auto lastOpIter = op->getParentRegion()->back().end();
        lastOpIter--;
        //cout << (*lastOpIter).getName().getStringRef().str() << endl;
        rewriter.setInsertionPoint(&(*lastOpIter));
        
        auto releaseloop = rewriter.create<scf::ForOp>(loc, start, end, step);
        
        for (Operation &nested : *releaseloop.getBody())
            rewriter.eraseOp(&nested);
        
        rewriter.setInsertionPointToEnd(releaseloop.getBody());

        auto arg = releaseloop.getInductionVar();
        auto idx = rewriter.create<mlir::arith::IndexCastOp>(loc, IntegerType::get(rewriter.getContext(), 64), arg);
    
        auto ele_ptr = rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(ele_type), 
                          allocatedElementPtr, llvm::ArrayRef<mlir::Value>({idx}));
        
        auto val = rewriter.create<LLVM::LoadOp>(loc, ele_type, ele_ptr);

        auto func_name = LLVMQuantumFunc::getOrInsertReleaseQubit(rewriter, parentModule);
        rewriter.create<CallOp>(loc, func_name, llvm::None, val.getResult());
        
        rewriter.create<scf::YieldOp>(loc);

    }

    return std::make_tuple(allocatedElementPtr, allocatedElementPtr);
  }