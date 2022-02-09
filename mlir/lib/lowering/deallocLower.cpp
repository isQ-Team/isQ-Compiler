#include "isq/lowering/deallocLower.h"
#include "isq/lowering/quantumFunc.h"
#include "isq/IR.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include <iostream>

using namespace std;
using namespace mlir;

LogicalResult DeallocOpLowering::matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
    // Insert the `free` declaration if it is not already present.
    auto freeFunc = LLVM::lookupOrCreateFreeFn(op->getParentOfType<ModuleOp>());
    MemRefDescriptor memref(adaptor.memref());
    
    auto loc = op.getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    MLIRContext *context = rewriter.getContext();

    auto memType = op.memref().getType().cast<mlir::MemRefType>();

    if (memType.getElementType().isa<isq::ir::QStateType>()){
      
      
      Value alignptr = memref.alignedPtr(rewriter, loc);

      auto size = memType.getShape().front();
      auto ele_type = LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque("Qubit", context));
      
      auto start = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto end = rewriter.create<arith::ConstantIndexOp>(loc, size);
      auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      
      auto releaseloop = rewriter.create<scf::ForOp>(loc, start, end, step);
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      
      for (Operation &nested : *releaseloop.getBody())
          rewriter.eraseOp(&nested);

      rewriter.setInsertionPointToEnd(releaseloop.getBody());

      auto arg = releaseloop.getInductionVar();
      auto idx = rewriter.create<mlir::arith::IndexCastOp>(loc, IntegerType::get(rewriter.getContext(), 64), arg);
    
      auto ele_ptr = rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(ele_type), 
                          alignptr, llvm::ArrayRef<mlir::Value>({idx}));
        
      auto val = rewriter.create<LLVM::LoadOp>(loc, ele_type, ele_ptr);

      auto func_name = LLVMQuantumFunc::getOrInsertReleaseQubit(rewriter, parentModule);
      rewriter.create<CallOp>(loc, func_name, llvm::None, val.getResult());
        
      rewriter.create<scf::YieldOp>(loc);
    }
    Value casted = rewriter.create<LLVM::BitcastOp>(
        loc, getVoidPtrType(),
        memref.allocatedPtr(rewriter, loc));

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange(), SymbolRefAttr::get(freeFunc), casted);
    return success();
  }