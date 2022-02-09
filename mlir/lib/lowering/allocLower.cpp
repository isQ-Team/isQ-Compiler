#include "isq/lowering/allocLower.h"
#include "isq/lowering/quantumFunc.h"
#include "isq/IR.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include <iostream>

using namespace std;
using namespace mlir;

std::tuple<Value, Value> AllocOpLowering::allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const {
    // Heap allocations.
    memref::AllocOp allocOp = cast<memref::AllocOp>(op);
    MemRefType memRefType = allocOp.getType();

    Value alignment;
    if (auto alignmentAttr = allocOp.alignment()) {
      alignment = createIndexConstant(rewriter, loc, *alignmentAttr);
    } else if (!memRefType.getElementType().isSignlessIntOrIndexOrFloat()) {
      // In the case where no alignment is specified, we may want to override
      // `malloc's` behavior. `malloc` typically aligns at the size of the
      // biggest scalar on a target HW. For non-scalars, use the natural
      // alignment of the LLVM type given by the LLVM DataLayout.
      alignment = getSizeInBytes(loc, memRefType.getElementType(), rewriter);
    }

    if (alignment) {
      // Adjust the allocation size to consider alignment.
      sizeBytes = rewriter.create<LLVM::AddOp>(loc, sizeBytes, alignment);
    }

    // Allocate the underlying buffer and store a pointer to it in the MemRef
    // descriptor.
    Type elementPtrType = this->getElementPtrType(memRefType);
    auto allocFuncOp = LLVM::lookupOrCreateMallocFn(
        allocOp->getParentOfType<ModuleOp>(), getIndexType());
    auto results = createLLVMCall(rewriter, loc, allocFuncOp, {sizeBytes},
                                  getVoidPtrType());
    Value allocatedPtr =
        rewriter.create<LLVM::BitcastOp>(loc, elementPtrType, results[0]);

    Value alignedPtr = allocatedPtr;
    if (alignment) {
      // Compute the aligned type pointer.
      Value allocatedInt =
          rewriter.create<LLVM::PtrToIntOp>(loc, getIndexType(), allocatedPtr);
      Value alignmentInt =
          createAligned(rewriter, loc, allocatedInt, alignment);
      alignedPtr =
          rewriter.create<LLVM::IntToPtrOp>(loc, elementPtrType, alignmentInt);
    }

    // init: classical var set 0, qubit alloc;

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    MLIRContext *context = rewriter.getContext();

    auto size = allocOp.getType().getShape().front();
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
    if (allocOp.getType().getElementType().isa<isq::ir::QStateType>()){  
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
                        alignedPtr, llvm::ArrayRef<mlir::Value>({idx}));
    rewriter.create<LLVM::StoreOp>(loc, val, ele_ptr);

    rewriter.create<scf::YieldOp>(loc);


    return std::make_tuple(allocatedPtr, alignedPtr);
  }