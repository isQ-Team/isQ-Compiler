#include "isq/lowering/globalLower.h"
#include "isq/lowering/quantumFunc.h"
#include "isq/IR.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include <iostream>
using namespace std;
using namespace mlir;

static Type convertGlobalMemrefTypeToLLVM(MemRefType type, LLVMTypeConverter &typeConverter) {
  // LLVM type for a global memref will be a multi-dimension array. For
  // declarations or uninitialized global memrefs, we can potentially flatten
  // this to a 1D array. However, for memref.global's with an initial value,
  // we do not intend to flatten the ElementsAttribute when going from std ->
  // LLVM dialect, so the LLVM type needs to me a multi-dimension array.
  Type elementType = type.getElementType();
  //if (elementType.isa<isq::ir::QStateType>())
      //return typeConverter.convertType(type);
  // Shape has the outermost dim at index 0, so need to walk it backwards
  Type arrayTy = typeConverter.convertType(elementType);
  for (int64_t dim : llvm::reverse(type.getShape()))
    arrayTy = LLVM::LLVMArrayType::get(arrayTy, dim);
  return arrayTy;
}

LogicalResult GlobalMemrefOpLowering::matchAndRewrite(memref::GlobalOp global, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
    
    MemRefType type = global.type();
    if (!isConvertibleAndHasIdentityMaps(type)){
        cout << "type is not convertable!" << endl;
        return failure();
    }

    Type arrayTy = convertGlobalMemrefTypeToLLVM(type, *getTypeConverter());

    LLVM::Linkage linkage =
        global.isPublic() ? LLVM::Linkage::External : LLVM::Linkage::Private;

    Attribute initialValue = nullptr;
    
    if (!global.isExternal() && !global.isUninitialized()) {
      auto elementsAttr = global.initial_value()->cast<ElementsAttr>();
      initialValue = elementsAttr;

      // For scalar memrefs, the global variable created is of the element type,
      // so unpack the elements attribute to extract the value.
      if (type.getRank() == 0)
        initialValue = elementsAttr.getSplatValue<Attribute>();
    }    

    auto loc = global.getLoc();
    uint64_t alignment = global.alignment().getValueOr(0);

    auto newGlobal = rewriter.replaceOpWithNewOp<LLVM::GlobalOp>(
        global, arrayTy, global.constant(), linkage, global.sym_name(),
        initialValue, alignment, type.getMemorySpaceAsInt());
    
    if (!global.isExternal() && global.isUninitialized()) {
      Block *blk = new Block();
      newGlobal.getInitializerRegion().push_back(blk);
      rewriter.setInsertionPointToStart(blk);
      

      ModuleOp parentModule = global->getParentOfType<ModuleOp>();
      MLIRContext *context = rewriter.getContext();

      mlir::Value res = rewriter.create<LLVM::UndefOp>(loc, arrayTy);
      rewriter.create<LLVM::ReturnOp>(loc, res);
      
    }
    return success();
  }

LogicalResult GlobalMemrefOpLowering::initAndRelease(memref::GlobalOp global, ConversionPatternRewriter &rewriter) const{
  
  for (auto &op : global->getParentRegion()->front()){
      auto funcOp = dyn_cast_or_null<mlir::FuncOp>(&op);
      if (funcOp){
          if (funcOp.sym_name().equals("main")){
              
              PatternRewriter::InsertionGuard insertGuard(rewriter);

              // init global var at the start of main function
              rewriter.setInsertionPointToStart(&funcOp.body().front());
              
              auto loc = global.getLoc();
              ModuleOp parentModule = global->getParentOfType<ModuleOp>();
              MLIRContext *context = rewriter.getContext();

              MemRefType type = global.type();
              unsigned memSpace = type.getMemorySpaceAsInt();
              Type arrayTy = convertGlobalMemrefTypeToLLVM(type, *getTypeConverter());
              auto addressOf = rewriter.create<LLVM::AddressOfOp>(
                  loc, LLVM::LLVMPointerType::get(arrayTy, memSpace), global.sym_name());

              // Get the address of the first element in the array by creating a GEP with
              // the address of the GV as the base, and (rank + 1) number of 0 indices.
              
              Type elementType = typeConverter->convertType(type.getElementType());
              Type elementPtrType = LLVM::LLVMPointerType::get(elementType, memSpace);

              SmallVector<Value> operands;
              operands.insert(operands.end(), type.getRank() + 1,
                              createIndexConstant(rewriter, loc, 0));
              auto gep = rewriter.create<LLVM::GEPOp>(loc, elementPtrType, addressOf, operands);

              
              // for loop to init
              auto start = rewriter.create<arith::ConstantIndexOp>(loc, 0);
              auto end = rewriter.create<arith::ConstantIndexOp>(loc, memSpace);
              auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
              
              auto initloop = rewriter.create<scf::ForOp>(loc, start, end, step);
              
              for (Operation &nested : *initloop.getBody()){
                  rewriter.eraseOp(&nested);
              }
              rewriter.setInsertionPointToEnd(initloop.getBody());
              
              mlir::Value val;
              if (type.getElementType().isa<isq::ir::QStateType>()){  
                  auto func_name = LLVMQuantumFunc::getOrInsertAllocQubit(rewriter, parentModule);
                  auto call = rewriter.create<CallOp>(loc, func_name, elementType, llvm::None);
                  val = call.getResult(0);
              }else{
                  val = rewriter.create<LLVM::ConstantOp>(loc, 
                            IntegerType::get(rewriter.getContext(), 64),
                            rewriter.getIntegerAttr(rewriter.getI64Type(), 0));
              }
              auto arg = initloop.getInductionVar();
              auto idx = rewriter.create<mlir::arith::IndexCastOp>(loc, IntegerType::get(rewriter.getContext(), 64), arg);
              auto ele_ptr = rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(elementType), 
                                  gep, llvm::ArrayRef<mlir::Value>({idx}));
              rewriter.create<LLVM::StoreOp>(loc, val, ele_ptr);
              
              rewriter.create<scf::YieldOp>(loc);
          }
      }
  }

  return mlir::success();

}