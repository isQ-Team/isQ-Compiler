#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "isq/IR.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/AllocLikeConversion.h"

using namespace mlir;

struct DeallocOpLowering : public ConvertOpToLLVMPattern<memref::DeallocOp> {
  using ConvertOpToLLVMPattern<memref::DeallocOp>::ConvertOpToLLVMPattern;

  explicit DeallocOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<memref::DeallocOp>(converter) {}

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};