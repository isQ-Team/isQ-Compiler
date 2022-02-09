#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "isq/IR.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

using namespace mlir;
struct GlobalMemrefOpLowering
    : public ConvertOpToLLVMPattern<memref::GlobalOp> {
  using ConvertOpToLLVMPattern<memref::GlobalOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::GlobalOp global, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
  
  LogicalResult initAndRelease(memref::GlobalOp global, ConversionPatternRewriter &rewriter) const;
};