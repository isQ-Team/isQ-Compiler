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

struct AllocOpLowering : public AllocLikeOpLLVMLowering {
  AllocOpLowering(LLVMTypeConverter &converter)
      : AllocLikeOpLLVMLowering(memref::AllocOp::getOperationName(),
                                converter) {}

  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const override;
};