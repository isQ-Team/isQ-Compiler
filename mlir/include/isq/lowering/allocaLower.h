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

struct AllocaOpLowering : public AllocLikeOpLLVMLowering {
  AllocaOpLowering(LLVMTypeConverter &converter)
      : AllocLikeOpLLVMLowering(memref::AllocaOp::getOperationName(),
                                converter) {}

  /// Allocates the underlying buffer using the right call. `allocatedBytePtr`
  /// is set to null for stack allocations. `accessAlignment` is set if
  /// alignment is needed post allocation (for eg. in conjunction with malloc).
  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const override;
};