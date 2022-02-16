#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "isq/IR.h"

using namespace mlir;


class StoreOpLowering : public ConversionPattern {
public:
  explicit StoreOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::AffineStoreOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;

};