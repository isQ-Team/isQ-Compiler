#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "isq/IR.h"

using namespace mlir;


class useGateOpLowering : public ConversionPattern {
public:
  explicit useGateOpLowering(MLIRContext *context)
      : ConversionPattern(isq::ir::UseGateOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;

};