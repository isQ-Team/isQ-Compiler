#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "isq/IR.h"
#include <set>

using namespace mlir;

class defgateOpLowering : public ConversionPattern {
public:
  explicit defgateOpLowering(MLIRContext *context)
      : ConversionPattern(isq::ir::DefgateOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;

};