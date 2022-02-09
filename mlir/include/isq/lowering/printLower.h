#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "isq/IR.h"

using namespace mlir;

/// Lowers `toy.print` to a loop nest calling `printf` on each of the individual
/// elements of the array.
class PrintOpLowering : public ConversionPattern {
public:
  explicit PrintOpLowering(MLIRContext *context)
      : ConversionPattern(isq::ir::PrintOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;

private:
  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module);

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module);
};