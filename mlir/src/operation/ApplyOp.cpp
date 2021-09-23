#include <isq/Operations.h>
#include <mlir/IR/PatternMatch.h>
#include <isq/passes/EliminateHermitianPairs.h>
namespace isq {
namespace ir {
void ApplyOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                          mlir::MLIRContext *context) {
    results.add<EliminateHermitianPairs>(context);
}
} // namespace ir
} // namespace isq