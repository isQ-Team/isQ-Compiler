#include <isq/Operations.h>
#include <mlir/IR/PatternMatch.h>
#include <isq/passes/EliminateHermitianPairs.h>
namespace isq {
namespace ir {
mlir::SmallVector<mlir::Type> getGateParameterType(GateType ty) {
    mlir::SmallVector<mlir::Type> args;
    for (auto i = 0; i < ty.getSize(); i++) {
        args.push_back(QStateType::get(ty.getContext()));
    }
    return args;
}
/*
void ApplyOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                      mlir::MLIRContext *context) {
results.add<EliminateHermitianPairs>(context);
}
*/
} // namespace ir
} // namespace isq