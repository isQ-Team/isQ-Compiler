#include <isq/Operations.h>
#include <mlir/IR/PatternMatch.h>
#include <isq/passes/canonicalization/CanonicalizeApplyGate.h>
#include <mlir/Support/LogicalResult.h>
namespace isq {
namespace ir {
mlir::SmallVector<mlir::Type> getGateParameterType(GateType ty) {
    mlir::SmallVector<mlir::Type> args;
    for (auto i = 0; i < ty.getSize(); i++) {
        args.push_back(QStateType::get(ty.getContext()));
    }
    return args;
}
bool isFromApplyGate(mlir::Value val) {
    return !!::mlir::dyn_cast_or_null<ApplyGateOp>(val.getDefiningOp());
}
bool isFromCallQOp(mlir::Value val) {
    return !!::mlir::dyn_cast_or_null<CallQOpOp>(val.getDefiningOp());
}

mlir::LogicalResult ApplyGateOp::verifyIR() {
    for (auto i = 0; i < this->getNumResults(); i++) {
        mlir::Value result = this->getResult(i);
        if (!(result.hasOneUse() || result.getUses().begin()==result.getUses().end() )) {
            this->emitError() << "Result #" << i
                              << " is used more than once.";
            return mlir::failure();
        }
    }
    for (auto i = 0; i < this->args().size(); i++) {
        mlir::Value arg = this->args()[i];
        if (!(arg.hasOneUse() || arg.getDefiningOp<ContribUndef>())) {
            this->emitError()
                << "Argument #" << i << " is used more than once and is not undef.";
            return mlir::failure();
        }
    }
    return mlir::success();
}

void ApplyGateOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                       mlir::MLIRContext *context) {
    patterns.add<passes::canonicalize::NoDowngradeApply>(context);
    patterns.add<passes::canonicalize::CorrectSymmetryApplyOrder>(context);
    patterns.add<passes::canonicalize::CancelHermitianUU>(context);
    patterns.add<passes::canonicalize::CancelUUAdj>(context);
}

/*
void ApplyOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                      mlir::MLIRContext *context) {
results.add<EliminateHermitianPairs>(context);
}
*/
} // namespace ir
} // namespace isq