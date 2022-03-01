#include <isq/Operations.h>
#include <mlir/IR/PatternMatch.h>
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
        if (!result.hasOneUse()) {
            this->emitError() << "Result #" << i
                              << " is used more than once or not used at all.";
            return mlir::failure();
        }
    }
    for (auto i = 0; i < this->args().size(); i++) {
        mlir::Value arg = this->args()[i];
        if (!arg.hasOneUse()) {
            this->emitError()
                << "Argument #" << i << " is used more than once.";
            return mlir::failure();
        }
    }
    return mlir::success();
}
/*
void ApplyOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                      mlir::MLIRContext *context) {
results.add<EliminateHermitianPairs>(context);
}
*/
} // namespace ir
} // namespace isq