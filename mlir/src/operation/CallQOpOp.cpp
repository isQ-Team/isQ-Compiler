#include <isq/Operations.h>
#include <mlir/IR/PatternMatch.h>
#include <isq/passes/EliminateHermitianPairs.h>
#include <mlir/Support/LogicalResult.h>
namespace isq {
namespace ir {
::mlir::LogicalResult
CallQOpOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
    auto symbol_def =
        symbolTable.lookupNearestSymbolFrom(*this, this->callee());
    if (auto qop = llvm::dyn_cast_or_null<DeclareQOpOp>(symbol_def)) {

        auto fn =
            mlir::FunctionType::get(this->getContext(), this->args().getTypes(),
                                    this->getResults().getTypes());
        if (fn == qop.getTypeWhenUsed()) {
            return mlir::success();
        } else {
            this->emitOpError()
                << "type mismatch, expected " << qop.getTypeWhenUsed();
            return mlir::failure();
        }
    }
    this->emitOpError() << "symbol `" << this->callee()
                        << "` not found or has wrong type";
    return mlir::failure();
}

/*
void ApplyOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                      mlir::MLIRContext *context) {
results.add<EliminateHermitianPairs>(context);
}
*/
} // namespace ir
} // namespace isq