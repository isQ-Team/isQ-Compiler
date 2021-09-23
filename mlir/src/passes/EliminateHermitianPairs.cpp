#include <isq/passes/EliminateHermitianPairs.h>
namespace isq {
namespace ir {
mlir::LogicalResult EliminateHermitianPairs::matchAndRewrite(
    ApplyOp applyop, mlir::PatternRewriter &rewriter) const {
  if (applyop.args().size() == 0) {
    return mlir::failure();
  }
  ApplyOp apply_2 = applyop.args().front().getDefiningOp<ApplyOp>();
  if (!apply_2)
    return mlir::failure();
  UseOp use_1 = applyop.op().getDefiningOp<UseOp>();
  if (!use_1)
    return mlir::failure();
  GateOp gate_1 = use_1.getOperand().getDefiningOp<GateOp>();
  if (!gate_1)
    return mlir::failure();

  auto info = std::get<1>(gate_1.gate_type().getGateInfo());
  if (std::find(info.begin(), info.end(), GateTypeHint::Hermitian) ==
      info.end()) {
    return mlir::failure();
  }
  auto apply_1_args = mlir::SmallVector<mlir::Value>(applyop.args());
  auto apply_2_results = mlir::SmallVector<mlir::Value>(apply_2.getResults());
  if (std::find(info.begin(), info.end(), GateTypeHint::Symmetric) !=
      info.end()) {
    std::sort(
        apply_1_args.begin(), apply_1_args.end(),
        [](mlir::Value a, mlir::Value b) { return a.getImpl() < b.getImpl(); });
    std::sort(
        apply_2_results.begin(), apply_2_results.end(),
        [](mlir::Value a, mlir::Value b) { return a.getImpl() < b.getImpl(); });
  }
  if (apply_1_args != apply_2_results) {

    return mlir::failure(); // on different qubits.
  }
  UseOp use_2 = apply_2.op().getDefiningOp<UseOp>();
  if (!use_2)
    return mlir::failure();
  if (use_2 != use_1) {
    return mlir::failure(); // not the same gate use.
  }
  // rewrite
  rewriter.replaceOp(applyop, apply_2.args());
  rewriter.eraseOp(apply_2);
  return mlir::success();

  /*
  def EliminateHermitianPairPattern: Pat<
  (ISQ_ApplyOp
  (ISQ_UseOp (ISQ_GateOp $name1, $attr1)),
  (ISQ_ApplyOp (ISQ_UseOp (ISQ_GateOp $name2, $attr2)), $qubits)),
  (replaceWithValue $qubits)>;
  */
}
} // namespace ir
} // namespace isq