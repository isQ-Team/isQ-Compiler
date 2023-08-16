#include <iostream>
#include <vector>

#include "isq/Dialect.h"
#include "isq/Lower.h"
#include "isq/Operations.h"
#include "isq/QTypes.h"
#include "isq/GateDefTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/IR/IRMapping.h>

namespace isq {
namespace ir {
namespace passes {

const void processRegion(SwitchOp op, mlir::PatternRewriter &rewriter, mlir::Region &region, int nqubit, int case_num) {
    mlir::Value arg = op.getArg();
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    auto range = region.getOps();
    for (mlir::Region::OpIterator it = range.begin(); it != range.end(); ) {
        if (ApplyGateOp apply_op = llvm::dyn_cast_or_null<ApplyGateOp>(*it)) {
            // Prepare the control array
            mlir::SmallVector<mlir::Attribute> ctrl;
            for (int k=nqubit-1; k>=0; k--) {
                ctrl.push_back(mlir::BoolAttr::get(ctx, (case_num >> k) & 1));
            }
            GateTrait trait = DecorateOp::computePostDecorateTrait(GateTrait::General, nqubit, false, case_num == (1 << nqubit) - 1);

            // Create a new gate that has additional control bits
            auto origin_gate = apply_op.getGate();
            auto origin_gate_size = origin_gate.getType().getSize();
            int new_gate_size = origin_gate_size + nqubit;
            auto new_type = GateType::get(ctx, new_gate_size, trait);
            auto new_gate = rewriter.create<DecorateOp>(loc, new_type, origin_gate, false, mlir::ArrayAttr::get(ctx, ctrl));

            // Apply the new gate
            mlir::SmallVector<mlir::Type> qtype(new_gate_size, QStateType::get(ctx));
            mlir::SmallVector<mlir::Value> states;
            for (int k=nqubit-1; k>=0; k--) {
                mlir::Value idx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, k);
                auto loaded = rewriter.create<mlir::AffineLoadOp>(loc, arg, mlir::ArrayRef<mlir::Value>({idx}));
                states.push_back(loaded);
            }
            for (mlir::Value state : apply_op.getArgs()) {
                states.push_back(state);
            }
            auto new_apply = rewriter.create<isq::ir::ApplyGateOp>(loc, qtype, new_gate.getResult(), states);
            for (int k=nqubit-1; k>=0; k--) {
                mlir::Value idx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, k);
                rewriter.create<mlir::AffineStoreOp>(loc, new_apply.getResult(nqubit - 1 - k), arg, mlir::ArrayRef<mlir::Value>({idx}));
            }
            for (int j=0; j<origin_gate_size; j++) {
                apply_op.getResult(j).replaceAllUsesWith(new_apply.getResult(j + nqubit));
            }

            it++;
            rewriter.eraseOp(apply_op);
        } else if (YieldOp yield_op = llvm::dyn_cast_or_null<YieldOp>(*it)) {
            it++;
        } else {
            auto old = it++;
            old->moveBefore(op);
        }
    }
}

class LowerSwitchOp: public mlir::OpRewritePattern<SwitchOp> {
public:
    LowerSwitchOp(mlir::MLIRContext* ctx): mlir::OpRewritePattern<SwitchOp>(ctx, 1) {}

    mlir::LogicalResult matchAndRewrite(isq::ir::SwitchOp op, mlir::PatternRewriter &rewriter) const override {
        mlir::Value arg = op.getArg();
        mlir::Type type = arg.getType();
        auto mem_type = type.dyn_cast_or_null<mlir::MemRefType>();
        int nqubit = mem_type.getDimSize(0);
        int dim = 1 << nqubit;
        std::vector<bool> shown(dim, false);

        // Process the cases
        mlir::ArrayAttr case_attr = op.getCases();
        int i = 0;
        for (auto attr : case_attr) {
            auto int_attr = attr.dyn_cast_or_null<mlir::IntegerAttr>();
            int case_num = int_attr.getInt();
            shown[case_num] = true;
            processRegion(op, rewriter, op.getCaseRegions()[i], nqubit, case_num);
            i++;
        }

        // Process default
        for (int i=0; i<dim; i++) {
            if (!shown[i]) {
                mlir::Region region;
                mlir::IRMapping map;
                op.getDefaultRegion().cloneInto(&region, map);
                processRegion(op, rewriter, region, nqubit, i);
            }
        }

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

/*
* Modified based on https://github.com/llvm/llvm-project/blob/6f3f600b2aa10df3b9de580e9fd9457b93e3d087/mlir/lib/Conversion/SCFToControlFlow/SCFToControlFlow.cpp#L618-L673
* Hope we can delete this code piece after MLIR fixing the bug!
*/
using namespace mlir;
struct IndexSwitchLowering : public OpRewritePattern<mlir::scf::IndexSwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::scf::IndexSwitchOp op,
                                PatternRewriter &rewriter) const override {
    // Split the block at the op.
  mlir::Block *condBlock = rewriter.getInsertionBlock();
  Block *continueBlock = rewriter.splitBlock(condBlock, Block::iterator(op));

  // Create the arguments on the continue block with which to replace the
  // results of the op.
  SmallVector<Value> results;
  results.reserve(op.getNumResults());
  for (Type resultType : op.getResultTypes())
    results.push_back(continueBlock->addArgument(resultType, op.getLoc()));

  // Handle the regions.
  auto convertRegion = [&](Region &region) -> FailureOr<Block *> {
    Block *block = &region.front();

    // Convert the yield terminator to a branch to the continue block.
    auto yield = cast<scf::YieldOp>(block->getTerminator());
    rewriter.setInsertionPoint(yield);
    rewriter.replaceOpWithNewOp<cf::BranchOp>(yield, continueBlock,
                                              yield.getOperands());

    // Inline the region.
    rewriter.inlineRegionBefore(region, continueBlock);
    return block;
  };

  // Convert the case regions.
  SmallVector<Block *> caseSuccessors;
  SmallVector<int32_t> caseValues;
  caseSuccessors.reserve(op.getCases().size());
  caseValues.reserve(op.getCases().size());
  for (auto [region, value] : llvm::zip(op.getCaseRegions(), op.getCases())) {
    FailureOr<Block *> block = convertRegion(region);
    if (failed(block))
      return failure();
    caseSuccessors.push_back(*block);
    caseValues.push_back(value);
  }

  // Convert the default region.
  FailureOr<Block *> defaultBlock = convertRegion(op.getDefaultRegion());
  if (failed(defaultBlock))
    return failure();

  // Create the switch.
  rewriter.setInsertionPointToEnd(condBlock);

  // Note: extra type conversion not contained in the original MLIR code!
  auto int_arg = rewriter.create<arith::IndexCastOp>(op.getLoc(), rewriter.getI32Type(), op.getArg());

  SmallVector<ValueRange> caseOperands(caseSuccessors.size(), {});
  rewriter.create<cf::SwitchOp>(
      op.getLoc(), int_arg, *defaultBlock, ValueRange(),
      rewriter.getDenseI32ArrayAttr(caseValues), caseSuccessors, caseOperands);
  rewriter.replaceOp(op, continueBlock->getArguments());
  return success();
}
};

struct LowerSwitchOpPass: public mlir::PassWrapper<LowerSwitchOpPass, mlir::OperationPass<mlir::ModuleOp>>{

    void runOnOperation() override{
        mlir::ModuleOp m = this->getOperation();
        auto ctx = m->getContext();
        
        mlir::RewritePatternSet rps(ctx);
        rps.add<LowerSwitchOp>(ctx);
        rps.add<IndexSwitchLowering>(ctx);
        mlir::FrozenRewritePatternSet frps(std::move(rps));
        (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
    }

    mlir::StringRef getArgument() const final{
        return "isq-lower-switch";
    }

    mlir::StringRef getDescription() const final{
        return "Lower the switch operation.";
    }
};

void registerLowerSwitchOp(){
    mlir::PassRegistration<LowerSwitchOpPass>();
}

}

}
}
