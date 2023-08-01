#include "isq/Dialect.h"
#include "isq/Lower.h"
#include "isq/Operations.h"
#include "isq/QTypes.h"
#include "isq/GateDefTypes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <optional>

namespace isq{
namespace ir{
namespace passes{

namespace remove_reset{

class RuleRemoveReset : public mlir::OpRewritePattern<CallQOpOp>{
    mlir::ModuleOp rootModule;
public:
    RuleRemoveReset(mlir::MLIRContext* ctx, mlir::ModuleOp module): mlir::OpRewritePattern<CallQOpOp>(ctx, 1), rootModule(module){}

    mlir::LogicalResult matchAndRewrite(CallQOpOp op, mlir::PatternRewriter &rewriter) const override{
        lower::QIRExternQuantumFunc utils;
        auto rootModule = this->rootModule;
        auto qop = rootModule.lookupSymbol<DeclareQOpOp>(op.getCallee());
        assert(qop);
        // First, we unwire next ops.
        #define UNWIRE \
        for(auto i=0; i<op.getSize(); i++){ \
            auto output = op->getResult(i); \
            auto input = op->getOperand(i); \
            output.replaceAllUsesWith(input); \
        }
        auto loc = op->getLoc();
        if(qop.getSymName() == "__isq__builtin__reset"){
            UNWIRE;
            rewriter.eraseOp(op);
            return mlir::success();
        }
        return mlir::failure();
    }
};

struct RemoveResetPass : public mlir::PassWrapper<RemoveResetPass, mlir::OperationPass<mlir::ModuleOp>>{
    void runOnOperation() override {
        mlir::ModuleOp m = this->getOperation();
        auto ctx = m->getContext();
        
        do {
            mlir::RewritePatternSet rps(ctx);
            rps.add<RuleRemoveReset>(ctx, m);
            mlir::FrozenRewritePatternSet frps(std::move(rps));
            (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
        } while(0);
    }
  mlir::StringRef getArgument() const final {
    return "isq-remove-reset";
  }
  mlir::StringRef getDescription() const final {
    return  "Remove reset operation to support QCIS etc.";
  }
};
}

void registerRemoveReset(){
    using namespace remove_reset;
    mlir::PassRegistration<RemoveResetPass>();
}

}
}
}