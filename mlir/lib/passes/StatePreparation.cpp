#include <iostream>
#include <vector>

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

namespace isq{
namespace ir{
namespace passes{

class StatePreparation: public mlir::OpRewritePattern<InitOp>{
    mlir::ModuleOp rootModule;
    inline static double EPS = 1e-10; // boundary adopting from Qiskit

public:
    StatePreparation(mlir::MLIRContext* ctx, mlir::ModuleOp module): mlir::OpRewritePattern<InitOp>(ctx, 1), rootModule(module){
    }

    static double arccos(double v) {
        if (1 < v && v < 1 + EPS) {
            return 0;
        }
        return acos(v);
    }

    /*
    * Quantum state preparation based on paper:
    *    Shende, V.V., S.S. Bullock, and I.L. Markov. “Synthesis of Quantum-Logic Circuits.” IEEE TCAD, 2006.
    */
    mlir::LogicalResult matchAndRewrite(isq::ir::InitOp op,  mlir::PatternRewriter &rewriter) const override {
        mlir::Value qubits = op.getQubits();
        auto mem_type = qubits.getType().dyn_cast<mlir::MemRefType>();
        assert(mem_type && "Qubits are not of MemRefType");
        int nqubits = mem_type.getDimSize(0);

        // Reset all the qubits
        mlir::Location loc = op.getLoc();
        mlir::MLIRContext *ctx = rewriter.getContext();
        for (int i=0; i<nqubits; i++) {
            mlir::Value idx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
            auto loaded = rewriter.create<mlir::AffineLoadOp>(loc, qubits, mlir::ArrayRef<mlir::Value>({idx}));
            auto resetted = rewriter.create<CallQOpOp>(loc, mlir::TypeRange{QStateType::get(ctx)}, 
                mlir::FlatSymbolRefAttr::get(rewriter.getStringAttr("__isq__builtin__reset")), mlir::ValueRange{loaded.getResult()},
                1, mlir::TypeAttr::get(rewriter.getFunctionType({}, {})));
            rewriter.create<mlir::AffineStoreOp>(loc, resetted.getResult(0), qubits, mlir::ValueRange{idx});
        }

        // Rescale user-input amplitude
        ::isq::ir::DenseComplexF64MatrixAttr state = op.getState();
        llvm::SmallVector<Eigen::dcomplex> val = state.toMatrixVal()[0];
        int64_t state_len = val.size();
        double norm_sum = 0;
        for (int i=0; i<state_len; i++) {
            norm_sum += norm(val[i]);
        }
        double scale = sqrt(norm_sum);
        std::vector<Eigen::dcomplex> amplitude(1 << nqubits, 0);
        for (int i=0; i<state_len; i++) {
            amplitude[i] = val[i] / scale;
        }

        // Disentangle qubit i and change it to |0>
        for (int i=0; i<nqubits; i++) {
            int nctrl = nqubits - i - 1;
            int nctrl_pow = 1 << nctrl;
            for (int j=0; j<nctrl_pow; j++) {
                Eigen::dcomplex zero = amplitude[2 * j];
                Eigen::dcomplex one = amplitude[2 * j + 1];
                double r = sqrt(norm(zero) + norm(one));

                // Avoid applying Rx/Rz gates if the norm is small
                if (r < EPS) {
                    amplitude[j] = 0;
                    continue;
                }

                // Calculate the angles on the Bloch sphere
                // |\psi> = re^{jt/2}[e^{-j\phi/2}cos(\theta/2)|0>+e^{j\phi/2}cos(\theta/2)|1>]
                // So that Ry(-\theta)Rz(-\phi)|\psi>=re^{jt/2}|0>
                double arg0 = arg(zero);
                double arg1 = arg(one);
                amplitude[j] = r * exp(Eigen::dcomplex(0, (arg0 + arg1) / 2));
                double theta = 2 * arccos(abs(zero) / r);
                double phi = arg1 - arg0;
                if (abs(theta) < EPS && abs(phi) < EPS) {
                    continue;
                }

                // Use the binrary representation of j as the control Boolean values
                mlir::SmallVector<mlir::Attribute> ctrl;
                for (int k=nctrl-1; k>=0; k--) {
                    ctrl.push_back(mlir::BoolAttr::get(ctx, (j >> k) & 1));
                }
                mlir::SmallVector<mlir::Type> qtype(nctrl + 1, QStateType::get(ctx));

                // Apply controlled-Ry(theta)
                auto value_theta = rewriter.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(theta), mlir::Float64Type::get(ctx));
                mlir::Value ry = rewriter.create<isq::ir::UseGateOp>(loc, isq::ir::GateType::get(ctx, 1, GateTrait::General),
                    mlir::FlatSymbolRefAttr::get(ctx, "Ry"), mlir::ValueRange{value_theta}).getResult();
                auto ry_type = GateType::get(ctx, nqubits - i, DecorateOp::computePostDecorateTrait(GateTrait::General, nctrl, false, j == nctrl_pow - 1));
                auto decorated_ry = rewriter.create<DecorateOp>(loc, ry_type, ry, false, mlir::ArrayAttr::get(ctx, ctrl));
                mlir::SmallVector<mlir::Value> states;
                for (int k=nqubits-1; k>=i; k--) {
                    mlir::Value idx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, k);
                    auto loaded = rewriter.create<mlir::AffineLoadOp>(loc, qubits, mlir::ArrayRef<mlir::Value>({idx}));
                    states.push_back(loaded);
                }
                auto applied_ry = rewriter.create<isq::ir::ApplyGateOp>(loc, qtype, decorated_ry.getResult(), states);

                // Apply controlled-Rz(phi)
                mlir::Value value_phi = rewriter.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(phi), mlir::Float64Type::get(ctx));
                mlir::Value rz = rewriter.create<isq::ir::UseGateOp>(loc, isq::ir::GateType::get(ctx, 1, GateTrait::General),
                    mlir::FlatSymbolRefAttr::get(ctx, "Rz"), mlir::ValueRange{value_phi}).getResult();
                auto rz_type = GateType::get(ctx, nqubits - i, DecorateOp::computePostDecorateTrait(GateTrait::General, nctrl, false, j == nctrl_pow - 1));
                auto decorated_rz = rewriter.create<DecorateOp>(loc, rz_type, rz, false, mlir::ArrayAttr::get(ctx, ctrl));
                auto applied_rz = rewriter.create<isq::ir::ApplyGateOp>(loc, qtype, decorated_rz.getResult(), applied_ry.getResults());
                for (int k=nqubits-1; k>=i; k--) {
                    mlir::Value idx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, k);
                    rewriter.create<mlir::AffineStoreOp>(loc, applied_rz.getResult(nqubits - 1 - k), qubits, mlir::ArrayRef<mlir::Value>({idx}));
                }

                // Inverse the circuit as inserting new gates
                rewriter.setInsertionPoint(value_theta);
            }
        }

        rewriter.eraseOp(op);
        return mlir::success();
    }
};


struct StatePreparationPass: public mlir::PassWrapper<StatePreparationPass, mlir::OperationPass<mlir::ModuleOp>>{

    void runOnOperation() override{
        mlir::ModuleOp m = this->getOperation();
        auto ctx = m->getContext();
        
        mlir::RewritePatternSet rps(ctx);
        rps.add<StatePreparation>(ctx, m);
        mlir::FrozenRewritePatternSet frps(std::move(rps));
        (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
    }

    mlir::StringRef getArgument() const final{
        return "isq-state-preparation";
    }

    mlir::StringRef getDescription() const final{
        return "Prepare qubits in the specified state.";
    }
};

void registerStatePreparation(){
    mlir::PassRegistration<StatePreparationPass>();
}

}

}
}
