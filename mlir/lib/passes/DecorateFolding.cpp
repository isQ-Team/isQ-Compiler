#include "isq/GateDefTypes.h"
#include "isq/Operations.h"
#include "isq/QAttrs.h"
#include "isq/QStructs.h"
#include "isq/QTypes.h"
#include "isq/passes/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Casting.h"
#include <mlir/IR/Attributes.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/TypeRange.h>
#include <optional>
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace isq{
namespace ir{
namespace passes{
namespace{

std::vector<std::vector<std::complex<double>>> appendMatrix(const std::vector<std::vector<std::complex<double>>>& mat, ::mlir::ArrayRef<bool> ctrl, bool adj){
    auto mat_qubit_num = (int)std::log2(mat.size());
    auto new_mat_size = ((1<<ctrl.size()) * mat.size());
    std::vector<std::vector<std::complex<double>>> new_matrix;
    new_matrix.resize(new_mat_size);
    for(auto i=0; i<new_mat_size; i++){
        new_matrix[i].resize(new_mat_size);
        new_matrix[i][i]=1.0;
    }
    uint64_t mat_mask = 0;
    for(auto i=0; i<ctrl.size(); i++){
        mat_mask = (mat_mask<<1) | (ctrl[i]?1:0);
    }
    mat_mask = mat_mask << mat_qubit_num;
    for(auto i=0; i<(1<<mat_qubit_num); i++){
        for(auto j=0; j<(1<<mat_qubit_num); j++){
            if(adj){
                new_matrix[i|mat_mask][j|mat_mask] = std::conj(mat[j][i]);
            }else{
                new_matrix[i|mat_mask][j|mat_mask] = mat[i][j];
            }
            
        }
    }
    return new_matrix;
}


struct DecorateFoldRewriteRule : public mlir::OpRewritePattern<isq::ir::ApplyGateOp>{
    mlir::ModuleOp rootModule;
    DecorateFoldRewriteRule(mlir::MLIRContext* ctx, mlir::ModuleOp module): mlir::OpRewritePattern<isq::ir::ApplyGateOp>(ctx, 1), rootModule(module){

    }

    mlir::LogicalResult createControlledDefgate(isq::ir::DefgateOp defgate, mlir::ArrayRef<bool> ctrl, bool adj, mlir::FlatSymbolRefAttr sym, mlir::PatternRewriter &rewriter) const{
        auto ctx = rewriter.getContext();
        mlir::SmallVector<mlir::Attribute> usefulGatedefs;
        auto id=0;
        auto new_qubit_num = (int)defgate.type().getSize() + ctrl.size();
        for(auto def: defgate.definition()->getAsRange<GateDefinition>()){
            auto d = AllGateDefs::parseGateDefinition(defgate, id, defgate.type(), def);
            if(d==std::nullopt) return mlir::failure();
            
            if(auto mat = llvm::dyn_cast_or_null<MatrixDefinition>(&**d)){
                auto& old_matrix = mat->getMatrix();

                // construct new matrix.
                
                auto new_matrix = appendMatrix(old_matrix, ctrl, adj);
                mlir::SmallVector<mlir::Attribute> matrix_attr;
                for(auto& row: new_matrix){
                    mlir::SmallVector<mlir::Attribute> row_attr;
                    for(auto column: row){
                        auto c = ComplexF64Attr::get(ctx, ::llvm::APFloat(column.real()), ::llvm::APFloat(column.imag()));
                        row_attr.push_back(c);
                    }
                    matrix_attr.push_back(::mlir::ArrayAttr::get(ctx, row_attr));
                }
                usefulGatedefs.push_back(GateDefinition::get(mlir::StringAttr::get(ctx, "unitary"), mlir::ArrayAttr::get(ctx, matrix_attr), ctx));
            }else if(auto decomp = llvm::dyn_cast_or_null<DecompositionDefinition>(&**d)){
                auto fn = decomp->getDecomposedFunc();
                // TODO: adjoint op support.
                // Do we need to "revert" all steps?
                if(adj) return mlir::failure();
                auto new_fn = fn.clone();
                mlir::ModuleOp rootModule = this->rootModule;
                rewriter.setInsertionPointToStart(rootModule.getBody());
                rewriter.insert(new_fn);
                rewriter.startRootUpdate(new_fn);
                auto new_fn_name = "$__isq__decomposition__"+sym.getValue();
                new_fn.sym_nameAttr(mlir::StringAttr::get(ctx, new_fn_name));
                auto ip = rewriter.saveInsertionPoint();

                rewriter.finalizeRootUpdate(new_fn);
                rewriter.restoreInsertionPoint(ip);
                
            }
            id++;
        }
        if(usefulGatedefs.size()==0){
            return mlir::failure();
        }
        auto ip = rewriter.saveInsertionPoint();
        mlir::ModuleOp rootModule = this->rootModule;
        rewriter.setInsertionPointToStart(rootModule.getBody());
        rewriter.create<DefgateOp>(::mlir::UnknownLoc::get(ctx), mlir::TypeAttr::get(GateType::get(ctx, new_qubit_num, GateTrait::General)), sym.getAttr(), mlir::StringAttr::get(ctx, "nested"), ::mlir::ArrayAttr{}, ::mlir::ArrayAttr::get(ctx, usefulGatedefs), ::mlir::ArrayAttr::get(ctx, ::llvm::ArrayRef<::mlir::Attribute>{}));
        rewriter.restoreInsertionPoint(ip);
        return mlir::success();
    }
    mlir::LogicalResult matchAndRewrite(isq::ir::ApplyGateOp op,  mlir::PatternRewriter &rewriter) const override{
        // Check if it is a use-decorate-apply pattern.
        auto decorate_op = mlir::dyn_cast_or_null<DecorateOp>(op.gate().getDefiningOp());
        if(!decorate_op) return mlir::failure();
        auto use_op = mlir::dyn_cast_or_null<UseGateOp>(decorate_op.args().getDefiningOp());
        if(!use_op) return mlir::failure();
        if(use_op.parameters().size()>0){
            return mlir::failure(); // Only matrix-gates are supported.
        }
        auto defgate = mlir::SymbolTable::lookupNearestSymbolFrom<DefgateOp>(use_op.getOperation(), use_op.name());
        assert(defgate);
        if(!defgate.definition()) return mlir::failure();

        // cx is cnot.
        if(isFamousGate(defgate, "CNOT") && decorate_op.ctrl().size()==1){
            auto ctx = getContext();
            rewriter.replaceOpWithNewOp<UseGateOp>(decorate_op, mlir::TypeRange{GateType::get(ctx, 1, GateTrait::General)}, mlir::FlatSymbolRefAttr::get(ctx, "__isq__builtin__cnot"), mlir::ValueRange{});
            return mlir::success();
        }

        // construct new matrix name.
        // TODO: resolve decorate-and-decorate problem.
        auto new_defgate_name = std::string(defgate.sym_name());
        if(decorate_op.ctrl().size()>0){
            new_defgate_name+="_ctrl_";
            for(auto c: decorate_op.ctrl().getAsValueRange<mlir::BoolAttr>()){
                new_defgate_name+= c?"1":"0";
            }
        }
        if(decorate_op.adjoint()){
            new_defgate_name += "_adj";
        }
        auto new_defgate_sym = mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(rewriter.getContext(), new_defgate_name));
        auto new_defgate = mlir::SymbolTable::lookupNearestSymbolFrom<DefgateOp>(op, new_defgate_sym);
            
        auto ctrl_array = mlir::SmallVector<bool>();
        for(auto attr: decorate_op.ctrl().getAsValueRange<mlir::BoolAttr>()){
            ctrl_array.push_back(attr);
        }
        if(!new_defgate){
            if(mlir::failed(createControlledDefgate(defgate, ctrl_array, decorate_op.adjoint(), new_defgate_sym, rewriter))){
                return mlir::failure();
            }
        }
        new_defgate = mlir::SymbolTable::lookupNearestSymbolFrom<DefgateOp>(op, new_defgate_sym);
        auto ctx = getContext();
        auto new_qubit_num = (int)defgate.type().getSize() + ctrl_array.size();
        auto ip = rewriter.saveInsertionPoint();
        rewriter.setInsertionPoint(op);
        auto new_use_gate = rewriter.create<UseGateOp>(op->getLoc(), GateType::get(ctx, new_qubit_num, GateTrait::General), new_defgate_sym, ::mlir::ValueRange{});
        rewriter.restoreInsertionPoint(ip);
        rewriter.replaceOpWithNewOp<ApplyGateOp>(op.getOperation(), op->getResultTypes(), new_use_gate.result(), op.args());
        return mlir::success();
    }
};



}

struct DecorateFoldingPass : public mlir::PassWrapper<DecorateFoldingPass, mlir::OperationPass<mlir::ModuleOp>>{
    void runOnOperation() override {
        mlir::ModuleOp m = this->getOperation();
        auto ctx = m->getContext();
        mlir::RewritePatternSet rps(ctx);
        rps.add<DecorateFoldRewriteRule>(ctx, m);
        mlir::FrozenRewritePatternSet frps(std::move(rps));
        (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
    }
  mlir::StringRef getArgument() const final {
    return "isq-fold-constant-decorated-gates";
  }
  mlir::StringRef getDescription() const final {
    return  "Constant folding for known and decorated gates.";
  }
};

void registerDecorateFolding(){
    mlir::PassRegistration<DecorateFoldingPass>();
}

}
}
}