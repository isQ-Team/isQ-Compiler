#include "isq/GateDefTypes.h"
#include "isq/Operations.h"
#include "isq/QAttrs.h"
#include "isq/QStructs.h"
#include "isq/QTypes.h"
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


struct DecorateFoldRewriteRule : mlir::OpRewritePattern<isq::ir::ApplyGateOp>{
    mlir::ModuleOp rootModule;
    DecorateFoldRewriteRule(mlir::MLIRContext* ctx, mlir::ModuleOp module): mlir::OpRewritePattern<isq::ir::ApplyGateOp>(ctx, 1), rootModule(module){

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
        int id = 0;
        for(auto def: defgate.definition()->getAsRange<GateDefinition>()){
            auto d = AllGateDefs::parseGateDefinition(defgate, id, defgate.type(), def);
            if(d==std::nullopt) return mlir::failure();
            auto mat = llvm::dyn_cast_or_null<MatrixDefinition>(&**d);
            if(!mat){
                id++;
                continue;
            }
            // construct new matrix name.
            auto new_matrix_name = std::string(defgate.sym_name());
            if(decorate_op.ctrl().size()>0){
                new_matrix_name+="_ctrl_";
                for(auto c: decorate_op.ctrl().getAsValueRange<mlir::BoolAttr>()){
                    new_matrix_name+= c?"1":"0";
                }
            }
            if(decorate_op.adjoint()){
                new_matrix_name += "_adj";
            }
            auto new_defgate_sym = mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(rewriter.getContext(), new_matrix_name));
            auto new_defgate = mlir::SymbolTable::lookupNearestSymbolFrom<DefgateOp>(op, new_defgate_sym);
            auto& old_matrix = mat->getMatrix();
            auto mat_qubit_num = (int)std::log2(old_matrix.size()) + decorate_op.ctrl().size();
            auto ctx = rewriter.getContext();
            if(!new_defgate){
                // construct new matrix.
                ::mlir::SmallVector<bool> ctrl;
                for(auto b: decorate_op.ctrl().getAsValueRange<mlir::BoolAttr>()){
                    ctrl.push_back(b);
                }
                
                auto new_matrix = appendMatrix(old_matrix, ctrl, decorate_op.adjoint());
                auto ip = rewriter.saveInsertionPoint();
                mlir::ModuleOp rootModule = this->rootModule;
                rewriter.setInsertionPointToStart(rootModule.getBody());
                
                
                mlir::SmallVector<mlir::Attribute> new_mat_defs;
                mlir::SmallVector<mlir::Attribute> matrix_attr;
                for(auto& row: new_matrix){
                    mlir::SmallVector<mlir::Attribute> row_attr;
                    for(auto column: row){
                        auto c = ComplexF64Attr::get(ctx, ::llvm::APFloat(column.real()), ::llvm::APFloat(column.imag()));
                        row_attr.push_back(c);
                    }
                    matrix_attr.push_back(::mlir::ArrayAttr::get(ctx, row_attr));
                }
                new_mat_defs.push_back(GateDefinition::get(mlir::StringAttr::get(ctx, "unitary"), mlir::ArrayAttr::get(ctx, matrix_attr), ctx));
                rewriter.create<DefgateOp>(::mlir::UnknownLoc::get(ctx), mlir::TypeAttr::get(GateType::get(ctx, mat_qubit_num, GateTrait::General)), mlir::StringAttr::get(ctx, new_matrix_name), mlir::StringAttr::get(ctx, "nested"), ::mlir::ArrayAttr{}, ::mlir::ArrayAttr::get(ctx, new_mat_defs), ::mlir::ArrayAttr::get(ctx, ::llvm::ArrayRef<::mlir::Attribute>{}));
                rewriter.restoreInsertionPoint(ip);
            }
            auto ip = rewriter.saveInsertionPoint();
            rewriter.setInsertionPoint(op);
            auto new_use_gate = rewriter.create<UseGateOp>(op->getLoc(), GateType::get(ctx, mat_qubit_num, GateTrait::General), mlir::FlatSymbolRefAttr::get(ctx,new_matrix_name), ::mlir::ValueRange{});
            rewriter.restoreInsertionPoint(ip);
            rewriter.replaceOpWithNewOp<ApplyGateOp>(op.getOperation(), op->getResultTypes(), new_use_gate.result(), op.args());
            return mlir::success();
        }
        return mlir::failure();
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
    return "fold-constant-decorated-gates";
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