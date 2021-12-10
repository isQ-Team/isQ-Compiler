#include "isq/QTypes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include <cstdint>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include "mlir/Analysis/AffineStructures.h"
#include "isq/passes/FoldAffineComparison.h"
#include "llvm/Support/raw_ostream.h"
namespace isq{
namespace ir{
namespace passes{



mlir::LogicalResult isBooleanConstraint(mlir::IntegerSet integer_set){
    if(integer_set.getNumDims()==0 && integer_set.getNumSymbols()==1){
        mlir::FlatAffineConstraints cons(integer_set);
        //integer_set.dump();
        mlir::SmallVector<int64_t> temp;
        temp.push_back(0);
        // The integer set should not contain 0.
        if(cons.containsPoint(temp)) return mlir::failure();
        temp[0]=-1;
        // The integer set should contain 1.
        if(!cons.containsPoint(temp)) return mlir::failure();
        // Then it becomes a boolean constraint.
        return mlir::success();
    }
    return mlir::failure();
}
FoldAffineComparison::FoldAffineComparison(mlir::MLIRContext *context)
    : OpRewritePattern<mlir::AffineIfOp>(context, 1) {
        eq = mlir::parseIntegerSet("()[s0, s1]: (s0-s1 == 0)", context);
        // s0-s1<0 s0-s1+1<=0
        slt = mlir::parseIntegerSet("()[s0, s1]: (s1-s0-1 >= 0)", context);
        // s0-s1<=0
        sle = mlir::parseIntegerSet("()[s0, s1]: (s1-s0 >= 0)", context);
        // s0-s1>0 s0-s1-1>=0
        sgt = mlir::parseIntegerSet("()[s0, s1]: (s0-s1-1 >= 0)", context);
        // s0-s1>=0
        sge = mlir::parseIntegerSet("()[s0, s1]: (s0-s1 >= 0)", context);
        //eq.dump();
        //slt.dump();
        //sle.dump();
        //sgt.dump();
        //sge.dump();
    }

mlir::LogicalResult FoldAffineComparison::match(mlir::AffineIfOp op) const {
    mlir::IntegerSet integer_set = op.getIntegerSet();
    if(integer_set.getNumDims()!=0 || integer_set.getNumSymbols()!=1){
        return mlir::failure();
    }
    auto first_operand = op->getOperand(0);
    if(!first_operand) return mlir::failure();
    auto previous_index_cast = first_operand.getDefiningOp<mlir::arith::IndexCastOp>();
    if(!previous_index_cast) return mlir::failure();
    auto input = previous_index_cast.getIn();
    // must be i1.
    auto bool_type = mlir::IntegerType::get(op.getContext(), 1);
    if(input.getType() != bool_type) return mlir::failure();
    auto cmp_insn = input.getDefiningOp<mlir::arith::CmpIOp>();
    if(!cmp_insn) return mlir::failure();
    // don't handle unsigned comparison.
    if(static_cast<uint64_t>(cmp_insn.predicate())>5){
        return mlir::failure();
    }
    auto index_type = mlir::IndexType::get(op->getContext());
    if(cmp_insn.getOperand(0).getType()!=index_type) return mlir::failure();
    if(mlir::failed(isBooleanConstraint(integer_set))) return mlir::failure();
    return mlir::success();
}


void FoldAffineComparison::rewrite(mlir::AffineIfOp op, mlir::PatternRewriter& rewriter) const{
    auto cmp_op = op.getOperand(0).getDefiningOp<mlir::arith::IndexCastOp>()
    .getIn().getDefiningOp<mlir::arith::CmpIOp>();
    assert(cmp_op);
    cmp_op.dump();
    mlir::SmallVector<mlir::Value> vals;
    vals.push_back(cmp_op.lhs());
    vals.push_back(cmp_op.rhs());
    auto cmp_ty = cmp_op.getPredicate();
    if(cmp_ty == mlir::arith::CmpIPredicate::eq){
        rewriter.startRootUpdate(op);
        op.setConditional(eq, vals);
        rewriter.finalizeRootUpdate(op);
    }else if(cmp_ty == mlir::arith::CmpIPredicate::ne){
        mlir::AffineIfOp new_if = op.cloneWithoutRegions();
        rewriter.cloneRegionBefore(op.thenRegion(), new_if.elseRegion(), new_if.elseRegion().begin());
        rewriter.cloneRegionBefore(op.elseRegion(), new_if.thenRegion(), new_if.elseRegion().begin());
        new_if.setConditional(eq, vals);
        rewriter.replaceOp(op, new_if.getResults());
    }else if(cmp_ty == mlir::arith::CmpIPredicate::slt){
        rewriter.startRootUpdate(op);
        op.setConditional(slt, vals);
        rewriter.finalizeRootUpdate(op);
    }else if(cmp_ty == mlir::arith::CmpIPredicate::sle){
        rewriter.startRootUpdate(op);
        op.setConditional(sle, vals);
        rewriter.finalizeRootUpdate(op);
    }else if(cmp_ty == mlir::arith::CmpIPredicate::sgt){
        rewriter.startRootUpdate(op);
        op.setConditional(sgt, vals);
        rewriter.finalizeRootUpdate(op);
    }else if(cmp_ty == mlir::arith::CmpIPredicate::sge){
        rewriter.startRootUpdate(op);
        op.setConditional(sge, vals);
        rewriter.finalizeRootUpdate(op);
    }
    
}

}
}
}
