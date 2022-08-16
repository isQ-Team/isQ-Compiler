#ifndef _ISQ_CONTRIB_AFFINE_H
#define _ISQ_CONTRIB_AFFINE_H
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include "mlir/Pass/Pass.h"
namespace isq::contrib{
namespace mlir{
    void affineScalarReplace(::mlir::FuncOp f, ::mlir::DominanceInfo &domInfo,
                         ::mlir::PostDominanceInfo &postDomInfo);
    void registerAffineScalarReplacementPass();
}
}
#endif