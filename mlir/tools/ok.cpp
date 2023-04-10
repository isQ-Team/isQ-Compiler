#include <llvm/Support/TypeName.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>


int main(int argc, char **argv) {
    mlir::MLIRContext ctx;
    mlir::RewritePatternSet rps(&ctx);
    mlir::AffineStoreOp::getCanonicalizationPatterns(rps, &ctx);
    mlir::AffineLoadOp::getCanonicalizationPatterns(rps, &ctx);
    for(auto& pattern: rps.getNativePatterns()){
        llvm::outs()<<pattern->getDebugName()<<"\n";
        for(auto& label: pattern->getDebugLabels()){
            llvm::outs()<<"    "<<label<<"\n";
        }
<<<<<<< HEAD
        
=======
        mlir::AffineStoreOp s;
        s.getIndices();
>>>>>>> merge
    }
    return 0;
}