//====- LowerToLLVM.cpp - Lowering from Toy+Affine+Std to LLVM ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements full lowering of Toy operations to LLVM MLIR dialect.
// 'toy.print' is lowered to a loop nest that calls `printf` on each element of
// the input array. The file also sets up the ToyToLLVMLoweringPass. This pass
// lowers the combination of Affine + SCF + Standard dialects to the LLVM one:
//
//                         Affine --
//                                  |
//                                  v
//                                  Standard --> LLVM (Dialect)
//                                  ^
//                                  |
//     'toy.print' --> Loop (SCF) --
//
//===----------------------------------------------------------------------===//

#include "isq/IR.h"

#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include <iostream>

#include "isq/lowering/printLower.h"
#include "isq/lowering/globalLower.h"
#include "isq/lowering/storeLower.h"
#include "isq/lowering/callqopLower.h"
#include "isq/lowering/moduleLower.h"
#include "isq/lowering/defgateLower.h"
#include "isq/lowering/applyLower.h"
#include "isq/lowering/decorateLower.h"
#include "isq/lowering/useLower.h"
#include "isq/lowering/allocLower.h"
#include "isq/lowering/deallocLower.h"

using namespace std;
using namespace mlir;

//===----------------------------------------------------------------------===//
// IsqToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {

struct IsQToLLVMLoweringPass
    : public PassWrapper<IsQToLLVMLoweringPass, OperationPass<ModuleOp>> {
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect, mlir::scf::SCFDialect, mlir::arith::ArithmeticDialect>();
    }
    void runOnOperation() final;
};
}

void IsQToLLVMLoweringPass::runOnOperation() {
    // The first thing to define is the conversion target. This will define the
    // final target for this lowering. For this lowering, we are only targeting
    // the LLVM dialect.
    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<mlir::ModuleOp>(
        [](mlir::ModuleOp op) {
            return !op.getName().hasValue();
        }
    );
    target.addIllegalOp<isq::ir::PrintOp>();
    target.addIllegalOp<isq::ir::CallQOpOp>();
    target.addIllegalOp<isq::ir::DeclareQOpOp>();
    target.addIllegalOp<isq::ir::DefgateOp>();
    target.addIllegalOp<isq::ir::ApplyGateOp>();
    target.addIllegalOp<isq::ir::DecorateOp>();
    target.addIllegalOp<isq::ir::UseGateOp>();

    target.addLegalDialect<LLVM::LLVMDialect>();
    
    LLVMTypeConverter typeConverter(&getContext());
    cout << typeConverter.getIndexTypeBitwidth() << endl;
    
    /*
    typeConverter.addConversion([&](isq::ir::QStateType type) { 
        return LLVM::LLVMStructType::getNewIdentified(&getContext(), StringRef("Qubit"), 
                        llvm::ArrayRef<Type>(IntegerType::get(&getContext(), 1)));
        
    });*/

    typeConverter.addConversion([&](isq::ir::QStateType type) {
        return LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque(StringRef("Qubit"), &getContext()));
    });

    /*
    typeConverter.addConversion([&](mlir::MemRefType type)-> Optional<Type> {
        if (type.getElementType().isa<isq::ir::QStateType>()){
            return LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque(StringRef("Array"), &getContext()));
        }
        return llvm::None;
    });*/

    /*
    cout << typeConverter.isLegal(isq::ir::QStateType::get(&getContext()));
    cout << LLVM::LLVMPointerType::isValidElementType(isq::ir::QStateType::get(&getContext()));

    auto ptr = LLVM::LLVMPointerType::get(isq::ir::QStateType::get(&getContext()), 2);
    */
    RewritePatternSet patterns(&getContext());
    
        
    populateAffineToStdConversionPatterns(patterns);
    populateLoopToStdConversionPatterns(patterns);
    
    patterns.add<moduleOpLowering>(&getContext());
    patterns.add<defgateOpLowering>(&getContext());
    patterns.add<callQOpLowering>(&getContext());
    
    patterns.add<applyGateOpLowering>(&getContext());
    patterns.add<decorateOpLowering>(&getContext());
    patterns.add<useGateOpLowering>(&getContext());

    patterns.add<StoreOpLowering>(&getContext());
    patterns.add<PrintOpLowering>(&getContext());
    
    populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
    
    //patterns.add<GlobalMemrefOpLowering>(typeConverter);
    patterns.add<AllocOpLowering>(typeConverter);
    patterns.add<DeallocOpLowering>(typeConverter);

    arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    
    
    
    auto module = getOperation();

    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::isqLower::createLowerToLLVMPass() {
    return std::make_unique<IsQToLLVMLoweringPass>();
}