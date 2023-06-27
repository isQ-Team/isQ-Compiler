#include "isq/Dialect.h"
#include "isq/Lower.h"
#include "isq/Operations.h"
#include "isq/QTypes.h"
#include "isq/GateDefTypes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
#include <mlir/Dialect/Arith/Transforms/Passes.h>
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
namespace isq{
namespace ir{
namespace passes{

namespace qir_rep_to_llvm{
using namespace mlir;

class RuleReplaceAssert : public mlir::OpRewritePattern<AssertOp>{
    mlir::ModuleOp rootModule;
public:
    RuleReplaceAssert(mlir::MLIRContext* ctx, mlir::ModuleOp module): mlir::OpRewritePattern<AssertOp>(ctx, 1), rootModule(module){}

    /// Return a symbol reference to the printf function, inserting it into the
    /// module if necessary.
    static mlir::FlatSymbolRefAttr getOrInsertPrintf(mlir::PatternRewriter &rewriter,
                                                mlir::ModuleOp module) {
        auto *context = module.getContext();
        if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf"))
            return mlir::SymbolRefAttr::get(context, "printf");

        // Create a function declaration for printf, the signature is:
        //   * `i32 (i8*, ...)`
        auto llvmI32Ty = mlir::IntegerType::get(context, 32);
        auto llvmI8PtrTy = mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8));
        auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                        /*isVarArg=*/true);

        // Insert the printf function into the body of the parent module.
        mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
        return mlir::SymbolRefAttr::get(context, "printf");
    }

    static mlir::FlatSymbolRefAttr getOrInsertExit(mlir::PatternRewriter &rewriter,
                                                mlir::ModuleOp module) {
        auto *context = module.getContext();
        if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("exit"))
            return mlir::SymbolRefAttr::get(context, "exit");

        auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(
            mlir::LLVM::LLVMVoidType::get(context), mlir::IntegerType::get(context, 32));
        mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "exit", llvmFnType);
        return mlir::SymbolRefAttr::get(context, "exit");
    }

    /// Return a value representing an access into a global string with the given
    /// name, creating the string if necessary.
    static mlir::Value getOrCreateGlobalString(mlir::Location loc, mlir::OpBuilder &builder,
                                        mlir::StringRef name, mlir::StringRef value,
                                        mlir::ModuleOp module) {
        // Create the global at the entry of the module.
        mlir::LLVM::GlobalOp global;
        if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
            mlir::OpBuilder::InsertionGuard insertGuard(builder);
            builder.setInsertionPointToStart(module.getBody());
            auto type = mlir::LLVM::LLVMArrayType::get(
                mlir::IntegerType::get(builder.getContext(), 8), value.size());
            global = builder.create<mlir::LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                mlir::LLVM::Linkage::External, name, builder.getStringAttr(value), /*alignment=*/0);
        }

        // Get the pointer to the first character in the global string.
        mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);
        mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                        builder.getIndexAttr(0));
        return builder.create<mlir::LLVM::GEPOp>(
            loc,
            mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(builder.getContext(), 8)),
            globalPtr, llvm::ArrayRef<mlir::Value>({cst0, cst0}));
    }

    static mlir::Value getErrorMessage(mlir::Location loc, mlir::OpBuilder &builder, uint64_t code, mlir::ModuleOp module) {
        switch (code) {
        case 1: return getOrCreateGlobalString(loc, builder, "__msg1", mlir::StringRef("Error: divide 0\0", 16), module);
        case 2: return getOrCreateGlobalString(loc, builder, "__msg2", mlir::StringRef("Error: index out of bound\0", 26), module);
        case 3: return getOrCreateGlobalString(loc, builder, "__msg3", mlir::StringRef("Error: failed assertion\0", 24), module);
        default: return getOrCreateGlobalString(loc, builder, "__msg0", mlir::StringRef("Error: unknown error code\0", 26), module);
        }
    }

    mlir::LogicalResult matchAndRewrite(AssertOp op, mlir::PatternRewriter &rewriter) const override{
        mlir::ModuleOp parentModule = this->rootModule; //op->getParentOfType<mlir::ModuleOp>();
        
        // Get a symbol reference to the printf function, inserting it if necessary.
        auto printfRef = getOrInsertPrintf(rewriter, parentModule);
        auto exitRef = getOrInsertExit(rewriter, parentModule);
        mlir::Location loc = op->getLoc();
        auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, op.getCond(), true);
        rewriter.eraseOp(op);
        rewriter.updateRootInPlace(ifOp, [&]{
            rewriter.setInsertionPointToStart(ifOp.elseBlock());

            // Print source file information
            auto terminator = std::string("\0", 1);
            auto locStr = getOrCreateGlobalString(loc, rewriter, "__loc",
                mlir::StringRef("%s line %d column %d\n" + terminator), parentModule);
            mlir::FileLineColLoc flc = loc.dyn_cast<mlir::FileLineColLoc>();
            auto fileName = flc.getFilename().strref();
            auto fileNameTerminated = fileName.str() + terminator;
            auto sourcePos = getOrCreateGlobalString(loc, rewriter, fileName, fileNameTerminated, parentModule);
            auto lineAttr = rewriter.getI32IntegerAttr(flc.getLine());
            auto *context = parentModule.getContext();
            mlir::Value line = rewriter.create<mlir::LLVM::ConstantOp>(loc, mlir::IntegerType::get(context, 32), lineAttr);
            auto columnAttr = rewriter.getI32IntegerAttr(flc.getColumn());
            mlir::Value column = rewriter.create<mlir::LLVM::ConstantOp>(loc, mlir::IntegerType::get(context, 32), columnAttr);
            rewriter.create<mlir::func::CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                llvm::ArrayRef<mlir::Value>({locStr, sourcePos, line, column}));

            // Print error message
            auto errorNum = op.getErrorNum();
            mlir::Value messageCst = getErrorMessage(loc, rewriter, errorNum, parentModule);
            rewriter.create<mlir::func::CallOp>(loc, printfRef, rewriter.getIntegerType(32), llvm::ArrayRef<mlir::Value>({messageCst}));
            auto attr = rewriter.getIntegerAttr(rewriter.getIntegerType(32), errorNum);
            mlir::Value code = rewriter.create<mlir::LLVM::ConstantOp>(loc, mlir::IntegerType::get(context, 32), attr);
            rewriter.create<mlir::func::CallOp>(loc, exitRef, llvm::ArrayRef<Type>({}), llvm::ArrayRef<mlir::Value>({code}));
        });
        return mlir::success();
    }
};

struct QIRRepToLLVMPass : public mlir::PassWrapper<QIRRepToLLVMPass, mlir::OperationPass<mlir::ModuleOp>>{

    void runOnOperation() override {
        ConversionTarget target(getContext());
        //target.addLegalDialect<LLVM::LLVMDialect, mlir::scf::SCFDialect, mlir::arith::ArithmeticDialect>();
        target.addLegalDialect<LLVM::LLVMDialect>();
        target.addLegalOp<mlir::ModuleOp>();
        LLVMTypeConverter typeConverter(&getContext());
        //llvm::errs() << "Host index bit width: " <<typeConverter.getIndexTypeBitwidth() << "\n";
        
        typeConverter.addConversion([&](isq::ir::QIRQubitType type) {
            return LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque(StringRef("Qubit"), &getContext()));
        });
        typeConverter.addConversion([&](isq::ir::QIRResultType type) {
            return LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getOpaque(StringRef("Result"), &getContext()));
        });

        RewritePatternSet patterns(&getContext());
        populateAffineToStdConversionPatterns(patterns);
        populateSCFToControlFlowConversionPatterns(patterns);
        populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
        arith::populateArithExpandOpsPatterns(patterns);
        arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
        cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        //populateStdToLLVMConversionPatterns(typeConverter, patterns);

        auto module = getOperation();
        auto ctx = module->getContext();
        patterns.add<RuleReplaceAssert>(ctx, module);

        if (failed(applyFullConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
  mlir::StringRef getArgument() const final {
    return "isq-lower-qir-rep-to-llvm";
  }
  mlir::StringRef getDescription() const final {
    return  "Lower QIR subdialect to standard LLVM.";
  }
};


}

void registerQIR2LLVM(){
    using namespace qir_rep_to_llvm;
    mlir::PassRegistration<QIRRepToLLVMPass>();
}

}
}
}