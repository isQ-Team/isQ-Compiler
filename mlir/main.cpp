#include <cstdio>
#include <cassert>
#include <mlir/IR/Types.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/Parser.h>
#include <mlir/Parser/AsmParserState.h>
#include <mlir/IR/DialectImplementation.h>
#include "llvm/ADT/TypeSwitch.h"
#include <mlir/InitAllDialects.h>
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include <mlir/InitAllPasses.h>
#include <mlir/IR/BuiltinTypes.h>

namespace isq{
    namespace ir{
        /*class QubitType : public mlir::Type::TypeBase<QubitType, mlir::Type, mlir::TypeStorage>{
        public:
            using Base::Base;
        };*/
        class QStateType : public mlir::Type::TypeBase<QStateType, mlir::Type, mlir::TypeStorage, mlir::detail::MemRefElementTypeInterfaceTrait>{
        public:
            using Base::Base;
        };
        
    }
}


#define GET_OP_FWD_DEFINES
#include <tblgen/ISQOps.h.inc>
namespace isq{
    namespace ir {
        enum class GateTypeHint{
            Symmetric,
            Diagonal,
            AntiDiagonal,
            Hermitian
        };
        llvm::hash_code hash_value(const GateTypeHint& arg){
            int val = -1;
            if(arg==GateTypeHint::Symmetric){
                val = 0;
            }else if(arg==GateTypeHint::Diagonal){
                val = 1;
            }else if(arg==GateTypeHint::AntiDiagonal){
                val = 2;
            }else if (arg==GateTypeHint::Hermitian){
                val = 3;
            }
            assert(val>=0);
            return llvm::hash_code(val);
        }
        using GateInfo = std::tuple<int64_t, mlir::ArrayRef<GateTypeHint>>;
        struct GateTypeStorage : public mlir::TypeStorage{
            using KeyTy = GateInfo;
            int64_t size;
            mlir::ArrayRef<GateTypeHint> hints;
        private:
            GateTypeStorage(const KeyTy& c, mlir::TypeStorageAllocator& allocator): size(std::get<0>(c)), hints(allocator.copyInto(std::get<1>(c))){}
        public:
            static llvm::hash_code hashKey(const KeyTy& key){
                return llvm::hash_value(key);
            }
            static GateTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key){
                return new (allocator.allocate<GateTypeStorage>()) GateTypeStorage(key, allocator);
            }
            bool operator==(const KeyTy &key) const {
                return key == KeyTy(size, hints);
            }
        };

        class GateType : public mlir::Type::TypeBase<GateType, mlir::Type, GateTypeStorage, mlir::detail::MemRefElementTypeInterfaceTrait>{
        public:
            using Base::Base;
            static GateType get(mlir::MLIRContext* ctx, GateInfo k){
                auto r = std::get<1>(k);
                assert(std::is_sorted(r.begin(), r.end()) && "hints should be sorted");
                assert(std::adjacent_find(r.begin(), r.end())==r.end() && "no duplicate elements should exist");
                return Base::get(ctx, k);
            }
            GateInfo getGateInfo(){
                return GateInfo(getImpl()->size, getImpl()->hints);
            }
        };
        
        struct QOpTypeStorage : public mlir::TypeStorage{
            using KeyTy = mlir::FunctionType;
            KeyTy funcType;
        private:
            QOpTypeStorage(const KeyTy& c): funcType(c){}
        public:
            static llvm::hash_code hashKey(const KeyTy& key){
                return mlir::hash_value(key);
            }
            static QOpTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key){
                return new (allocator.allocate<QOpTypeStorage>()) QOpTypeStorage(key);
            }
            bool operator==(const KeyTy &key) const {
                return key == funcType;
            }
        };
        class QOpType : public mlir::Type::TypeBase<QOpType, mlir::Type, QOpTypeStorage>{
        public:
            using Base::Base;
            static QOpType get(mlir::MLIRContext* ctx, mlir::FunctionType funcType){
                return Base::get(ctx, funcType);
            }
            mlir::FunctionType getFuncType(){
                return getImpl()->funcType;
            }
        };
        /*
        // ops parser, printer and verifier
        // $op `(` $args `)` attr-dict `:` type($op) type($args)type($output)
        static mlir::ParseResult parseApplyOp(mlir::OpAsmParser &parser, mlir::OperationState &result){
            mlir::OpAsmParser::OperandType invokedOp;
            mlir::SmallVector<mlir::OpAsmParser::OperandType> arguments;
            mlir::Type funcType;
            if(parser.parseLParen() || parser.parseOperandList(arguments) || parser.parseRParen() || parser.parseColon()) return mlir::failure();
            auto typeLoc = parser.getCurrentLocation();
            if(parser.parseType(funcType)) return mlir::failure();
            mlir::FunctionType fType = funcType.cast<mlir::FunctionType>();
            if(!fType){
                parser.emitError(typeLoc, "the applyop type should be a function.");
                return mlir::failure();
            }
            auto qop = QOpType::get(result.getContext(), fType);

            return mlir::success();
        }

        static void print(mlir::OpAsmPrinter& printer, ApplyOp op){
            
        }
        */
       /*
       static mlir::Type constructQOpType(mlir::MLIRContext* ctx, mlir::OperandRange::type_range args, mlir::OperandRange::type_range results){
           auto tyfun = mlir::FunctionType::get(ctx, args, results);
           auto tyqop = QOpType::get(ctx, tyfun);
           return tyqop;
       }
       */

        static QOpType gateToQOp(GateType gate){
            auto info = gate.getGateInfo();
            auto qubit_size = std::get<0>(info);
            auto tyinout = mlir::SmallVector<mlir::Type>();
            for(auto i=0; i<qubit_size; i++){
                tyinout.push_back(QStateType::get(gate.getContext()));
            }
            return QOpType::get(gate.getContext(), mlir::FunctionType::get(gate.getContext(), tyinout, tyinout));
        }
        //mlir::LogicalResult verify(GateOp op);

    }
}

#include <tblgen/ISQDialect.h.inc>
#define GET_OP_CLASSES
#include <tblgen/ISQOps.h.inc>
#include <tblgen/ISQDialect.cpp.inc>



namespace isq{
    namespace ir{
        static void foo(GateOp o1){
        }
        mlir::LogicalResult verify(GateOp op){
            if(op.gate_type()!=op.getResult().getType()){
                op.emitOpError("operation dimension mismatch.");
                return mlir::failure();
            }
            return mlir::success();

        }
        mlir::LogicalResult verify(DeclareOp op){
            if(op.op_type()!=op.getResult().getType()){
                op.emitOpError("operation signature mismatch.");
                return mlir::failure();
            }
            return mlir::success();

        }
        mlir::LogicalResult verify(DowngradeGateOp op){
            auto result = op.getResult().getType().cast<GateType>().getGateInfo();
            auto operand = op.getOperand().getType().cast<GateType>().getGateInfo();
            auto vr = std::get<1>(result);
            auto vo = std::get<1>(operand);
            if(!std::includes(vo.begin(), vo.end(), vr.begin(), vr.end())){
                op.emitOpError("downgraded gate contains new trait(s) compared with original input.");
                return mlir::failure();
            }
            return mlir::success();
        }

        struct EliminateHermitianPairs : public mlir::OpRewritePattern<ApplyOp> {
            /// We register this pattern to match every toy.transpose in the IR.
            /// The "benefit" is used by the framework to order the patterns and process
            /// them in order of profitability.
            EliminateHermitianPairs(mlir::MLIRContext *context)
                : OpRewritePattern<ApplyOp>(context, /*benefit=*/1) {}

            /// This method is attempting to match a pattern and rewrite it. The rewriter
            /// argument is the orchestrator of the sequence of rewrites. It is expected
            /// to interact with it to perform any changes to the IR from here.
            mlir::LogicalResult
            matchAndRewrite(ApplyOp applyop,
                            mlir::PatternRewriter &rewriter) const {
            //applyop.emitRemark("Foo");
            if(applyop.args().size()==0){
                return mlir::failure();
            }
            ApplyOp apply_2 = applyop.args().front().getDefiningOp<ApplyOp>();
            //applyop.emitRemark("Foo1");
            if(!apply_2) return mlir::failure();
            UseOp use_1 = applyop.op().getDefiningOp<UseOp>();
            //applyop.emitRemark("Foo2");
            if(!use_1) return mlir::failure();
            GateOp gate_1 = use_1.getOperand().getDefiningOp<GateOp>();
            //applyop.emitRemark("Foo3");
            if(!gate_1) return mlir::failure();
            
            auto info = std::get<1>(gate_1.gate_type().getGateInfo());
            //applyop.emitRemark("Foo4");
            if(std::find(info.begin(), info.end(), GateTypeHint::Hermitian)==info.end()){
                
                return mlir::failure(); // not hermitian.
            }
            auto apply_1_args = mlir::SmallVector<mlir::Value>(applyop.args());
            auto apply_2_results = mlir::SmallVector<mlir::Value>(apply_2.getResults());
            if(std::find(info.begin(), info.end(), GateTypeHint::Symmetric)!=info.end()){
                std::sort(apply_1_args.begin(), apply_1_args.end(), [](mlir::Value a, mlir::Value b){
                    return a.getImpl() < b.getImpl();
                });
                std::sort(apply_2_results.begin(), apply_2_results.end(), [](mlir::Value a, mlir::Value b){
                    return a.getImpl() < b.getImpl();
                });
            }
            //applyop.emitRemark("Foo5");
            if(apply_1_args != apply_2_results){
                
                return mlir::failure(); // on different qubits.
            }
            //applyop.emitRemark("Foo6");
            UseOp use_2 = apply_2.op().getDefiningOp<UseOp>();
            if(!use_2) return mlir::failure();
            //applyop.emitRemark("Foo7");
            if(use_2 != use_1){
                return mlir::failure(); // not the same gate use.
            }
            //applyop.emitRemark("Foo8");
            // rewrite
            rewriter.replaceOp(applyop, apply_2.args());
            rewriter.eraseOp(apply_2);
            return mlir::success();

            /*
            def EliminateHermitianPairPattern: Pat<
            (ISQ_ApplyOp 
            (ISQ_UseOp (ISQ_GateOp $name1, $attr1)), 
            (ISQ_ApplyOp (ISQ_UseOp (ISQ_GateOp $name2, $attr2)), $qubits)), (replaceWithValue $qubits)>;
            */
            
            }
        };
        void ApplyOp::getCanonicalizationPatterns(
            mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
            results.add<EliminateHermitianPairs>(context);
        }
        void ISQDialect::initialize(){
            addTypes<QStateType, GateType, QOpType>();
            #define GET_OP_LIST
            addOperations<
            #include <tblgen/ISQOPs.cpp.inc>
            >();
        }
        mlir::Type ISQDialect::parseType(mlir::DialectAsmParser& parser) const{
            mlir::StringRef kw;
            auto kwLoc = parser.getCurrentLocation();
            if(parser.parseKeyword(&kw)){
                return nullptr;
            }
            if(kw=="qstate"){
                return QStateType::get(this->getContext());
            }else if(kw=="gate"){
                // gate<n, [traits]>
                if(parser.parseLess()) return nullptr;
                auto typeLoc = parser.getCurrentLocation();
                int64_t gate_size = 0;
                if(parser.parseInteger(gate_size)) return nullptr;
                mlir::SmallVector<GateTypeHint> hints;
                while(mlir::succeeded(parser.parseOptionalComma())){
                    auto traitLoc = parser.getCurrentLocation();
                    mlir::StringRef next_trait;
                    if(parser.parseKeyword(&next_trait)) return nullptr;
                    if(next_trait == "symmetric"){
                        hints.push_back(GateTypeHint::Symmetric);
                    }else if(next_trait == "diagonal"){
                        hints.push_back(GateTypeHint::Diagonal);
                    }else if(next_trait == "antidiagonal"){
                        hints.push_back(GateTypeHint::AntiDiagonal);
                    }else if(next_trait == "hermitian"){
                        hints.push_back(GateTypeHint::Hermitian);
                    }else {
                        parser.emitError(traitLoc, "unknown gate trait \""+next_trait+"\"");
                        return nullptr;
                    }
                }
                if(parser.parseGreater()) return nullptr;

                std::sort(hints.begin(), hints.end());
                hints.erase(std::unique(hints.begin(), hints.end()), hints.end());
                if(gate_size<=0){
                    parser.emitError(typeLoc, "gate size should be positive.");
                    return nullptr;
                }
                return GateType::get(this->getContext(), GateInfo(gate_size, hints));
                
            }else if(kw=="qop"){
                // qop<functiontype>
                mlir::Type f;
                if(parser.parseLess()) return nullptr;
                auto typeLoc = parser.getCurrentLocation();
                if(parser.parseType(f) || parser.parseGreater()) return nullptr;
                auto ft = f.cast<mlir::FunctionType>();
                if(!ft){
                    parser.emitError(typeLoc, "QOp internal type should be a FunctionType.");
                    return nullptr;
                }
                return QOpType::get(this->getContext(), ft);
            }else{
                parser.emitError(kwLoc, "unrecognized type: "+kw);
                return nullptr;
            }
        }
        void ISQDialect::printType(::mlir::Type type,::mlir::DialectAsmPrinter &printer) const{
            auto result = llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(type)
            .Case<QStateType>([&](QStateType t){
                printer<<"qstate";
                return mlir::success();
            })
            .Case<QOpType>([&](QOpType t){
                printer<<"qop<";
                printer<<t.getFuncType();
                printer<<">";
                return mlir::success();
            })
            .Case<GateType>([&](GateType t){
                auto info = t.getGateInfo();
                printer<<"gate<"<<std::get<0>(info);
                for(auto& trait: std::get<1>(info)){
                    printer<<",";
                    if(trait==GateTypeHint::Symmetric){
                        printer<<"symmetric";
                    }else if(trait==GateTypeHint::Diagonal){
                        printer<<"diagonal";
                    }else if(trait==GateTypeHint::AntiDiagonal){
                        printer<<"antidiagonal";
                    }else if(trait==GateTypeHint::Hermitian){
                        printer<<"hermitian";
                    }else{
                        llvm_unreachable("unexpected Gate Type Hint");
                    }
                }
                printer<<">";
                return mlir::success();
            })
            .Default([&](mlir::Type t){
                return mlir::failure();
            });
            if(mlir::failed(result)){
                llvm_unreachable("unexpected 'ISQ' type kind");
            }
        }
    }
    

}
#define GET_OP_CLASSES
#include <tblgen/ISQOPs.cpp.inc>

int isq_mlir_opt_main(int argc, char** argv){
    mlir::registerAllPasses();
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    registry.insert<isq::ir::ISQDialect>();
    return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR modular optimizer driver for ISQ dialect\n", registry,
                        /*preloadDialectsInContext=*/false));
}

int main(int argc, char** argv){
    return isq_mlir_opt_main(argc, argv);
}