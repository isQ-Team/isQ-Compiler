#ifndef _ISQ_QTYPES_H
#define _ISQ_QTYPES_H
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Types.h>
#include <isq/Enums.h>
namespace isq {
namespace ir {
/*class QubitType : public mlir::Type::TypeBase<QubitType, mlir::Type,
mlir::TypeStorage>{ public: using Base::Base;
};*/
class QStateType : public mlir::Type::TypeBase<
                       QStateType, mlir::Type, mlir::TypeStorage,
                       mlir::detail::MemRefElementTypeInterfaceTrait> {
public:
    using Base::Base;
};

using GateInfo = std::tuple<int64_t, GateTraitAttr>;
struct GateTypeStorage : public mlir::TypeStorage {
    using KeyTy = GateInfo;
    int64_t size;
    GateTraitAttr hints;

private:
    GateTypeStorage(const KeyTy &c, mlir::TypeStorageAllocator &allocator)
        : size(std::get<0>(c)), hints(std::get<1>(c)) {}

public:
    static llvm::hash_code hashKey(const KeyTy &key);
    static GateTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key);
    bool operator==(const KeyTy &key) const;
};

class GateType : public mlir::Type::TypeBase<
                     GateType, mlir::Type, GateTypeStorage,
                     mlir::detail::MemRefElementTypeInterfaceTrait> {
public:
    using Base::Base;
    static GateType get(mlir::MLIRContext *ctx, GateInfo k);
    GateInfo getGateInfo();
    int64_t gateSize();
    GateTrait hints();
    bool hasHint(GateTrait t);
};

struct QOpTypeStorage : public mlir::TypeStorage {
    using KeyTy = mlir::FunctionType;
    KeyTy funcType;

private:
    QOpTypeStorage(const KeyTy &c) : funcType(c) {}

public:
    static llvm::hash_code hashKey(const KeyTy &key);
    static QOpTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key);
    bool operator==(const KeyTy &key) const;
};
class QOpType
    : public mlir::Type::TypeBase<QOpType, mlir::Type, QOpTypeStorage> {
public:
    using Base::Base;
    static QOpType get(mlir::MLIRContext *ctx, mlir::FunctionType funcType);
    mlir::FunctionType getFuncType();
};
/*
// ops parser, printer and verifier
// $op `(` $args `)` attr-dict `:` type($op) type($args)type($output)
static mlir::ParseResult parseApplyOp(mlir::OpAsmParser &parser,
mlir::OperationState &result){ mlir::OpAsmParser::OperandType invokedOp;
    mlir::SmallVector<mlir::OpAsmParser::OperandType> arguments;
    mlir::Type funcType;
    if(parser.parseLParen() || parser.parseOperandList(arguments) ||
parser.parseRParen() || parser.parseColon()) return mlir::failure(); auto
typeLoc = parser.getCurrentLocation(); if(parser.parseType(funcType)) return
mlir::failure(); mlir::FunctionType fType = funcType.cast<mlir::FunctionType>();
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
static mlir::Type constructQOpType(mlir::MLIRContext* ctx,
mlir::OperandRange::type_range args, mlir::OperandRange::type_range results){
    auto tyfun = mlir::FunctionType::get(ctx, args, results);
    auto tyqop = QOpType::get(ctx, tyfun);
    return tyqop;
}
*/

QOpType gateToQOp(GateType gate);
// mlir::LogicalResult verify(GateOp op);

} // namespace ir
} // namespace isq
#endif