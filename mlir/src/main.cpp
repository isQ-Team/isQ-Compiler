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
#include <algorithm>

#include <isq/IR.h>

namespace isq {
namespace ir {
static void foo(GateOp o1) {}

mlir::Type ISQDialect::parseType(mlir::DialectAsmParser &parser) const {
    mlir::StringRef kw;
    auto kwLoc = parser.getCurrentLocation();
    if (parser.parseKeyword(&kw)) {
        return nullptr;
    }
    if (kw == "qstate") {
        return QStateType::get(this->getContext());
    } else if (kw == "gate") {
        // gate<n, [traits]>
        if (parser.parseLess())
            return nullptr;
        auto typeLoc = parser.getCurrentLocation();
        int64_t gate_size = 0;
        if (parser.parseInteger(gate_size))
            return nullptr;
        mlir::SmallVector<GateTypeHint> hints;
        while (mlir::succeeded(parser.parseOptionalComma())) {
            auto traitLoc = parser.getCurrentLocation();
            mlir::StringRef next_trait;
            if (parser.parseKeyword(&next_trait))
                return nullptr;
            if (next_trait == "symmetric") {
                hints.push_back(GateTypeHint::Symmetric);
            } else if (next_trait == "diagonal") {
                hints.push_back(GateTypeHint::Diagonal);
            } else if (next_trait == "antidiagonal") {
                hints.push_back(GateTypeHint::AntiDiagonal);
            } else if (next_trait == "hermitian") {
                hints.push_back(GateTypeHint::Hermitian);
            } else {
                parser.emitError(traitLoc,
                                 "unknown gate trait \"" + next_trait + "\"");
                return nullptr;
            }
        }
        if (parser.parseGreater())
            return nullptr;

        std::sort(hints.begin(), hints.end());
        hints.erase(std::unique(hints.begin(), hints.end()), hints.end());
        if (gate_size <= 0) {
            parser.emitError(typeLoc, "gate size should be positive.");
            return nullptr;
        }
        return GateType::get(this->getContext(), GateInfo(gate_size, hints));

    } else if (kw == "qop") {
        // qop<functiontype>
        mlir::Type f;
        if (parser.parseLess())
            return nullptr;
        auto typeLoc = parser.getCurrentLocation();
        if (parser.parseType(f) || parser.parseGreater())
            return nullptr;
        auto ft = f.cast<mlir::FunctionType>();
        if (!ft) {
            parser.emitError(typeLoc,
                             "QOp internal type should be a FunctionType.");
            return nullptr;
        }
        return QOpType::get(this->getContext(), ft);
    } else {
        parser.emitError(kwLoc, "unrecognized type: " + kw);
        return nullptr;
    }
}
void ISQDialect::printType(::mlir::Type type,
                           ::mlir::DialectAsmPrinter &printer) const {
    auto result = llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(type)
                      .Case<QStateType>([&](QStateType t) {
                          printer << "qstate";
                          return mlir::success();
                      })
                      .Case<QOpType>([&](QOpType t) {
                          printer << "qop<";
                          printer << t.getFuncType();
                          printer << ">";
                          return mlir::success();
                      })
                      .Case<GateType>([&](GateType t) {
                          auto info = t.getGateInfo();
                          printer << "gate<" << std::get<0>(info);
                          for (auto &trait : std::get<1>(info)) {
                              printer << ",";
                              if (trait == GateTypeHint::Symmetric) {
                                  printer << "symmetric";
                              } else if (trait == GateTypeHint::Diagonal) {
                                  printer << "diagonal";
                              } else if (trait == GateTypeHint::AntiDiagonal) {
                                  printer << "antidiagonal";
                              } else if (trait == GateTypeHint::Hermitian) {
                                  printer << "hermitian";
                              } else {
                                  llvm_unreachable("unexpected Gate Type Hint");
                              }
                          }
                          printer << ">";
                          return mlir::success();
                      })
                      .Default([&](mlir::Type t) { return mlir::failure(); });
    if (mlir::failed(result)) {
        llvm_unreachable("unexpected 'ISQ' type kind");
    }
}
} // namespace ir

} // namespace isq

int isq_mlir_opt_main(int argc, char **argv) {
    mlir::registerAllPasses();
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    registry.insert<isq::ir::ISQDialect>();
    return mlir::asMainReturnCode(mlir::MlirOptMain(
        argc, argv, "MLIR modular optimizer driver for ISQ dialect\n", registry,
        /*preloadDialectsInContext=*/false));
}

int main(int argc, char **argv) { return isq_mlir_opt_main(argc, argv); }