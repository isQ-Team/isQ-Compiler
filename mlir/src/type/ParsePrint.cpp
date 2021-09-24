#include <isq/IR.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Types.h>
#include <isq/ParsePrint.h>
namespace isq {
namespace ir {

template <class... Cases> class ISQTypeParserPrinter {};
// Edge case:
template <> class ISQTypeParserPrinter<> {
public:
    ::mlir::Type parseType(::mlir::StringRef kw, ::llvm::SMLoc kwLoc,
                           ::mlir::DialectAsmParser &parser) const {
        parser.emitError(kwLoc, "unrecognized type: " + kw);
        return nullptr;
    }
    void printType(::mlir::Type type,
                   ::mlir::DialectAsmPrinter &printer) const {
        llvm_unreachable("unexpected 'ISQ' type kind");
    }
};
// Parser/Codegen P:
template <class P, class... Cases> class ISQTypeParserPrinter<P, Cases...> {
public:
    typedef typename std::enable_if<
        std::is_base_of<ISQTypeParserPrinterCase, P>::value, P>::type PP;
    ::mlir::Type parseType(::mlir::StringRef kw, ::llvm::SMLoc kwLoc,
                           ::mlir::DialectAsmParser &parser) const {
        PP p;
        auto name = p.keyword();
        if (name == kw) {
            return p.parseType(kwLoc, parser);
        } else {
            ISQTypeParserPrinter<Cases...> q;
            return q.parseType(kw, kwLoc, parser);
        }
    }
    void printType(::mlir::Type type,
                   ::mlir::DialectAsmPrinter &printer) const {
        PP p;
        if (p.isa(type)) {
            p.printType(type, printer);
        } else {
            ISQTypeParserPrinter<Cases...> q;
            q.printType(type, printer);
        }
    }
};

typedef ISQTypeParserPrinter<QStateParsePrint, GateOpParsePrint, QOpParsePrint>
    MainParserPrinter;
mlir::Type ISQDialect::parseType(mlir::DialectAsmParser &parser) const {
    mlir::StringRef kw;
    auto kwLoc = parser.getCurrentLocation();
    if (parser.parseKeyword(&kw)) {
        return nullptr;
    }
    MainParserPrinter p;
    return p.parseType(kw, kwLoc, parser);
}
void ISQDialect::printType(::mlir::Type type,
                           ::mlir::DialectAsmPrinter &printer) const {
    MainParserPrinter p;
    p.printType(type, printer);
}

} // namespace ir

} // namespace isq
