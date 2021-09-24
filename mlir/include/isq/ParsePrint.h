#ifndef _ISQ_PARSEPRINT_H
#define _ISQ_PARSEPRINT_H
#include <mlir/IR/Types.h>
#include <mlir/IR/DialectImplementation.h>
namespace isq {
namespace ir {
class ISQTypeParserPrinterCase {
public:
    virtual const char *keyword() const = 0;
    virtual bool isa(::mlir::Type me) const = 0;
    virtual ::mlir::Type parseType(::llvm::SMLoc kwLoc,
                                   ::mlir::DialectAsmParser &parser) const = 0;
    virtual void printType(::mlir::Type type,
                           ::mlir::DialectAsmPrinter &printer) const = 0;
};
class QStateParsePrint : public ISQTypeParserPrinterCase {
public:
    const char *keyword() const;
    bool isa(::mlir::Type me) const;
    ::mlir::Type parseType(::llvm::SMLoc kwLoc,
                           ::mlir::DialectAsmParser &parser) const;
    void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const;
};
class GateOpParsePrint : public ISQTypeParserPrinterCase {
public:
    const char *keyword() const;
    bool isa(::mlir::Type me) const;
    ::mlir::Type parseType(::llvm::SMLoc kwLoc,
                           ::mlir::DialectAsmParser &parser) const;
    void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const;
};
class QOpParsePrint : public ISQTypeParserPrinterCase {
public:
    const char *keyword() const;
    bool isa(::mlir::Type me) const;
    ::mlir::Type parseType(::llvm::SMLoc kwLoc,
                           ::mlir::DialectAsmParser &parser) const;
    void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const;
};
} // namespace ir
} // namespace isq

#endif