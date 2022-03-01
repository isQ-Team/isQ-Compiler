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
    virtual ::mlir::Type parseType(::mlir::AsmParser &parser) const = 0;
    virtual void printType(::mlir::Type type,
                           ::mlir::AsmPrinter &printer) const = 0;
};
class QStateParsePrint : public ISQTypeParserPrinterCase {
public:
    const char *keyword() const;
    bool isa(::mlir::Type me) const;
    ::mlir::Type parseType(::mlir::AsmParser &parser) const;
    void printType(::mlir::Type type, ::mlir::AsmPrinter &printer) const;
};
class GateParsePrint : public ISQTypeParserPrinterCase {
public:
    const char *keyword() const;
    bool isa(::mlir::Type me) const;
    ::mlir::Type parseType(::mlir::AsmParser &parser) const;
    void printType(::mlir::Type type, ::mlir::AsmPrinter &printer) const;
};
class QIRQubitParsePrint : public ISQTypeParserPrinterCase {
public:
    const char *keyword() const;
    bool isa(::mlir::Type me) const;
    ::mlir::Type parseType(::mlir::AsmParser &parser) const;
    void printType(::mlir::Type type, ::mlir::AsmPrinter &printer) const;
};
class QIRResultParsePrint : public ISQTypeParserPrinterCase {
public:
    const char *keyword() const;
    bool isa(::mlir::Type me) const;
    ::mlir::Type parseType(::mlir::AsmParser &parser) const;
    void printType(::mlir::Type type, ::mlir::AsmPrinter &printer) const;
};
} // namespace ir
} // namespace isq

#endif