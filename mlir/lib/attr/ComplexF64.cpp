#include <isq/IR.h>
#include <complex>
#include <mlir/Parser.h>
#include <mlir/Support/LogicalResult.h>
namespace isq {
namespace ir {
::std::complex<double> ComplexF64Attr::complexValue() {
    return ::std::complex<double>(this->getReal().convertToDouble(),
                                  this->getImag().convertToDouble());
}
::mlir::Attribute ComplexF64Attr::parseIR(::mlir::AsmParser &parser) {
    double real, imag;
    if (parser.parseLess() || parser.parseFloat(real) || parser.parseComma() ||
        parser.parseFloat(imag) || parser.parseGreater()) {
        return nullptr;
    }
    return get(parser.getBuilder().getContext(), ::llvm::APFloat(real),
               ::llvm::APFloat(imag));
}
void ComplexF64Attr::printIR(::mlir::AsmPrinter &p) const {
    p << "complex<";
    p << this->getReal();
    p << ", ";
    p << this->getImag();
    p << ">";
}
} // namespace ir
} // namespace isq
