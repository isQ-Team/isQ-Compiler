#ifndef _ISQ_MATH_H
#define _ISQ_MATH_H
#include <cstdint>
#include <mlir/Support/LLVM.h>
#include <complex>
namespace isq {
namespace ir {
namespace math {
template <class T> struct Fwd {
public:
    using Ty = T;
    std::unique_ptr<T> body;
    Ty *operator->() { return &*body; }
};
// Row-first input matrix
struct InputMatrix
    : public Fwd<std::vector<std::vector<std::complex<double>>>> {};
struct Mat;
struct MatDel {
    void operator()(Mat *m);
};
::mlir::Optional<size_t> checkDimensionality(InputMatrix &mat);
std::unique_ptr<Mat, MatDel> toEigenMatrix(InputMatrix &mat);
bool isUnitary(Mat &mat, double eps = 1e-5);
bool isHermitian(Mat &mat, double eps = 1e-5);
bool isSymmetric(Mat &mat, double eps = 1e-5);
bool isDiagonal(Mat &mat, double eps = 1e-5);
bool isAntiDiagonal(Mat &mat, double eps = 1e-5);
} // namespace math
} // namespace ir
} // namespace isq
#endif