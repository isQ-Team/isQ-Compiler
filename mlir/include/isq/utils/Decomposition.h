#ifndef _ISQ_UTILS_DECOMPOSITION_H
#define _ISQ_UTILS_DECOMPOSITION_H
#include <complex>
namespace isq{
    namespace ir{
        struct ZYZDecomposition{
            double phase;
            double theta;
            double phi;
            double lam;
        };
        ZYZDecomposition zyzDecomposition(std::complex<double> matrix[2][2]);
    }
}
#endif