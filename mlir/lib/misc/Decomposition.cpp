#include <cmath>
#include <complex>
#include <Eigen/Dense>
#include "isq/Operations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Pass/Pass.h"
#include "isq/utils/Decomposition.h"
#include "llvm/Support/raw_ostream.h"
namespace isq{
    namespace ir{
        ZYZDecomposition zyzDecomposition(std::complex<double> matrix[2][2]){
            Eigen::Matrix2cd mat {
                {matrix[0][0], matrix[0][1]},
                {matrix[1][0], matrix[1][1]}
            };
            if(!mat.isUnitary()){
                throw "matrix is not unitary";
            }
            auto coeff = std::pow(mat.determinant(), -0.5);
            auto phase = -std::arg(coeff);
            mat = coeff * mat;
            auto theta = 2*std::atan2(std::abs(mat(1,0)),std::abs(mat(0,0)));
            auto phi_p_lambda_2 = std::arg(mat(1,1));
            auto phi_m_lambda_2 = std::arg(mat(1,0));
            auto phi = phi_p_lambda_2 + phi_m_lambda_2;
            auto lam = phi_p_lambda_2 - phi_m_lambda_2;
            return ZYZDecomposition{phase,theta,phi,lam};
        }/*
        struct KAKDecomposition{
            ZYZDecomposition a;
            ZYZDecomposition b;
        };
        KAKDecomposition kakDecomposition(std::complex<double> matrix[2][2]){
            Eigen::Matrix2cd mat {
                {matrix[0][0], matrix[0][1]},
                {matrix[1][0], matrix[1][1]}
            };
            Eigen::Matrix2cd magic{
                {1, 1i, 0, 0},
                {0, 0, 1i, 1},
                {0, 0, 1i, -1},
                {1, -1i, 0, 0}
            };
            magic = magic * 0.70710678;
            auto ab = magic * mat * magic.getAdjoint();

        }
        */
    }
}

