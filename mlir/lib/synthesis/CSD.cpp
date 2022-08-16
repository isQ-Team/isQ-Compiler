#include "isq/QSynthesis.h"

using namespace isq::ir::synthesis;
/*
Matrix2cd qsyn::Rx(double angle) {
    Matrix2cd _Rx {
        {cos(angle / 2.), complex(0., -sin(angle / 2.))},
        {complex(0., sin(angle / 2.)), cos(angle / 2.)}
    };
    return _Rx;
}

Matrix2cd qsyn::Ry(double angle) {
    Matrix2cd _Ry {
        {cos(angle / 2.), -sin(angle / 2.)},
        {sin(angle / 2.), cos(angle / 2.)}
    };
    return _Ry;
}

Matrix2cd qsyn::Rz(double angle) {
    Matrix2cd _Rz {
        {complex(cos(angle / 2.), -sin(angle / 2.)), 0.},
        {0., complex(cos(angle / 2.), -sin(angle / 2.))}
    };
    return _Rz;
}
*/

CSD::CSD(MatrixXcd U) {
    vector<MatrixXcd> _csd;
    int n = log2(U.rows());
    MatrixXcd U1 = U.block(0, 0, 1<<(n-1), 1<<(n-1));
    MatrixXcd U2 = U.block(0, 1<<(n-1), 1<<(n-1), 1<<(n-1));
    MatrixXcd U3 = U.block(1<<(n-1), 0, 1<<(n-1), 1<<(n-1));
    MatrixXcd U4 = U.block(1<<(n-1), 1<<(n-1), 1<<(n-1), 1<<(n-1));
    if (/*n <= 5*/ true) {
        JacobiSVD<MatrixXcd> svdofU1(U1, ComputeThinU | ComputeThinV);
        JacobiSVD<MatrixXcd> svdofU3(U3 * svdofU1.matrixV(), ComputeThinU | ComputeThinV);
        S = svdofU3.singularValues().asDiagonal();
        B1 = svdofU3.matrixU();
        A2 = svdofU3.matrixV().conjugate().transpose() * svdofU1.matrixV().conjugate().transpose();
        C = svdofU3.matrixV().conjugate().transpose() * svdofU1.singularValues().asDiagonal() * svdofU3.matrixV();
        A1 = svdofU1.matrixU() * svdofU3.matrixV();
        B2 = C * B1.conjugate().transpose() * U4 - S * A1.conjugate().transpose() * U2;
    } else {
        BDCSVD<MatrixXcd> svdofU1(U1, ComputeThinU | ComputeThinV);
        BDCSVD<MatrixXcd> svdofU3(U3 * svdofU1.matrixV().transpose(), ComputeThinU | ComputeThinV);
        S = svdofU3.singularValues().asDiagonal();
        B1 = svdofU3.matrixU();
        A2 = svdofU3.matrixV().conjugate().transpose() * svdofU1.matrixV().conjugate().transpose();
        C = svdofU3.matrixV().conjugate().transpose() * svdofU1.singularValues().asDiagonal() * svdofU3.matrixV();
        A1 = svdofU1.matrixU() * svdofU3.matrixV();
        B2 = C * B1.conjugate().transpose() * U4 - S * A1.conjugate().transpose() * U2;
    }
}