#include <cmath>
#include <iostream>
#include "isq/qsyn.h"

using namespace Eigen;
using std::get;
qsyn::qsyn::qsyn(int n, UnitaryVector uvector, double e) {
    using namespace std;
    GateMatrix gate(1<<n, 1<<n);
    for (int j=0; j<(1<<n); j++) {
        for (int k=0; k<(1<<n); k++) {
            gate(j,k) = dcomplex(uvector[j*(1<<n)+k].first, uvector[j*(1<<n)+k].second);
        }
    }
    GateType gtype = NONE;
    GateLocation glocation;
    for (int j=0; j<n; j++) {
        glocation.push_back(j);
    }

    remain_gates.push_back(Gate(gtype, glocation, gate, 0.));
    phase = 0.;
    eps = e;

    QSD();
}

void qsyn::qsyn::QSD() {
    using namespace std;
    if (remain_gates.size() == 0) return;

    Gate gate = remain_gates.back();
    remain_gates.pop_back();
    GateType gtype = get<0>(gate);
    GateLocation glocation = get<1>(gate);
    int n = get<1>(gate).size();

    // Quantum Shannon Decomposition
    if (gtype == NONE) {
        if (n == 1) {
            AddDecomposedGate(gate);
            return QSD();
        } else {
            CSD csdofgate(get<2>(gate));
            ComplexSchur<MatrixXcd> schurofa1b1(csdofgate.A1 * csdofgate.B1.conjugate().transpose());
            MatrixXcd V1 = schurofa1b1.matrixU();
            MatrixXcd D1 = schurofa1b1.matrixT().diagonal().array().sqrt().matrix().asDiagonal();
            MatrixXcd W1 = D1 * V1.conjugate().transpose() * csdofgate.B1;

            ComplexSchur<MatrixXcd> schurofa2b2(csdofgate.A2 * csdofgate.B2.conjugate().transpose());
            MatrixXcd V2 = schurofa2b2.matrixU();
            MatrixXcd D2 = schurofa2b2.matrixT().diagonal().array().sqrt().matrix().asDiagonal();
            MatrixXcd W2 = D2 * V2.conjugate().transpose() * csdofgate.B2;

            GateLocation dlocation(glocation.begin()+1, glocation.end());

            remain_gates.push_back(Gate(NONE, dlocation, V1, 0.));
            remain_gates.push_back(Gate(MZ, glocation, -2. * D1.diagonal().array().arg().matrix().asDiagonal(), 0.));
            remain_gates.push_back(Gate(NONE, dlocation, W1, 0.));
            MatrixXcd temp = csdofgate.C + csdofgate.S*dcomplex(0.,1.);
            remain_gates.push_back(Gate(MY, glocation, 2. * temp.diagonal().array().arg().matrix().asDiagonal(), 0.));
            remain_gates.push_back(Gate(NONE, dlocation, V2, 0.));
            remain_gates.push_back(Gate(MZ, glocation, -2. * D2.diagonal().array().arg().matrix().asDiagonal(), get<3>(gate)));
            remain_gates.push_back(Gate(NONE, dlocation, W2, 0.));
            return QSD();
        }
    }

    if (gtype == CNOT) {
        AddDecomposedGate(gate);
        return QSD();
    }

    // Multiplexed-Pauli Decomposition
    if (n == 1) {
        AddDecomposedGate(gate);
        return QSD();
    }

    ArrayXcd A = get<2>(gate).diagonal().array();
    ArrayXcd A1 = A(seqN(0,1<<(n-2),2));
    ArrayXcd A2 = A(seqN(1,1<<(n-2),2));
    ArrayXcd B1 = (A1 + A2) / 2.;
    ArrayXcd B2 = (A1 - A2) / 2.;

    if (n == 2) {
        GateLocation ulocation(glocation.begin(), glocation.end()-1);
        GateLocation clocation;
        clocation.push_back(glocation.back());
        clocation.push_back(glocation[0]);
        phase += get<3>(gate);
        if (gtype == MZ) {
            AddDecomposedGate(Gate(RZ, ulocation, B1.matrix().asDiagonal(), 0.));
            AddDecomposedGate(Gate(CNOT, clocation, Matrix2cd(), 0.));
            AddDecomposedGate(Gate(RZ, ulocation, B2.matrix().asDiagonal(), 0.));
            AddDecomposedGate(Gate(CNOT, clocation, Matrix2cd(), 0.));
        } else if (gtype == MY) {
            AddDecomposedGate(Gate(RY, ulocation, B1.matrix().asDiagonal(), 0.));
            AddDecomposedGate(Gate(CNOT, clocation, Matrix2cd(), 0.));
            AddDecomposedGate(Gate(RY, ulocation, B2.matrix().asDiagonal(), 0.));
            AddDecomposedGate(Gate(CNOT, clocation, Matrix2cd(), 0.));
        }
        return QSD();
    }

    ArrayXcd C1 = B1(seqN(0,1<<(n-3),2));
    ArrayXcd C2 = B1(seqN(1,1<<(n-3),2));
    ArrayXcd D1 = (C1 + C2) / 2.;
    ArrayXcd D2 = (C1 - C2) / 2.;

    ArrayXcd E1 = B2(seqN(0,1<<(n-3),2));
    ArrayXcd E2 = B2(seqN(1,1<<(n-3),2));
    ArrayXcd F1 = (E1 - E2) / 2.;
    ArrayXcd F2 = (E1 + E2) / 2.;

    GateLocation u2location(glocation.begin(), glocation.end()-2);
    GateLocation c1location;
    c1location.push_back(glocation.back());
    c1location.push_back(glocation[0]);
    GateLocation c2location;
    c2location.push_back(glocation[glocation.size()-2]);
    c2location.push_back(glocation[0]);

    remain_gates.push_back(Gate(CNOT, c1location, GateMatrix(), 0.));
    remain_gates.push_back(Gate(gtype, u2location, F2.matrix().asDiagonal(), 0.));
    remain_gates.push_back(Gate(CNOT, c2location, GateMatrix(), 0.));
    remain_gates.push_back(Gate(gtype, u2location, F1.matrix().asDiagonal(), 0.));
    remain_gates.push_back(Gate(CNOT, c1location, GateMatrix(), 0.));
    remain_gates.push_back(Gate(gtype, u2location, D2.matrix().asDiagonal(), 0.));
    remain_gates.push_back(Gate(CNOT, c2location, GateMatrix(), 0.));
    remain_gates.push_back(Gate(gtype, u2location, D1.matrix().asDiagonal(), get<3>(gate)));

    return QSD();
}

void qsyn::qsyn::AddDecomposedGate(Gate gate) {
    using namespace std;
    GateType gtype = get<0>(gate);
    GateLocation glocation = get<1>(gate);
    phase += get<3>(gate);

    if (gtype == NONE) {
        CSD csdofgate(get<2>(gate));
        double a1 = arg(csdofgate.A1(0,0));
        double b1 = arg(csdofgate.B1(0,0));
        double a2 = arg(csdofgate.A2(0,0));
        double b2 = arg(csdofgate.B2(0,0));
        double c = arg(csdofgate.C(0)+csdofgate.S(0)*dcomplex(0.,1.));
        phase += a1 + a2;
        gates.push_back(ElementGate(NONE, glocation, 2. * c, b1 - a1, b2 - a2));
        return;
    }

    if (gtype == CNOT) {
        gates.push_back(ElementGate(CNOT, glocation, 0., 0., 0.));
        return;
    }

    GateAngle angle = real(get<2>(gate)(0,0));
    //if (fabs(angle) < eps) return;
    if (gtype == RZ || gtype == MZ) {
        phase -= angle/2.;
        gates.push_back(ElementGate(RZ, glocation, 0., 0., angle));
    } else if (gtype == RY || gtype == MY) {
        gates.push_back(ElementGate(RY, glocation, angle, 0., 0.));
    }
    return;
}


qsyn::DecomposedGates qsyn::simplify(DecomposedGates& gates){
        
    double esp = 1e-6;
    DecomposedGates sim_gates;

    for (int i = 0; i < gates.size(); i++){
        auto type = get<0>(gates[i]);
        auto pos = get<1>(gates[i]);
        bool flag = true;
        // adjacent cnot
        if (type == CNOT){
            if (sim_gates.size() > 0){
                auto pre = sim_gates.back();
                if (get<0>(pre) == CNOT){
                    auto pre_pos = get<1>(pre);
                    if (pos[0] == pre_pos[0] && pos[1] == pre_pos[1]){
                        sim_gates.pop_back();
                        flag = false;
                    }
                }
            }
        }else{
            // Identity
            
            if (abs(get<2>(gates[i])) < esp && abs(get<3>(gates[i])) < esp && abs(get<4>(gates[i])) < esp){
                flag = false;
            }
            else{  // adjacent U
                if (sim_gates.size() > 0){
                    auto pre = sim_gates.back();
                    auto pre_pos = get<1>(pre);
                    if (get<0>(pre) == type && pos[0] == pre_pos[0] && abs(get<2>(gates[i]) - get<2>(pre)) < esp && abs(get<3>(gates[i]) - get<3>(pre)) < esp && abs(get<3>(gates[i]) - get<3>(pre)) < esp){
                        sim_gates.pop_back();
                        flag = false;
                    }
                }
            }
        }

        if (flag)
            sim_gates.push_back(gates[i]);
    }

    return sim_gates;
}

MatrixXcd kron(MatrixXcd A, MatrixXcd B) {
    MatrixXcd AB = MatrixXcd::Zero(A.rows()* B.rows(), A.cols() * B.cols());
    for (int i=0; i<A.rows(); i++) {
        for (int j=0; j<A.cols(); j++) {
            AB.block(i*B.rows(), j*B.cols(), B.rows(), B.cols()) = A(i,j) * B;
        }
    }
    return AB;
}

Matrix2cd U3(double theta, double phi, double lambda) {
    Matrix2cd U {
        {cos(theta / 2.), -dcomplex(cos(lambda), sin(lambda))*sin(theta / 2.)},
        {dcomplex(cos(phi), sin(phi))*sin(theta / 2.), dcomplex(cos(phi + lambda), sin(phi + lambda))*cos(theta / 2.)}
    };
    return U;
}

bool qsyn::verify(int n, UnitaryVector& Uvector, DecomposedGates& gates, double phase) {
        
    MatrixXcd U(1<<n, 1<<n);
    for (int j=0; j<(1<<n); j++) {
        for (int k=0; k<(1<<n); k++) {
            U(j,k) = dcomplex(Uvector[j*(1<<n)+k].first, Uvector[j*(1<<n)+k].second);
        }
    }
    double esp = 1e-6;
    MatrixXcd I = MatrixXcd::Identity(1<<n, 1<<n);
    MatrixXcd V = MatrixXcd::Identity(1<<n, 1<<n);
    MatrixXcd I1 {
        {1.,0.},
        {0.,1.}
    };
    MatrixXcd X {
        {0.,1.},
        {1.,0.}
    };
    MatrixXcd a00 {
        {1.,0.},
        {0.,0.}
    };
    MatrixXcd a11 {
        {0.,0.},
        {0.,1.}
    };
    for (int j=0; j<gates.size(); j++) {
        //printf("%d  %d %lf\t%lf\t%lf\n", get<0>(gates[j]), get<1>(gates[j])[0], get<2>(gates[j]), get<3>(gates[j]), get<4>(gates[j]));
        
        if (get<0>(gates[j]) == CNOT) {
            MatrixXcd Temp = MatrixXd::Zero(4,2*n);
            for (int k=0; k<n; k++) {
                if (k == get<1>(gates[j])[0]) {
                    Temp.block(0,2*k,2,2) = a00;
                    Temp.block(2,2*k,2,2) = a11;
                } else if (k == get<1>(gates[j])[1]) {
                    Temp.block(0,2*k,2,2) = I1;
                    Temp.block(2,2*k,2,2) = X;
                } else {
                    Temp.block(0,2*k,2,2) = I1;
                    Temp.block(2,2*k,2,2) = I1;
                }
            }
            MatrixXcd Temp1 = MatrixXd::Identity(1,1);
            MatrixXcd Temp2 = MatrixXd::Identity(1,1);
            for (int k=0; k<n; k++) {
                Temp1 = kron(Temp1, Temp.block(0,2*k,2,2));
                Temp2 = kron(Temp2, Temp.block(2,2*k,2,2));
            }
            V = (Temp1 + Temp2) * V;
        } else {
            MatrixXcd Temp = MatrixXd::Zero(2,2*n);
            for (int k=0; k<n; k++) {
                if (k == get<1>(gates[j])[0]) {
                    Temp.block(0,2*k,2,2) = U3(get<2>(gates[j]), get<3>(gates[j]), get<4>(gates[j]));
                } else {
                    Temp.block(0,2*k,2,2) = I1;
                }
            }
            MatrixXcd Temp1 = MatrixXd::Identity(1,1);
            for (int k=0; k<n; k++) {
                Temp1 = kron(Temp1, Temp.block(0,2*k,2,2));
            }
            V = Temp1 * V;
        }
    }
    MatrixXcd A = dcomplex(cos(phase), sin(phase)) * V * U.conjugate().transpose();
    //std::cout << "Gate sequence * U^dagger" << std::endl << A.diagonal() << std::endl;
    A += -1.0*I;
    auto s = A.sum();
    if (abs(s.real()) < esp && abs(s.imag()) < esp)
        return true;
    return false;
}

