#ifndef _ISQ_QSYNTHESIS_H
#define _ISQ_QSYNTHESIS_H

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <vector>


namespace isq {
namespace ir{
namespace synthesis{
using namespace Eigen;
    using std::vector;
    using std::tuple;
    using std::pair;

    enum GateType {
        NONE, RX, RY, RZ, CNOT, MX, MY, MZ, TOFFOLI, H, X
    };

    typedef vector<int> GateLocation;
    typedef double GatePhase;
    typedef double GateAngle;
    typedef MatrixXcd GateMatrix;
    typedef tuple<GateType, GateLocation, GateMatrix, GatePhase> Gate;
    typedef vector<Gate> GateSequence;
    typedef pair<double, double> ComplexPair;
    typedef vector<ComplexPair> UnitaryVector;
    typedef tuple<GateType, GateLocation, GateAngle, GateAngle, GateAngle> ElementGate;
    typedef vector<ElementGate> DecomposedGates;
    typedef tuple<double, double, double, double> UAngle;

    // Single Pauli matrices
    // Matrix2cd Rx(double angle);
    // Matrix2cd Ry(double angle);
    // Matrix2cd Rz(double angle);

    // Cosine-Sine Decomposition
    class CSD {
        public:
            MatrixXcd A1;
            MatrixXcd A2;
            MatrixXcd B1;
            MatrixXcd B2;
            MatrixXcd C;
            MatrixXcd S;
            CSD(MatrixXcd U);
    };

    // Multiplexed-Pauli Decomposition
    // GateSequence MPD(std::vector<double> angles, GateLocation labQ, GateType P);

    // Quantum Shannon Decomposition
    // GateSequence& QSD(GateSequence& decomposed_gates, GateSequence& remain_gates);

    DecomposedGates simplify(DecomposedGates &gates);
    
    bool verify(int n, UnitaryVector& Uvector, DecomposedGates& gates, double phase);

    class QSynthesis {
        public:
            DecomposedGates gates;
            GatePhase phase;
            double eps;
            QSynthesis(int n, UnitaryVector uvector, double e=1e-16);
            void QSD();
        private:
            void AddDecomposedGate(Gate gate);
            GateSequence remain_gates;
            bool QSDBody();
    };

    DecomposedGates mcdecompose_u(UnitaryVector uvector, std::string ctrl);
    DecomposedGates mcdecompose_addone(int n);

}    

    //DecomposedGates Universal(int n, UnitaryVector uvector);
}
}

#endif