#include <cmath>
#include <iostream>
#include "isq/QSynthesis.h"
#include <complex>
#include <assert.h>
#include <numeric>

using namespace Eigen;
using namespace std;

using namespace isq::ir::synthesis;

double eps = 1e-6;


void printGate(DecomposedGates &gatelist){

    for (auto gate: gatelist){
        
        switch (get<0>(gate))
        {
        case RX:
            cout << "rx";
            break;
        case RY:
            cout << "ry";
            break;
        case RZ:
            cout << "rz";
            break;
        case CPHASE:
            cout << "cphase";
            break;
        case H:
            cout << "h";
            break;
        case X:
            cout << "x";
            break;
        case CNOT:
            cout << "cnot";
            break;
        case NONE:
            cout << "u3";
            break;
        case TOFFOLI:
            cout << "toffoli";
            break;
        default:
            break;
        }

        cout << '(';
        for (auto loc: get<1>(gate)){
            cout << loc << ',';
        }
        cout << ")\n";

    }

}

Matrix2cd Rz(double theta) {
    Matrix2cd U {
        {dcomplex(cos(-1*theta / 2.), sin(-1*theta / 2.)), 0},
        {0, dcomplex(cos(theta / 2.), sin(theta / 2.))}
    };
    return U;
}

Matrix2cd Ry(double theta){
    Matrix2cd U {
        {cos(theta / 2.), -sin(theta / 2.)},
        {sin(theta / 2.), cos(theta / 2.)}
    };
    return U;
}

Matrix2cd getX(){
    Matrix2cd U {
        {0., 1.},
        {1., 0.}
    };
    return U;
}

bool very_close(Matrix2cd A, Matrix2cd B){
    auto C = A - B;
    auto s = C.cwiseAbs().sum();
    if (s > eps)
        return false;
    return true;
}

/*
U = eiα * Rz(β) * Ry(γ) * Rz(δ)
*/
UAngle ZYDecompose(UnitaryVector uvector){
    if (uvector.size() != 4){
        cout << "gate size error, only support 1 qbit gate\n";
        return UAngle(0., 0., 0., 0.);
    }
    
    double alpha=0. , beta=0., gamma =0. , delta = 0.;
    complex<double> a(uvector[0].first, uvector[0].second);
    complex<double> b(uvector[1].first, uvector[1].second);
    complex<double> c(uvector[2].first, uvector[2].second);
    complex<double> d(uvector[3].first, uvector[3].second);
    
    Matrix2cd U{
        {a, b},{c, d}
    };

    if (abs(b) + abs(c) < eps){
        beta = arg(d / a);
        alpha = arg(sqrt(d * a));
    }else if (abs(a) + abs(d) < eps){
        gamma = M_PI;
        beta = arg(c / (dcomplex(-1., 0.)*b));
        alpha = arg(sqrt(dcomplex(-1., 0.)*b*c));
    }else{
        beta = arg(sqrt(c*d / (dcomplex(-1., 0.)*a*b)));
        gamma = acos(real(dcomplex(2., 0.)*a*d/(a*d-b*c))-1);
        delta = arg(sqrt(dcomplex(-1., 0.)*b*d / (a*c)));
        alpha = arg(sqrt(a*d-b*c));
    }

    //cout << alpha << ' ' << beta << ' ' << gamma << ' ' << delta << endl;
    Matrix2cd Up = dcomplex(cos(alpha), sin(alpha)) * Rz(beta) * Ry(gamma) * Rz(delta);
    /*
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            cout << U(i, j) << ' ' << Up(i, j) << endl;
        }
    }*/
    //cout << "===============\n";
    if (!very_close(Up, U)){
        if (very_close(U, -1.*Up)){
            alpha += M_PI;
        }else{
            if (norm(U(0,0)-Up(0,0)) < eps){
                gamma = -gamma;
            }else if (norm(U(0,1)-Up(0,1)) < eps){
                gamma = 2.*M_PI - gamma;
            }else if (norm(U(0,1)-dcomplex(0.,1.)*Up(0,1)) < eps){
                delta += M_PI;
                if (!(norm(U(0,0)-dcomplex(0.,-1.)*Up(0,0)) < eps)){
                    gamma = 2.*M_PI - gamma;
                }
            }else{
                beta += M_PI;
                if (!(norm(U(0,0)-dcomplex(0.,-1.)*Up(0,0)) < eps)){
                    gamma = 2.*M_PI - gamma;
                }
            }
        }
    }

    //cout << alpha << ' ' << beta << ' ' << gamma << ' ' << delta << endl;
    Up = dcomplex(cos(alpha), sin(alpha)) * Rz(beta) * Ry(gamma) * Rz(delta);
    /*
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            cout << U(i, j) << ' ' << Up(i, j) << endl;
        }
    }
    */
    //cout<<"alpha:"<< alpha<<"\n";
    //cout<<"beta:"<< beta<<"\n";
    //cout<<"gamma:"<< gamma<<"\n";
    //cout<<"delta:"<< delta<<"\n";
    if (!very_close(Up, U)){
        assert(0 && "z-y decompose failed");
        cout << "z-y decompose error\n";
        return UAngle(0.,0.,0.,0.);
    }
    return UAngle(alpha, beta, gamma, delta);
}

// decopose multi control X with multi ancilla
/*
A,B,C,D,E is control qbit, T is target qbit, x1, x2, x3 is ancilla

A   --●--    -----------●-----------------●--
      |                 |                 |
B   --●--    -----------●-----------------●--
      |                 |                 |
x1  -----    --------●--⊕--●-----------●--⊕--●
      |              |     |           |     |
C   --●--    --------●-----●-----------●-----●
      |              |     |           |     |
x2  ----- =  -----●--⊕-----⊕--●-----●--⊕-----⊕--●
      |           |           |     |           |
D   --●--    -----●-----------●-----●-----------●
      |           |           |     |           |
x3  -----    --●--⊕-----------⊕--●--⊕-----------⊕
      |        |                 |
E   --●--    --●-----------------●---------------
      |        |                 |
T   --⊕--    --⊕-----------------⊕---------------  
*/
DecomposedGates mcdecompose_with_mcancilla(GateLocation q, GateLocation a){

    DecomposedGates gatelist;
    int size = q.size();

    if (size == 2){
        assert(q.size()==2);
        gatelist.push_back(ElementGate(CNOT, q, 0., 0., 0.));
    }else if (size == 3){
        assert(q.size()==3);
        gatelist.push_back(ElementGate(TOFFOLI, q, 0., 0., 0.));
    }else{
        int iq = size - 3;
        int ia = 1;
        DecomposedGates midgate;
        while (iq > 1){
            midgate.push_back(ElementGate(TOFFOLI, {q[iq], a[ia], a[ia-1]}, 0., 0., 0.));
            iq -= 1;
            ia += 1;
        }
        gatelist.push_back(ElementGate(TOFFOLI, {q.at(size-2), a.at(0), q.at(size-1)}, 0., 0., 0.));
        for(auto& g: midgate){
            gatelist.push_back(g);
        }
        //gatelist.insert(gatelist.end(), midgate.begin(), midgate.end());
        gatelist.push_back(ElementGate(TOFFOLI, {q.at(0), q.at(1), a.at(ia-1)}, 0., 0., 0.));
        reverse(midgate.begin(), midgate.end());
        for(auto& g: midgate){
            gatelist.push_back(std::move(g));
        }
        //gatelist.insert(gatelist.end(), midgate.begin(), midgate.end());
        auto gatelist_new = gatelist;
        for(auto& g: gatelist_new){
            gatelist.push_back(std::move(g));
        }
        //gatelist.insert(gatelist.end(), gatelist.begin(), gatelist.end());
        
    }

    return gatelist;
}

// decompose multi control X with an ancilla
/*
A,B,C,D,E is control qbit, T is target qbit, a is ancilla

A   --●--    --●-----●-------
      |        |     |
B   --●--    --●-----●-------
      |        |     |
C   --●--    --●-----●-------
      |        |     |
D   --●--  = -----●-----●----
      |        |  |  |  |
E   --●--    -----●-----●----
      |        |  |  |  |
T   --⊕--    -----⊕-----⊕---- 
               |  |  |  |
a            --⊕--●--⊕--●----
*/
DecomposedGates mcdecompose_with_ancilla(GateLocation q, int a){
    
    DecomposedGates gatelist;
    int size = q.size();
    
    if (size <= 3){
        return mcdecompose_with_mcancilla(q, {a});
    }else{
        int l = (size + 1) / 2;
        GateLocation lq(q.begin(), q.begin()+l);
        GateLocation rq(q.begin()+l, q.end());
        lq.push_back(a);
        auto lgate = mcdecompose_with_mcancilla(lq, rq);
        lq.pop_back();
        rq.insert(rq.begin(), a);
        auto rgate = mcdecompose_with_mcancilla(rq, lq);

        gatelist.insert(gatelist.end(), lgate.begin(), lgate.end());
        gatelist.insert(gatelist.end(), rgate.begin(), rgate.end());
        gatelist.insert(gatelist.end(), lgate.begin(), lgate.end());
        gatelist.insert(gatelist.end(), rgate.begin(), rgate.end());
    }

    return gatelist;
}

// decompose +1 with one ancilla
/*                                          __
v0 --●--●--●--⊕     -----------●--------●--|  |
     |  |  |                   |        |  |+1|
v1 --●--●--⊕---     -----------●--------●--|__|
     |  |              __      |   __   |
v2 --●--⊕------  =  --|  |--⊕-----|  |-----⊕--
     |                |+1|  |  |  |+1|  |  |
v3 --⊕---------     --|__|--⊕-----|__|-----⊕--
                        |   |  |    |   |  |
g  ------------     ----●---●--⊕----●---⊕--●--

                 __
g   ---●--    --|  |--⊕
      _|_       |  |
v1  --|  |  = --|+1|--
      |+1|      |  |
v2  --|__|    --|__|--
*/
DecomposedGates mcdecompose_add_one(GateLocation q, int a){

    DecomposedGates gatelist;
    int size = q.size();
    
    if (size == 1){
        gatelist.push_back(ElementGate(X, {q[0]}, 0., 0., 0.));
    }else if (size == 2){
        gatelist.push_back(ElementGate(CNOT, q, 0., 0., 0.));
        gatelist.push_back(ElementGate(X, {q[0]}, 0., 0., 0.));
    }else if (size == 3){
        gatelist.push_back(ElementGate(TOFFOLI, q, 0., 0., 0.));
        GateLocation loc1(q.begin(), q.end()-1);
        gatelist.push_back(ElementGate(CNOT, loc1, 0., 0., 0.));
        gatelist.push_back(ElementGate(X, {q[0]}, 0., 0., 0.));
    }else{
        int l = (size+1) / 2;
        
        GateLocation rq(q.begin()+l, q.end());
        rq.insert(rq.begin(), a);
        auto radd = mcdecompose_add_one(rq, q[0]);
        radd.push_back(ElementGate(X, {a}, 0., 0., 0.));
        
        GateLocation lq(q.begin(), q.begin()+l);
        lq.push_back(a);
        auto clist = mcdecompose_with_ancilla(lq, q[l]);
        lq.pop_back();

        gatelist.insert(gatelist.end(), radd.begin(), radd.end());
        for (int i = l; i < size; i++){
            gatelist.push_back(ElementGate(CNOT, {a, q[i]}, 0., 0., 0.));
        }
        gatelist.insert(gatelist.end(), clist.begin(), clist.end());
        gatelist.insert(gatelist.end(), radd.begin(), radd.end());
        gatelist.insert(gatelist.end(), clist.begin(), clist.end());
        for (int i = l; i < size; i++){
            gatelist.push_back(ElementGate(CNOT, {a, q[i]}, 0., 0., 0.));
        }

        auto ladd = mcdecompose_add_one(lq, a);
        gatelist.insert(gatelist.end(), ladd.begin(), ladd.end());
    }

    return gatelist;
}

// decompose multi control z
// note!!! Here Z is actually controlled-GPhase.
// Z = |111><111|e^{i\theta} + \sum |k><k|
// This should not be used to decompose real controlled-Z gate.
/*
A(n qbit array) is control qbit, T is target qbit
                                                  __             __
                                               --|  |-----------|  |--Z(θ/8)--
A   n--●--    n----●-----------●---Z(θ/2)-     --|+1|--Z(-θ/8)--|-1|--Z(θ/8)--
       |    =      |           |            =  --|  |--Z(-θ/4)--|  |--Z(θ/4)--
T    -Z(θ)-    ----⊕--Z(-θ/2)--⊕---Z(θ/2)-     --|__|--Z(-θ/2)--|__|--Z(θ/2)--

*/

DecomposedGates mcdecompose_z(double theta, GateLocation q, int a){

    DecomposedGates gatelist;
    int size = q.size();
    
    if (size == 1){
        gatelist.push_back(ElementGate(CPHASE, q, theta, 0., 0.));
    }else{
        auto add = mcdecompose_add_one(q, a);
        gatelist.insert(gatelist.end(), add.begin(), add.end());

        double n_theta = theta;
        for (int i = size-1; i > 0; i--){
            n_theta /= 2.0;
            gatelist.push_back(ElementGate(CPHASE, {q[i]}, -n_theta, 0., 0.));
        }
        
        reverse(add.begin(), add.end());
        gatelist.insert(gatelist.end(), add.begin(), add.end());
        
        n_theta = theta;
        for (int i = size-1; i > 0; i--){
            n_theta /= 2.0;
            gatelist.push_back(ElementGate(CPHASE, {q[i]}, n_theta, 0., 0.));
        }
        gatelist.push_back(ElementGate(CPHASE, {q[0]}, n_theta, 0., 0.));
    }

    return gatelist;
}

// decompose multi control x without ancilla
/*
A(n qbit array),B, is control qbit, T is target qbit

A   n--●--    n--●----------●--------------●-----
       |         |          |              |
B    --●--  =  --⊕----●-----⊕-----●--------------
       |              |           |        |
T    --⊕--     -H----√Z† ------- √Z ----- √Z --H- 

B,T can be ancilla when do next decompose

*/
DecomposedGates mcdecompose_x(GateLocation q){

    DecomposedGates gatelist;
    int size = q.size();

    if (size == 2){
        gatelist.push_back(ElementGate(CNOT, q, 0., 0., 0.));
    }else if (size == 3){
        gatelist.push_back(ElementGate(TOFFOLI, q, 0., 0., 0.));
    }else{
        gatelist.push_back(ElementGate(H, {q[size-1]}, 0., 0., 0.));

        GateLocation loc(q.begin(), q.end()-1);
        auto xlist = mcdecompose_with_ancilla(loc, q[size-1]);
        gatelist.insert(gatelist.end(), xlist.begin(), xlist.end());

        auto zlist1 = mcdecompose_z(-M_PI / 2., {q[size-2], q[size-1]}, q[0]);
        gatelist.insert(gatelist.end(), zlist1.begin(), zlist1.end());

        gatelist.insert(gatelist.end(), xlist.begin(), xlist.end());

        auto zlist2 = mcdecompose_z(M_PI / 2., {q[size-2], q[size-1]}, q[0]);
        gatelist.insert(gatelist.end(), zlist2.begin(), zlist2.end());
        
        GateLocation loc2(q.begin(), q.end()-2);
        loc2.push_back(q[size-1]);
        auto zlist3 = mcdecompose_z(M_PI / 2., loc2, q[size-2]);
        gatelist.insert(gatelist.end(), zlist3.begin(), zlist3.end());

        gatelist.push_back(ElementGate(H, {q[size-1]}, 0., 0., 0.));
    }

    return gatelist;
}


// decompose c-ry/c-rz
/*
X     --●--    --●------------●---
        |    =   |            |       
T   --Rz(θ)--  --⊕--Rz(-θ/2)--⊕--Rz(θ/2)--
*/
// decompose c-rx
/*
Rx(θ) = Rz(-pi/2)Ry(θ)Rz(pi/2)

X     --●--    --------------●------------●---Z(pi/2)---
        |    =               |            |       
T   --Rx(θ)--  --Rz(-pi/2)---⊕--Ry(-θ/2)--⊕--Ry(θ/2)Rz(pi/2)--
*/
// decompose c-cphase
/*

X    --●--     ----●------------●--Z(θ/2)- 
       |    =      |            |            
T    -Z(θ)-    ----⊕--RZ(-θ/2)--⊕--RZ(θ/2)-
*/
DecomposedGates ctrl_r(GateType g, GateLocation q, double theta){
    assert(((g == RY) || (g == RZ) || (g == RX) || (g == CPHASE)));
    DecomposedGates gatelist;
    if (g == RX){
        gatelist.push_back(ElementGate(RZ, {q[1]}, -1.*M_PI / 2, 0., 0.));
        gatelist.push_back(ElementGate(CNOT, {q[0], q[1]}, 0., 0., 0.));
        gatelist.push_back(ElementGate(RY, {q[1]}, -1.*theta / 2, 0., 0.));
        gatelist.push_back(ElementGate(CNOT, {q[0], q[1]}, 0., 0., 0.));
        gatelist.push_back(ElementGate(RY, {q[1]}, theta / 2, 0., 0.));
        gatelist.push_back(ElementGate(RZ, {q[1]}, M_PI / 2, 0., 0.));
    }else if (g == CPHASE) {
        gatelist.push_back(ElementGate(CNOT, {q[0], q[1]}, 0., 0., 0.));
        gatelist.push_back(ElementGate(RZ, {q[1]}, -1.*theta / 2, 0., 0.));
        gatelist.push_back(ElementGate(CNOT, {q[0], q[1]}, 0., 0., 0.));
        gatelist.push_back(ElementGate(RZ, {q[1]}, theta / 2, 0., 0.));
        gatelist.push_back(ElementGate(CPHASE, {q[0]}, theta / 2, 0., 0.));
    }else{
        gatelist.push_back(ElementGate(CNOT, {q[0], q[1]}, 0., 0., 0.));
        gatelist.push_back(ElementGate(g, {q[1]}, -1.*theta / 2, 0., 0.));
        gatelist.push_back(ElementGate(CNOT, {q[0], q[1]}, 0., 0., 0.));
        gatelist.push_back(ElementGate(g, {q[1]}, theta / 2, 0., 0.));
    }
    return gatelist;
}

vector<int> isq::ir::synthesis::generate_gray_code(int num_bit){
    
    assert(num_bit >= 1);
    vector<int> gray_code = {0};
    
    for (auto i=0; i < num_bit; i++){
        for (auto j=(1<<i)-1; j >= 0; j--){
            gray_code.push_back(gray_code[j]+(1<<i));
        }
    }

    return gray_code;
}

int isq::ir::synthesis::last_one_idx(int x, int n){
    for (auto i=n-1; i>=0; i--){
        if (((1 << i) & x) > 0) return (n-1-i);
    }
    return -1;
}

int isq::ir::synthesis::get_one_count(int x, int n){
    int cnt = 0;
    for (auto i=n-1; i>=0; i--){
        if (((1 << i) & x) > 0) cnt += 1;
    }
    return cnt;
}

DecomposedGates mcr_graycode(GateType g,  double theta, GateLocation ctrl, int target){

    DecomposedGates gatelist;

    int n = ctrl.size();
    if (n == 0){
        gatelist.push_back(ElementGate(g, {target}, theta, 0., 0.));
    }
    auto gray_code = generate_gray_code(n);
    int last_pattern = -1;

    for (auto i = 0; i < (1 << n); i++){
        if (i == 0) continue;
        int pattern = gray_code[i];

        if (last_pattern == -1) last_pattern = pattern;
        int lm_pos = last_one_idx(pattern, n);
        int pos = last_one_idx(last_pattern ^ pattern, n);
        //cout << last_pattern << ' ' << pattern << ' ' << lm_pos << ' ' << pos << '\n';
        if (pos > -1){
            if (lm_pos != pos){
                gatelist.push_back(ElementGate(CNOT, {ctrl[pos], ctrl[lm_pos]}, 0., 0., 0.));
            }else{
                for (auto j=n-1; j>=0; j--){
                    if (((1 << j) & pattern) > 0){
                        if ((n-1-j) == lm_pos) continue;
                        gatelist.push_back(ElementGate(CNOT, {ctrl[n-1-j], ctrl[lm_pos]}, 0., 0., 0.));
                    }
                }
            }
        }

        int cnt = get_one_count(pattern, n);
        auto cr_theta = theta;
        if (cnt % 2 == 0){
            cr_theta *= -1.0;
        }
        auto sub = ctrl_r(g, {ctrl[lm_pos], target}, cr_theta);
        gatelist.insert(gatelist.end(), sub.begin(), sub.end());

        last_pattern = pattern;
    }

    return gatelist;

}

// decompose multi control U without ancilla
/*
X(n qbit array) is control qbit, T is target qbit

ABC = I
eiθ*AXBXC = U

X   n--●--    n------●-------●---Z(θ)-
       |    =        |       |       
T    --U--     --C---⊕---B---⊕---A-----
*/
DecomposedGates isq::ir::synthesis::mcdecompose_u(UnitaryVector uvector, string ctrl){

    assert(uvector.size() == 4);
    
    DecomposedGates gatelist;

    for (int i = 0; i < ctrl.size(); i++){
        if (ctrl[i] == 'f'){
            gatelist.push_back(ElementGate(X, {i}, 0., 0., 0.));
        }
    }

    complex<double> a(uvector[0].first, uvector[0].second);
    complex<double> b(uvector[1].first, uvector[1].second);
    complex<double> c(uvector[2].first, uvector[2].second);
    complex<double> d(uvector[3].first, uvector[3].second);
    
    Matrix2cd U{
        {a, b},{c, d}
    };

    int m = ctrl.size();

    if (m <= 8){
        //use mcr_graycode
        // U = eiα * Rz(β) * Ry(γ) * Rz(δ)
        // C(m)U = C(m-1)CPHASE(α) * C(m)Rz(β) * C(m)Ry(γ) * C(m)Rz(δ)
        auto angle = ZYDecompose(uvector);
        auto alpha = get<0>(angle);
        auto beta = get<1>(angle);
        auto gamma = get<2>(angle);
        auto delta = get<3>(angle);

        GateLocation q(m);
        iota(q.begin(), q.end(), 0);

        if (abs(delta) > eps){
            delta /= (1 << (m-1));
            auto gl = mcr_graycode(RZ, delta, q, m);
            gatelist.insert(gatelist.end(), gl.begin(), gl.end());
        }
        if (abs(gamma) > eps){
            gamma /= (1 << (m-1));
            auto gl = mcr_graycode(RY, gamma, q, m);
            gatelist.insert(gatelist.end(), gl.begin(), gl.end());
        }
        if (abs(beta) > eps){
            beta /= (1 << (m-1));
            auto gl = mcr_graycode(RZ, beta, q, m);
            gatelist.insert(gatelist.end(), gl.begin(), gl.end());
        }
        if (abs(alpha) > eps){
            if (m == 1) gatelist.push_back(ElementGate(CPHASE, {0}, alpha, 0., 0.));
            else{
                alpha /= (1 << (m-2));
                GateLocation loc(q.begin(), q.end()-1);
                auto gl = mcr_graycode(CPHASE, alpha, loc, q[m-1]);
                gatelist.insert(gatelist.end(), gl.begin(), gl.end());
            }
        }
    }
    else{

        int shape = ctrl.size()+1;
        GateLocation q(shape);
        iota(q.begin(), q.end(), 0);

        if (very_close(U, getX())){
            auto xlist = mcdecompose_x(q);
            gatelist.insert(gatelist.end(), xlist.begin(), xlist.end());
        }else{
        
            auto xlist = mcdecompose_x(q);
            
            auto angle = ZYDecompose(uvector);
            auto alpha = get<0>(angle);
            auto beta = get<1>(angle);
            auto gamma = get<2>(angle);
            auto delta = get<3>(angle);
            //cout << get<0>(uvector[0])<<alpha << ' ' << beta << ' ' << gamma << ' ' << delta << endl;
            gatelist.push_back(ElementGate(NONE, {shape-1}, 0., (delta - beta) / 2., 0.));
            gatelist.insert(gatelist.end(), xlist.begin(), xlist.end());
            gatelist.push_back(ElementGate(NONE, {shape-1}, -1.*gamma / 2., 0., -1.*(delta+beta) / 2.));
            gatelist.insert(gatelist.end(), xlist.begin(), xlist.end());
            gatelist.push_back(ElementGate(NONE, {shape-1}, gamma / 2., beta, 0.));
            
            GateLocation loc(q.begin(), q.end()-1);
            auto zlist = mcdecompose_z(get<0>(angle), loc, shape-1);
            gatelist.insert(gatelist.end(), zlist.begin(), zlist.end());
        }
        
    }

    for (int i = 0; i < ctrl.size(); i++){
        if (ctrl[i] == 'f'){
            gatelist.push_back(ElementGate(X, {i}, 0., 0., 0.));
        }
    }

    //printGate(gatelist);
    return gatelist;
}

// external addone function, use one ancilla (the last one)
DecomposedGates isq::ir::synthesis::mcdecompose_addone(int n){

    if (n <= 2){
        return {};
    }
    
    GateLocation q(n-1);
    iota(q.begin(), q.end(), 0);
    return mcdecompose_add_one(q, n-1);
}