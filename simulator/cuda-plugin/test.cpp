#include <cstdlib>
#include <cstdio>
#include "qsim_kernel.h"
#include <complex>
#include <vector>
#include <cassert>
#include <algorithm>
#include <random>

class Test{
protected:
    const char* const testType;
    qstate* state;
    virtual bool run()=0;
    std::vector<std::complex<float>> debugState(){
        std::vector<std::complex<float>> result;
        auto size = 1ULL<<qstate_size(state);
        result.resize(size);
        qstate_debug_amps(state, (float*)&result[0], size);
        return result;
    }
    void dumpState(){
        auto state = debugState();
        for(auto i=0; i<state.size(); i++){
            printf("%f+(%f)i ", state[i].real(), state[i].imag());
        }
        printf("\n");
    }
public:
    Test(const char* testType, qint_t capacity): testType(testType){
        state = (qstate*)malloc(qstate_struct_size());
        qstate_init(state, capacity);
    }
    void start(){
        printf(">>>>>>>> Testing case: %s\n", testType);
        bool result = run();
        if(!result){
            fprintf(stderr,"<<<<<<<< Test failed! %s\n", testType);
            fflush(stderr);
            abort();
        }else{
            printf("<<<<<<<< Test success! %s\n", testType);
        }
    }
    virtual ~Test(){
        qstate_deinit(state);
    }
};

#define TEST(name, reason, capacity, body) \
class name : public Test{ \
public: \
    name () : Test(reason, capacity){} \
    bool run(){ \
        body \
        return true;\
    }\
};
#define CASE(...) __VA_ARGS__

TEST(TestHadamard, "Testing on one qubit.", 1, CASE({
    dumpState();
    auto qubit = qstate_alloc(state);
    dumpState();
    auto is2 = 0.7071067811865476;
    float hadamard[8] = {0.7071067811865476,0.0,
                        0.7071067811865476,0.0,
                        0.7071067811865476,0.0,
                        -0.7071067811865476,0.0};
    qstate_u3(state, qubit, hadamard);
    dumpState();
}));
TEST(TestBell, "Testing on two qubits.", 2, CASE({
    dumpState();
    auto q1 = qstate_alloc(state);
    auto q2 = qstate_alloc(state);
    dumpState();
    auto is2 = 0.7071067811865476;
    float hadamard[8] = {0.7071067811865476,0.0,
                        0.7071067811865476,0.0,
                        0.7071067811865476,0.0,
                        -0.7071067811865476,0.0};
    qstate_u3(state, q1, hadamard);
    dumpState();
    qstate_cnot(state, q1, q2);
    dumpState();
}));
TEST(TestGHZ, "Testing on three qubits.", 3, CASE({
    dumpState();
    auto q1 = qstate_alloc(state);
    auto q2 = qstate_alloc(state);
    auto q3 = qstate_alloc(state);
    dumpState();
    auto is2 = 0.7071067811865476;
    float hadamard[8] = {0.7071067811865476,0.0,
                        0.7071067811865476,0.0,
                        0.7071067811865476,0.0,
                        -0.7071067811865476,0.0};
    qstate_u3(state, q1, hadamard);
    dumpState();
    qstate_cnot(state, q1, q2);
    qstate_cnot(state, q2, q3);
    dumpState();
}));
TEST(TestGHZ20, "Testing on 28 qubits.", 28, CASE({
    auto q0 = qstate_alloc(state);
    float hadamard[8] = {0.7071067811865476,0.0,
                        0.7071067811865476,0.0,
                        0.7071067811865476,0.0,
                        -0.7071067811865476,0.0};
    qstate_u3(state, q0, hadamard);
    qint_t q27 = -1;
    for(auto i=1; i<27; i++){
        auto qi = qstate_alloc(state);
        qstate_cnot(state, qi-1, qi);
        q27 = qi;
    }
    //dumpState();
    auto host_buf = debugState();
    printf("%f %f\n", std::norm(host_buf[0]), std::norm(host_buf[host_buf.size()-1]));
    assert(std::abs(std::norm(host_buf[0])-0.5)<1e-3);
    assert(std::abs(std::norm(host_buf[host_buf.size()-1])-0.5)<1e-3);
}));
TEST(TestSwap, "Testing swapping a lot.", 10, CASE({
    

    std::vector<int> arr;
    std::vector<int> l2p;
    std::vector<int> p2l;
    for(auto i=0; i<10; i++){
        arr.push_back(qstate_alloc(state));
        l2p.push_back(i);
        p2l.push_back(i);
    }
    
    std::shuffle(arr.begin(), arr.end(), std::random_device());
    for(auto i: arr){
        printf("%d ", i);
    }
    puts("\n");
    auto q0 = arr[0];
    float hadamard[8] = {0.7071067811865476,0.0,
                        0.7071067811865476,0.0,
                        0.7071067811865476,0.0,
                        -0.7071067811865476,0.0};
    printf("Starting from %d\n", q0);
    qstate_u3(state, q0, hadamard);
    for(auto iq=arr.begin()+1; iq!=arr.end(); iq++){
        auto q=*iq;
        printf("Running from %d(%d) to %d(%d)\n", q0, l2p[q0], q, l2p[q]);
        qstate_cnot(state, l2p[q], l2p[q0]);
        qstate_cnot(state, l2p[q0], l2p[q]);
        qstate_cnot(state, l2p[q], l2p[q0]);
        dumpState();
        // trace out q0.
        float f;
        auto old_msb = qstate_swap_to_msb_and_free(state, l2p[q0], (rand()%100)/100.0, &f);
        printf("Prob[0] = %f\n", f);
        l2p[p2l[old_msb]] = l2p[q0];
        p2l[l2p[q0]] = p2l[old_msb];
        q0=q;
        dumpState();
    }
    assert(l2p[q0]==0);
    dumpState();
}));
int main(){
    TestHadamard().start();
    TestBell().start();
    TestGHZ().start();
    TestSwap().start();
    TestGHZ20().start();
}