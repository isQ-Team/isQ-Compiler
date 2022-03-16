#include "qsim_kernel.h"
#include <type_traits>
#include <cuda/std/complex>
#include <array>
#include <cassert>
#include <cstdio>
#include <algorithm>

using qamp_t = cuda::std::complex<qamph_t>;
static_assert(std::is_standard_layout<qamp_t>::value, "qamp_t must be standard layout.");

// assertion utilities.
#define gpuErrchk(ans) do{ gpuAssert((ans), __FILE__, __LINE__); }while(0)
#define gpuAssertOk() gpuErrchk(cudaPeekAtLastError())
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s at file %s, line %d\n", cudaGetErrorString(code), file, line);
      fflush(stderr);
      if (abort) exit(code);
   }
}


// Some qubits counting from LSB are considered warp-local.
// A measurement lying on non-warp-local qubits can perform amplitude sum-up by warp-shuffle.
// A single-qubit gate lying on warp-local qubits can update pairs of amplitude by a warp-shuffle.
// (A multi-controlled-single-qubit-gate, multi-controllers non-warp-local and controllee warp-local, can update by a warp-shuffle.)
#define WARP_SIZE 32
static const qint_t warpLocalQubits = (qint_t)log2(WARP_SIZE);
//static const qint_t maxIndexLength = 64;
constexpr __inline__ qint_t isWarpLocal(qint_t q){
    return q<warpLocalQubits;
}

struct qubit_enum_sq{
    qindex_t mask_hi;
    qindex_t mask_lo;
    qint_t offset;
    __device__
    qindex_t operator()(qindex_t id, bool flag){
        qindex_t s = flag;
        return ((id&mask_hi)<<1) | (s<<offset) | (id&mask_lo);
    }
    qubit_enum_sq(qint_t qubit){
        mask_lo = (((qindex_t)1)<<qubit)-1;
        offset = qubit;
        mask_hi = ~mask_lo;
    }
};
struct qubit_enum_2q{
    qindex_t mask_hi;
    qindex_t mask_mi;
    qindex_t mask_lo;
    qint_t offset_hi;
    qint_t offset_lo;
    __device__
    qindex_t operator()(qindex_t id, bool flag_hi, bool flag_lo){
        qindex_t s_hi = flag_hi;
        qindex_t s_lo = flag_lo;
        return ((id&mask_hi)<<2) | (s_hi<<offset_hi) | ((id&mask_mi)<<1) | (s_lo<<offset_lo) | (id&mask_lo);
    }
    qubit_enum_2q(qint_t qhi, qint_t qlo){
        assert(qhi>qlo);
        qindex_t one = 1;
        mask_lo = (one<<qlo)-1;
        mask_mi = ((one<<(qhi-1))-1) ^ mask_lo;
        mask_hi = ~(mask_lo | mask_mi);
        offset_lo = qlo;
        offset_hi = qhi;
    }
};
static_assert(std::is_standard_layout<qubit_enum_sq>::value, "qubit_enum_sq must be standard layout to pass to GPU.");
static_assert(std::is_standard_layout<qubit_enum_2q>::value, "qubit_enum_2q must be standard layout to pass to GPU.");

// matrix should be layouted in {m[0][0], m[0][1], m[1][1], m[1][0]}.
__inline__ __device__
qamp_t warpVec2Mul(qamp_t val, qamp_t m[4], qint_t offset, bool f){
    qamph_t other_val_r = __shfl_xor_sync(0xffffffff, val.real(), 1<<offset);
    qamph_t other_val_i = __shfl_xor_sync(0xffffffff, val.imag(), 1<<offset);
    qamp_t other_val(other_val_r, other_val_i);
    int base = ((int)f)<<1;
    return val*m[base+0]+other_val*m[base+1];
}

__inline__ __device__
qamp_t warpCNOT(qamp_t val, qindex_t amp_id, qint_t offset, qint_t ctrl_offset){
    unsigned mask = __ballot_sync(0xffffffff, amp_id & (1<<ctrl_offset));
    if(amp_id & (1<<ctrl_offset)){
        qamph_t other_val_r = __shfl_xor_sync(mask, val.real(), 1<<offset);
        qamph_t other_val_i = __shfl_xor_sync(mask, val.imag(), 1<<offset);
        qamp_t other_val(other_val_r, other_val_i);
        return other_val;
    }else{
        return val;
    }

}

struct m4{qamp_t m[4];};
// One thread is created for every amplitude.
// m is passed as {0,1,3,2}.
__global__ void single_qubit_gate_warped_qubit(qamp_t* data, qindex_t size, qint_t qubit, m4 m){
    qindex_t i = ((qindex_t)blockIdx.x)*((qindex_t)blockDim.x)+(qindex_t)threadIdx.x;
    qamp_t amp = 0;
    if(i<size){
        amp = data[i];
    }
    amp = warpVec2Mul(amp, m.m, qubit, (bool)(i&1));
    if(i<size){
        data[i]=amp;
    }
}

// One thread is created for half amplitudes.
// m is passed row-first.
__global__ void single_qubit_gate_nonwarped_qubit(qamp_t* data, qindex_t size, qindex_t thread_tot_count, qubit_enum_sq c, m4 m){
    qindex_t global_id = ((qindex_t)blockIdx.x)*((qindex_t)blockDim.x)+(qindex_t)threadIdx.x;
    if(global_id>=thread_tot_count) return;
    qindex_t i_0 = c(global_id, 0);
    qindex_t i_1 = c(global_id, 1);
    qamp_t a_0 = data[i_0];
    qamp_t a_1 = data[i_1];
    qamp_t b_0 = m.m[0]*a_0 + m.m[1]*a_1;
    qamp_t b_1 = m.m[2]*a_0 + m.m[3]*a_1;
    data[i_0]=b_0;
    data[i_1]=b_1;
}

// One thread is created for every amplitude.
__global__ void cnot_warped_qubit(qamp_t* data, qindex_t size, qint_t a, qint_t b){
    qindex_t i = ((qindex_t)blockIdx.x)*((qindex_t)blockDim.x)+(qindex_t)threadIdx.x;
    qamp_t amp = 0;
    if(i<size){
        amp = data[i];
    }
    amp = warpCNOT(amp, i, b, a);
    if(i<size){
        data[i]=amp;
    }
}

// One thread is created for 1/4 amplitudes.
// m is passed row-first.
template<bool IS_HI_CONTROLLER>
__global__ void cnot_nonwarped_qubit(qamp_t* data, qindex_t size, qindex_t thread_tot_count, qubit_enum_2q c){
    qindex_t global_id = ((qindex_t)blockIdx.x)*((qindex_t)blockDim.x)+(qindex_t)threadIdx.x;
    if(global_id>=thread_tot_count) return;
    qindex_t i_0;
    qindex_t i_1;
    if(IS_HI_CONTROLLER){
        i_0 = c(global_id, 1, 0);
        i_1 = c(global_id, 1, 1);
    }else{
        i_0 = c(global_id, 0, 1);
        i_1 = c(global_id, 1, 1);
    }

    qamp_t a_0 = data[i_0];
    qamp_t a_1 = data[i_1];
    data[i_0]=a_1;
    data[i_1]=a_0;
}

// Using warp-local ops for reduce-sum.
__inline__ __device__
float warpReduceSum(float val, int warpSize=warpSize) {
    #pragma unroll 10 // is enough.
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// One thread is created for half amplitudes of all data.
// Equation: BLOCK_SIZE*2*GRID_SIZE*GRID_DIM=total size
template<qindex_t BLOCK_SIZE, qindex_t GRID_SIZE, bool SIZE_NO_GREATER_THAN_TWO_BLOCKSIZE>
__global__ void sum_measured_amplitudes(qamp_t* data, qindex_t size, qamph_t* out){
    qindex_t tid = threadIdx.x;
    qindex_t global_id = ((qindex_t)blockIdx.x)*BLOCK_SIZE*2+(qindex_t)tid;
    __shared__ qamph_t prob[BLOCK_SIZE];
    prob[tid]=0;
    qamph_t block_prob = 0;
    if(SIZE_NO_GREATER_THAN_TWO_BLOCKSIZE){
        if(global_id<size){
            block_prob=cuda::std::norm(data[global_id]);
        }
    }else{
        qindex_t goffset=0;
        qindex_t gridElemCount = BLOCK_SIZE*2*GRID_SIZE;
        for(auto i=0; i<gridDim.x; i++){
            block_prob+=cuda::std::norm(data[goffset + global_id]);
            block_prob+=cuda::std::norm(data[goffset + global_id + BLOCK_SIZE]);
            goffset+=gridElemCount;
        }
    }

    prob[tid]=block_prob;
    __syncthreads();
    if(BLOCK_SIZE>=512){
        if(tid<256) prob[tid]+=prob[tid+256]; __syncthreads();
    }
    if(BLOCK_SIZE>=256){
        if(tid<128) prob[tid]+=prob[tid+128]; __syncthreads();
    }
    if(BLOCK_SIZE>=128){
        if(tid<64) prob[tid]+=prob[tid+64]; __syncthreads();
    }
    qamph_t result = 0.0;
    if(BLOCK_SIZE>=64){
        result+=warpReduceSum(prob[tid+32]);
    }
    result+=warpReduceSum(prob[tid]);
    if(tid==0){
        out[blockIdx.x] = result;
    }
}

// Collapse one measurement result.
// If the measured qubit is warp-local, use in-warp op.
// The operation is on every amplitude.
__global__ void measure_collapse_warped_qubit(qamp_t* data, qindex_t size, qint_t q, qamph_t prob_invsqrt, bool measurement_result, bool is_reset){
    qindex_t i = ((qindex_t)blockIdx.x)*((qindex_t)blockDim.x)+(qindex_t)threadIdx.x;
    qamp_t amp = 0;
    if(i<size){
        amp = data[i];
    }
    // First, zero out unmatched amplitude.
    bool matched_amplitude = ((bool)(threadIdx.x&(1<<q))) == measurement_result;
    if(!matched_amplitude){
        amp = 0;
    }
    
    if(is_reset && measurement_result){
        // Reset the qubit by performing X.
        qamph_t other_val_r = __shfl_xor_sync(0xffffffff, amp.real(), 1<<q);
        qamph_t other_val_i = __shfl_xor_sync(0xffffffff, amp.imag(), 1<<q);
        qamp_t other_val(other_val_r, other_val_i);
        amp=other_val;
    }
    if(i<size){
        data[i]=amp*prob_invsqrt;
    }
}
__global__ void measure_collapse_nonwarped_qubit(qamp_t* data, qindex_t size, qindex_t thread_tot_count, qubit_enum_sq c, qamph_t prob_invsqrt, bool measurement_result, bool is_reset){
    qindex_t global_id = ((qindex_t)blockIdx.x)*((qindex_t)blockDim.x)+(qindex_t)threadIdx.x;
    if(global_id>=thread_tot_count) return;
    qindex_t i_0 = c(global_id, 0);
    qindex_t i_1 = c(global_id, 1);
    qamp_t a_0 = data[i_0];
    qamp_t a_1 = data[i_1];
    qamp_t b_0 = a_0;
    qamp_t b_1 = a_1;
    if(measurement_result){
        b_0=0;
    }else{
        b_1=0;
    }
    if(measurement_result && is_reset){
        cuda::std::swap(b_0, b_1);
    }
    data[i_0]=b_0*prob_invsqrt;
    data[i_1]=b_1*prob_invsqrt;
}

typedef struct qstate{
    qint_t capacity;
    qint_t current_size;
    qamp_t* amplitudes;
    qamph_t* prob_buffer;
    qamph_t* host_prob_buffer;
    qindex_t amplitude_size() const{
        return ((qindex_t)1)<<((qindex_t)current_size);
    }
} qstate;


static_assert(std::is_pod<qstate>::value, "qstate must be a POD type.");
extern "C" {
size_t qstate_struct_size(){
    return sizeof(struct qstate);
}
size_t qstate_align_size(){
    return alignof(struct qstate);
}
void qstate_init(qstate* s, qint_t capacity){
    assert(capacity>0 && capacity<=MAX_SIMULATE_QUBITS);
    s->capacity = capacity;
    s->current_size = 0;
    s->amplitudes = nullptr;
    s->prob_buffer = nullptr;
    // Allocate all memory at once.
    if(capacity>0){
        gpuErrchk(cudaMalloc(&s->amplitudes, sizeof(qamp_t) * (((qindex_t)1)<<capacity)));
        gpuErrchk(cudaMemset(s->amplitudes, 0, sizeof(qamp_t)* (((qindex_t)1)<<capacity) /4));
        qamph_t one = 1.0;
        gpuErrchk(cudaMemcpy(s->amplitudes, &one, sizeof(qamph_t)*1, cudaMemcpyHostToDevice));
    }
    gpuErrchk(cudaMalloc(&s->prob_buffer, sizeof(qamph_t) * PROB_BUFFER_SIZE));
    s->host_prob_buffer = (qamph_t*)malloc(sizeof(qamph_t)*PROB_BUFFER_SIZE);
    assert(s->host_prob_buffer);
}
void qstate_deinit(qstate* s){
    assert(s->amplitudes);
    if(s->amplitudes){
        gpuErrchk(cudaFree(s->amplitudes));
    }
    if(s->prob_buffer){
        gpuErrchk(cudaFree(s->prob_buffer));
    }
}
qint_t qstate_size(qstate* state){
    assert(state->current_size<=MAX_SIMULATE_QUBITS);
    return state->current_size;
}

qint_t qstate_measure(qstate* state, qint_t q, int is_reset, qamph_t seed, qamph_t* out_probs){
    assert(q<state->current_size);
    // Compute on first half.
    auto size = state->amplitude_size();
    auto half_size = state->amplitude_size()/2;
    auto half_size_no_greater_than_two_blocksize = half_size<= USED_BLOCK_SIZE*2;
    if(half_size_no_greater_than_two_blocksize){
        sum_measured_amplitudes<USED_BLOCK_SIZE, PROB_BUFFER_SIZE, true> <<<PROB_BUFFER_SIZE, USED_BLOCK_SIZE>>>(state->amplitudes, half_size, state->prob_buffer);
        gpuAssertOk();
    }else{
        sum_measured_amplitudes<USED_BLOCK_SIZE, PROB_BUFFER_SIZE, false> <<<PROB_BUFFER_SIZE, USED_BLOCK_SIZE>>>(state->amplitudes, half_size, state->prob_buffer);
        gpuAssertOk();
    }
    gpuErrchk(cudaMemcpy(state->host_prob_buffer, state->prob_buffer, sizeof(qamph_t)*PROB_BUFFER_SIZE, cudaMemcpyDeviceToHost));
    double prob_zero = 0;
    for(auto i=0; i<PROB_BUFFER_SIZE; i++){
        prob_zero += state->host_prob_buffer[i];
    }
    assert(prob_zero >=0 && prob_zero < 1+1e-4);
    if(prob_zero>1.0) prob_zero=1.0;
    bool measured=seed>prob_zero;
    auto prob = measured?(1-prob_zero):prob_zero;
    auto prob_invsqrt = 1.0/std::sqrt(prob);
    if(out_probs){
        *out_probs=(qamph_t)prob_zero;
    }

    if(isWarpLocal(q)){
        auto block_size = std::max((qindex_t)WARP_SIZE, std::min(size, (qindex_t)USED_BLOCK_SIZE));
        auto grid_size = (size+block_size-1)/block_size;
        measure_collapse_warped_qubit<<<(unsigned int)grid_size, (unsigned int)block_size>>>(state->amplitudes, size, q, (qamph_t)prob_invsqrt, measured, is_reset);
        gpuAssertOk();
    }else{
        auto thread_tot_count = state->amplitude_size()/2;
        auto block_size = std::max((qindex_t)WARP_SIZE, std::min(thread_tot_count, (qindex_t)USED_BLOCK_SIZE));
        auto grid_size = (thread_tot_count+block_size-1)/block_size;
        measure_collapse_nonwarped_qubit<<<(unsigned int)grid_size, (unsigned int)block_size>>>(state->amplitudes, size,  thread_tot_count, q, (qamph_t)prob_invsqrt, measured, is_reset);
        gpuAssertOk();
    }
    return measured;
}

void qstate_u3(qstate* state, qint_t q, qamph_t mat[8]){
    assert(q<state->current_size);
    auto size = state->amplitude_size();

    struct m4 a_m4;
    qamp_t* m = a_m4.m;
    for(auto i=0; i<4; i++){
        m[i]=qamp_t(mat[2*i], mat[2*i+1]);
    }
    if(isWarpLocal(q)){
        // Create required mat.
        std::swap(m[2], m[3]);
        auto block_size = std::max((qindex_t)WARP_SIZE, std::min(size, (qindex_t)USED_BLOCK_SIZE));
        auto grid_size = (size+block_size-1)/block_size;
        single_qubit_gate_warped_qubit<<<(unsigned int)grid_size, (unsigned int)block_size>>>(state->amplitudes, size, q, a_m4);
        gpuAssertOk();
    }else{
        auto thread_tot_count = state->amplitude_size()/2;
        auto block_size = std::max((qindex_t)WARP_SIZE, std::min(thread_tot_count, (qindex_t)USED_BLOCK_SIZE));
        auto grid_size = (thread_tot_count+block_size-1)/block_size;
        single_qubit_gate_nonwarped_qubit<<<(unsigned int)grid_size, (unsigned int)block_size>>>(state->amplitudes, size, thread_tot_count, q, a_m4);
        gpuAssertOk();
    }
}
void qstate_cnot(qstate* state, qint_t q1, qint_t q2){
    assert(q1<state->current_size);
    assert(q2<state->current_size);
    assert(q1!=q2);
    auto size = state->amplitude_size();
    if(isWarpLocal(q1) && isWarpLocal(q2)){
        auto block_size = std::max((qindex_t)WARP_SIZE, std::min(size, (qindex_t)USED_BLOCK_SIZE));
        auto grid_size = (size+block_size-1)/block_size;
        cnot_warped_qubit<<<(unsigned int)grid_size, (unsigned int)block_size>>>(state->amplitudes, size, q1, q2);
        gpuAssertOk();
    }else{
        auto thread_tot_count = state->amplitude_size()/4;
        auto block_size = std::max((qindex_t)WARP_SIZE, std::min(thread_tot_count, (qindex_t)USED_BLOCK_SIZE));
        auto grid_size = (thread_tot_count+block_size-1)/block_size;
        auto hi = std::max(q1, q2);
        auto lo = std::min(q1, q2);
        auto is_hi_ctrl = hi==q1;
        if(is_hi_ctrl){
            cnot_nonwarped_qubit<true> <<<(unsigned int)grid_size, (unsigned int)block_size>>>(state->amplitudes, size, thread_tot_count, qubit_enum_2q(hi, lo));
            gpuAssertOk();
        }else{
            cnot_nonwarped_qubit<false> <<<(unsigned int)grid_size, (unsigned int)block_size>>>(state->amplitudes, size, thread_tot_count, qubit_enum_2q(hi, lo));
            gpuAssertOk();
        }
    }
}

void qstate_debug_amps(qstate* state, qamph_t* amps, qindex_t size){
    auto real_size = state->amplitude_size();
    assert(size>=real_size);
    gpuErrchk(cudaMemcpy(amps, state->amplitudes, sizeof(qamp_t)*real_size, cudaMemcpyDeviceToHost));
}

qint_t qstate_alloc(qstate* state){
    assert(state->current_size<=state->capacity);
    auto sz = state->current_size;
    auto half_real_size = state->amplitude_size();
    state->current_size++;
    assert(state->current_size<=state->capacity);
    // Memset the top half.
    gpuErrchk(cudaMemset(state->amplitudes + half_real_size, 0, sizeof(qamp_t)*half_real_size/4));
    return sz;
}

__global__ void transpose_two_qubits(qamp_t* data, qindex_t size, qindex_t thread_tot_count, qubit_enum_2q c){
    qindex_t global_id = ((qindex_t)blockIdx.x)*((qindex_t)blockDim.x)+(qindex_t)threadIdx.x;
    if(global_id>=thread_tot_count) return;
    qindex_t i_0;
    qindex_t i_1;
    i_0 = c(global_id, 1, 0);
    i_1 = c(global_id, 0, 1);
    qamp_t a_0 = data[i_0];
    qamp_t a_1 = data[i_1];
    data[i_0]=a_1;
    data[i_1]=a_0;
}

// The problem with freeing: it is more convenient to rotate the qubit onto MSB.
qint_t qstate_swap_to_msb_and_free(qstate* state, qint_t q, qamph_t seed, qamph_t* out_probs){
    assert(q>=0 && q<state->current_size);
    auto hi = state->current_size-1;
    auto lo = q;
    if(lo!=hi){
        // TODO: a better transpose algorithm, or a better way to free qubits?
        // Maybe qubit resource estimation can one day become a feature of isQ analysis...
        auto size = state->amplitude_size();
        auto thread_tot_count = state->amplitude_size()/4;
        auto block_size = std::max((qindex_t)WARP_SIZE, std::min(thread_tot_count, (qindex_t)USED_BLOCK_SIZE));
        auto grid_size = (thread_tot_count+block_size-1)/block_size;
        transpose_two_qubits <<<(unsigned int)grid_size, (unsigned int)block_size>>>(state->amplitudes, size, thread_tot_count, qubit_enum_2q(hi, lo));
        gpuAssertOk();
    }
    // Measure highest qubit out.
    qstate_measure(state, hi, true, seed, out_probs);
    // Erase highest amplitudes.
    state->current_size--;
    return hi;
}


}