#ifndef _ISQ_QSIM_KERNEL_H
#define _ISQ_QSIM_KERNEL_H

// Only 61 qubits are supported, since we only have 64-bit addressing anyway.
#define MAX_SIMULATE_QUBITS 61
// This is the grid size that should be used.

#define USED_GRID_SIZE 1024
#define USED_BLOCK_SIZE 512
#define PROB_BUFFER_SIZE USED_GRID_SIZE

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#define NULLABLE = nullptr
#else
#include <stddef.h>
#include <stdint.h>
#define NULLABLE
#endif

// Type for indexing qubit.
typedef int qint_t;
// Type for indexing one amplitude.
typedef uint64_t qindex_t;
// Type for real/imag part of amplitude. Also for probability.
typedef float qamph_t;
#ifdef __cplusplus
extern "C" {
#endif

// Struct for a well-hidden qubit-state simulation.
struct qstate;
typedef struct qstate qstate;

size_t qstate_struct_size();
size_t qstate_align_size();
void qstate_init(qstate* state, qint_t capacity);
void qstate_deinit(qstate* state);
qint_t qstate_size(qstate* state);
qint_t qstate_measure(qstate* state, qint_t q, int is_reset, qamph_t seed, qamph_t* out_probs NULLABLE);
void qstate_u3(qstate* state, qint_t q, qamph_t mat[8]);
void qstate_cnot(qstate* state, qint_t q1, qint_t q2);
void qstate_debug_amps(qstate* state, qamph_t* amps, qindex_t size);
qint_t qstate_alloc(qstate* state);
qint_t qstate_swap_to_msb_and_free(qstate* state, qint_t q, qamph_t seed, qamph_t* out_probs NULLABLE);
#ifdef __cplusplus
}
#endif
#endif
