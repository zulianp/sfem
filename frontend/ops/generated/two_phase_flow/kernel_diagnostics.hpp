#ifndef SFEM_CODEGEN_KERNEL_DIAGNOSTICS_HPP
#define SFEM_CODEGEN_KERNEL_DIAGNOSTICS_HPP

#include <stddef.h>
#include <stdio.h>

#ifndef SFEM_INLINE
#define SFEM_INLINE inline
#endif

namespace sfem {
namespace codegen {

struct KernelDiagnostics {
    const char *kernel_name;
    const char *element_type;
    int dim;
    int n_qp;
    int n_shape;
    int vector_size;
    int quadrature_order;
    long add_instructions_per_qp_lane;
    long mul_instructions_per_qp_lane;
    long div_instructions_per_qp_lane;
    long sqrt_instructions_per_qp_lane;
    long pow_instructions_per_qp_lane;
    long exp_instructions_per_qp_lane;
    long log_instructions_per_qp_lane;
    long trig_instructions_per_qp_lane;
    long load_instructions_per_qp_lane;
    long store_instructions_per_qp_lane;
    long flops_per_qp_lane;
    long temporaries;
    long estimated_registers;
    int geometry_streams;
    int reference_scalars;
    int quadrature_weight_scalars;
    int material_scalars;
    int u_streams;
    int h_streams;
    int output_streams;
    int output_reads_per_element;
    int output_writes_per_element;
    double add_cpi;
    double mul_cpi;
    double div_cpi;
    double sqrt_cpi;
    double pow_cpi;
    double exp_cpi;
    double log_cpi;
    double trig_cpi;
    double load_cpi;
    double store_cpi;
};

static SFEM_INLINE double KernelDiagnostics_total_flops(
        const KernelDiagnostics *const d,
        const ptrdiff_t nelements) {
    const double n = nelements > 0 ? (double)nelements : 0.0;
    return n * (double)d->n_qp * (double)d->flops_per_qp_lane;
}

static SFEM_INLINE size_t KernelDiagnostics_total_bytes(
        const KernelDiagnostics *const d,
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    (void)accumulator_bytes;
    const size_t n = nelements > 0 ? (size_t)nelements : (size_t)0;
    const size_t geometry_bytes = n * (size_t)d->n_qp * (size_t)d->geometry_streams * scalar_bytes;
    const size_t field_bytes = n * (size_t)(d->u_streams + d->h_streams) * real_bytes;
    const size_t output_bytes = n * (size_t)d->output_streams * (size_t)(d->output_reads_per_element + d->output_writes_per_element) * real_bytes;
    const size_t reference_bytes = ((size_t)d->reference_scalars + (size_t)d->quadrature_weight_scalars + (size_t)d->material_scalars) * scalar_bytes;
    return geometry_bytes + field_bytes + output_bytes + reference_bytes;
}

static SFEM_INLINE double KernelDiagnostics_arithmetic_intensity(
        const KernelDiagnostics *const d,
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    const size_t bytes = KernelDiagnostics_total_bytes(d, nelements, scalar_bytes, real_bytes, accumulator_bytes);
    return bytes ? KernelDiagnostics_total_flops(d, nelements) / (double)bytes : 0.0;
}

static SFEM_INLINE void KernelDiagnostics_print_rate(
        const char *const name,
        const KernelDiagnostics *const d,
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    const double seconds_per_call = repeat > 0 ? elapsed / (double)repeat : 0.0;
    const double element_rate = seconds_per_call > 0.0 ? 1e-6 * (double)nelements / seconds_per_call : 0.0;
    const double dof_rate = seconds_per_call > 0.0 ? 1e-6 * (double)ndofs / seconds_per_call : 0.0;
    const double ai = KernelDiagnostics_arithmetic_intensity(
            d, nelements, scalar_bytes, real_bytes, accumulator_bytes);
    const double gflops = seconds_per_call > 0.0
            ? 1e-9 * KernelDiagnostics_total_flops(d, nelements) / seconds_per_call
            : 0.0;
    printf("%-72s %12.6e %16.3f %13.3f %10.3f %13.3f\n",
           name ? name : d->kernel_name,
           seconds_per_call, element_rate, dof_rate, ai, gflops);
}

} // namespace codegen
} // namespace sfem

#endif
