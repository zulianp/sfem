#ifndef SFEM_CODEGEN_KERNEL_DIAGNOSTICS_CUH
#define SFEM_CODEGEN_KERNEL_DIAGNOSTICS_CUH

#include <stddef.h>
#include <cstdio>
#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define SFEM_CODEGEN_HAS_DEVICE_RUNTIME
#endif
#endif

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif

#ifndef SFEM_FAILURE
#define SFEM_FAILURE 1
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define SFEM_CODEGEN_DEVICE_INLINE __host__ __device__ __forceinline__
#define SFEM_CODEGEN_HOST_INLINE __host__ __forceinline__
#else
#define SFEM_CODEGEN_DEVICE_INLINE inline
#define SFEM_CODEGEN_HOST_INLINE inline
#endif

namespace sfem {
namespace codegen {

//! Reports a dispatch that has no kernel for this combination.
//!
//! One function rather than the five-line `std::fprintf` every
//! dispatch entry point used to carry: there were 248 copies of it,
//! differing only in the name they print.
static SFEM_CODEGEN_HOST_INLINE int unsupported_dispatch(
    const char *const name,
    const int element_type,
    const int real_type) {
  std::fprintf(stderr,
      "%s does not support element type %d with real type %d\n",
      name, element_type, real_type);
  return SFEM_FAILURE;
}

#ifdef SFEM_CODEGEN_HAS_DEVICE_RUNTIME
//! Reports a kernel launch that did not start.
//!
//! Without this an entry point returns `SFEM_SUCCESS` for a launch it
//! never checked, and the sticky error surfaces much later, somewhere
//! unrelated, as a wrong number rather than as a failure.
static SFEM_CODEGEN_HOST_INLINE int launch_status(const char *const name) {
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s launch failed: %s\n", name, cudaGetErrorString(status));
    return SFEM_FAILURE;
  }
  return SFEM_SUCCESS;
}
#endif

struct KernelDiagnostics {
  const char *kernel_name;
  const char *element_type;
  int dim;
  int n_qp;
  int n_shape;
  int vector_size;
  int quadrature_order;
  long add_instructions_per_qp_scalar;
  long mul_instructions_per_qp_scalar;
  long div_instructions_per_qp_scalar;
  long sqrt_instructions_per_qp_scalar;
  long pow_instructions_per_qp_scalar;
  long exp_instructions_per_qp_scalar;
  long log_instructions_per_qp_scalar;
  long trig_instructions_per_qp_scalar;
  long load_instructions_per_qp_scalar;
  long store_instructions_per_qp_scalar;
  long flops_per_qp_scalar;
  long affine_mesh_flops_per_element;
  long isoparametric_mesh_flops_per_element;
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

static SFEM_CODEGEN_DEVICE_INLINE double KernelDiagnostics_total_flops(
    const KernelDiagnostics *const d,
    const ptrdiff_t nelements) {
  const double n = nelements > 0 ? (double)nelements : 0.0;
  return n * ((double)d->n_qp * (double)d->flops_per_qp_scalar + (double)d->isoparametric_mesh_flops_per_element);
}

static SFEM_CODEGEN_DEVICE_INLINE double KernelDiagnostics_total_flops_affine_mesh(
    const KernelDiagnostics *const d,
    const ptrdiff_t nelements) {
  const double n = nelements > 0 ? (double)nelements : 0.0;
  return n * ((double)d->n_qp * (double)d->flops_per_qp_scalar + (double)d->affine_mesh_flops_per_element);
}

static SFEM_CODEGEN_DEVICE_INLINE double KernelDiagnostics_total_flops_isoparametric_mesh(
    const KernelDiagnostics *const d,
    const ptrdiff_t nelements) {
  const double n = nelements > 0 ? (double)nelements : 0.0;
  return n * ((double)d->n_qp * (double)d->flops_per_qp_scalar + (double)d->isoparametric_mesh_flops_per_element);
}

static SFEM_CODEGEN_DEVICE_INLINE size_t KernelDiagnostics_total_bytes(
    const KernelDiagnostics *const d,
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t) {
  const size_t n = nelements > 0 ? (size_t)nelements : (size_t)0;
  const size_t geometry_bytes = n * (size_t)d->n_qp * (size_t)d->geometry_streams * scalar_bytes;
  const size_t field_bytes = n * (size_t)(d->u_streams + d->h_streams) * real_bytes;
  const size_t output_bytes = n * (size_t)(d->output_reads_per_element + d->output_writes_per_element) * real_bytes;
  const size_t reference_bytes = ((size_t)d->reference_scalars + (size_t)d->quadrature_weight_scalars + (size_t)d->material_scalars) * scalar_bytes;
  return geometry_bytes + field_bytes + output_bytes + reference_bytes;
}

static SFEM_CODEGEN_DEVICE_INLINE size_t KernelDiagnostics_total_bytes_affine_mesh(
    const KernelDiagnostics *const d,
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t) {
  const size_t n = nelements > 0 ? (size_t)nelements : (size_t)0;
  const size_t geometry_bytes = n * (size_t)(d->dim * d->dim + 1) * scalar_bytes;
  const size_t field_bytes = n * (size_t)(d->u_streams + d->h_streams) * real_bytes;
  const size_t output_bytes = n * (size_t)(d->output_reads_per_element + d->output_writes_per_element) * real_bytes;
  const size_t reference_bytes = ((size_t)d->reference_scalars + (size_t)d->quadrature_weight_scalars + (size_t)d->material_scalars) * scalar_bytes;
  return geometry_bytes + field_bytes + output_bytes + reference_bytes;
}

static SFEM_CODEGEN_DEVICE_INLINE size_t KernelDiagnostics_total_bytes_isoparametric_mesh(
    const KernelDiagnostics *const d,
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t) {
  const size_t n = nelements > 0 ? (size_t)nelements : (size_t)0;
  const size_t geometry_bytes = n * (size_t)d->dim * (size_t)d->n_shape * scalar_bytes;
  const size_t field_bytes = n * (size_t)(d->u_streams + d->h_streams) * real_bytes;
  const size_t output_bytes = n * (size_t)(d->output_reads_per_element + d->output_writes_per_element) * real_bytes;
  const size_t reference_bytes = ((size_t)d->reference_scalars + (size_t)d->quadrature_weight_scalars + (size_t)d->material_scalars) * scalar_bytes;
  return geometry_bytes + field_bytes + output_bytes + reference_bytes;
}

static SFEM_CODEGEN_DEVICE_INLINE double KernelDiagnostics_arithmetic_intensity(
    const KernelDiagnostics *const d,
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  const size_t bytes = KernelDiagnostics_total_bytes(d, nelements, scalar_bytes, real_bytes, accumulator_bytes);
  return bytes ? KernelDiagnostics_total_flops(d, nelements) / (double)bytes : 0.0;
}

static SFEM_CODEGEN_DEVICE_INLINE double KernelDiagnostics_arithmetic_intensity_affine_mesh(
    const KernelDiagnostics *const d,
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  const size_t bytes = KernelDiagnostics_total_bytes_affine_mesh(d, nelements, scalar_bytes, real_bytes, accumulator_bytes);
  return bytes ? KernelDiagnostics_total_flops_affine_mesh(d, nelements) / (double)bytes : 0.0;
}

static SFEM_CODEGEN_DEVICE_INLINE double KernelDiagnostics_arithmetic_intensity_isoparametric_mesh(
    const KernelDiagnostics *const d,
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  const size_t bytes = KernelDiagnostics_total_bytes_isoparametric_mesh(d, nelements, scalar_bytes, real_bytes, accumulator_bytes);
  return bytes ? KernelDiagnostics_total_flops_isoparametric_mesh(d, nelements) / (double)bytes : 0.0;
}

static SFEM_CODEGEN_DEVICE_INLINE void KernelDiagnostics_print_rate_with_ai(
    const char *const name,
    const KernelDiagnostics *const d,
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs,
    const double ai,
    const double total_flops) {
  const double element_rate = elapsed > 0.0 ? 1e-6 * (double)nelements / elapsed : 0.0;
  const double dof_rate = elapsed > 0.0 ? 1e-6 * (double)ndofs / elapsed : 0.0;
  const double gflops = elapsed > 0.0
      ? 1e-9 * total_flops / elapsed
      : 0.0;
  printf("%-72s %12.6e %16.3f %13.3f %10.3f %13.3f\n",
           name ? name : d->kernel_name,
           elapsed, element_rate, dof_rate, ai, gflops);
}

static SFEM_CODEGEN_DEVICE_INLINE void KernelDiagnostics_print_rate(
    const char *const name,
    const KernelDiagnostics *const d,
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  const double ai = KernelDiagnostics_arithmetic_intensity(
      d, nelements, scalar_bytes, real_bytes, accumulator_bytes);
  const double total_flops = KernelDiagnostics_total_flops(d, nelements);
  KernelDiagnostics_print_rate_with_ai(name, d, elapsed, nelements, ndofs, ai, total_flops);
}

static SFEM_CODEGEN_DEVICE_INLINE void KernelDiagnostics_print_rate_affine_mesh(
    const char *const name,
    const KernelDiagnostics *const d,
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  const double ai = KernelDiagnostics_arithmetic_intensity_affine_mesh(
      d, nelements, scalar_bytes, real_bytes, accumulator_bytes);
  const double total_flops = KernelDiagnostics_total_flops_affine_mesh(d, nelements);
  KernelDiagnostics_print_rate_with_ai(name, d, elapsed, nelements, ndofs, ai, total_flops);
}

static SFEM_CODEGEN_DEVICE_INLINE void KernelDiagnostics_print_rate_isoparametric_mesh(
    const char *const name,
    const KernelDiagnostics *const d,
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  const double ai = KernelDiagnostics_arithmetic_intensity_isoparametric_mesh(
      d, nelements, scalar_bytes, real_bytes, accumulator_bytes);
  const double total_flops = KernelDiagnostics_total_flops_isoparametric_mesh(d, nelements);
  KernelDiagnostics_print_rate_with_ai(name, d, elapsed, nelements, ndofs, ai, total_flops);
}

} // namespace codegen
} // namespace sfem

#endif
