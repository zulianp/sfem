#include <cstdio>
#include <type_traits>
#include "../laplace_d2_simplex_local.hpp"
#include "../laplace_d2_simplex_hessian.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdint>
#include <cstdlib>
#include "../../../packed_thread_scratch.hpp"
#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif
#ifndef SFEM_FAILURE
#define SFEM_FAILURE 1
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
SFEM_INLINE const s_t *ageom_stream(
    const int,
    const g_t *const RSTR source,
    s_t *const RSTR,
    std::true_type) {
  return source;
}

template <typename s_t, typename g_t, int VS>
SFEM_INLINE const s_t *ageom_stream(
    const int ne,
    const g_t *const RSTR source,
    s_t *const RSTR converted,
    std::false_type) {
  #pragma omp simd
  for (int lane = 0; lane < ne; ++lane) {
    converted[lane] = s_t(source[lane]);
  }
  return converted;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {


template <typename s_t>
struct laplace_tri3_affine_reference_data {
  static const s_t *shape() {
    static const s_t data[3] = {s_t(0.33333333333333343), s_t(0.33333333333333331), s_t(0.33333333333333331)};
    return data;
  }
  static const s_t *grad_ref_x() {
    static const s_t data[3] = {s_t(-1), s_t(1), s_t(0)};
    return data;
  }
  static const s_t *grad_ref_y() {
    static const s_t data[3] = {s_t(-1), s_t(0), s_t(1)};
    return data;
  }
  static const s_t *q_weight() {
    static const s_t data[1] = {s_t(0.5)};
    return data;
  }
};

template <typename s_t>
struct laplace_tri3_isoparametric_reference_data {
  static const s_t *shape() {
    static const s_t data[3] = {s_t(0.33333333333333343), s_t(0.33333333333333331), s_t(0.33333333333333331)};
    return data;
  }
  static const s_t *grad_ref_x() {
    static const s_t data[3] = {s_t(-1), s_t(1), s_t(0)};
    return data;
  }
  static const s_t *grad_ref_y() {
    static const s_t data[3] = {s_t(-1), s_t(0), s_t(1)};
    return data;
  }
  static const s_t *q_weight() {
    static const s_t data[1] = {s_t(0.5)};
    return data;
  }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_tri3_objective_soa_diagnostics_data = {
  "laplace_tri3_objective_soa",
  "TRI3",
  2,
  1,
  3,
  16,
  1,
  1,
  2,
  0,
  0,
  2,
  0,
  0,
  0,
  2,
  1,
  5,
  0,
  0,
  0,
  3,
  5,
  6,
  1,
  2,
  6,
  0,
  1,
  1,
  1,
  1.0,
  1.0,
  8.0,
  12.0,
  16.0,
  20.0,
  20.0,
  24.0,
  1.0,
  1.0
};

} // namespace codegen
} // namespace sfem

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tri3_objective_soa_diagnostics(void) {
  return &sfem::codegen::laplace_tri3_objective_soa_diagnostics_data;
}

extern "C" double laplace_tri3_objective_soa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::laplace_tri3_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_tri3_objective_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "laplace_tri3_objective_soa",
      &sfem::codegen::laplace_tri3_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tri3_objective_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "laplace_tri3_objective_soa_float",
      &sfem::codegen::laplace_tri3_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tri3_objective_affine_mesh_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "laplace_tri3_objective_affine_mesh_soa",
      &sfem::codegen::laplace_tri3_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tri3_objective_affine_mesh_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "laplace_tri3_objective_affine_mesh_soa_float",
      &sfem::codegen::laplace_tri3_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tri3_objective_isoparametric_mesh_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "laplace_tri3_objective_isoparametric_mesh_soa",
      &sfem::codegen::laplace_tri3_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tri3_objective_isoparametric_mesh_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "laplace_tri3_objective_isoparametric_mesh_soa_float",
      &sfem::codegen::laplace_tri3_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int laplace_tri3_objective_steps_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const g_t *const RSTR g_met0,
        const g_t *const RSTR g_met1,
        const g_t *const RSTR g_met2,
        const s_t kappa,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const int nsteps,
        const s_t *const RSTR steps,
        s_t *const RSTR value
) {
  (void)nnodes;

  #pragma omp parallel for schedule(static)
  for (ptrdiff_t element = 0; element < nelements; ++element) {
    const idx_t ev0 = elements[0][element];
    const idx_t ev1 = elements[1][element];
    const idx_t ev2 = elements[2][element];
    const s_t x0 = ux[ev0 * u_stride];
    const s_t x1 = ux[ev1 * u_stride];
    const s_t x2 = ux[ev2 * u_stride];
    const s_t h0 = hx[ev0 * h_stride];
    const s_t h1 = hx[ev1 * h_stride];
    const s_t h2 = hx[ev2 * h_stride];
    const s_t fff0 = kappa * s_t(g_met0[element]);
    const s_t fff1 = kappa * s_t(g_met1[element]);
    const s_t fff2 = kappa * s_t(g_met2[element]);
    for (int step = 0; step < nsteps; ++step) {
      const s_t alpha = steps[step];
      const s_t u0 = x0 + alpha * h0;
      const s_t u1 = x1 + alpha * h1;
      const s_t u2 = x2 + alpha * h2;
      const s_t t0 = -u0 + u1;
      const s_t t1 = -u0 + u2;
      value[(ptrdiff_t)step * nelements + element] = ((s_t(1) / s_t(2)))*t0*(fff0*t0 + fff1*t1) + ((s_t(1) / s_t(2)))*t1*(fff1*t0 + fff2*t1);
    }
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tri3_objective_steps_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
) {
  return sfem::codegen::laplace_tri3_objective_steps_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

extern "C" int laplace_tri3_objective_steps_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
) {
  return sfem::codegen::laplace_tri3_objective_steps_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_tri3_gradient_soa_diagnostics_data = {
  "laplace_tri3_gradient_soa",
  "TRI3",
  2,
  1,
  3,
  16,
  1,
  0,
  2,
  0,
  0,
  0,
  0,
  0,
  0,
  2,
  2,
  2,
  0,
  0,
  0,
  2,
  5,
  6,
  1,
  2,
  6,
  0,
  3,
  3,
  3,
  1.0,
  1.0,
  8.0,
  12.0,
  16.0,
  20.0,
  20.0,
  24.0,
  1.0,
  1.0
};

} // namespace codegen
} // namespace sfem

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tri3_gradient_soa_diagnostics(void) {
  return &sfem::codegen::laplace_tri3_gradient_soa_diagnostics_data;
}

extern "C" double laplace_tri3_gradient_soa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::laplace_tri3_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_tri3_gradient_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "laplace_tri3_gradient_soa",
      &sfem::codegen::laplace_tri3_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tri3_gradient_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "laplace_tri3_gradient_soa_float",
      &sfem::codegen::laplace_tri3_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tri3_gradient_affine_mesh_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "laplace_tri3_gradient_affine_mesh_soa",
      &sfem::codegen::laplace_tri3_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tri3_gradient_affine_mesh_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "laplace_tri3_gradient_affine_mesh_soa_float",
      &sfem::codegen::laplace_tri3_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tri3_gradient_isoparametric_mesh_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "laplace_tri3_gradient_isoparametric_mesh_soa",
      &sfem::codegen::laplace_tri3_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tri3_gradient_isoparametric_mesh_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "laplace_tri3_gradient_isoparametric_mesh_soa_float",
      &sfem::codegen::laplace_tri3_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int laplace_tri3_gradient_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const g_t *const RSTR g_met0,
        const g_t *const RSTR g_met1,
        const g_t *const RSTR g_met2,
        const s_t kappa,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx
) {
  (void)nnodes;

  #pragma omp parallel for schedule(static)
  for (ptrdiff_t element = 0; element < nelements; ++element) {
    const idx_t ev0 = elements[0][element];
    const idx_t ev1 = elements[1][element];
    const idx_t ev2 = elements[2][element];
    const s_t u0 = ux[ev0 * u_stride];
    const s_t u1 = ux[ev1 * u_stride];
    const s_t u2 = ux[ev2 * u_stride];
    const s_t fff0 = kappa * s_t(g_met0[element]);
    const s_t fff1 = kappa * s_t(g_met1[element]);
    const s_t fff2 = kappa * s_t(g_met2[element]);
    const s_t t0 = -u0 + u1;
    const s_t t1 = -u0 + u2;
    const s_t t2 = fff0*t0 + fff1*t1;
    const s_t t3 = fff1*t0 + fff2*t1;
    const s_t e0 = -t2 - t3;
    #pragma omp atomic update
    outx[ev0 * out_stride] += e0;
    const s_t e1 = t2;
    #pragma omp atomic update
    outx[ev1 * out_stride] += e1;
    const s_t e2 = t3;
    #pragma omp atomic update
    outx[ev2 * out_stride] += e2;
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tri3_gradient_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
) {
  return sfem::codegen::laplace_tri3_gradient_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_tri3_gradient_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
) {
  return sfem::codegen::laplace_tri3_gradient_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, u_stride, ux, out_stride, outx);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_tri3_apply_soa_diagnostics_data = {
  "laplace_tri3_apply_soa",
  "TRI3",
  2,
  1,
  3,
  16,
  1,
  0,
  2,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  2,
  2,
  0,
  0,
  0,
  2,
  5,
  6,
  1,
  2,
  0,
  6,
  3,
  3,
  3,
  1.0,
  1.0,
  8.0,
  12.0,
  16.0,
  20.0,
  20.0,
  24.0,
  1.0,
  1.0
};

} // namespace codegen
} // namespace sfem

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tri3_apply_soa_diagnostics(void) {
  return &sfem::codegen::laplace_tri3_apply_soa_diagnostics_data;
}

extern "C" double laplace_tri3_apply_soa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::laplace_tri3_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_tri3_apply_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "laplace_tri3_apply_soa",
      &sfem::codegen::laplace_tri3_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tri3_apply_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "laplace_tri3_apply_soa_float",
      &sfem::codegen::laplace_tri3_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tri3_apply_affine_mesh_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "laplace_tri3_apply_affine_mesh_soa",
      &sfem::codegen::laplace_tri3_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tri3_apply_affine_mesh_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "laplace_tri3_apply_affine_mesh_soa_float",
      &sfem::codegen::laplace_tri3_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tri3_apply_isoparametric_mesh_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "laplace_tri3_apply_isoparametric_mesh_soa",
      &sfem::codegen::laplace_tri3_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tri3_apply_isoparametric_mesh_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "laplace_tri3_apply_isoparametric_mesh_soa_float",
      &sfem::codegen::laplace_tri3_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int laplace_tri3_apply_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const g_t *const RSTR g_met0,
        const g_t *const RSTR g_met1,
        const g_t *const RSTR g_met2,
        const s_t kappa,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx
) {
  (void)nnodes;

  #pragma omp parallel for schedule(static)
  for (ptrdiff_t element = 0; element < nelements; ++element) {
    const idx_t ev0 = elements[0][element];
    const idx_t ev1 = elements[1][element];
    const idx_t ev2 = elements[2][element];
    const s_t u0 = hx[ev0 * h_stride];
    const s_t u1 = hx[ev1 * h_stride];
    const s_t u2 = hx[ev2 * h_stride];
    const s_t fff0 = kappa * s_t(g_met0[element]);
    const s_t fff1 = kappa * s_t(g_met1[element]);
    const s_t fff2 = kappa * s_t(g_met2[element]);
    const s_t t0 = -u0 + u1;
    const s_t t1 = -u0 + u2;
    const s_t t2 = fff0*t0 + fff1*t1;
    const s_t t3 = fff1*t0 + fff2*t1;
    const s_t e0 = -t2 - t3;
    #pragma omp atomic update
    outx[ev0 * out_stride] += e0;
    const s_t e1 = t2;
    #pragma omp atomic update
    outx[ev1 * out_stride] += e1;
    const s_t e2 = t3;
    #pragma omp atomic update
    outx[ev2 * out_stride] += e2;
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tri3_apply_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
) {
  return sfem::codegen::laplace_tri3_apply_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_tri3_apply_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
) {
  return sfem::codegen::laplace_tri3_apply_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, h_stride, hx, out_stride, outx);
}


namespace sfem {
namespace codegen {

static SFEM_INLINE void laplace_tri3_hessian_isoparametric_mesh_soa_find_cols(
    const idx_t *const RSTR targets,
    const idx_t *const RSTR row,
    const int lenrow,
    idx_t *const RSTR ks) {
#pragma unroll(3)
  for (int d = 0; d < 3; ++d) {
    ks[d] = 0;
  }
  for (int k = 0; k < lenrow; ++k) {
#pragma unroll(3)
    for (int d = 0; d < 3; ++d) {
      ks[d] += row[k] < targets[d];
    }
  }
}

template <typename s_t>
static SFEM_INLINE void laplace_tri3_hessian_isoparametric_mesh_soa_scatter_bsr(
    const idx_t *const RSTR ev,
    const s_t *const RSTR element_matrix,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values) {
  static constexpr int NC = 1;
  static constexpr int NS = 3;
  count_t entries[NS * NS];
  idx_t ks[NS];
  for (int i = 0; i < NS; ++i) {
    const idx_t dof_i = ev[i];
    const count_t row_begin = rowptr[dof_i];
    const int lenrow = (int)(rowptr[dof_i + 1] - row_begin);
    const idx_t *const RSTR cols = &colidx[row_begin];
    laplace_tri3_hessian_isoparametric_mesh_soa_find_cols(ev, cols, lenrow, ks);
    for (int j = 0; j < NS; ++j) {
      entries[i * NS + j] = row_begin + ks[j];
    }
  }
  for (int i = 0; i < NS; ++i) {
    for (int j = 0; j < NS; ++j) {
      s_t *const block = &values[entries[i * NS + j] * NC * NC];
      for (int bi = 0; bi < NC; ++bi) {
        const int row = bi * NS + i;
        for (int bj = 0; bj < NC; ++bj) {
          const int col = bj * NS + j;
#pragma omp atomic update
          block[bi * NC + bj] += element_matrix[row * (NC * NS) + col];
        }
      }
    }
  }
}

template <typename s_t>
static SFEM_INLINE void laplace_tri3_hessian_isoparametric_mesh_soa_scatter_crs(
    const idx_t *const RSTR ev,
    const s_t *const RSTR element_matrix,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values) {
  static constexpr int NC = 1;
  static constexpr int NS = 3;
  count_t row_begin[NS];
  int lenrow[NS];
  int local_col[NS * NS];
  idx_t ks[NS];
  for (int i = 0; i < NS; ++i) {
    row_begin[i] = rowptr[ev[i]];
    lenrow[i] = (int)(rowptr[ev[i] + 1] - row_begin[i]);
    const idx_t *const RSTR cols = &colidx[row_begin[i]];
    laplace_tri3_hessian_isoparametric_mesh_soa_find_cols(ev, cols, lenrow[i], ks);
    for (int j = 0; j < NS; ++j) {
      local_col[i * NS + j] = (int)ks[j];
    }
  }
  for (int i = 0; i < NS; ++i) {
    const count_t rb = row_begin[i];
    const int lr = lenrow[i];
    for (int j = 0; j < NS; ++j) {
      const int lc = local_col[i * NS + j];
      for (int bi = 0; bi < NC; ++bi) {
        const int row = bi * NS + i;
        s_t *const row_values = &values[rb * NC * NC + bi * lr * NC];
        for (int bj = 0; bj < NC; ++bj) {
          const int col = bj * NS + j;
#pragma omp atomic update
          row_values[lc * NC + bj] += element_matrix[row * (NC * NS) + col];
        }
      }
    }
  }
}

template <typename s_t, typename g_t, int FORMAT>
static int laplace_tri3_hessian_isoparametric_mesh_soa_assemble_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const g_t *const *const RSTR points,
    const s_t kappa,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values,
    const int *const RSTR diag_offsets,
    const ptrdiff_t ndiag,
    const ptrdiff_t coo_nnz,
    const idx_t *const RSTR coo_rows,
    const idx_t *const RSTR coo_cols,
    idx_t *const RSTR coo_triplet_rows,
    idx_t *const RSTR coo_triplet_cols) {
  static constexpr int NC = 1;
  static constexpr int ND = 2;
  static constexpr int NQ = 1;
  static constexpr int NS = 3;
  static constexpr int VS = 1;
  static constexpr int NDOFS = NC * NS;
  (void)nnodes;
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::laplace_tri3_isoparametric_reference_data<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::laplace_tri3_isoparametric_reference_data<s_t>::grad_ref_y();
  const s_t *const isoparametric_q_weight = sfem::codegen::laplace_tri3_isoparametric_reference_data<s_t>::q_weight();

  int unsupported_matrix_format = 0;
#pragma omp parallel for schedule(static) reduction(|:unsupported_matrix_format)
  for (ptrdiff_t element = 0; element < nelements; ++element) {
    idx_t ev[NS];
    s_t element_matrix[NDOFS * NDOFS];
    s_t bh_data[NS * NC][VS];
    s_t bout_data[NS * NC][VS];
    s_t bcoordinate_data[NS * ND][VS];
    static constexpr int ne = VS;
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t bdet0[NQ * VS];
    s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3};
    const s_t *bh_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bh_streams[stream] = bh_data[stream];
    }
    s_t *bout_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bout_streams[stream] = bout_data[stream];
    }

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t node = elements[shape][element];
      ev[shape] = node;
      for (int d = 0; d < ND; ++d) {
        bcoordinate_data[shape * ND + d][0] = s_t(points[d][node]);
      }
    }


    for (int q = 0; q < NQ; ++q) {
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3};
      s_t J00_values[VS];
      s_t J01_values[VS];
      s_t J10_values[VS];
      s_t J11_values[VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        J00_values[lane] = s_t(0);
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        J01_values[lane] = s_t(0);
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        J10_values[lane] = s_t(0);
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        J11_values[lane] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = isoparametric_grad_ref_x[q * NS + shape];
        const s_t g1 = isoparametric_grad_ref_y[q * NS + shape];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J00_values[lane] += bcoordinate_data[shape * 2 + 0][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J01_values[lane] += bcoordinate_data[shape * 2 + 0][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J10_values[lane] += bcoordinate_data[shape * 2 + 1][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J11_values[lane] += bcoordinate_data[shape * 2 + 1][lane] * g1;
        }
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = J00_values[lane];
        const s_t J01 = J01_values[lane];
        const s_t J10 = J10_values[lane];
        const s_t J11 = J11_values[lane];
        geometry_jacobian_adjugate_and_determinant_2<s_t>(
            J00, J01, J10, J11, badj_streams, bdet0, q * VS + lane);
      }
    }

    laplace_d2_simplex_direct_hessian_reference_element_matrix<s_t, NQ, NS, VS>(badj0, badj1, badj2, badj3, bdet0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_q_weight, kappa, element_matrix);

    if constexpr (FORMAT == 1) {
      laplace_tri3_hessian_isoparametric_mesh_soa_scatter_bsr(ev, element_matrix, rowptr, colidx, values);
    } else if constexpr (FORMAT == 0) {
      laplace_tri3_hessian_isoparametric_mesh_soa_scatter_crs(ev, element_matrix, rowptr, colidx, values);
    } else {
      unsupported_matrix_format |= 1;
    }
  }

  return unsupported_matrix_format ? SFEM_FAILURE : SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tri3_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
) {
  return sfem::codegen::laplace_tri3_hessian_isoparametric_mesh_soa_assemble_impl<double, geom_t, 0>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int laplace_tri3_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
) {
  return sfem::codegen::laplace_tri3_hessian_isoparametric_mesh_soa_assemble_impl<float, geom_t, 0>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int laplace_tri3_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
) {
  return sfem::codegen::laplace_tri3_hessian_isoparametric_mesh_soa_assemble_impl<double, geom_t, 1>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int laplace_tri3_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
) {
  return sfem::codegen::laplace_tri3_hessian_isoparametric_mesh_soa_assemble_impl<float, geom_t, 1>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
