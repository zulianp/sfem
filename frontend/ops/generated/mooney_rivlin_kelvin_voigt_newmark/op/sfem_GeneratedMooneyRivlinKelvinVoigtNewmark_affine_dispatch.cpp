#include "sfem_GeneratedMooneyRivlinKelvinVoigtNewmark_c_abi.hpp"
#include <cstdio>

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif
#ifndef SFEM_FAILURE
#define SFEM_FAILURE 1
#endif
#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_apply_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_apply_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_apply_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_apply_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_gradient_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_gradient_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_gradient_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_gradient_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_steps_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_steps_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_objective_steps_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_objective_steps_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_objective_steps_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_objective_steps_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_steps_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_steps_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const ptrdiff_t direction_stride,
    const double *const RSTR u0_direction,
    const double *const RSTR u1_direction,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const ptrdiff_t direction_stride,
    const float *const RSTR u0_direction,
    const float *const RSTR u1_direction,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const ptrdiff_t direction_stride,
    const double *const RSTR u0_direction,
    const double *const RSTR u1_direction,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const ptrdiff_t direction_stride,
    const float *const RSTR u0_direction,
    const float *const RSTR u1_direction,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_jacobian_action_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const ptrdiff_t direction_stride,
    const double *const RSTR u0_direction,
    const double *const RSTR u1_direction,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_jacobian_action_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const ptrdiff_t direction_stride,
    const float *const RSTR u0_direction,
    const float *const RSTR u1_direction,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const double *const RSTR u2,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const double *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const double *const RSTR u0_direction,
    const double *const RSTR u1_direction,
    const double *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out,
    double *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const float *const RSTR u2,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const float *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const float *const RSTR u0_direction,
    const float *const RSTR u1_direction,
    const float *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out,
    float *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const double *const RSTR u2,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const double *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const double *const RSTR u0_direction,
    const double *const RSTR u1_direction,
    const double *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out,
    double *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const float *const RSTR u2,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const float *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const float *const RSTR u0_direction,
    const float *const RSTR u1_direction,
    const float *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out,
    float *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const double *const RSTR u2,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const double *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const double *const RSTR u0_direction,
    const double *const RSTR u1_direction,
    const double *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out,
    double *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const float *const RSTR u2,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const float *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const float *const RSTR u0_direction,
    const float *const RSTR u1_direction,
    const float *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out,
    float *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const double *const RSTR u2,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const double *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const double *const RSTR u0_direction,
    const double *const RSTR u1_direction,
    const double *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out,
    double *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const float *const RSTR u2,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const float *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const float *const RSTR u0_direction,
    const float *const RSTR u1_direction,
    const float *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out,
    float *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_residual_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_residual_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const double *const RSTR u2,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const double *const RSTR u2_old,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out,
    double *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const float *const RSTR u2,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const float *const RSTR u2_old,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out,
    float *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const double *const RSTR u2,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const double *const RSTR u2_old,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out,
    double *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const float *const RSTR u2,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const float *const RSTR u2_old,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out,
    float *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const double *const RSTR u2,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const double *const RSTR u2_old,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out,
    double *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const float *const RSTR u2,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const float *const RSTR u2_old,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out,
    float *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const double *const RSTR u2,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const double *const RSTR u2_old,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out,
    double *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const float *const RSTR u2,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const float *const RSTR u2_old,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out,
    float *const RSTR u2_out
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TRI3: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, out_stride, (double *)outx, (double *)outy);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, out_stride, (float *)outx, (float *)outy);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_a_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_apply_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_apply_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        default:
          break;
      }
      break;
    }
    case smesh::PROTEUS_HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        default:
          break;
      }
      break;
    }
    case smesh::TET10: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_apply_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_apply_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        default:
          break;
      }
      break;
    }
    case smesh::TET4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_a_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TRI3: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, out_stride, (double *)outx, (double *)outy);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, out_stride, (float *)outx, (float *)outy);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_a_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_gradient_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_gradient_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        default:
          break;
      }
      break;
    }
    case smesh::PROTEUS_HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        default:
          break;
      }
      break;
    }
    case smesh::TET10: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_gradient_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_gradient_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        default:
          break;
      }
      break;
    }
    case smesh::TET4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_a_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_objective_steps_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TRI3: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_steps_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_steps_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "mooney_rivlin_kelvin_voigt_newmark_elastic_objective_steps_2d_a_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_objective_steps_3d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_objective_steps_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_objective_steps_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    case smesh::PROTEUS_HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    case smesh::TET10: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_objective_steps_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_objective_steps_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    case smesh::TET4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_steps_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_steps_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "mooney_rivlin_kelvin_voigt_newmark_elastic_objective_steps_3d_a_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, out_stride, (double *)u0_out, (double *)u1_out);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, out_stride, (float *)u0_out, (float *)u1_out);
        default:
          break;
      }
      break;
    }
    case smesh::QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, out_stride, (double *)u0_out, (double *)u1_out);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, out_stride, (float *)u0_out, (float *)u1_out);
        default:
          break;
      }
      break;
    }
    case smesh::TRI3: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_jacobian_action_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, out_stride, (double *)u0_out, (double *)u1_out);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_jacobian_action_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, out_stride, (float *)u0_out, (float *)u1_out);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_a_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const void *const RSTR u2,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const void *const RSTR u2_old,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const void *const RSTR u2_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        default:
          break;
      }
      break;
    }
    case smesh::PROTEUS_HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        default:
          break;
      }
      break;
    }
    case smesh::TET10: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        default:
          break;
      }
      break;
    }
    case smesh::TET4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_a_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, out_stride, (double *)u0_out, (double *)u1_out);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, out_stride, (float *)u0_out, (float *)u1_out);
        default:
          break;
      }
      break;
    }
    case smesh::QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, out_stride, (double *)u0_out, (double *)u1_out);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, out_stride, (float *)u0_out, (float *)u1_out);
        default:
          break;
      }
      break;
    }
    case smesh::TRI3: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_residual_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, out_stride, (double *)u0_out, (double *)u1_out);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_residual_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, out_stride, (float *)u0_out, (float *)u1_out);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_a_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const void *const RSTR u2,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const void *const RSTR u2_old,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        default:
          break;
      }
      break;
    }
    case smesh::PROTEUS_HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        default:
          break;
      }
      break;
    }
    case smesh::TET10: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        default:
          break;
      }
      break;
    }
    case smesh::TET4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_a_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}
