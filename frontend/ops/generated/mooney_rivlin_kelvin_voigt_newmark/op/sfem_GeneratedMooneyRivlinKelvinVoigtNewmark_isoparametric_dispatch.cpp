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

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri6_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri6_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex27_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex27_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex27_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex27_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex64_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex64_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri6_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri6_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex27_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex27_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex27_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex27_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex64_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex64_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri6_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri6_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex27_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex27_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex27_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex27_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex64_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex64_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri6_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri6_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u0_direction,
        const double *const SFEM_RESTRICT u1_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u0_direction,
        const float *const SFEM_RESTRICT u1_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u0_direction,
        const double *const SFEM_RESTRICT u1_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u0_direction,
        const float *const SFEM_RESTRICT u1_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u0_direction,
        const double *const SFEM_RESTRICT u1_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u0_direction,
        const float *const SFEM_RESTRICT u1_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri6_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u0_direction,
        const double *const SFEM_RESTRICT u1_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri6_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u0_direction,
        const float *const SFEM_RESTRICT u1_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex27_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex27_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex64_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex64_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex27_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u0_direction,
        const double *const SFEM_RESTRICT u1_direction,
        const double *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex27_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u0_direction,
        const float *const SFEM_RESTRICT u1_direction,
        const float *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u0_direction,
        const double *const SFEM_RESTRICT u1_direction,
        const double *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u0_direction,
        const float *const SFEM_RESTRICT u1_direction,
        const float *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u0_direction,
        const double *const SFEM_RESTRICT u1_direction,
        const double *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u0_direction,
        const float *const SFEM_RESTRICT u1_direction,
        const float *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex64_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u0_direction,
        const double *const SFEM_RESTRICT u1_direction,
        const double *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex64_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u0_direction,
        const float *const SFEM_RESTRICT u1_direction,
        const float *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u0_direction,
        const double *const SFEM_RESTRICT u1_direction,
        const double *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u0_direction,
        const float *const SFEM_RESTRICT u1_direction,
        const float *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u0_direction,
        const double *const SFEM_RESTRICT u1_direction,
        const double *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u0_direction,
        const float *const SFEM_RESTRICT u1_direction,
        const float *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u0_direction,
        const double *const SFEM_RESTRICT u1_direction,
        const double *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u0_direction,
        const float *const SFEM_RESTRICT u1_direction,
        const float *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri6_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri6_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri6_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri6_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex27_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex27_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex64_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex64_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex27_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex27_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex64_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex64_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, out_stride, (double *)outx, (double *)outy);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, out_stride, (float *)outx, (float *)outy);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, out_stride, (double *)outx, (double *)outy);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, out_stride, (float *)outx, (float *)outy);
                default:
                    break;
            }
            break;
        }
        case smesh::TRI3: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, out_stride, (double *)outx, (double *)outy);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, out_stride, (float *)outx, (float *)outy);
                default:
                    break;
            }
            break;
        }
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tri6_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, out_stride, (double *)outx, (double *)outy);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tri6_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, out_stride, (float *)outx, (float *)outy);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const void *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy,
        void *const SFEM_RESTRICT outz
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_hex27_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_hex27_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex27_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex27_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX64: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex64_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex64_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::TET4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, out_stride, (double *)outx, (double *)outy);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, out_stride, (float *)outx, (float *)outy);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, out_stride, (double *)outx, (double *)outy);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, out_stride, (float *)outx, (float *)outy);
                default:
                    break;
            }
            break;
        }
        case smesh::TRI3: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, out_stride, (double *)outx, (double *)outy);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, out_stride, (float *)outx, (float *)outy);
                default:
                    break;
            }
            break;
        }
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tri6_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, out_stride, (double *)outx, (double *)outy);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tri6_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, out_stride, (float *)outx, (float *)outy);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy,
        void *const SFEM_RESTRICT outz
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_hex27_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_hex27_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex27_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex27_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX64: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex64_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex64_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::TET4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_objective_steps_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const int nsteps,
        const void *const SFEM_RESTRICT steps,
        void *const SFEM_RESTRICT value
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::TRI3: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tri6_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tri6_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_elastic_objective_steps_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_objective_steps_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const void *const SFEM_RESTRICT hz,
        const int nsteps,
        const void *const SFEM_RESTRICT steps,
        void *const SFEM_RESTRICT value
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_hex27_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_hex27_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex27_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex27_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX64: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex64_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex64_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::TET4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_elastic_objective_steps_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT previous,
        const void *const SFEM_RESTRICT direction,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::TRI3: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tri6_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tri6_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u0,
        const void *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u0_old,
        const void *const SFEM_RESTRICT u1_old,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u0_direction,
        const void *const SFEM_RESTRICT u1_direction,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u0_out,
        void *const SFEM_RESTRICT u1_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, out_stride, (double *)u0_out, (double *)u1_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, out_stride, (float *)u0_out, (float *)u1_out);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, out_stride, (double *)u0_out, (double *)u1_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, out_stride, (float *)u0_out, (float *)u1_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TRI3: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, out_stride, (double *)u0_out, (double *)u1_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, out_stride, (float *)u0_out, (float *)u1_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tri6_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, out_stride, (double *)u0_out, (double *)u1_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tri6_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, out_stride, (float *)u0_out, (float *)u1_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT previous,
        const void *const SFEM_RESTRICT direction,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex27_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex27_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX64: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex64_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex64_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::TET4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u0,
        const void *const SFEM_RESTRICT u1,
        const void *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u0_old,
        const void *const SFEM_RESTRICT u1_old,
        const void *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u0_direction,
        const void *const SFEM_RESTRICT u1_direction,
        const void *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u0_out,
        void *const SFEM_RESTRICT u1_out,
        void *const SFEM_RESTRICT u2_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex27_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex27_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX64: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex64_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex64_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TET4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT previous,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::TRI3: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tri6_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tri6_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u0,
        const void *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u0_old,
        const void *const SFEM_RESTRICT u1_old,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u0_out,
        void *const SFEM_RESTRICT u1_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, out_stride, (double *)u0_out, (double *)u1_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, out_stride, (float *)u0_out, (float *)u1_out);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, out_stride, (double *)u0_out, (double *)u1_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, out_stride, (float *)u0_out, (float *)u1_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TRI3: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, out_stride, (double *)u0_out, (double *)u1_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, out_stride, (float *)u0_out, (float *)u1_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tri6_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, out_stride, (double *)u0_out, (double *)u1_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tri6_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, out_stride, (float *)u0_out, (float *)u1_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT previous,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex27_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex27_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX64: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex64_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex64_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::TET4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u0,
        const void *const SFEM_RESTRICT u1,
        const void *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u0_old,
        const void *const SFEM_RESTRICT u1_old,
        const void *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u0_out,
        void *const SFEM_RESTRICT u1_out,
        void *const SFEM_RESTRICT u2_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex27_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex27_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX64: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex64_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex64_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TET4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}
