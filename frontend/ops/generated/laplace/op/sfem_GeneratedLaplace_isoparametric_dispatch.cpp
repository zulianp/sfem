#include "sfem_GeneratedLaplace_c_abi.hpp"
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

extern "C" int laplace_proteus_quad4_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_quad4_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_tri3_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_tri6_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_quad4_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_quad4_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_tri3_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_tri6_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_hex27_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_hex8_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex125_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex27_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex64_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex729_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex8_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_tet10_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_tet4_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_hex27_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_hex8_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex125_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex27_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex64_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex729_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex8_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_tet10_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_tet4_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_quad4_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_quad4_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_tri3_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_tri6_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_quad4_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_quad4_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_tri3_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_tri6_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_hex27_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_hex8_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex125_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex27_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex64_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex729_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex8_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_tet10_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_tet4_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_hex27_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_hex8_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex125_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex27_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex64_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex729_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex8_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_tet10_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_tet4_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_quad4_hessian_dia_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_quad4_hessian_dia_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_tri3_hessian_dia_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_tri6_hessian_dia_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_quad4_hessian_dia_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_quad4_hessian_dia_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_tri3_hessian_dia_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_tri6_hessian_dia_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_hex27_hessian_dia_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_hex8_hessian_dia_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex125_hessian_dia_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex27_hessian_dia_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex64_hessian_dia_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex729_hessian_dia_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex8_hessian_dia_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_tet10_hessian_dia_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_tet4_hessian_dia_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        double *const SFEM_RESTRICT values
);
extern "C" int laplace_hex27_hessian_dia_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_hex8_hessian_dia_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex125_hessian_dia_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex27_hessian_dia_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex64_hessian_dia_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex729_hessian_dia_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_hex8_hessian_dia_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_tet10_hessian_dia_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_tet4_hessian_dia_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        float *const SFEM_RESTRICT values
);
extern "C" int laplace_proteus_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_tri3_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_tri6_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_tri3_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_tri6_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_tri3_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_tri6_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_tri3_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_tri6_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_hex27_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex125_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex27_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex64_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex729_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_tet10_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_tet4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_hex27_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex125_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex27_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex64_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex729_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_tet10_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_tet4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_hex27_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex125_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex64_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex729_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_tet10_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_tet4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_hex27_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex125_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex64_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex729_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_tet10_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_tet4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_tri3_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_tri6_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_tri3_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_tri6_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_tri3_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_tri6_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_tri3_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_tri6_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_hex27_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex125_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex27_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex64_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex729_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_tet10_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_tet4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
);
extern "C" int laplace_hex27_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex125_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex27_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex64_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex729_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_proteus_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_tet10_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_tet4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
);
extern "C" int laplace_hex27_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex125_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex27_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex64_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex729_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_tet10_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_tet4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);
extern "C" int laplace_hex27_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex125_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex27_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex64_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex729_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_tet10_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);
extern "C" int laplace_tet4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_hessian_bsr_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return laplace_proteus_quad4_hessian_bsr_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::QUAD4:
            return laplace_quad4_hessian_bsr_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::TRI3:
            return laplace_tri3_hessian_bsr_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::TRI6:
            return laplace_tri6_hessian_bsr_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        default:
            std::fprintf(stderr, "laplace_hessian_bsr_2d_isoparametric_mesh_soa does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_hessian_bsr_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return laplace_proteus_quad4_hessian_bsr_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::QUAD4:
            return laplace_quad4_hessian_bsr_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::TRI3:
            return laplace_tri3_hessian_bsr_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::TRI6:
            return laplace_tri6_hessian_bsr_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        default:
            std::fprintf(stderr, "laplace_hessian_bsr_2d_isoparametric_mesh_soa_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_hessian_bsr_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    switch (element_type) {
        case smesh::HEX27:
            return laplace_hex27_hessian_bsr_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::HEX8:
            return laplace_hex8_hessian_bsr_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX125:
            return laplace_proteus_hex125_hessian_bsr_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX27:
            return laplace_proteus_hex27_hessian_bsr_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX64:
            return laplace_proteus_hex64_hessian_bsr_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX729:
            return laplace_proteus_hex729_hessian_bsr_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX8:
            return laplace_proteus_hex8_hessian_bsr_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::TET10:
            return laplace_tet10_hessian_bsr_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::TET4:
            return laplace_tet4_hessian_bsr_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        default:
            std::fprintf(stderr, "laplace_hessian_bsr_3d_isoparametric_mesh_soa does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_hessian_bsr_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    switch (element_type) {
        case smesh::HEX27:
            return laplace_hex27_hessian_bsr_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::HEX8:
            return laplace_hex8_hessian_bsr_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX125:
            return laplace_proteus_hex125_hessian_bsr_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX27:
            return laplace_proteus_hex27_hessian_bsr_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX64:
            return laplace_proteus_hex64_hessian_bsr_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX729:
            return laplace_proteus_hex729_hessian_bsr_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX8:
            return laplace_proteus_hex8_hessian_bsr_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::TET10:
            return laplace_tet10_hessian_bsr_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::TET4:
            return laplace_tet4_hessian_bsr_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        default:
            std::fprintf(stderr, "laplace_hessian_bsr_3d_isoparametric_mesh_soa_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_hessian_crs_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return laplace_proteus_quad4_hessian_crs_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::QUAD4:
            return laplace_quad4_hessian_crs_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::TRI3:
            return laplace_tri3_hessian_crs_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::TRI6:
            return laplace_tri6_hessian_crs_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        default:
            std::fprintf(stderr, "laplace_hessian_crs_2d_isoparametric_mesh_soa does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_hessian_crs_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return laplace_proteus_quad4_hessian_crs_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::QUAD4:
            return laplace_quad4_hessian_crs_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::TRI3:
            return laplace_tri3_hessian_crs_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::TRI6:
            return laplace_tri6_hessian_crs_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        default:
            std::fprintf(stderr, "laplace_hessian_crs_2d_isoparametric_mesh_soa_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_hessian_crs_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    switch (element_type) {
        case smesh::HEX27:
            return laplace_hex27_hessian_crs_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::HEX8:
            return laplace_hex8_hessian_crs_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX125:
            return laplace_proteus_hex125_hessian_crs_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX27:
            return laplace_proteus_hex27_hessian_crs_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX64:
            return laplace_proteus_hex64_hessian_crs_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX729:
            return laplace_proteus_hex729_hessian_crs_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX8:
            return laplace_proteus_hex8_hessian_crs_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::TET10:
            return laplace_tet10_hessian_crs_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::TET4:
            return laplace_tet4_hessian_crs_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        default:
            std::fprintf(stderr, "laplace_hessian_crs_3d_isoparametric_mesh_soa does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_hessian_crs_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    switch (element_type) {
        case smesh::HEX27:
            return laplace_hex27_hessian_crs_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::HEX8:
            return laplace_hex8_hessian_crs_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX125:
            return laplace_proteus_hex125_hessian_crs_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX27:
            return laplace_proteus_hex27_hessian_crs_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX64:
            return laplace_proteus_hex64_hessian_crs_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX729:
            return laplace_proteus_hex729_hessian_crs_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::PROTEUS_HEX8:
            return laplace_proteus_hex8_hessian_crs_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::TET10:
            return laplace_tet10_hessian_crs_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        case smesh::TET4:
            return laplace_tet4_hessian_crs_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
        default:
            std::fprintf(stderr, "laplace_hessian_crs_3d_isoparametric_mesh_soa_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_hessian_dia_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        double *const SFEM_RESTRICT values
) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return laplace_proteus_quad4_hessian_dia_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::QUAD4:
            return laplace_quad4_hessian_dia_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::TRI3:
            return laplace_tri3_hessian_dia_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::TRI6:
            return laplace_tri6_hessian_dia_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        default:
            std::fprintf(stderr, "laplace_hessian_dia_2d_isoparametric_mesh_soa does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_hessian_dia_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        float *const SFEM_RESTRICT values
) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return laplace_proteus_quad4_hessian_dia_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::QUAD4:
            return laplace_quad4_hessian_dia_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::TRI3:
            return laplace_tri3_hessian_dia_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::TRI6:
            return laplace_tri6_hessian_dia_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        default:
            std::fprintf(stderr, "laplace_hessian_dia_2d_isoparametric_mesh_soa_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_hessian_dia_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        double *const SFEM_RESTRICT values
) {
    switch (element_type) {
        case smesh::HEX27:
            return laplace_hex27_hessian_dia_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::HEX8:
            return laplace_hex8_hessian_dia_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::PROTEUS_HEX125:
            return laplace_proteus_hex125_hessian_dia_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::PROTEUS_HEX27:
            return laplace_proteus_hex27_hessian_dia_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::PROTEUS_HEX64:
            return laplace_proteus_hex64_hessian_dia_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::PROTEUS_HEX729:
            return laplace_proteus_hex729_hessian_dia_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::PROTEUS_HEX8:
            return laplace_proteus_hex8_hessian_dia_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::TET10:
            return laplace_tet10_hessian_dia_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::TET4:
            return laplace_tet4_hessian_dia_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        default:
            std::fprintf(stderr, "laplace_hessian_dia_3d_isoparametric_mesh_soa does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_hessian_dia_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        float *const SFEM_RESTRICT values
) {
    switch (element_type) {
        case smesh::HEX27:
            return laplace_hex27_hessian_dia_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::HEX8:
            return laplace_hex8_hessian_dia_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::PROTEUS_HEX125:
            return laplace_proteus_hex125_hessian_dia_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::PROTEUS_HEX27:
            return laplace_proteus_hex27_hessian_dia_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::PROTEUS_HEX64:
            return laplace_proteus_hex64_hessian_dia_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::PROTEUS_HEX729:
            return laplace_proteus_hex729_hessian_dia_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::PROTEUS_HEX8:
            return laplace_proteus_hex8_hessian_dia_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::TET10:
            return laplace_tet10_hessian_dia_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        case smesh::TET4:
            return laplace_tet4_hessian_dia_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
        default:
            std::fprintf(stderr, "laplace_hessian_dia_3d_isoparametric_mesh_soa_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return laplace_proteus_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::QUAD4:
            return laplace_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::TRI3:
            return laplace_tri3_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::TRI6:
            return laplace_tri6_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, direction, output);
        default:
            std::fprintf(stderr, "laplace_jacobian_action_2d_isoparametric_mesh_aos does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_jacobian_action_2d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return laplace_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::QUAD4:
            return laplace_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::TRI3:
            return laplace_tri3_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::TRI6:
            return laplace_tri6_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, direction, output);
        default:
            std::fprintf(stderr, "laplace_jacobian_action_2d_isoparametric_mesh_aos_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return laplace_proteus_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::QUAD4:
            return laplace_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::TRI3:
            return laplace_tri3_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::TRI6:
            return laplace_tri6_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        default:
            std::fprintf(stderr, "laplace_jacobian_action_2d_isoparametric_mesh_soa does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_jacobian_action_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return laplace_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::QUAD4:
            return laplace_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::TRI3:
            return laplace_tri3_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::TRI6:
            return laplace_tri6_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        default:
            std::fprintf(stderr, "laplace_jacobian_action_2d_isoparametric_mesh_soa_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    switch (element_type) {
        case smesh::HEX27:
            return laplace_hex27_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::HEX8:
            return laplace_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::PROTEUS_HEX125:
            return laplace_proteus_hex125_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::PROTEUS_HEX27:
            return laplace_proteus_hex27_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::PROTEUS_HEX64:
            return laplace_proteus_hex64_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::PROTEUS_HEX729:
            return laplace_proteus_hex729_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::PROTEUS_HEX8:
            return laplace_proteus_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::TET10:
            return laplace_tet10_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::TET4:
            return laplace_tet4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, direction, output);
        default:
            std::fprintf(stderr, "laplace_jacobian_action_3d_isoparametric_mesh_aos does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_jacobian_action_3d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    switch (element_type) {
        case smesh::HEX27:
            return laplace_hex27_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::HEX8:
            return laplace_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::PROTEUS_HEX125:
            return laplace_proteus_hex125_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::PROTEUS_HEX27:
            return laplace_proteus_hex27_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::PROTEUS_HEX64:
            return laplace_proteus_hex64_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::PROTEUS_HEX729:
            return laplace_proteus_hex729_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::PROTEUS_HEX8:
            return laplace_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::TET10:
            return laplace_tet10_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, direction, output);
        case smesh::TET4:
            return laplace_tet4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, direction, output);
        default:
            std::fprintf(stderr, "laplace_jacobian_action_3d_isoparametric_mesh_aos_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    switch (element_type) {
        case smesh::HEX27:
            return laplace_hex27_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::HEX8:
            return laplace_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::PROTEUS_HEX125:
            return laplace_proteus_hex125_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::PROTEUS_HEX27:
            return laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::PROTEUS_HEX64:
            return laplace_proteus_hex64_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::PROTEUS_HEX729:
            return laplace_proteus_hex729_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::PROTEUS_HEX8:
            return laplace_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::TET10:
            return laplace_tet10_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::TET4:
            return laplace_tet4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        default:
            std::fprintf(stderr, "laplace_jacobian_action_3d_isoparametric_mesh_soa does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_jacobian_action_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    switch (element_type) {
        case smesh::HEX27:
            return laplace_hex27_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::HEX8:
            return laplace_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::PROTEUS_HEX125:
            return laplace_proteus_hex125_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::PROTEUS_HEX27:
            return laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::PROTEUS_HEX64:
            return laplace_proteus_hex64_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::PROTEUS_HEX729:
            return laplace_proteus_hex729_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::PROTEUS_HEX8:
            return laplace_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::TET10:
            return laplace_tet10_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        case smesh::TET4:
            return laplace_tet4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
        default:
            std::fprintf(stderr, "laplace_jacobian_action_3d_isoparametric_mesh_soa_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return laplace_proteus_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::QUAD4:
            return laplace_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::TRI3:
            return laplace_tri3_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::TRI6:
            return laplace_tri6_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, current, output);
        default:
            std::fprintf(stderr, "laplace_residual_2d_isoparametric_mesh_aos does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_residual_2d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return laplace_proteus_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::QUAD4:
            return laplace_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::TRI3:
            return laplace_tri3_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::TRI6:
            return laplace_tri6_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, current, output);
        default:
            std::fprintf(stderr, "laplace_residual_2d_isoparametric_mesh_aos_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return laplace_proteus_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::QUAD4:
            return laplace_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::TRI3:
            return laplace_tri3_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::TRI6:
            return laplace_tri6_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        default:
            std::fprintf(stderr, "laplace_residual_2d_isoparametric_mesh_soa does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_residual_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return laplace_proteus_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::QUAD4:
            return laplace_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::TRI3:
            return laplace_tri3_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::TRI6:
            return laplace_tri6_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        default:
            std::fprintf(stderr, "laplace_residual_2d_isoparametric_mesh_soa_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
) {
    switch (element_type) {
        case smesh::HEX27:
            return laplace_hex27_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::HEX8:
            return laplace_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::PROTEUS_HEX125:
            return laplace_proteus_hex125_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::PROTEUS_HEX27:
            return laplace_proteus_hex27_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::PROTEUS_HEX64:
            return laplace_proteus_hex64_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::PROTEUS_HEX729:
            return laplace_proteus_hex729_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::PROTEUS_HEX8:
            return laplace_proteus_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::TET10:
            return laplace_tet10_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::TET4:
            return laplace_tet4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, parameters, current, output);
        default:
            std::fprintf(stderr, "laplace_residual_3d_isoparametric_mesh_aos does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_residual_3d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
) {
    switch (element_type) {
        case smesh::HEX27:
            return laplace_hex27_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::HEX8:
            return laplace_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::PROTEUS_HEX125:
            return laplace_proteus_hex125_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::PROTEUS_HEX27:
            return laplace_proteus_hex27_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::PROTEUS_HEX64:
            return laplace_proteus_hex64_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::PROTEUS_HEX729:
            return laplace_proteus_hex729_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::PROTEUS_HEX8:
            return laplace_proteus_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::TET10:
            return laplace_tet10_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, current, output);
        case smesh::TET4:
            return laplace_tet4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, parameters, current, output);
        default:
            std::fprintf(stderr, "laplace_residual_3d_isoparametric_mesh_aos_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    switch (element_type) {
        case smesh::HEX27:
            return laplace_hex27_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::HEX8:
            return laplace_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::PROTEUS_HEX125:
            return laplace_proteus_hex125_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::PROTEUS_HEX27:
            return laplace_proteus_hex27_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::PROTEUS_HEX64:
            return laplace_proteus_hex64_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::PROTEUS_HEX729:
            return laplace_proteus_hex729_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::PROTEUS_HEX8:
            return laplace_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::TET10:
            return laplace_tet10_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::TET4:
            return laplace_tet4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        default:
            std::fprintf(stderr, "laplace_residual_3d_isoparametric_mesh_soa does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_residual_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    switch (element_type) {
        case smesh::HEX27:
            return laplace_hex27_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::HEX8:
            return laplace_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::PROTEUS_HEX125:
            return laplace_proteus_hex125_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::PROTEUS_HEX27:
            return laplace_proteus_hex27_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::PROTEUS_HEX64:
            return laplace_proteus_hex64_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::PROTEUS_HEX729:
            return laplace_proteus_hex729_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::PROTEUS_HEX8:
            return laplace_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::TET10:
            return laplace_tet10_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        case smesh::TET4:
            return laplace_tet4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
        default:
            std::fprintf(stderr, "laplace_residual_3d_isoparametric_mesh_soa_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}
