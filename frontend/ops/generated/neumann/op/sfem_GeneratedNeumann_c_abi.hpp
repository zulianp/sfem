#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_CODEGEN_OP_HAS_SFEM_BASE
#endif
#endif

#ifndef SFEM_CODEGEN_OP_HAS_SFEM_BASE
typedef ptrdiff_t idx_t;
typedef ptrdiff_t element_idx_t;
typedef double real_t;
typedef double geom_t;
#endif

#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif

#include "../kernel_diagnostics.hpp"

extern "C" int neumann_hex27_quadshell9_boundary_residual_sideset_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2);

extern "C" int neumann_hex27_quadshell9_boundary_residual_sideset_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2);

extern "C" int neumann_hex27_quadshell9_boundary_residual_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2);

extern "C" int neumann_hex27_quadshell9_boundary_residual_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2);

extern "C" int neumann_hex8_quadshell4_boundary_residual_sideset_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2);

extern "C" int neumann_hex8_quadshell4_boundary_residual_sideset_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2);

extern "C" int neumann_hex8_quadshell4_boundary_residual_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2);

extern "C" int neumann_hex8_quadshell4_boundary_residual_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2);

extern "C" int neumann_proteus_hex125_proteus_quadshell25_boundary_residual_sideset_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2);

extern "C" int neumann_proteus_hex125_proteus_quadshell25_boundary_residual_sideset_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2);

extern "C" int neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2);

extern "C" int neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2);

extern "C" int neumann_proteus_hex27_proteus_quadshell9_boundary_residual_sideset_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2);

extern "C" int neumann_proteus_hex27_proteus_quadshell9_boundary_residual_sideset_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2);

extern "C" int neumann_proteus_hex27_proteus_quadshell9_boundary_residual_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2);

extern "C" int neumann_proteus_hex27_proteus_quadshell9_boundary_residual_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2);

extern "C" int neumann_proteus_hex64_proteus_quadshell16_boundary_residual_sideset_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2);

extern "C" int neumann_proteus_hex64_proteus_quadshell16_boundary_residual_sideset_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2);

extern "C" int neumann_proteus_hex64_proteus_quadshell16_boundary_residual_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2);

extern "C" int neumann_proteus_hex64_proteus_quadshell16_boundary_residual_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2);

extern "C" int neumann_proteus_hex8_proteus_quadshell4_boundary_residual_sideset_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2);

extern "C" int neumann_proteus_hex8_proteus_quadshell4_boundary_residual_sideset_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2);

extern "C" int neumann_proteus_hex8_proteus_quadshell4_boundary_residual_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2);

extern "C" int neumann_proteus_hex8_proteus_quadshell4_boundary_residual_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2);

extern "C" int neumann_quad4_edgeshell2_boundary_residual_sideset_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1);

extern "C" int neumann_quad4_edgeshell2_boundary_residual_sideset_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1);

extern "C" int neumann_quad4_edgeshell2_boundary_residual_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1);

extern "C" int neumann_quad4_edgeshell2_boundary_residual_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1);

extern "C" int neumann_tet10_trishell6_boundary_residual_sideset_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2);

extern "C" int neumann_tet10_trishell6_boundary_residual_sideset_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2);

extern "C" int neumann_tet10_trishell6_boundary_residual_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2);

extern "C" int neumann_tet10_trishell6_boundary_residual_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2);

extern "C" int neumann_tet4_trishell3_boundary_residual_sideset_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2);

extern "C" int neumann_tet4_trishell3_boundary_residual_sideset_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2);

extern "C" int neumann_tet4_trishell3_boundary_residual_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2);

extern "C" int neumann_tet4_trishell3_boundary_residual_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2);

extern "C" int neumann_tri3_edgeshell2_boundary_residual_sideset_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1);

extern "C" int neumann_tri3_edgeshell2_boundary_residual_sideset_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1);

extern "C" int neumann_tri3_edgeshell2_boundary_residual_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1);

extern "C" int neumann_tri3_edgeshell2_boundary_residual_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1);
