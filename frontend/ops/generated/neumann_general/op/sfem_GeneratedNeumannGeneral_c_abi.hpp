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
typedef ptrdiff_t count_t;
typedef double real_t;
typedef double geom_t;
#endif

#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif

#include "../../kernel_diagnostics.hpp"

#include "smesh_mesh.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" int neumann_general_edgeshell2_boundary_residual_2d_sideset_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t0_010,
        const real_t t0_100,
        const real_t t1,
        const real_t t1_010,
        const real_t t1_100,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1
);

extern "C" int neumann_general_edgeshell2_boundary_residual_2d_sideset_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t0_010,
        const float t0_100,
        const float t1,
        const float t1_010,
        const float t1_100,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1
);

extern "C" int neumann_general_proteus_quadshell16_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t0_001,
        const real_t t0_010,
        const real_t t0_100,
        const real_t t1,
        const real_t t1_001,
        const real_t t1_010,
        const real_t t1_100,
        const real_t t2,
        const real_t t2_001,
        const real_t t2_010,
        const real_t t2_100,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2
);

extern "C" int neumann_general_proteus_quadshell16_boundary_residual_3d_sideset_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t0_001,
        const float t0_010,
        const float t0_100,
        const float t1,
        const float t1_001,
        const float t1_010,
        const float t1_100,
        const float t2,
        const float t2_001,
        const float t2_010,
        const float t2_100,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2
);

extern "C" int neumann_general_proteus_quadshell25_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t0_001,
        const real_t t0_010,
        const real_t t0_100,
        const real_t t1,
        const real_t t1_001,
        const real_t t1_010,
        const real_t t1_100,
        const real_t t2,
        const real_t t2_001,
        const real_t t2_010,
        const real_t t2_100,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2
);

extern "C" int neumann_general_proteus_quadshell25_boundary_residual_3d_sideset_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t0_001,
        const float t0_010,
        const float t0_100,
        const float t1,
        const float t1_001,
        const float t1_010,
        const float t1_100,
        const float t2,
        const float t2_001,
        const float t2_010,
        const float t2_100,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2
);

extern "C" int neumann_general_proteus_quadshell4_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t0_001,
        const real_t t0_010,
        const real_t t0_100,
        const real_t t1,
        const real_t t1_001,
        const real_t t1_010,
        const real_t t1_100,
        const real_t t2,
        const real_t t2_001,
        const real_t t2_010,
        const real_t t2_100,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2
);

extern "C" int neumann_general_proteus_quadshell4_boundary_residual_3d_sideset_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t0_001,
        const float t0_010,
        const float t0_100,
        const float t1,
        const float t1_001,
        const float t1_010,
        const float t1_100,
        const float t2,
        const float t2_001,
        const float t2_010,
        const float t2_100,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2
);

extern "C" int neumann_general_proteus_quadshell9_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t0_001,
        const real_t t0_010,
        const real_t t0_100,
        const real_t t1,
        const real_t t1_001,
        const real_t t1_010,
        const real_t t1_100,
        const real_t t2,
        const real_t t2_001,
        const real_t t2_010,
        const real_t t2_100,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2
);

extern "C" int neumann_general_proteus_quadshell9_boundary_residual_3d_sideset_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t0_001,
        const float t0_010,
        const float t0_100,
        const float t1,
        const float t1_001,
        const float t1_010,
        const float t1_100,
        const float t2,
        const float t2_001,
        const float t2_010,
        const float t2_100,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2
);

extern "C" int neumann_general_quadshell4_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t0_001,
        const real_t t0_010,
        const real_t t0_100,
        const real_t t1,
        const real_t t1_001,
        const real_t t1_010,
        const real_t t1_100,
        const real_t t2,
        const real_t t2_001,
        const real_t t2_010,
        const real_t t2_100,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2
);

extern "C" int neumann_general_quadshell4_boundary_residual_3d_sideset_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t0_001,
        const float t0_010,
        const float t0_100,
        const float t1,
        const float t1_001,
        const float t1_010,
        const float t1_100,
        const float t2,
        const float t2_001,
        const float t2_010,
        const float t2_100,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2
);

extern "C" int neumann_general_quadshell9_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t0_001,
        const real_t t0_010,
        const real_t t0_100,
        const real_t t1,
        const real_t t1_001,
        const real_t t1_010,
        const real_t t1_100,
        const real_t t2,
        const real_t t2_001,
        const real_t t2_010,
        const real_t t2_100,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2
);

extern "C" int neumann_general_quadshell9_boundary_residual_3d_sideset_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t0_001,
        const float t0_010,
        const float t0_100,
        const float t1,
        const float t1_001,
        const float t1_010,
        const float t1_100,
        const float t2,
        const float t2_001,
        const float t2_010,
        const float t2_100,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2
);

extern "C" int neumann_general_trishell3_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t0_001,
        const real_t t0_010,
        const real_t t0_100,
        const real_t t1,
        const real_t t1_001,
        const real_t t1_010,
        const real_t t1_100,
        const real_t t2,
        const real_t t2_001,
        const real_t t2_010,
        const real_t t2_100,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2
);

extern "C" int neumann_general_trishell3_boundary_residual_3d_sideset_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t0_001,
        const float t0_010,
        const float t0_100,
        const float t1,
        const float t1_001,
        const float t1_010,
        const float t1_100,
        const float t2,
        const float t2_001,
        const float t2_010,
        const float t2_100,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2
);

extern "C" int neumann_general_trishell6_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t0_001,
        const real_t t0_010,
        const real_t t0_100,
        const real_t t1,
        const real_t t1_001,
        const real_t t1_010,
        const real_t t1_100,
        const real_t t2,
        const real_t t2_001,
        const real_t t2_010,
        const real_t t2_100,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2
);

extern "C" int neumann_general_trishell6_boundary_residual_3d_sideset_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t0_001,
        const float t0_010,
        const float t0_100,
        const float t1,
        const float t1_001,
        const float t1_010,
        const float t1_100,
        const float t2,
        const float t2_001,
        const float t2_010,
        const float t2_100,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2
);
