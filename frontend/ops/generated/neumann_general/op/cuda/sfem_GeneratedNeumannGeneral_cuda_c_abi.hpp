#pragma once

#include <cstddef>
#include <cstdint>

#include <stddef.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_GENERATED_SCALAR_T
#endif
#endif

#ifndef SFEM_GENERATED_SCALAR_T
#define SFEM_GENERATED_SCALAR_T
typedef double real_t;
typedef ptrdiff_t idx_t;
typedef ptrdiff_t element_idx_t;
typedef ptrdiff_t count_t;
typedef double geom_t;
#endif

#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif
#ifndef RSTR
#define RSTR SFEM_RESTRICT
#endif

#include "../../../cuda/kernel_diagnostics.cuh"

#include "smesh_mesh.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" int cu_neumann_general_edgeshell2_boundary_residual_2d_ss_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
        const real_t t0,
        const real_t t0_010,
        const real_t t0_100,
        const real_t t1,
        const real_t t1_010,
        const real_t t1_100,
        const int out_stride,
        real_t *const RSTR out0,
        real_t *const RSTR out1,
        void *const stream
);

extern "C" int cu_neumann_general_proteus_quadshell4_boundary_residual_3d_ss_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
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
        real_t *const RSTR out0,
        real_t *const RSTR out1,
        real_t *const RSTR out2,
        void *const stream
);

extern "C" int cu_neumann_general_quadshell4_boundary_residual_3d_ss_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
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
        real_t *const RSTR out0,
        real_t *const RSTR out1,
        real_t *const RSTR out2,
        void *const stream
);

extern "C" int cu_neumann_general_trishell3_boundary_residual_3d_ss_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
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
        real_t *const RSTR out0,
        real_t *const RSTR out1,
        real_t *const RSTR out2,
        void *const stream
);

extern "C" int cu_neumann_general_trishell6_boundary_residual_3d_ss_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
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
        real_t *const RSTR out0,
        real_t *const RSTR out1,
        real_t *const RSTR out2,
        void *const stream
);
