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
#ifndef RSTR
#define RSTR SFEM_RESTRICT
#endif

#include "../../kernel_diagnostics.hpp"

#include "smesh_mesh.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" int neumann_edgeshell2_boundary_residual_2d_sideset_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
        const real_t t0,
        const real_t t1,
        const int out_stride,
        real_t *const RSTR out0,
        real_t *const RSTR out1
);

extern "C" int neumann_proteus_quadshell4_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
        const real_t t0,
        const real_t t1,
        const real_t t2,
        const int out_stride,
        real_t *const RSTR out0,
        real_t *const RSTR out1,
        real_t *const RSTR out2
);

extern "C" int neumann_quadshell4_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
        const real_t t0,
        const real_t t1,
        const real_t t2,
        const int out_stride,
        real_t *const RSTR out0,
        real_t *const RSTR out1,
        real_t *const RSTR out2
);

extern "C" int neumann_trishell3_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
        const real_t t0,
        const real_t t1,
        const real_t t2,
        const int out_stride,
        real_t *const RSTR out0,
        real_t *const RSTR out1,
        real_t *const RSTR out2
);

extern "C" int neumann_trishell6_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
        const real_t t0,
        const real_t t1,
        const real_t t2,
        const int out_stride,
        real_t *const RSTR out0,
        real_t *const RSTR out1,
        real_t *const RSTR out2
);
