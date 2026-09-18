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

#include "../../kernel_diagnostics.hpp"

#include "smesh_mesh.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" const sfem::codegen::KernelDiagnostics *body_force_jacobian_action_2d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *body_force_jacobian_action_3d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int body_force_residual_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const real_t g0,
        const real_t g1,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out
);

extern "C" const sfem::codegen::KernelDiagnostics *body_force_residual_2d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int body_force_residual_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
);

extern "C" int body_force_residual_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const real_t g0,
        const real_t g1,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out
);

extern "C" int body_force_residual_3d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const real_t g0,
        const real_t g1,
        const real_t g2,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out
);

extern "C" const sfem::codegen::KernelDiagnostics *body_force_residual_3d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int body_force_residual_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
);

extern "C" int body_force_residual_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const real_t g0,
        const real_t g1,
        const real_t g2,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out
);
