#include "sfem_GeneratedNeumann_c_abi.hpp"
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

extern "C" int neumann_proteus_quad4_edgeshell2_boundary_residual_sideset_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1);
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
extern "C" int neumann_proteus_quad4_edgeshell2_boundary_residual_sideset_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1);
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

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neumann_edgeshell2_boundary_residual_2d_sideset_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t1,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1
) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return neumann_proteus_quad4_edgeshell2_boundary_residual_sideset_soa(nsides, nnodes, elements, parent, side_idx, points, t0, t1, out_stride, out0, out1);
        case smesh::QUAD4:
            return neumann_quad4_edgeshell2_boundary_residual_sideset_soa(nsides, nnodes, elements, parent, side_idx, points, t0, t1, out_stride, out0, out1);
        case smesh::TRI3:
            return neumann_tri3_edgeshell2_boundary_residual_sideset_soa(nsides, nnodes, elements, parent, side_idx, points, t0, t1, out_stride, out0, out1);
        default:
            std::fprintf(stderr, "neumann_edgeshell2_boundary_residual_2d_sideset_soa does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neumann_edgeshell2_boundary_residual_2d_sideset_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t1,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1
) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return neumann_proteus_quad4_edgeshell2_boundary_residual_sideset_soa_float(nsides, nnodes, elements, parent, side_idx, points, t0, t1, out_stride, out0, out1);
        case smesh::QUAD4:
            return neumann_quad4_edgeshell2_boundary_residual_sideset_soa_float(nsides, nnodes, elements, parent, side_idx, points, t0, t1, out_stride, out0, out1);
        case smesh::TRI3:
            return neumann_tri3_edgeshell2_boundary_residual_sideset_soa_float(nsides, nnodes, elements, parent, side_idx, points, t0, t1, out_stride, out0, out1);
        default:
            std::fprintf(stderr, "neumann_edgeshell2_boundary_residual_2d_sideset_soa_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neumann_proteus_quadshell16_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t1,
        const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2
) {
    switch (element_type) {
        case smesh::PROTEUS_HEX64:
            return neumann_proteus_hex64_proteus_quadshell16_boundary_residual_sideset_soa(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
        default:
            std::fprintf(stderr, "neumann_proteus_quadshell16_boundary_residual_3d_sideset_soa does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neumann_proteus_quadshell16_boundary_residual_3d_sideset_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t1,
        const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2
) {
    switch (element_type) {
        case smesh::PROTEUS_HEX64:
            return neumann_proteus_hex64_proteus_quadshell16_boundary_residual_sideset_soa_float(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
        default:
            std::fprintf(stderr, "neumann_proteus_quadshell16_boundary_residual_3d_sideset_soa_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neumann_proteus_quadshell25_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t1,
        const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2
) {
    switch (element_type) {
        case smesh::PROTEUS_HEX125:
            return neumann_proteus_hex125_proteus_quadshell25_boundary_residual_sideset_soa(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
        default:
            std::fprintf(stderr, "neumann_proteus_quadshell25_boundary_residual_3d_sideset_soa does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neumann_proteus_quadshell25_boundary_residual_3d_sideset_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t1,
        const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2
) {
    switch (element_type) {
        case smesh::PROTEUS_HEX125:
            return neumann_proteus_hex125_proteus_quadshell25_boundary_residual_sideset_soa_float(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
        default:
            std::fprintf(stderr, "neumann_proteus_quadshell25_boundary_residual_3d_sideset_soa_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neumann_proteus_quadshell4_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t1,
        const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2
) {
    switch (element_type) {
        case smesh::PROTEUS_HEX8:
            return neumann_proteus_hex8_proteus_quadshell4_boundary_residual_sideset_soa(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
        default:
            std::fprintf(stderr, "neumann_proteus_quadshell4_boundary_residual_3d_sideset_soa does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neumann_proteus_quadshell4_boundary_residual_3d_sideset_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t1,
        const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2
) {
    switch (element_type) {
        case smesh::PROTEUS_HEX8:
            return neumann_proteus_hex8_proteus_quadshell4_boundary_residual_sideset_soa_float(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
        default:
            std::fprintf(stderr, "neumann_proteus_quadshell4_boundary_residual_3d_sideset_soa_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neumann_proteus_quadshell9_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t1,
        const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2
) {
    switch (element_type) {
        case smesh::PROTEUS_HEX27:
            return neumann_proteus_hex27_proteus_quadshell9_boundary_residual_sideset_soa(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
        default:
            std::fprintf(stderr, "neumann_proteus_quadshell9_boundary_residual_3d_sideset_soa does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neumann_proteus_quadshell9_boundary_residual_3d_sideset_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t1,
        const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2
) {
    switch (element_type) {
        case smesh::PROTEUS_HEX27:
            return neumann_proteus_hex27_proteus_quadshell9_boundary_residual_sideset_soa_float(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
        default:
            std::fprintf(stderr, "neumann_proteus_quadshell9_boundary_residual_3d_sideset_soa_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neumann_quadshell4_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t1,
        const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2
) {
    switch (element_type) {
        case smesh::HEX8:
            return neumann_hex8_quadshell4_boundary_residual_sideset_soa(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
        default:
            std::fprintf(stderr, "neumann_quadshell4_boundary_residual_3d_sideset_soa does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neumann_quadshell4_boundary_residual_3d_sideset_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t1,
        const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2
) {
    switch (element_type) {
        case smesh::HEX8:
            return neumann_hex8_quadshell4_boundary_residual_sideset_soa_float(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
        default:
            std::fprintf(stderr, "neumann_quadshell4_boundary_residual_3d_sideset_soa_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neumann_quadshell9_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t1,
        const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2
) {
    switch (element_type) {
        case smesh::HEX27:
            return neumann_hex27_quadshell9_boundary_residual_sideset_soa(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
        default:
            std::fprintf(stderr, "neumann_quadshell9_boundary_residual_3d_sideset_soa does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neumann_quadshell9_boundary_residual_3d_sideset_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t1,
        const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2
) {
    switch (element_type) {
        case smesh::HEX27:
            return neumann_hex27_quadshell9_boundary_residual_sideset_soa_float(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
        default:
            std::fprintf(stderr, "neumann_quadshell9_boundary_residual_3d_sideset_soa_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neumann_trishell3_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t1,
        const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2
) {
    switch (element_type) {
        case smesh::TET4:
            return neumann_tet4_trishell3_boundary_residual_sideset_soa(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
        default:
            std::fprintf(stderr, "neumann_trishell3_boundary_residual_3d_sideset_soa does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neumann_trishell3_boundary_residual_3d_sideset_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t1,
        const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2
) {
    switch (element_type) {
        case smesh::TET4:
            return neumann_tet4_trishell3_boundary_residual_sideset_soa_float(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
        default:
            std::fprintf(stderr, "neumann_trishell3_boundary_residual_3d_sideset_soa_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neumann_trishell6_boundary_residual_3d_sideset_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t t0,
        const real_t t1,
        const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2
) {
    switch (element_type) {
        case smesh::TET10:
            return neumann_tet10_trishell6_boundary_residual_sideset_soa(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
        default:
            std::fprintf(stderr, "neumann_trishell6_boundary_residual_3d_sideset_soa does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neumann_trishell6_boundary_residual_3d_sideset_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float t0,
        const float t1,
        const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2
) {
    switch (element_type) {
        case smesh::TET10:
            return neumann_tet10_trishell6_boundary_residual_sideset_soa_float(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
        default:
            std::fprintf(stderr, "neumann_trishell6_boundary_residual_3d_sideset_soa_float does not support element type %d\n", (int)element_type);
            return SFEM_FAILURE;
    }
}
