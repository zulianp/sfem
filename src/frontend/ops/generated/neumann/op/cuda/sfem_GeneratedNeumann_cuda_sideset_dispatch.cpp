#include "sfem_GeneratedNeumann_cuda_c_abi.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

static_assert((int)smesh::SMESH_FLOAT64 == (int)sizeof(double),
              "the generated kernels select their scalar by width");
static_assert((int)smesh::SMESH_FLOAT32 == (int)sizeof(float),
              "the generated kernels select their scalar by width");

extern "C" int cu_neumann_proteus_quad4_edgeshell2_boundary_residual_ss_soa(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const real_t t0, const real_t t1,
    const int out_stride,
    real_t *const RSTR out0,
    real_t *const RSTR out1,
    void *const stream);
extern "C" int cu_neumann_proteus_quad4_edgeshell2_boundary_residual_ss_soa_float(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const float t0, const float t1,
    const int out_stride,
    float *const RSTR out0,
    float *const RSTR out1,
    void *const stream);
extern "C" int cu_neumann_quad4_edgeshell2_boundary_residual_ss_soa(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const real_t t0, const real_t t1,
    const int out_stride,
    real_t *const RSTR out0,
    real_t *const RSTR out1,
    void *const stream);
extern "C" int cu_neumann_quad4_edgeshell2_boundary_residual_ss_soa_float(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const float t0, const float t1,
    const int out_stride,
    float *const RSTR out0,
    float *const RSTR out1,
    void *const stream);
extern "C" int cu_neumann_tri3_edgeshell2_boundary_residual_ss_soa(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const real_t t0, const real_t t1,
    const int out_stride,
    real_t *const RSTR out0,
    real_t *const RSTR out1,
    void *const stream);
extern "C" int cu_neumann_tri3_edgeshell2_boundary_residual_ss_soa_float(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const float t0, const float t1,
    const int out_stride,
    float *const RSTR out0,
    float *const RSTR out1,
    void *const stream);
extern "C" int cu_neumann_proteus_hex8_proteus_quadshell4_boundary_residual_ss_soa(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const real_t t0, const real_t t1, const real_t t2,
    const int out_stride,
    real_t *const RSTR out0,
    real_t *const RSTR out1,
    real_t *const RSTR out2,
    void *const stream);
extern "C" int cu_neumann_proteus_hex8_proteus_quadshell4_boundary_residual_ss_soa_float(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const float t0, const float t1, const float t2,
    const int out_stride,
    float *const RSTR out0,
    float *const RSTR out1,
    float *const RSTR out2,
    void *const stream);
extern "C" int cu_neumann_hex8_quadshell4_boundary_residual_ss_soa(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const real_t t0, const real_t t1, const real_t t2,
    const int out_stride,
    real_t *const RSTR out0,
    real_t *const RSTR out1,
    real_t *const RSTR out2,
    void *const stream);
extern "C" int cu_neumann_hex8_quadshell4_boundary_residual_ss_soa_float(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const float t0, const float t1, const float t2,
    const int out_stride,
    float *const RSTR out0,
    float *const RSTR out1,
    float *const RSTR out2,
    void *const stream);
extern "C" int cu_neumann_tet4_trishell3_boundary_residual_ss_soa(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const real_t t0, const real_t t1, const real_t t2,
    const int out_stride,
    real_t *const RSTR out0,
    real_t *const RSTR out1,
    real_t *const RSTR out2,
    void *const stream);
extern "C" int cu_neumann_tet4_trishell3_boundary_residual_ss_soa_float(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const float t0, const float t1, const float t2,
    const int out_stride,
    float *const RSTR out0,
    float *const RSTR out1,
    float *const RSTR out2,
    void *const stream);
extern "C" int cu_neumann_tet10_trishell6_boundary_residual_ss_soa(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const real_t t0, const real_t t1, const real_t t2,
    const int out_stride,
    real_t *const RSTR out0,
    real_t *const RSTR out1,
    real_t *const RSTR out2,
    void *const stream);
extern "C" int cu_neumann_tet10_trishell6_boundary_residual_ss_soa_float(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const float t0, const float t1, const float t2,
    const int out_stride,
    float *const RSTR out0,
    float *const RSTR out1,
    float *const RSTR out2,
    void *const stream);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_neumann_edgeshell2_boundary_residual_2d_ss_soa(
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
        real_t *const RSTR out1,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return cu_neumann_proteus_quad4_edgeshell2_boundary_residual_ss_soa(nsides, nnodes, elements, parent, side_idx, points, t0, t1, out_stride, (double *)out0, (double *)out1, stream);
        case smesh::SMESH_FLOAT32:
          return cu_neumann_proteus_quad4_edgeshell2_boundary_residual_ss_soa_float(nsides, nnodes, elements, parent, side_idx, points, t0, t1, out_stride, (float *)out0, (float *)out1, stream);
        default:
          break;
      }
      break;
    }
    case smesh::QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return cu_neumann_quad4_edgeshell2_boundary_residual_ss_soa(nsides, nnodes, elements, parent, side_idx, points, t0, t1, out_stride, (double *)out0, (double *)out1, stream);
        case smesh::SMESH_FLOAT32:
          return cu_neumann_quad4_edgeshell2_boundary_residual_ss_soa_float(nsides, nnodes, elements, parent, side_idx, points, t0, t1, out_stride, (float *)out0, (float *)out1, stream);
        default:
          break;
      }
      break;
    }
    case smesh::TRI3: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return cu_neumann_tri3_edgeshell2_boundary_residual_ss_soa(nsides, nnodes, elements, parent, side_idx, points, t0, t1, out_stride, (double *)out0, (double *)out1, stream);
        case smesh::SMESH_FLOAT32:
          return cu_neumann_tri3_edgeshell2_boundary_residual_ss_soa_float(nsides, nnodes, elements, parent, side_idx, points, t0, t1, out_stride, (float *)out0, (float *)out1, stream);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "neumann_edgeshell2_boundary_residual_2d_ss_soa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_neumann_proteus_quadshell4_boundary_residual_3d_ss_soa(
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
        real_t *const RSTR out2,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return cu_neumann_proteus_hex8_proteus_quadshell4_boundary_residual_ss_soa(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, (double *)out0, (double *)out1, (double *)out2, stream);
        case smesh::SMESH_FLOAT32:
          return cu_neumann_proteus_hex8_proteus_quadshell4_boundary_residual_ss_soa_float(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, (float *)out0, (float *)out1, (float *)out2, stream);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "neumann_proteus_quadshell4_boundary_residual_3d_ss_soa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_neumann_quadshell4_boundary_residual_3d_ss_soa(
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
        real_t *const RSTR out2,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return cu_neumann_hex8_quadshell4_boundary_residual_ss_soa(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, (double *)out0, (double *)out1, (double *)out2, stream);
        case smesh::SMESH_FLOAT32:
          return cu_neumann_hex8_quadshell4_boundary_residual_ss_soa_float(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, (float *)out0, (float *)out1, (float *)out2, stream);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "neumann_quadshell4_boundary_residual_3d_ss_soa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_neumann_trishell3_boundary_residual_3d_ss_soa(
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
        real_t *const RSTR out2,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TET4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return cu_neumann_tet4_trishell3_boundary_residual_ss_soa(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, (double *)out0, (double *)out1, (double *)out2, stream);
        case smesh::SMESH_FLOAT32:
          return cu_neumann_tet4_trishell3_boundary_residual_ss_soa_float(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, (float *)out0, (float *)out1, (float *)out2, stream);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "neumann_trishell3_boundary_residual_3d_ss_soa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_neumann_trishell6_boundary_residual_3d_ss_soa(
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
        real_t *const RSTR out2,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TET10: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return cu_neumann_tet10_trishell6_boundary_residual_ss_soa(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, (double *)out0, (double *)out1, (double *)out2, stream);
        case smesh::SMESH_FLOAT32:
          return cu_neumann_tet10_trishell6_boundary_residual_ss_soa_float(nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, (float *)out0, (float *)out1, (float *)out2, stream);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "neumann_trishell6_boundary_residual_3d_ss_soa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}
