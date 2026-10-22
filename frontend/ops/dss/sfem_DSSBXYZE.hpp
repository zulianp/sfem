#pragma once

#include "sfem_aliases.hpp"
#include "sfem_Operator.hpp"
#include "sfem_dss_bxyze_affine_laplacian.hpp"
#include "sfem_dss_bxyze_geometry.hpp"

namespace sfem {

struct DSSBXYZEBuffer {
    int                 level{0};
    int                 element_block_size{0};
    ptrdiff_t           nelements{0};
    ptrdiff_t           nvalues{0};
    SharedBuffer<real_t> values;
};

struct DSSBXYZEGeometry {
    sfem_dss_bxyze_layout_t layout{};
    ptrdiff_t              block_stride{0};
    ptrdiff_t              nvalues{0};
    SharedBuffer<scalar_t> values;
};

int dss_bxyze_affine_geometry(const SharedMesh &mesh,
                              const sfem_dss_bxyze_layout_t &layout,
                              DSSBXYZEGeometry *out);

// Setup-only inverse multiplicities: fill ones, DSS, reciprocal. Full blocks;
// same structured element ordering and local orientations as structured DSS.
SharedBuffer<real_t> dss_bxyze_structured_weights(const sfem_dss_bxyze_layout_t &layout,
                                                ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz);

// Prepared host operator for homogeneous Dirichlet CG: volume, DSS, then
// identity rows at the constrained cellwise offsets (all copies, no duplicates).
// Input vectors must vanish at those offsets. The operator retains all buffers
// and a copy of op; no allocations or dimension dispatch during apply.
// workspace must cover the maximum thread count used for every apply.
// One workspace cannot be used by concurrent applies.
std::shared_ptr<Operator<real_t>> dss_bxyze_laplacian_affine_operator(
        const sfem_dss_bxyze_laplacian_affine_volumetric_t &op,
        ptrdiff_t nx, ptrdiff_t ny, ptrdiff_t nz,
        const SharedBuffer<scalar_t> &S, const SharedBuffer<scalar_t> &D,
        const SharedBuffer<scalar_t> &qw, const SharedBuffer<scalar_t> &geometry,
        const SharedBuffer<scalar_t> &workspace,
        const SharedBuffer<ptrdiff_t> &dirichlet = nullptr);

int dss_global_to_bxyze(const SharedMesh &mesh,
                        int element_block_size,
                        const real_t *const global,
                        DSSBXYZEBuffer *const out);

int dss_bxyze_structured_inplace(ptrdiff_t nx,
                                 ptrdiff_t ny,
                                 ptrdiff_t nz,
                                 DSSBXYZEBuffer *const buffer);

int dss_bxyze_laplacian_cartesian_apply(ptrdiff_t nx,
                                        ptrdiff_t ny,
                                        ptrdiff_t nz,
                                        real_t hx,
                                        real_t hy,
                                        real_t hz,
                                        real_t kappa,
                                        const DSSBXYZEBuffer &x,
                                        DSSBXYZEBuffer *const out);

}  // namespace sfem
