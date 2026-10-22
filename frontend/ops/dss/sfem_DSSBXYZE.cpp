#include "sfem_DSSBXYZE.hpp"

#include "sfem_dss_bxyze.hpp"
#include "sfem_dss_bxyze_laplacian.hpp"
#include "sfem_dss_bxyze_structured.hpp"
#include "smesh_semistructured.hpp"
#include "smesh_sshex8.hpp"

#include <cstdio>
#include <limits>

#ifdef SFEM_ENABLE_OPENMP
#include <omp.h>
#endif

namespace sfem {

namespace {

bool structured_shape_matches(const sfem_dss_bxyze_layout_t &layout,
                              const ptrdiff_t nx, const ptrdiff_t ny, const ptrdiff_t nz) {
    const ptrdiff_t limit = std::numeric_limits<ptrdiff_t>::max();
    return nx > 0 && ny > 0 && nz > 0 && nx <= limit / ny && nx * ny <= limit / nz &&
           nx * ny * nz == layout.nelements;
}

class DSSBXYZEAffineOperator final : public Operator<real_t> {
public:
    sfem_dss_bxyze_laplacian_affine_volumetric_t op{};
    ptrdiff_t nx, ny, nz, workspace_threads;
    SharedBuffer<scalar_t> S, D, qw, geometry, workspace;
    SharedBuffer<ptrdiff_t> dirichlet;

    ptrdiff_t rows() const override { return op.layout.nblocks * op.layout.block_stride; }
    ptrdiff_t cols() const override { return rows(); }
    ExecutionSpace execution_space() const override { return EXECUTION_SPACE_HOST; }

    int apply(const real_t *const x, real_t *const y) override {
#ifdef SFEM_ENABLE_OPENMP
        if (omp_get_max_threads() > workspace_threads) return SFEM_FAILURE;
#endif
        int status = sfem_dss_bxyze_laplacian_affine_volumetric_apply(
                &op, S->data(), D->data(), qw->data(), geometry->data(), x, y, workspace->data());
        if (status != SFEM_SUCCESS) return status;
        status = sfem_dss_bxyze_structured_inplace(
                op.layout.level, nx, ny, nz, op.layout.element_block_size, y);
        if (status != SFEM_SUCCESS) return status;
        if (dirichlet && dirichlet->size()) {
            const ptrdiff_t *const ids = dirichlet->data();
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < (ptrdiff_t)dirichlet->size(); ++i) y[ids[i]] = x[ids[i]];
        }
        return SFEM_SUCCESS;
    }
};

}  // namespace

SharedBuffer<real_t> dss_bxyze_structured_weights(const sfem_dss_bxyze_layout_t &layout,
                                                const ptrdiff_t nx, const ptrdiff_t ny, const ptrdiff_t nz) {
    if (!structured_shape_matches(layout, nx, ny, nz) ||
        sfem_dss_bxyze_laplacian_affine_volumetric_workspace_n_values(&layout, layout.n) < 0) return nullptr;
    const ptrdiff_t n = layout.nblocks * layout.block_stride;
    auto weights = create_host_buffer<real_t>((size_t)n);
    if (!weights || !weights->data()) return nullptr;
    real_t *const w = weights->data();
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; ++i) w[i] = 1;
    if (sfem_dss_bxyze_structured_inplace(layout.level, nx, ny, nz, layout.element_block_size, w) != SFEM_SUCCESS)
        return nullptr;
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; ++i) w[i] = 1 / w[i];
    return weights;
}

std::shared_ptr<Operator<real_t>> dss_bxyze_laplacian_affine_operator(
        const sfem_dss_bxyze_laplacian_affine_volumetric_t &op,
        const ptrdiff_t nx, const ptrdiff_t ny, const ptrdiff_t nz,
        const SharedBuffer<scalar_t> &S, const SharedBuffer<scalar_t> &D,
        const SharedBuffer<scalar_t> &qw, const SharedBuffer<scalar_t> &geometry,
        const SharedBuffer<scalar_t> &workspace, const SharedBuffer<ptrdiff_t> &dirichlet) {
    const ptrdiff_t stride = sfem_dss_bxyze_laplacian_affine_volumetric_workspace_n_values(&op.layout, op.Q);
    const auto enough = [](const auto &buffer, const size_t n) {
        return buffer && buffer->mem_space() == MEMORY_SPACE_HOST && buffer->size() >= n;
    };
    int threads = 1;
#ifdef SFEM_ENABLE_OPENMP
    threads = omp_get_max_threads();
#endif
    if (!op.kernel || stride <= 0 || stride != op.workspace_stride ||
        !structured_shape_matches(op.layout, nx, ny, nz) ||
        !enough(S, (size_t)op.Q * op.layout.n) || !enough(D, (size_t)op.Q * op.layout.n) ||
        !enough(qw, (size_t)op.Q) || !enough(geometry, (size_t)6 * op.layout.nelements) ||
        !enough(workspace, (size_t)threads * stride) ||
        (dirichlet && dirichlet->mem_space() != MEMORY_SPACE_HOST)) return nullptr;
    const ptrdiff_t n = op.layout.nblocks * op.layout.block_stride;
    if (dirichlet) {
        for (size_t i = 0; i < dirichlet->size(); ++i) {
            if (dirichlet->data()[i] < 0 || dirichlet->data()[i] >= n) return nullptr;
        }
    }
    auto ret = std::make_shared<DSSBXYZEAffineOperator>();
    ret->op = op;
    ret->nx = nx; ret->ny = ny; ret->nz = nz;
    ret->workspace_threads = workspace->size() / stride;
    ret->S = S; ret->D = D; ret->qw = qw;
    ret->geometry = geometry; ret->workspace = workspace; ret->dirichlet = dirichlet;
    return ret;
}

int dss_bxyze_affine_geometry(const SharedMesh &mesh,
                              const sfem_dss_bxyze_layout_t &layout,
                              DSSBXYZEGeometry *const out) {
    if (!out) {
        std::fprintf(stderr, "dss_bxyze_affine_geometry: output buffer descriptor is null\n");
        return SFEM_FAILURE;
    }
    const sfem_dss_bxyze_layout_t geometry_layout = layout;
    *out = DSSBXYZEGeometry{};

    if (!mesh || mesh->n_blocks() != 1 || !smesh::is_semistructured_type(mesh->element_type(0)) ||
        !smesh::is_hex_ss_family(mesh->element_type(0))) {
        std::fprintf(stderr, "dss_bxyze_affine_geometry: expected one semi-structured hex mesh block\n");
        return SFEM_FAILURE;
    }

    const auto block = mesh->block(0);
    const ptrdiff_t nvalues = sfem_dss_bxyze_affine_geometry_n_values(&geometry_layout);
    if (!block || nvalues < 0 || smesh::semistructured_level(*mesh) != geometry_layout.level ||
        block->n_elements() != geometry_layout.nelements) {
        std::fprintf(stderr, "dss_bxyze_affine_geometry: mesh and BXYZE layout are inconsistent\n");
        return SFEM_FAILURE;
    }

    auto values = create_host_buffer<scalar_t>((size_t)nvalues);
    if (!values) {
        std::fprintf(stderr, "dss_bxyze_affine_geometry: failed to allocate geometry buffer\n");
        return SFEM_FAILURE;
    }

    const int err = sfem_dss_bxyze_affine_geometry(
            &geometry_layout, block->elements()->data(), mesh->points()->data(), values->data());
    if (err != SFEM_SUCCESS) {
        return err;
    }

    out->layout       = geometry_layout;
    out->block_stride = (ptrdiff_t)6 * geometry_layout.element_block_size;
    out->nvalues      = nvalues;
    out->values       = values;
    return SFEM_SUCCESS;
}

int dss_global_to_bxyze(const SharedMesh &mesh,
                        const int         element_block_size,
                        const real_t *const global,
                        DSSBXYZEBuffer *const out) {
    if (!out) {
        std::fprintf(stderr, "dss_global_to_bxyze: output buffer descriptor is null\n");
        return SFEM_FAILURE;
    }

    *out = DSSBXYZEBuffer{};

    if (!mesh || !global || element_block_size <= 0) {
        std::fprintf(stderr, "dss_global_to_bxyze: invalid input\n");
        return SFEM_FAILURE;
    }

    if (mesh->n_blocks() != 1) {
        std::fprintf(stderr,
                     "dss_global_to_bxyze: expected exactly one mesh block, got %zu\n",
                     mesh->n_blocks());
        return SFEM_FAILURE;
    }

    const auto element_type = mesh->element_type(0);
    if (!smesh::is_semistructured_type(element_type) || !smesh::is_hex_ss_family(element_type)) {
        std::fprintf(stderr,
                     "dss_global_to_bxyze: expected semi-structured hex element type, got %s\n",
                     smesh::type_to_string(element_type));
        return SFEM_FAILURE;
    }

    auto block = mesh->block(0);
    if (!block) {
        std::fprintf(stderr, "dss_global_to_bxyze: mesh block 0 is null\n");
        return SFEM_FAILURE;
    }

    const int       level   = smesh::semistructured_level(*mesh);
    const ptrdiff_t ne      = block->n_elements();
    const ptrdiff_t nvalues = sfem_dss_bxyze_n_values(level, ne, element_block_size);

    if (nvalues < 0) {
        std::fprintf(stderr, "dss_global_to_bxyze: invalid BXYZE layout\n");
        return SFEM_FAILURE;
    }

    auto values = create_host_buffer<real_t>((size_t)nvalues);
    if (!values) {
        std::fprintf(stderr, "dss_global_to_bxyze: failed to allocate BXYZE buffer\n");
        return SFEM_FAILURE;
    }

    const int err = sfem_dss_global_to_bxyze(level,
                                             ne,
                                             element_block_size,
                                             block->elements()->data(),
                                             global,
                                             values->data());

    if (err != SFEM_SUCCESS) {
        return err;
    }

    out->level              = level;
    out->element_block_size = element_block_size;
    out->nelements          = ne;
    out->nvalues            = nvalues;
    out->values             = values;

    return SFEM_SUCCESS;
}

int dss_bxyze_structured_inplace(const ptrdiff_t nx,
                                 const ptrdiff_t ny,
                                 const ptrdiff_t nz,
                                 DSSBXYZEBuffer *const buffer) {
    if (!buffer || !buffer->values) {
        std::fprintf(stderr, "dss_bxyze_structured_inplace: invalid BXYZE buffer\n");
        return SFEM_FAILURE;
    }

    if (nx <= 0 || ny <= 0 || nz <= 0) {
        std::fprintf(stderr, "dss_bxyze_structured_inplace: invalid structured dimensions\n");
        return SFEM_FAILURE;
    }

    const ptrdiff_t ne = nx * ny * nz;
    if (buffer->nelements != ne) {
        std::fprintf(stderr,
                     "dss_bxyze_structured_inplace: dimensions imply %ld elements, buffer has %ld\n",
                     (long)ne,
                     (long)buffer->nelements);
        return SFEM_FAILURE;
    }

    const ptrdiff_t nvalues =
            sfem_dss_bxyze_n_values(buffer->level, buffer->nelements, buffer->element_block_size);

    if (nvalues != buffer->nvalues || (ptrdiff_t)buffer->values->size() != buffer->nvalues) {
        std::fprintf(stderr, "dss_bxyze_structured_inplace: inconsistent BXYZE buffer metadata\n");
        return SFEM_FAILURE;
    }

    return sfem_dss_bxyze_structured_inplace(buffer->level,
                                            nx,
                                            ny,
                                            nz,
                                            buffer->element_block_size,
                                            buffer->values->data());
}

int dss_bxyze_laplacian_cartesian_apply(const ptrdiff_t        nx,
                                        const ptrdiff_t        ny,
                                        const ptrdiff_t        nz,
                                        const real_t           hx,
                                        const real_t           hy,
                                        const real_t           hz,
                                        const real_t           kappa,
                                        const DSSBXYZEBuffer  &x,
                                        DSSBXYZEBuffer *const  out) {
    if (!out) {
        std::fprintf(stderr, "dss_bxyze_laplacian_cartesian_apply: output buffer descriptor is null\n");
        return SFEM_FAILURE;
    }

    *out = DSSBXYZEBuffer{};

    if (!x.values || nx <= 0 || ny <= 0 || nz <= 0 || hx <= 0 || hy <= 0 || hz <= 0) {
        std::fprintf(stderr, "dss_bxyze_laplacian_cartesian_apply: invalid input\n");
        return SFEM_FAILURE;
    }

    const ptrdiff_t ne = nx * ny * nz;
    if (x.nelements != ne) {
        std::fprintf(stderr,
                     "dss_bxyze_laplacian_cartesian_apply: dimensions imply %ld elements, buffer has %ld\n",
                     (long)ne,
                     (long)x.nelements);
        return SFEM_FAILURE;
    }

    const ptrdiff_t nvalues = sfem_dss_bxyze_n_values(x.level, x.nelements, x.element_block_size);
    if (nvalues != x.nvalues || (ptrdiff_t)x.values->size() != x.nvalues) {
        std::fprintf(stderr, "dss_bxyze_laplacian_cartesian_apply: inconsistent BXYZE buffer metadata\n");
        return SFEM_FAILURE;
    }

    auto values = create_host_buffer<real_t>((size_t)x.nvalues);
    if (!values) {
        std::fprintf(stderr, "dss_bxyze_laplacian_cartesian_apply: failed to allocate output buffer\n");
        return SFEM_FAILURE;
    }

    const int err = sfem_dss_bxyze_laplacian_cartesian_apply(x.level,
                                                            nx,
                                                            ny,
                                                            nz,
                                                            x.element_block_size,
                                                            hx,
                                                            hy,
                                                            hz,
                                                            kappa,
                                                            x.values->data(),
                                                            values->data());

    if (err != SFEM_SUCCESS) {
        return err;
    }

    out->level              = x.level;
    out->element_block_size = x.element_block_size;
    out->nelements          = x.nelements;
    out->nvalues            = x.nvalues;
    out->values             = values;

    return SFEM_SUCCESS;
}

}  // namespace sfem
