#include "sfem_test.hpp"

#include "sfem_DSSBXYZE.hpp"
#include "sfem_dss_bxyze.hpp"
#include "sfem_dss_bxyze_structured.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"
#include "smesh_sshex8.hpp"

#include <algorithm>

static real_t local_value(const ptrdiff_t e, const int ix, const int iy, const int iz) {
    return (real_t)(1 + 1000 * e + 100 * iz + 10 * iy + ix);
}

static double tolerance() { return sizeof(real_t) == sizeof(float) ? 1e-5 : 1e-12; }

static int fill_reference(const int                  level,
                          const ptrdiff_t            ne,
                          const int                  bsize,
                          const idx_t *const *const  elements,
                          const real_t *const        original,
                          real_t *const              assembled,
                          real_t *const              reference) {
    const int n = level + 1;

    for (ptrdiff_t e = 0; e < ne; ++e) {
        for (int iz = 0; iz < n; ++iz) {
            for (int iy = 0; iy < n; ++iy) {
                for (int ix = 0; ix < n; ++ix) {
                    const int       lidx = smesh::sshex8_lidx(level, ix, iy, iz);
                    const idx_t     gid  = elements[lidx][e];
                    const ptrdiff_t in   = sfem_dss_bxyze_index(level, bsize, e, ix, iy, iz);

                    assembled[gid] += original[in];
                }
            }
        }
    }

    for (ptrdiff_t e = 0; e < ne; ++e) {
        for (int iz = 0; iz < n; ++iz) {
            for (int iy = 0; iy < n; ++iy) {
                for (int ix = 0; ix < n; ++ix) {
                    const int       lidx = smesh::sshex8_lidx(level, ix, iy, iz);
                    const idx_t     gid  = elements[lidx][e];
                    const ptrdiff_t out  = sfem_dss_bxyze_index(level, bsize, e, ix, iy, iz);

                    reference[out] = assembled[gid];
                }
            }
        }
    }

    return SFEM_TEST_SUCCESS;
}

static int run_structured_dss_case(const ptrdiff_t nx,
                                   const ptrdiff_t ny,
                                   const ptrdiff_t nz,
                                   const int       level,
                                   const int       bsize,
                                   const bool      use_frontend_wrapper) {
    auto macro_mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), nx, ny, nz, 0, 0, 0, 1, 1, 1);
    auto mesh       = smesh::to_semistructured(level, macro_mesh, false, false);
    auto block      = mesh->block(0);
    auto elements   = block->elements()->data();

    const ptrdiff_t ne      = block->n_elements();
    const ptrdiff_t nvalues = sfem_dss_bxyze_n_values(level, ne, bsize);

    SFEM_TEST_EQ(ne, nx * ny * nz);
    SFEM_TEST_ASSERT(nvalues > 0);

    auto original  = sfem::create_host_buffer<real_t>((size_t)nvalues);
    auto work      = sfem::create_host_buffer<real_t>((size_t)nvalues);
    auto reference = sfem::create_host_buffer<real_t>((size_t)nvalues);
    auto assembled = sfem::create_host_buffer<real_t>((size_t)mesh->n_nodes());

    std::fill(original->data(), original->data() + nvalues, (real_t)-1);
    std::fill(work->data(), work->data() + nvalues, (real_t)-1);
    std::fill(reference->data(), reference->data() + nvalues, (real_t)-2);
    std::fill(assembled->data(), assembled->data() + mesh->n_nodes(), (real_t)0);

    for (ptrdiff_t e = 0; e < ne; ++e) {
        for (int iz = 0; iz <= level; ++iz) {
            for (int iy = 0; iy <= level; ++iy) {
                for (int ix = 0; ix <= level; ++ix) {
                    const ptrdiff_t out = sfem_dss_bxyze_index(level, bsize, e, ix, iy, iz);
                    original->data()[out] = local_value(e, ix, iy, iz);
                    work->data()[out]     = original->data()[out];
                }
            }
        }
    }

    SFEM_TEST_EQ(fill_reference(level,
                                ne,
                                bsize,
                                elements,
                                original->data(),
                                assembled->data(),
                                reference->data()),
                 SFEM_TEST_SUCCESS);

    if (use_frontend_wrapper) {
        sfem::DSSBXYZEBuffer buffer;
        buffer.level              = level;
        buffer.element_block_size = bsize;
        buffer.nelements          = ne;
        buffer.nvalues            = nvalues;
        buffer.values             = work;

        SFEM_TEST_EQ(sfem::dss_bxyze_structured_inplace(nx, ny, nz, &buffer), SFEM_SUCCESS);
    } else {
        SFEM_TEST_EQ(sfem_dss_bxyze_structured_inplace(level, nx, ny, nz, bsize, work->data()), SFEM_SUCCESS);
    }

    const double tol = tolerance();
    for (ptrdiff_t e = 0; e < ne; ++e) {
        for (int iz = 0; iz <= level; ++iz) {
            for (int iy = 0; iy <= level; ++iy) {
                for (int ix = 0; ix <= level; ++ix) {
                    const ptrdiff_t out = sfem_dss_bxyze_index(level, bsize, e, ix, iy, iz);
                    SFEM_TEST_APPROXEQ((double)work->data()[out], (double)reference->data()[out], tol);
                }
            }
        }
    }

    return SFEM_TEST_SUCCESS;
}

static int test_structured_dss_crosses_data_blocks() {
    return run_structured_dss_case(5, 4, 3, 3, 4, false);
}

static int test_structured_dss_bsize_one() {
    return run_structured_dss_case(3, 2, 2, 4, 1, false);
}

static int test_structured_dss_frontend_buffer_wrapper() {
    return run_structured_dss_case(6, 3, 2, 2, 8, true);
}

static int test_structured_dss_rejects_invalid_inputs() {
    SFEM_TEST_EQ(sfem_dss_bxyze_structured_inplace(0, 1, 1, 1, 4, nullptr), SFEM_FAILURE);
    SFEM_TEST_EQ(sfem_dss_bxyze_structured_inplace(2, 0, 1, 1, 4, nullptr), SFEM_FAILURE);
    SFEM_TEST_EQ(sfem_dss_bxyze_structured_inplace(2, 1, 1, 1, 0, nullptr), SFEM_FAILURE);

    sfem::DSSBXYZEBuffer buffer;
    SFEM_TEST_EQ(sfem::dss_bxyze_structured_inplace(1, 1, 1, &buffer), SFEM_FAILURE);

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_structured_dss_crosses_data_blocks);
    SFEM_RUN_TEST(test_structured_dss_bsize_one);
    SFEM_RUN_TEST(test_structured_dss_frontend_buffer_wrapper);
    SFEM_RUN_TEST(test_structured_dss_rejects_invalid_inputs);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
