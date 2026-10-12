#include "sfem_test.hpp"

#include "sfem_aliases.hpp"
#include "sfem_dss_bxyze.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"
#include "smesh_sshex8.hpp"

#include <cstdint>
#include <vector>

static_assert(sizeof(idx_t) == sizeof(int32_t), "This BXYZE test expects int32 SFEM connectivity.");

static double global_value64(const ptrdiff_t gid) { return 0.25 + 1.25 * (double)gid; }

static float global_value32(const ptrdiff_t gid) { return (float)(0.5 + 0.75 * (double)gid); }

static int test_global_to_bxyze_real64_hex_high_order() {
    const ptrdiff_t nx    = 3;
    const ptrdiff_t ny    = 2;
    const ptrdiff_t nz    = 1;
    const int       level = 4;
    const int       bsize = 4;

    auto macro_mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), nx, ny, nz, 0, 0, 0, 1, 1, 1);
    auto mesh       = smesh::to_semistructured(level, macro_mesh, false, false);
    auto block      = mesh->block(0);
    auto elements   = block->elements()->data();

    const ptrdiff_t ne  = block->n_elements();
    const int       nxe = smesh::sshex8_nxe(level);

    SFEM_TEST_EQ(ne, nx * ny * nz);
    SFEM_TEST_EQ(block->n_nodes_per_element(), nxe);

    const ptrdiff_t nblocks          = (ne + bsize - 1) / bsize;
    const ptrdiff_t expected_nvalues = nblocks * bsize * nxe;
    const ptrdiff_t nvalues          = sfem_dss_bxyze_n_values(level, ne, bsize);
    SFEM_TEST_EQ(nvalues, expected_nvalues);

    std::vector<double> global((size_t)mesh->n_nodes());
    for (ptrdiff_t gid = 0; gid < mesh->n_nodes(); ++gid) {
        global[(size_t)gid] = global_value64(gid);
    }

    std::vector<double> bxyze((size_t)nvalues, -777.0);

    auto elements32 = reinterpret_cast<const int32_t *const *>(elements);
    SFEM_TEST_EQ(sfem_dss_global_to_bxyze_real64(level, ne, bsize, elements32, global.data(), bxyze.data()), SFEM_SUCCESS);

    for (ptrdiff_t e = 0; e < ne; ++e) {
        for (int iz = 0; iz <= level; ++iz) {
            for (int iy = 0; iy <= level; ++iy) {
                for (int ix = 0; ix <= level; ++ix) {
                    const int       lidx = smesh::sshex8_lidx(level, ix, iy, iz);
                    const idx_t     gid  = elements[lidx][e];
                    const ptrdiff_t out  = sfem_dss_bxyze_index(level, bsize, e, ix, iy, iz);

                    SFEM_TEST_APPROXEQ(bxyze[(size_t)out], global[(size_t)gid], 0.0);
                }
            }
        }
    }

    for (ptrdiff_t e = ne; e < nblocks * bsize; ++e) {
        for (int iz = 0; iz <= level; ++iz) {
            for (int iy = 0; iy <= level; ++iy) {
                for (int ix = 0; ix <= level; ++ix) {
                    const ptrdiff_t out = sfem_dss_bxyze_index(level, bsize, e, ix, iy, iz);
                    SFEM_TEST_APPROXEQ(bxyze[(size_t)out], 0.0, 0.0);
                }
            }
        }
    }

    for (int iz = 0; iz <= level; ++iz) {
        for (int iy = 0; iy <= level; ++iy) {
            const ptrdiff_t left_face  = sfem_dss_bxyze_index(level, bsize, 0, level, iy, iz);
            const ptrdiff_t right_face = sfem_dss_bxyze_index(level, bsize, 1, 0, iy, iz);

            SFEM_TEST_APPROXEQ(bxyze[(size_t)left_face], bxyze[(size_t)right_face], 0.0);
        }
    }

    return SFEM_TEST_SUCCESS;
}

static int test_global_to_bxyze_real32_hex_high_order() {
    const ptrdiff_t nx    = 2;
    const ptrdiff_t ny    = 1;
    const ptrdiff_t nz    = 1;
    const int       level = 3;
    const int       bsize = 8;

    auto macro_mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), nx, ny, nz, 0, 0, 0, 1, 1, 1);
    auto mesh       = smesh::to_semistructured(level, macro_mesh, false, false);
    auto block      = mesh->block(0);
    auto elements   = block->elements()->data();

    const ptrdiff_t ne      = block->n_elements();
    const ptrdiff_t nvalues = sfem_dss_bxyze_n_values(level, ne, bsize);

    std::vector<float> global((size_t)mesh->n_nodes());
    for (ptrdiff_t gid = 0; gid < mesh->n_nodes(); ++gid) {
        global[(size_t)gid] = global_value32(gid);
    }

    std::vector<float> bxyze((size_t)nvalues, -1.0f);

    auto elements32 = reinterpret_cast<const int32_t *const *>(elements);
    SFEM_TEST_EQ(sfem_dss_global_to_bxyze_real32(level, ne, bsize, elements32, global.data(), bxyze.data()), SFEM_SUCCESS);

    for (ptrdiff_t e = 0; e < ne; ++e) {
        for (int iz = 0; iz <= level; ++iz) {
            for (int iy = 0; iy <= level; ++iy) {
                for (int ix = 0; ix <= level; ++ix) {
                    const int       lidx = smesh::sshex8_lidx(level, ix, iy, iz);
                    const idx_t     gid  = elements[lidx][e];
                    const ptrdiff_t out  = sfem_dss_bxyze_index(level, bsize, e, ix, iy, iz);

                    SFEM_TEST_APPROXEQ((double)bxyze[(size_t)out], (double)global[(size_t)gid], 1e-6);
                }
            }
        }
    }

    return SFEM_TEST_SUCCESS;
}

static int test_bxyze_rejects_invalid_inputs() {
    SFEM_TEST_EQ(sfem_dss_bxyze_n_values(0, 1, 4), (ptrdiff_t)-1);
    SFEM_TEST_EQ(sfem_dss_bxyze_n_values(2, -1, 4), (ptrdiff_t)-1);
    SFEM_TEST_EQ(sfem_dss_bxyze_n_values(2, 1, 0), (ptrdiff_t)-1);
    SFEM_TEST_EQ(sfem_dss_bxyze_index(0, 4, 0, 0, 0, 0), (ptrdiff_t)-1);
    SFEM_TEST_EQ(sfem_dss_bxyze_index(2, 4, 0, 3, 0, 0), (ptrdiff_t)-1);
    SFEM_TEST_EQ(sfem_dss_global_to_bxyze_real64(0, 1, 4, nullptr, nullptr, nullptr), SFEM_FAILURE);

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_global_to_bxyze_real64_hex_high_order);
    SFEM_RUN_TEST(test_global_to_bxyze_real32_hex_high_order);
    SFEM_RUN_TEST(test_bxyze_rejects_invalid_inputs);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
