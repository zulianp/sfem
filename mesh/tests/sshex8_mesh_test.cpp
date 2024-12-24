
#include "sfem_API.hpp"

#include <stdio.h>

#include "sfem_test.h"

#include "sfem_hex8_mesh_graph.h"

int test_sshex8_hierarchical_renumbering() {
    const ptrdiff_t nelements = 2;
    const ptrdiff_t nnodes    = 27 + 18;

    auto elements = sfem::h_buffer<idx_t>(27, nelements);

    for (ptrdiff_t i = 0; i < 27; i++) {
        elements->data()[i][0] = i;
        elements->data()[i][1] = 18 + i;
    }

    int       L               = 24;
    const int nxe             = proteus_hex8_nxe(L);
    auto      sshex8_elements = sfem::h_buffer<idx_t>(nxe, nelements);

    ptrdiff_t sshex_nnodes   = -1;
    ptrdiff_t interior_start = -1;

    SFEM_TEST_ASSERT(
            sshex8_generate_elements(
                    L, elements->extent(1), nnodes, elements->data(), sshex8_elements->data(), &sshex_nnodes, &interior_start) ==
            SFEM_SUCCESS);

    int  nlevels = sshex8_hierarchical_n_levels(L);
    auto levels  = sfem::h_buffer<int>(nlevels);
    sshex8_hierarchical_mesh_levels(L, nlevels, levels->data());

    SFEM_TEST_ASSERT(nlevels == 4);
    SFEM_TEST_ASSERT(levels->data()[0] == 1);
    SFEM_TEST_ASSERT(levels->data()[1] == 6);
    SFEM_TEST_ASSERT(levels->data()[2] == 12);
    SFEM_TEST_ASSERT(levels->data()[3] == 24);

    SFEM_TEST_ASSERT(sshex8_hierarchical_renumbering(
                             L, nlevels, levels->data(), nelements, sshex_nnodes, sshex8_elements->data()) == SFEM_SUCCESS);

    // Check that original nodes are in range
    for (int zi = 0; zi <= 1; zi++) {
        for (int yi = 0; yi <= 1; yi++) {
            for (int xi = 0; xi <= 1; xi++) {
                int v = proteus_hex8_lidx(L, xi * L, yi * L, zi * L);

                for (ptrdiff_t e = 0; e < nelements; e++) {
                    SFEM_TEST_ASSERT(sshex8_elements->data()[v][e] < nnodes);
                }
            }
        }
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT();
    SFEM_RUN_TEST(test_sshex8_hierarchical_renumbering);
    return SFEM_UNIT_TEST_FINALIZE();
}
