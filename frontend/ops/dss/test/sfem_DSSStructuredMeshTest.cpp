#include "sfem_test.hpp"

#include "sfem_FunctionSpace.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"
#include "smesh_sshex8.hpp"

#include <cstddef>
#include <vector>

static int test_hex8_to_high_order_keeps_unique_global_nodes() {
    const ptrdiff_t nx    = 2;
    const ptrdiff_t ny    = 1;
    const ptrdiff_t nz    = 1;
    const int       level = 4;

    auto macro_mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), nx, ny, nz, 0, 0, 0, 1, 1, 1);
    SFEM_TEST_ASSERT(macro_mesh != nullptr);
    SFEM_TEST_EQ(macro_mesh->element_type(0), smesh::HEX8);
    SFEM_TEST_EQ(macro_mesh->n_elements(), nx * ny * nz);

    auto mesh = smesh::to_semistructured(level, macro_mesh, false, false);
    SFEM_TEST_ASSERT(mesh != nullptr);
    SFEM_TEST_EQ(mesh->n_blocks(), (size_t)1);
    SFEM_TEST_ASSERT(smesh::is_semistructured_type(mesh->element_type(0)));
    SFEM_TEST_ASSERT(smesh::is_hex_ss_family(mesh->element_type(0)));
    SFEM_TEST_EQ(smesh::semistructured_level(*mesh), level);

    const ptrdiff_t expected_nodes = (nx * level + 1) * (ny * level + 1) * (nz * level + 1);
    SFEM_TEST_EQ(mesh->n_nodes(), expected_nodes);

    auto            block = mesh->block(0);
    const int       nxe   = smesh::sshex8_nxe(level);
    const ptrdiff_t ne    = block->n_elements();

    SFEM_TEST_EQ(block->n_nodes_per_element(), nxe);
    SFEM_TEST_EQ((ptrdiff_t)block->elements()->extent(0), (ptrdiff_t)nxe);
    SFEM_TEST_EQ(ne, nx * ny * nz);

    auto elements = block->elements()->data();

    std::vector<unsigned char> seen((size_t)mesh->n_nodes(), 0);
    ptrdiff_t                  unique_nodes_seen = 0;

    for (int lidx = 0; lidx < nxe; ++lidx) {
        for (ptrdiff_t e = 0; e < ne; ++e) {
            const idx_t gid = elements[lidx][e];
            SFEM_TEST_ASSERT(gid >= 0);
            SFEM_TEST_ASSERT((ptrdiff_t)gid < mesh->n_nodes());

            if (!seen[(size_t)gid]) {
                seen[(size_t)gid] = 1;
                unique_nodes_seen++;
            }
        }
    }

    SFEM_TEST_EQ(unique_nodes_seen, mesh->n_nodes());

    for (int iz = 0; iz <= level; ++iz) {
        for (int iy = 0; iy <= level; ++iy) {
            const int   left_face_lidx  = smesh::sshex8_lidx(level, level, iy, iz);
            const int   right_face_lidx = smesh::sshex8_lidx(level, 0, iy, iz);
            const idx_t left_gid        = elements[left_face_lidx][0];
            const idx_t right_gid       = elements[right_face_lidx][1];
            SFEM_TEST_EQ(left_gid, right_gid);
        }
    }

    return SFEM_TEST_SUCCESS;
}

static int test_high_order_function_space_uses_unique_nodal_dofs() {
    const ptrdiff_t nx    = 2;
    const ptrdiff_t ny    = 2;
    const ptrdiff_t nz    = 1;
    const int       level = 3;

    auto macro_mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), nx, ny, nz, 0, 0, 0, 1, 1, 1);
    auto mesh       = smesh::to_semistructured(level, macro_mesh, false, false);
    auto space      = sfem::FunctionSpace::create(mesh, 1);

    SFEM_TEST_ASSERT(space != nullptr);
    SFEM_TEST_ASSERT(space->has_semi_structured_mesh());
    SFEM_TEST_EQ(space->block_size(), 1);
    SFEM_TEST_EQ(space->n_dofs(), mesh->n_nodes());
    SFEM_TEST_EQ(space->n_dofs_global(), mesh->n_nodes());

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_hex8_to_high_order_keeps_unique_global_nodes);
    SFEM_RUN_TEST(test_high_order_function_space_uses_unique_nodal_dofs);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
