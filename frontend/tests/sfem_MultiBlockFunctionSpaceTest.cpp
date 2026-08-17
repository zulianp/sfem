#include "sfem_test.hpp"

#include "sfem_FunctionSpace.hpp"
#include "smesh_mesh.hpp"

#include <iostream>
#include <memory>
#include <vector>

int test_single_block_mesh() {
    
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 2, 2);
    auto space = sfem::FunctionSpace::create(mesh, 1);

    SFEM_TEST_ASSERT(!mesh->is_distributed());
    
    // Test basic properties
    SFEM_TEST_ASSERT(space->n_blocks() == 1);
    SFEM_TEST_ASSERT(!space->is_multi_block());
    SFEM_TEST_ASSERT(space->element_type(0) == smesh::HEX8);
    
    // Test fallback behavior for non-existent blocks
    SFEM_TEST_ASSERT(space->element_type(1) == smesh::INVALID);  // Out of range should be smesh::INVALID
    SFEM_TEST_ASSERT(space->element_type(-1) == smesh::INVALID); // Out of range should be smesh::INVALID
    SFEM_TEST_ASSERT(space->element_type(10) == smesh::INVALID); // Out of range should be smesh::INVALID
    
    // Test block size
    SFEM_TEST_ASSERT(space->block_size() == 1);
    
    // Test mesh properties
    SFEM_TEST_ASSERT(space->mesh_ptr() == mesh);
    SFEM_TEST_ASSERT(!space->has_semi_structured_mesh());
    
    return SFEM_TEST_SUCCESS;
}

int test_multi_block_fallback() {
    
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 2, 2);
    
    // Test with different block sizes
    auto space1 = sfem::FunctionSpace::create(mesh, 1);
    auto space3 = sfem::FunctionSpace::create(mesh, 3);
    
    SFEM_TEST_ASSERT(space1->block_size() == 1);
    SFEM_TEST_ASSERT(space3->block_size() == 3);
    
    // Test that element types are consistent across block sizes
    SFEM_TEST_ASSERT(space1->element_type(0) == space3->element_type(0));
    
    // Test that requesting invalid blocks returns smesh::INVALID
    SFEM_TEST_ASSERT(space1->element_type(999) == smesh::INVALID);
    SFEM_TEST_ASSERT(space3->element_type(999) == smesh::INVALID);
    
    return SFEM_TEST_SUCCESS;
}

int test_semi_structured_promotion() {
    
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 2, 2);
    
    auto space = sfem::FunctionSpace::create(mesh, 1);
    
    // Initially should not have semi-structured mesh
    SFEM_TEST_ASSERT(!space->has_semi_structured_mesh());
    SFEM_TEST_ASSERT(space->element_type(0) == smesh::HEX8);
    
    // Build the semi-structured mesh explicitly, then create a new function space on top of it.
    auto ssmesh = smesh::to_semistructured(2, mesh, true, false);
    space = sfem::FunctionSpace::create(ssmesh, 1);

    
    // Should now have semi-structured mesh
    SFEM_TEST_ASSERT(space->has_semi_structured_mesh());
    SFEM_TEST_ASSERT(is_semistructured_type(space->element_type(0)));
    
    // Test that fallback still works
    SFEM_TEST_ASSERT(space->element_type(1) == smesh::INVALID);
    
    return SFEM_TEST_SUCCESS;
}

int test_vector_creation() {
    
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 2, 2);
    auto space = sfem::FunctionSpace::create(mesh, 1);
    
    ptrdiff_t nlocal, nglobal;
    real_t *values;
    
    // Test vector creation
    int result = space->create_vector(&nlocal, &nglobal, &values);
    SFEM_TEST_ASSERT(result == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(values != nullptr);
    SFEM_TEST_ASSERT(nlocal == space->n_dofs());
    SFEM_TEST_ASSERT(nglobal == space->n_dofs_global());
    
    // Test vector destruction
    result = space->destroy_vector(values);
    SFEM_TEST_ASSERT(result == SFEM_SUCCESS);
    
    return SFEM_TEST_SUCCESS;
}

// int test_lor_function_space() {
    
//     MPI_Comm comm = MPI_COMM_WORLD;
//     auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::world(), 2, 2, 2);
//     auto space = sfem::FunctionSpace::create(mesh, 1);
    
//     // Create LOR function space
//     auto lor_space = space->lor();
//     SFEM_TEST_ASSERT(lor_space != nullptr);
    
//     // LOR space should have same mesh and block size
//     SFEM_TEST_ASSERT(lor_space->mesh_ptr() == space->mesh_ptr());
//     SFEM_TEST_ASSERT(lor_space->block_size() == space->block_size());
    
//     // Element type should be different (macro variant)
//     SFEM_TEST_ASSERT(lor_space->element_type(0) != space->element_type(0));
    
//     return SFEM_TEST_SUCCESS;
// }

int test_derefine_function_space() {
    
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 2, 2);
    mesh = smesh::to_semistructured(2, mesh, true, false);
    auto space = sfem::FunctionSpace::create(mesh, 1);
    
    SFEM_TEST_ASSERT(space->has_semi_structured_mesh());
    
    // Test derefine
    auto derefined = space->derefine(1);
    SFEM_TEST_ASSERT(derefined != nullptr);
    
    // Derefined space should have same block size
    SFEM_TEST_ASSERT(derefined->block_size() == space->block_size());
    
    return SFEM_TEST_SUCCESS;
}

int test_edge_cases() {
    
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 1, 1, 1);
    auto space = sfem::FunctionSpace::create(mesh, 1);
    
    // Test with minimal mesh
    SFEM_TEST_ASSERT(space->n_dofs() > 0);
    SFEM_TEST_ASSERT(space->element_type(0) != smesh::INVALID);
    
    // Test with large block size
    auto space_large = sfem::FunctionSpace::create(mesh, 10);
    SFEM_TEST_ASSERT(space_large->block_size() == 10);
    SFEM_TEST_ASSERT(space_large->n_dofs() == space->n_dofs() * 10);
    
    // Test element type consistency
    SFEM_TEST_ASSERT(space->element_type(0) == space_large->element_type(0));
    
    return SFEM_TEST_SUCCESS;
}

int test_checkerboard_function_space() {
    auto mesh  = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), 2, 2, 2);
    auto space = sfem::FunctionSpace::create(mesh, 1);

    SFEM_TEST_ASSERT(space->n_blocks() == 2);
    SFEM_TEST_ASSERT(space->is_multi_block());
    SFEM_TEST_ASSERT(space->element_type(0) == smesh::HEX8);
    SFEM_TEST_ASSERT(space->element_type(1) == smesh::HEX8);

    const ptrdiff_t expected = mesh->n_nodes();
    SFEM_TEST_ASSERT(space->n_dofs() == expected);
    SFEM_TEST_ASSERT(space->n_owned_dofs() == expected);
    SFEM_TEST_ASSERT(space->n_dofs_global() == expected);

    auto space3 = sfem::FunctionSpace::create(mesh, 3);
    SFEM_TEST_ASSERT(space3->n_dofs() == expected * 3);
    SFEM_TEST_ASSERT(space3->n_owned_dofs() == expected * 3);
    SFEM_TEST_ASSERT(space3->n_dofs_global() == expected * 3);

    return SFEM_TEST_SUCCESS;
}

int test_hex8_tet4_function_space() {
    auto mesh  = sfem::Mesh::create_hex8_tet4_cube(sfem::Communicator::self(), 2, 2, 2);
    auto space = sfem::FunctionSpace::create(mesh, 1);

    SFEM_TEST_ASSERT(space->n_blocks() == 2);
    SFEM_TEST_ASSERT(space->element_type(0) == smesh::HEX8);
    SFEM_TEST_ASSERT(space->element_type(1) == smesh::TET4);

    const ptrdiff_t expected = mesh->n_nodes();
    SFEM_TEST_ASSERT(space->n_dofs() == expected);
    SFEM_TEST_ASSERT(space->n_owned_dofs() == expected);
    SFEM_TEST_ASSERT(space->n_dofs_global() == expected);
    SFEM_TEST_ASSERT(space->n_dofs() != mesh->n_elements());

    auto space3 = sfem::FunctionSpace::create(mesh, 3);
    SFEM_TEST_ASSERT(space3->n_dofs() == expected * 3);
    SFEM_TEST_ASSERT(space3->element_type(0) == smesh::HEX8);
    SFEM_TEST_ASSERT(space3->element_type(1) == smesh::TET4);

    return SFEM_TEST_SUCCESS;
}

int test_packed_mesh_function_space() {
    auto mesh        = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), 2, 2, 2);
    auto packed_mesh = sfem::FunctionSpace::PackedMesh::create(mesh, {}, true);
    SFEM_TEST_ASSERT(packed_mesh != nullptr);

    auto space = sfem::FunctionSpace::create(packed_mesh, 1);
    SFEM_TEST_ASSERT(space->n_blocks() == 2);
    SFEM_TEST_ASSERT(space->element_type(0) == smesh::HEX8);
    SFEM_TEST_ASSERT(space->element_type(1) == smesh::HEX8);
    SFEM_TEST_ASSERT(space->n_dofs() == mesh->n_nodes());
    SFEM_TEST_ASSERT(space->n_owned_dofs() == mesh->n_nodes());
    SFEM_TEST_ASSERT(space->n_dofs_global() == mesh->n_nodes());

    auto mixed        = sfem::Mesh::create_hex8_tet4_cube(sfem::Communicator::self(), 2, 2, 2);
    auto packed_mixed = sfem::FunctionSpace::PackedMesh::create(mixed, {}, true);
    SFEM_TEST_ASSERT(packed_mixed != nullptr);
    auto mixed_space = sfem::FunctionSpace::create(packed_mixed, 2);
    SFEM_TEST_ASSERT(mixed_space->n_blocks() == 2);
    SFEM_TEST_ASSERT(mixed_space->element_type(0) == smesh::HEX8);
    SFEM_TEST_ASSERT(mixed_space->element_type(1) == smesh::TET4);
    SFEM_TEST_ASSERT(mixed_space->n_dofs() == mixed->n_nodes() * 2);

    return SFEM_TEST_SUCCESS;
}

int test_override_element_types_multi_block() {
    auto mesh  = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), 2, 2, 2);
    auto space = sfem::FunctionSpace::create(mesh, 1, smesh::HEX8);

    SFEM_TEST_ASSERT(space->n_blocks() == 2);
    SFEM_TEST_ASSERT(space->element_type(0) == smesh::HEX8);
    SFEM_TEST_ASSERT(space->element_type(1) == smesh::HEX8);
    SFEM_TEST_ASSERT(space->n_dofs() == mesh->n_nodes());
    SFEM_TEST_ASSERT(space->n_owned_dofs() == mesh->n_nodes());
    SFEM_TEST_ASSERT(space->n_dofs_global() == mesh->n_nodes());

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_single_block_mesh);
    SFEM_RUN_TEST(test_multi_block_fallback);
    SFEM_RUN_TEST(test_semi_structured_promotion);
    SFEM_RUN_TEST(test_vector_creation);
    // SFEM_RUN_TEST(test_lor_function_space); // TODO: Implement LOR function space
    SFEM_RUN_TEST(test_derefine_function_space);
    SFEM_RUN_TEST(test_edge_cases);
    SFEM_RUN_TEST(test_checkerboard_function_space);
    SFEM_RUN_TEST(test_hex8_tet4_function_space);
    SFEM_RUN_TEST(test_packed_mesh_function_space);
    SFEM_RUN_TEST(test_override_element_types_multi_block);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}

