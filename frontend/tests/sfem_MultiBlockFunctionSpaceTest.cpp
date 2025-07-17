#include "sfem_test.h"

#include "sfem_FunctionSpace.hpp"
#include "sfem_Mesh.hpp"

#include <iostream>
#include <memory>

int test_single_block_mesh() {
    
    MPI_Comm comm = MPI_COMM_WORLD;
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::wrap(comm), 2, 2, 2);
    auto space = sfem::FunctionSpace::create(mesh, 1);
    
    // Test basic properties
    SFEM_TEST_ASSERT(space->n_blocks() == 1);
    SFEM_TEST_ASSERT(!space->is_multi_block());
    SFEM_TEST_ASSERT(space->element_type(0) == HEX8);
    
    // Test fallback behavior for non-existent blocks
    SFEM_TEST_ASSERT(space->element_type(1) == INVALID);  // Out of range should be INVALID
    SFEM_TEST_ASSERT(space->element_type(-1) == INVALID); // Out of range should be INVALID
    SFEM_TEST_ASSERT(space->element_type(10) == INVALID); // Out of range should be INVALID
    
    // Test block size
    SFEM_TEST_ASSERT(space->block_size() == 1);
    
    // Test mesh properties
    SFEM_TEST_ASSERT(space->mesh_ptr() == mesh);
    SFEM_TEST_ASSERT(!space->has_semi_structured_mesh());
    
    return SFEM_TEST_SUCCESS;
}

int test_multi_block_fallback() {
    
    MPI_Comm comm = MPI_COMM_WORLD;
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::wrap(comm), 2, 2, 2);
    
    // Test with different block sizes
    auto space1 = sfem::FunctionSpace::create(mesh, 1);
    auto space3 = sfem::FunctionSpace::create(mesh, 3);
    
    SFEM_TEST_ASSERT(space1->block_size() == 1);
    SFEM_TEST_ASSERT(space3->block_size() == 3);
    
    // Test that element types are consistent across block sizes
    SFEM_TEST_ASSERT(space1->element_type(0) == space3->element_type(0));
    
    // Test that requesting invalid blocks returns INVALID
    SFEM_TEST_ASSERT(space1->element_type(999) == INVALID);
    SFEM_TEST_ASSERT(space3->element_type(999) == INVALID);
    
    return SFEM_TEST_SUCCESS;
}

int test_semi_structured_promotion() {
    
    MPI_Comm comm = MPI_COMM_WORLD;
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::wrap(comm), 2, 2, 2);
    auto space = sfem::FunctionSpace::create(mesh, 1);
    
    // Initially should not have semi-structured mesh
    SFEM_TEST_ASSERT(!space->has_semi_structured_mesh());
    SFEM_TEST_ASSERT(space->element_type(0) == HEX8);
    
    // Promote to semi-structured
    int result = space->promote_to_semi_structured(2);
    SFEM_TEST_ASSERT(result == SFEM_SUCCESS);
    
    // Should now have semi-structured mesh
    SFEM_TEST_ASSERT(space->has_semi_structured_mesh());
    SFEM_TEST_ASSERT(space->element_type(0) == SSHEX8);
    
    // Test that fallback still works
    SFEM_TEST_ASSERT(space->element_type(1) == INVALID);
    
    return SFEM_TEST_SUCCESS;
}

int test_vector_creation() {
    
    MPI_Comm comm = MPI_COMM_WORLD;
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::wrap(comm), 2, 2, 2);
    auto space = sfem::FunctionSpace::create(mesh, 1);
    
    ptrdiff_t nlocal, nglobal;
    real_t *values;
    
    // Test vector creation
    int result = space->create_vector(&nlocal, &nglobal, &values);
    SFEM_TEST_ASSERT(result == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(values != nullptr);
    SFEM_TEST_ASSERT(nlocal == space->n_dofs());
    SFEM_TEST_ASSERT(nglobal == space->n_dofs());
    
    // Test vector destruction
    result = space->destroy_vector(values);
    SFEM_TEST_ASSERT(result == SFEM_SUCCESS);
    
    return SFEM_TEST_SUCCESS;
}

// int test_lor_function_space() {
    
//     MPI_Comm comm = MPI_COMM_WORLD;
//     auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::wrap(comm), 2, 2, 2);
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
    
    MPI_Comm comm = MPI_COMM_WORLD;
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::wrap(comm), 2, 2, 2);
    auto space = sfem::FunctionSpace::create(mesh, 1);
    
    // Promote to semi-structured first
    int result = space->promote_to_semi_structured(2);
    SFEM_TEST_ASSERT(result == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(space->has_semi_structured_mesh());
    
    // Test derefine
    auto derefined = space->derefine(1);
    SFEM_TEST_ASSERT(derefined != nullptr);
    
    // Derefined space should have same block size
    SFEM_TEST_ASSERT(derefined->block_size() == space->block_size());
    
    return SFEM_TEST_SUCCESS;
}

int test_edge_cases() {
    
    MPI_Comm comm = MPI_COMM_WORLD;
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::wrap(comm), 1, 1, 1);
    auto space = sfem::FunctionSpace::create(mesh, 1);
    
    // Test with minimal mesh
    SFEM_TEST_ASSERT(space->n_dofs() > 0);
    SFEM_TEST_ASSERT(space->element_type(0) != INVALID);
    
    // Test with large block size
    auto space_large = sfem::FunctionSpace::create(mesh, 10);
    SFEM_TEST_ASSERT(space_large->block_size() == 10);
    SFEM_TEST_ASSERT(space_large->n_dofs() == space->n_dofs() * 10);
    
    // Test element type consistency
    SFEM_TEST_ASSERT(space->element_type(0) == space_large->element_type(0));
    
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
    
    
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
} 