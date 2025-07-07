#include <stdio.h>

#include "sfem_test.h"

#include "sfem_Mesh.hpp"
#include "sfem_CRSGraph.hpp"
#include "sfem_glob.hpp"


int test_mesh_blocks_basic() {
    // Create a mesh with default block
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 2, 2);
    
    // Test basic block functionality
    SFEM_TEST_EQ(mesh->n_blocks(), (size_t)1);
    SFEM_TEST_EQ(mesh->block(0)->name(), std::string("default"));
    SFEM_TEST_EQ(mesh->block(0)->element_type(), HEX8);
    SFEM_TEST_EQ(mesh->n_elements(), (ptrdiff_t)8);
    
    // Test backward compatibility
    SFEM_TEST_EQ(mesh->elements()->extent(1), (size_t)8);
    SFEM_TEST_EQ(mesh->element_type(), HEX8);
    
    return SFEM_TEST_SUCCESS;
}

int test_mesh_blocks_add_remove() {
    // Create a mesh with default block
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 2, 2);
    
    // Add a second block
    auto elements_buffer = sfem::create_host_buffer<idx_t>(4, 1); // One quad4 element
    elements_buffer->data()[0][0] = 0;
    elements_buffer->data()[1][0] = 1;
    elements_buffer->data()[2][0] = 2;
    elements_buffer->data()[3][0] = 3;
    
    mesh->add_block("quad4_block", QUAD4, elements_buffer);
    
    // Test adding block
    SFEM_TEST_EQ(mesh->n_blocks(), (size_t)2);
    SFEM_TEST_EQ(mesh->block(1)->name(), std::string("quad4_block"));
    SFEM_TEST_EQ(mesh->block(1)->element_type(), QUAD4);
    SFEM_TEST_EQ(mesh->n_elements(), (ptrdiff_t)9); // 8 hex8 + 1 quad4
    
    // Test removing a block
    mesh->remove_block(1); // Remove the second block
    SFEM_TEST_EQ(mesh->n_blocks(), (size_t)1);
    SFEM_TEST_EQ(mesh->n_elements(), (ptrdiff_t)8); // Back to original count
    
    return SFEM_TEST_SUCCESS;
}

int test_mesh_blocks_checkerboard() {
    // Create a checkerboard cube with 2x2x2 elements (must be even dimensions)
    auto mesh = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), 2, 2, 2);
    
    // Test that it creates exactly 2 blocks
    SFEM_TEST_EQ(mesh->n_blocks(), (size_t)2);
    
    // Test block names
    SFEM_TEST_EQ(mesh->block(0)->name(), std::string("white"));
    SFEM_TEST_EQ(mesh->block(1)->name(), std::string("black"));
    
    // Test element types
    SFEM_TEST_EQ(mesh->block(0)->element_type(), HEX8);
    SFEM_TEST_EQ(mesh->block(1)->element_type(), HEX8);
    
    // Test element counts - 2x2x2 = 8 total elements, split evenly
    SFEM_TEST_EQ(mesh->block(0)->elements()->extent(1), (size_t)4); // 4 white elements
    SFEM_TEST_EQ(mesh->block(1)->elements()->extent(1), (size_t)4); // 4 black elements
    SFEM_TEST_EQ(mesh->n_elements(), (ptrdiff_t)8); // Total elements
    
    // Test backward compatibility - should use first block (white)
    SFEM_TEST_EQ(mesh->elements()->extent(1), (size_t)4);
    SFEM_TEST_EQ(mesh->element_type(), HEX8);

    // Test node_to_node_graph
    auto node_to_node_graph = mesh->node_to_node_graph();
    SFEM_TEST_EQ(node_to_node_graph->n_nodes(), (ptrdiff_t)27);
    SFEM_TEST_EQ(node_to_node_graph->n_nodes(), (ptrdiff_t)27);
    SFEM_TEST_EQ(node_to_node_graph->nnz(), (ptrdiff_t)343);
    
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    
    SFEM_RUN_TEST(test_mesh_blocks_basic);
    SFEM_RUN_TEST(test_mesh_blocks_add_remove);
    SFEM_RUN_TEST(test_mesh_blocks_checkerboard);
    
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
} 
