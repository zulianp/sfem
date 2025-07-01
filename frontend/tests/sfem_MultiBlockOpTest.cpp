#include "sfem_test.h"

#include "sfem_FunctionSpace.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_API.hpp"

#include <iostream>
#include <memory>

int test_multi_block_op() {
    
    MPI_Comm comm = MPI_COMM_WORLD;
    auto mesh = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::wrap(comm), 4, 4, 4);
    auto space = sfem::FunctionSpace::create(mesh, 1);

    auto f = sfem::Function::create(space);
    auto op = sfem::Factory::create_op(space, "Laplacian");
    op->initialize();
    f->add_operator(op);
    
    auto lop = sfem::create_linear_operator("MF", f, nullptr, f->execution_space());

    auto cg = sfem::create_cg(lop, f->execution_space());
    auto x = sfem::create_buffer<real_t>(space->n_dofs(), f->execution_space());
    auto b = sfem::create_buffer<real_t>(space->n_dofs(), f->execution_space());

    SFEM_TEST_ASSERT(cg->apply(x->data(), b->data()) == SFEM_SUCCESS);

    sfem::create_directory("test_multi_block_op");
    SFEM_TEST_ASSERT(mesh->write("test_multi_block_op") == SFEM_SUCCESS);

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    
    
    SFEM_RUN_TEST(test_multi_block_op);
    
    
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
} 