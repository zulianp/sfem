#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_FunctionSpace.hpp"
#include "smesh_mesh.hpp"

#include <memory>

int test_multi_block_op() {
    auto     mesh  = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), 4, 4, 4);
    auto     space = sfem::FunctionSpace::create(mesh, 1);

    auto f  = sfem::Function::create(space);
    auto op = sfem::Factory::create_op(space, "Laplacian");
    op->initialize();
    f->add_operator(op);

    auto lop = sfem::create_linear_operator(sfem::op_type::MATRIX_FREE, f, nullptr, f->execution_space());

    auto cg = sfem::create_cg(lop, f->execution_space());
    auto x  = sfem::create_buffer<real_t>(space->n_dofs(), f->execution_space());
    auto b  = sfem::create_buffer<real_t>(space->n_dofs(), f->execution_space());

    SFEM_TEST_ASSERT(cg->apply(x->data(), b->data()) == SFEM_SUCCESS);

    smesh::create_directory("test_multi_block_op");
    SFEM_TEST_ASSERT(mesh->write(smesh::Path("test_multi_block_op")) == SFEM_SUCCESS);

    return SFEM_TEST_SUCCESS;
}

int test_hex8_tet4_laplacian_apply() {
    auto mesh = sfem::Mesh::create_hex8_tet4_cube(sfem::Communicator::self(), 4, 4, 2);
    SFEM_TEST_ASSERT(mesh != nullptr);
    SFEM_TEST_ASSERT(!mesh->is_distributed());
    SFEM_TEST_EQ(mesh->n_blocks(), static_cast<size_t>(2));
    SFEM_TEST_EQ(mesh->element_type(0), smesh::HEX8);
    SFEM_TEST_EQ(mesh->element_type(1), smesh::TET4);

    auto space = sfem::FunctionSpace::create(mesh, 1);
    auto f     = sfem::Function::create(space);
    auto op    = sfem::Factory::create_op(space, "Laplacian");
    SFEM_TEST_ASSERT(op != nullptr);
    SFEM_TEST_ASSERT(op->initialize() == SFEM_SUCCESS);
    f->add_operator(op);

    const ptrdiff_t n = space->n_dofs();
    auto            x = sfem::create_host_buffer<real_t>(n);
    auto            y = sfem::create_host_buffer<real_t>(n);
    auto            points = mesh->points()->data();
    for (ptrdiff_t i = 0; i < n; ++i) {
        const geom_t px = points[0][i];
        const geom_t py = points[1][i];
        const geom_t pz = points[2][i];
        x->data()[i]    = px * px + real_t(0.5) * py * py - real_t(0.25) * pz * pz;
        y->data()[i]    = 0;
    }

    SFEM_TEST_ASSERT(f->apply(nullptr, x->data(), y->data()) == SFEM_SUCCESS);

    real_t nrm2 = 0;
    for (ptrdiff_t i = 0; i < n; ++i) {
        nrm2 += y->data()[i] * y->data()[i];
    }
    SFEM_TEST_ASSERT(nrm2 > 0);

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_multi_block_op);
    SFEM_RUN_TEST(test_hex8_tet4_laplacian_apply);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}

