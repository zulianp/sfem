#include <memory>

#include "sfem_test.h"

#include "sfem_Function.hpp"

#include "sfem_Buffer.hpp"
#include "sfem_base.h"
#include "sfem_crs_SpMV.hpp"
#include "spmv.h"

#include "matrixio_array.h"

#include "sfem_API.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_solver.hpp"
#endif

#include "sfem_ssmgc.hpp"

int test_contact() {
    MPI_Comm comm = MPI_COMM_WORLD;

    sfem::ExecutionSpace es = sfem::EXECUTION_SPACE_HOST;

    int SFEM_BASE_RESOLUTION = 2;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    auto m = sfem::Mesh::create_hex8_cube(
            comm, SFEM_BASE_RESOLUTION * 1, SFEM_BASE_RESOLUTION * 1, SFEM_BASE_RESOLUTION * 1, 0, 0, 0, 1, 1, 1);

    const int block_size = m->spatial_dimension();

    auto fs = sfem::FunctionSpace::create(m, block_size);

    int SFEM_ELEMENT_REFINE_LEVEL = 2;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    SFEM_TEST_ASSERT(SFEM_ELEMENT_REFINE_LEVEL > 1);

    fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
    fs->semi_structured_mesh().apply_hierarchical_renumbering();

#ifdef SFEM_ENABLE_CUDA
    {
        auto elements = fs->device_elements();
        if (!elements) {
            elements = create_device_elements(fs, fs->element_type());
            fs->set_device_elements(elements);
        }
    }
#endif

    auto top_ss = sfem::Sideset::create_from_selector(
            m, [=](const geom_t /*x*/, const geom_t y, const geom_t z) -> bool { return y > (1 - 1e-5) && y < (1 + 1e-5); });
    
    sfem::DirichletConditions::Condition xtop{.sideset = top_ss, .value = 0, .component = 0};
    sfem::DirichletConditions::Condition ytop{.sideset = top_ss, .value = -0.2, .component = 1};
    sfem::DirichletConditions::Condition ztop{.sideset = top_ss, .value = 0, .component = 2};

    auto conds = sfem::create_dirichlet_conditions(fs, {xtop, ytop, ztop}, es);

    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, "LinearElasticity", es);
    op->initialize();
    f->add_constraint(conds);
    f->add_operator(op);

    auto bottom_ss = sfem::Sideset::create_from_selector(
            m, [=](const geom_t /*x*/, const geom_t y, const geom_t z) -> bool { return y > -1e-5 && y < 1e-5; });

    auto sdf = sfem::create_sdf(comm,
                                SFEM_BASE_RESOLUTION * 10,
                                SFEM_BASE_RESOLUTION * 10,
                                SFEM_BASE_RESOLUTION * 10,
                                -0.5,
                                -0.5,
                                -0.5,
                                1.5,
                                1.5,
                                1.5,
                                [](const geom_t x, const geom_t y, const geom_t z) -> geom_t {
                                    const geom_t plane = -0.1;
                                    return plane - y;
                                });

    sfem::create_directory("test_contact");
    sdf->to_file("test_contact/sdf");

    auto contact_conds = sfem::ContactConditions::create(fs, sdf, bottom_ss, es);

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            x     = sfem::create_buffer<real_t>(ndofs, es);
    auto            rhs   = sfem::create_buffer<real_t>(ndofs, es);

    contact_conds->update(x->data());

    auto solver = sfem::create_ssmgc(f, contact_conds, es, nullptr);

    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());
    solver->apply(rhs->data(), x->data());
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

#ifdef SFEM_ENABLE_CUDA
    sfem::register_device_ops();
#endif

    SFEM_RUN_TEST(test_contact);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
