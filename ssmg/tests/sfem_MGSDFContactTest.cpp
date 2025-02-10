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

    int SFEM_BASE_RESOLUTION = 1;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    const geom_t y_top = 0.2;

    auto m = sfem::Mesh::create_hex8_cube(
            comm, SFEM_BASE_RESOLUTION * 5, SFEM_BASE_RESOLUTION * 1, SFEM_BASE_RESOLUTION * 5, 0, 0, 0, 1, y_top, 1);

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

    auto top_ss = sfem::Sideset::create_from_selector(m, [=](const geom_t /*x*/, const geom_t y, const geom_t z) -> bool {
        return y > (y_top - 1e-5) && y < (y_top + 1e-5);
    });

    sfem::DirichletConditions::Condition xtop{.sideset = top_ss, .value = 0, .component = 0};
    sfem::DirichletConditions::Condition ytop{.sideset = top_ss, .value = -0.08, .component = 1};
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
                                SFEM_ELEMENT_REFINE_LEVEL * SFEM_BASE_RESOLUTION * 5,
                                SFEM_ELEMENT_REFINE_LEVEL * SFEM_BASE_RESOLUTION * 1,
                                SFEM_ELEMENT_REFINE_LEVEL * SFEM_BASE_RESOLUTION * 5,
                                -0.1,
                                -0.2,
                                -0.1,
                                1.1,
                                y_top + 0.2,
                                1.1,
                                [](const geom_t x, const geom_t y, const geom_t z) -> geom_t {
                                    const geom_t cx = 0.6 * (1 - (x - .5) * (x - .5));
                                    const geom_t cz = 0.6 * (1 - (z - .5) * (z - .5));
                                    
                                    geom_t fx = 0.1 * cos(cx * 3.14 * 8) * cx * cx + 0.02 * cos(cx * 3.14 * 16);
                                    geom_t fz = 0.1 * cos(cz * 3.14 * 8) * cz * cz + 0.02 * cos(cx * 3.14 * 16);
                                    fx += 0.005 * cos(cx * 3.14 * 32);
                                    fz += 0.005 * cos(cz * 3.14 * 32);
                                    fx += 0.0025 * cos(cx * 3.14 * 64);
                                    fz += 0.0025 * cos(cz * 3.14 * 64);

                                    const geom_t obstacle = -0.1 - fx - fz;
                                    // const geom_t obstacle = -0.1;
                                    return obstacle - y;
                                });

    sfem::create_directory("test_contact");
    sdf->to_file("test_contact/sdf");

    auto contact_conds = sfem::ContactConditions::create(fs, sdf, bottom_ss, es);

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            x     = sfem::create_buffer<real_t>(ndofs, es);
    auto            rhs   = sfem::create_buffer<real_t>(ndofs, es);
    auto            gap   = sfem::create_buffer<real_t>(ndofs, es);

    f->apply_constraints(rhs->data());
    contact_conds->update(x->data());
    contact_conds->signed_distance_for_mesh_viz(x->data(), gap->data());

    fs->mesh_ptr()->write("test_contact/coarse_mesh");
    fs->semi_structured_mesh().export_as_standard("test_contact/mesh");
    auto out = f->output();
    out->set_output_dir("test_contact/out");
    out->enable_AoS_to_SoA(true);
    out->write("gap", gap->data());
    out->write("rhs", rhs->data());

#if 1  // FIXME
       // #if 1
    auto solver = sfem::create_ssmgc(f, contact_conds, es, nullptr);
    // auto solver = sfem::create_shifted_penalty(f, contact_conds, es, nullptr); // This works!
    f->apply_constraints(x->data());
    solver->apply(rhs->data(), x->data());

    out->write("disp", x->data());
#endif
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
