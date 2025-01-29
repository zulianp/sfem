#include <memory>

#include "mg_builder.hpp"
#include "sfem_Stationary.hpp"
#include "sfem_test.h"

#include "sfem_Function.hpp"

#include "sfem_Buffer.hpp"
#include "sfem_base.h"
#include "sfem_crs_SpMV.hpp"
#include "spmv.h"

#include "matrixio_array.h"

#include "sfem_API.hpp"
#include "sfem_ShiftedPenalty.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_solver.hpp"
#endif

int test_amg_poisson() {
    MPI_Comm comm = MPI_COMM_WORLD;

    sfem::ExecutionSpace es = sfem::EXECUTION_SPACE_HOST;

    int SFEM_MESH_RESOLUTION = 25;
    SFEM_READ_ENV(SFEM_MESH_RESOLUTION, atoi);

    auto m = sfem::Mesh::create_hex8_cube(
            comm, SFEM_MESH_RESOLUTION * 1, SFEM_MESH_RESOLUTION * 1, SFEM_MESH_RESOLUTION * 1, 0, 0, 0, 1, 1, 1);

    const int block_size = 1;
    auto      fs         = sfem::FunctionSpace::create(m, block_size);

    auto top_ss = sfem::Sideset::create_from_selector(
            m, [=](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool { return y > (1 - 1e-5) && y < (1 + 1e-5); });

    auto left_ss = sfem::Sideset::create_from_selector(
            m, [=](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool { return x > -1e-5 && x < 1e-5; });

    sfem::DirichletConditions::Condition top{.sideset = top_ss, .value = 1, .component = 0};
    sfem::DirichletConditions::Condition left{.sideset = left_ss, .value = -1, .component = 0};

    auto conds = sfem::create_dirichlet_conditions(fs, {top, left}, es);

    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, "Laplacian", es);
    op->initialize();
    f->add_constraint(conds);
    f->add_operator(op);

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            x     = sfem::create_buffer<real_t>(ndofs, es);
    auto            rhs   = sfem::create_buffer<real_t>(ndofs, es);

    auto linear_op = sfem::hessian_crs(f, x, es);

#if 0
    auto solver = sfem::create_cg<real_t>(linear_op, es);
#else
    // FIXME use AMG instead (e.g., sfem::create_amg<real_t>(linear_op, ..., es))
    auto mask = sfem::create_buffer<mask_t>(mask_count(fs->n_dofs()), es);
    f->constaints_mask(mask->data());
    count_t *row_ptr     = linear_op->row_ptr->data();
    idx_t   *col_indices = linear_op->col_idx->data();
    real_t  *values      = linear_op->values->data();
    auto     near_null   = sfem::create_buffer<real_t>(fs->n_dofs(), es);

    real_t coarsening_factor = 7.5;
    auto   amg               = builder_sa(coarsening_factor, mask, near_null, linear_op);

    if (!amg->test_interp()) {
        printf("tests passed\n");
    } else {
        printf("FAILEDDDDDD\n");
    }

    amg->set_max_it(100);
    amg->verbose = true;
    // amg->debug = true;
    auto solver = amg;
#endif

    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());
    solver->apply(rhs->data(), x->data());

    SFEM_TEST_ASSERT(sfem::create_directory("test_amg_poisson") == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(m->write("test_amg_poisson/mesh") == SFEM_SUCCESS);

    auto out = f->output();
    out->set_output_dir("test_amg_poisson/out");
    if (block_size > 1) out->enable_AoS_to_SoA(true);  // Needed only for vector problem
    SFEM_TEST_ASSERT(out->write("x", x->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(out->write("rhs", rhs->data()) == SFEM_SUCCESS);

    return SFEM_TEST_SUCCESS;
}

int test_amg_sqp() {
    MPI_Comm comm = MPI_COMM_WORLD;

    sfem::ExecutionSpace es = sfem::EXECUTION_SPACE_HOST;

    int SFEM_MESH_RESOLUTION = 8;
    SFEM_READ_ENV(SFEM_MESH_RESOLUTION, atoi);

    auto m = sfem::Mesh::create_hex8_cube(
            comm, SFEM_MESH_RESOLUTION * 1, SFEM_MESH_RESOLUTION * 1, SFEM_MESH_RESOLUTION * 1, 0, 0, 0, 1, 1, 1);

    const int block_size = 1;
    auto      fs         = sfem::FunctionSpace::create(m, block_size);

    auto top_ss = sfem::Sideset::create_from_selector(
            m, [=](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool { return y > (1 - 1e-5) && y < (1 + 1e-5); });

    sfem::DirichletConditions::Condition top{.sideset = top_ss, .value = 1, .component = 0};
    auto                                 conds = sfem::create_dirichlet_conditions(fs, {top}, es);

    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, "Laplacian", es);
    op->initialize();
    f->add_constraint(conds);
    f->add_operator(op);

    const ptrdiff_t ndofs       = fs->n_dofs();
    auto            x           = sfem::create_buffer<real_t>(ndofs, es);
    auto            rhs         = sfem::create_buffer<real_t>(ndofs, es);
    auto            upper_bound = sfem::create_buffer<real_t>(ndofs, es);

    auto bottom_ss = sfem::Sideset::create_from_selector(
            m, [=](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool { return y > -1e-5 && y < 1e-5; });

    // Indices of potential contact boundary nodes
    auto bottom = sfem::create_nodeset_from_sideset(fs, bottom_ss);

    // FIXME not GPU ready
    {
        auto idx = bottom->data();
        auto u   = upper_bound->data();

        for (ptrdiff_t i = 0; i < ndofs; i++) {
            u[i] = 10000;  // kind of infinity
        }

        for (ptrdiff_t i = 0; i < bottom->size(); i++) {
            u[idx[i]] = -1;
        }
    }

    auto linear_op = sfem::hessian_crs(f, x, es);

#if 1
    auto solver     = std::make_shared<sfem::ShiftedPenalty<real_t>>();
    solver->verbose = true;
    solver->set_op(linear_op);
    solver->default_init();
    solver->set_atol(1e-12);
    solver->set_max_it(100);
    solver->set_max_inner_it(30);
    solver->set_damping(1);
    solver->set_penalty_param(10);
    solver->set_upper_bound(upper_bound);

    {  // Set-up linear solver
        auto cg     = sfem::create_cg<real_t>(linear_op, es);
        cg->verbose = false;
        auto diag   = sfem::create_buffer<real_t>(fs->n_dofs(), es);
        f->hessian_diag(x->data(), diag->data());
        auto sj = sfem::create_shiftable_jacobi(diag, es);
        cg->set_preconditioner_op(sj);
        cg->set_atol(1e-12);
        cg->set_rtol(1e-4);
        cg->set_max_it(20000);
        solver->set_linear_solver(cg);
    }

#else
    // FIXME use AMG instead (e.g., sfem::create_spamg<real_t>(linear_op, ..., es))
#endif

    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());
    solver->apply(rhs->data(), x->data());

    SFEM_TEST_ASSERT(sfem::create_directory("test_amg_sqp") == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(m->write("test_amg_sqp/mesh") == SFEM_SUCCESS);

    auto out = f->output();
    out->set_output_dir("test_amg_sqp/out");
    if (block_size > 1) out->enable_AoS_to_SoA(true);
    SFEM_TEST_ASSERT(out->write("x", x->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(out->write("rhs", rhs->data()) == SFEM_SUCCESS);

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

#ifdef SFEM_ENABLE_CUDA
    sfem::register_device_ops();
#endif

    SFEM_RUN_TEST(test_amg_poisson);
    SFEM_RUN_TEST(test_amg_sqp);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
