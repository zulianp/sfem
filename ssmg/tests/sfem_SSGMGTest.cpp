#include <memory>

#include "sfem_test.h"

#include "sfem_Function.hpp"

#include "sfem_Buffer.hpp"
#include "sfem_base.h"
#include "sfem_crs_SpMV.hpp"
#include "spmv.h"

#include "matrixio_array.h"

#include "sfem_API.hpp"
#include "sfem_ssgmg.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_solver.hpp"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>

int test_linear_problem(const std::shared_ptr<sfem::Function> &f, const std::string &name) {
    auto fs  = f->space();
    auto x   = sfem::create_buffer<real_t>(fs->n_dofs(), f->execution_space());
    auto rhs = sfem::create_buffer<real_t>(fs->n_dofs(), f->execution_space());
    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());

    auto mg = create_ssgmg(f, f->execution_space());
    SFEM_TEST_ASSERT(mg->apply(rhs->data(), x->data()) == SFEM_SUCCESS);

#if 0
    sfem::create_directory(name.c_str());
    sfem::create_directory((name +"/fields").c_str());

    SFEM_TEST_ASSERT(fs->semi_structured_mesh().export_as_standard((name +"/mesh").c_str()) == SFEM_SUCCESS);

    sfem::Output out(fs);
    out.enable_AoS_to_SoA(true);

    out.set_output_dir((name +"/fields").c_str());
    SFEM_TEST_ASSERT(out.write("u", x->data()) == SFEM_SUCCESS);
#endif
    return SFEM_TEST_SUCCESS;
}

int test_ssgmg_poisson_cube() {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );

    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    const char *SFEM_OPERATOR = "Laplacian";
    // const char *SFEM_OPERATOR       = "em:Laplacian";
    const char *SFEM_FINE_OP_TYPE   = "MF";
    const char *SFEM_COARSE_OP_TYPE = "MF";

    int SFEM_ELEMENT_REFINE_LEVEL = 4;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    int SFEM_BASE_RESOLUTION = 4;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    geom_t Lx = 1;
    auto   m  = sfem::Mesh::create_hex8_cube(
            comm, SFEM_BASE_RESOLUTION * 1, SFEM_BASE_RESOLUTION * 1, SFEM_BASE_RESOLUTION * 1, 0, 0, 0, Lx, 1, 1);

    int  block_size = 1;
    auto fs         = sfem::FunctionSpace::create(m, block_size);
    fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
    fs->semi_structured_mesh().apply_hierarchical_renumbering();

    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, SFEM_OPERATOR, es);
    op->initialize();
    f->add_operator(op);

    auto bottom_ss = sfem::Sideset::create_from_selector(
            m, [=](const geom_t /*x*/, const geom_t y, const geom_t z) -> bool { return y > -1e-5 && y < 1e-5; });

    auto right_ss = sfem::Sideset::create_from_selector(
            m, [=](const geom_t x, const geom_t /*y*/, const geom_t z) -> bool { return x > (Lx - 1e-5) && x < (Lx + 1e-5); });

    sfem::DirichletConditions::Condition left{.sideset = bottom_ss, .value = -1, .component = 0};
    sfem::DirichletConditions::Condition right{.sideset = right_ss, .value = 1, .component = 0};

    auto conds = sfem::create_dirichlet_conditions(fs, {left, right}, es);
    f->add_constraint(conds);

    return test_linear_problem(f, "test_ssgmg_poisson_cube");
}

int test_ssgmg_linear_elasticity_cube() {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );

    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    const char *SFEM_OPERATOR       = "LinearElasticity";
    const char *SFEM_FINE_OP_TYPE   = "MF";
    const char *SFEM_COARSE_OP_TYPE = "MF";

    SFEM_READ_ENV(SFEM_COARSE_OP_TYPE, );
    SFEM_READ_ENV(SFEM_FINE_OP_TYPE, );

    int SFEM_ELEMENT_REFINE_LEVEL = 4;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    int SFEM_BASE_RESOLUTION = 4;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    geom_t Lx = 1;
    auto   m  = sfem::Mesh::create_hex8_cube(
            comm, SFEM_BASE_RESOLUTION * 1, SFEM_BASE_RESOLUTION * 1, SFEM_BASE_RESOLUTION * 1, 0, 0, 0, Lx, 1, 1);

    int  block_size = 3;
    auto fs         = sfem::FunctionSpace::create(m, block_size);
    fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
    fs->semi_structured_mesh().apply_hierarchical_renumbering();

    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, SFEM_OPERATOR, es);
    op->initialize();
    f->add_operator(op);

    auto left_ss = sfem::Sideset::create_from_selector(
            m, [=](const geom_t x, const geom_t y, const geom_t z) -> bool { return x > -1e-5 && x < 1e-5; });

    auto right_ss = sfem::Sideset::create_from_selector(
            m, [=](const geom_t x, const geom_t /*y*/, const geom_t z) -> bool { return x > (Lx - 1e-5) && x < (Lx + 1e-5); });

    sfem::DirichletConditions::Condition left{.sideset = left_ss, .value = -1, .component = 0};
    sfem::DirichletConditions::Condition right0{.sideset = right_ss, .value = 1, .component = 0};
    sfem::DirichletConditions::Condition right1{.sideset = right_ss, .value = 0, .component = 1};
    sfem::DirichletConditions::Condition right2{.sideset = right_ss, .value = 0, .component = 2};

    auto conds = sfem::create_dirichlet_conditions(fs, {left, right0, right1, right2}, es);
    f->add_constraint(conds);

    return test_linear_problem(f, "test_ssgmg_linear_elasticity_cube");
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_ssgmg_poisson_cube);
    SFEM_RUN_TEST(test_ssgmg_linear_elasticity_cube);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
