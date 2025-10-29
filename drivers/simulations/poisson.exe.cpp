#include <stdio.h>

#include "sfem_defs.h"

#include "sfem_API.hpp"
#include "sfem_Env.hpp"
#include "sfem_Function.hpp"
#include "sfem_SFC.hpp"
#include "sfem_P1toP2.hpp"

int lsolve(const std::shared_ptr<sfem::Function> &f, const std::string &output_dir) {
    auto es        = f->execution_space();
    auto fs        = f->space();
    auto m         = fs->mesh_ptr();
    auto linear_op = sfem::create_linear_operator(MATRIX_FREE, f, nullptr, es);
    auto cg        = sfem::create_cg<real_t>(linear_op, es);

    int    SFEM_MAX_IT             = sfem::Env::read<int>("SFEM_MAX_IT", 20000);
    bool   SFEM_USE_PRECONDITIONER = sfem::Env::read<bool>("SFEM_USE_PRECONDITIONER", false);
    bool   SFEM_VERBOSE            = sfem::Env::read<bool>("SFEM_VERBOSE", true);
    real_t SFEM_RTOL               = sfem::Env::read<real_t>("SFEM_RTOL", 1e-6);

    cg->set_max_it(SFEM_MAX_IT);
    cg->verbose = SFEM_VERBOSE;
    cg->set_op(linear_op);
    cg->set_rtol(SFEM_RTOL);

    if (SFEM_USE_PRECONDITIONER) {
        auto diag = sfem::create_buffer<real_t>(fs->n_dofs(), es);
        f->hessian_diag(nullptr, diag->data());
        auto preconditioner = sfem::create_shiftable_jacobi(diag, es);
        cg->set_preconditioner_op(preconditioner);
    }

    auto x   = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto rhs = sfem::create_buffer<real_t>(fs->n_dofs(), es);

    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());

    double tick = MPI_Wtime();
    cg->apply(rhs->data(), x->data());
    double tock = MPI_Wtime();

    if (SFEM_VERBOSE) {
        printf("---------------------\n");
        printf("%s #dofs %ld (%g seconds)\n", output_dir.c_str(), fs->n_dofs(), tock - tick);
        printf("---------------------\n");
    }

    bool SFEM_ENABLE_OUTPUT = sfem::Env::read<bool>("SFEM_ENABLE_OUTPUT", true);

    if (SFEM_ENABLE_OUTPUT) {
        if (SFEM_VERBOSE) {
            printf("Writing output in %s\n", output_dir.c_str());
        }

        sfem::create_directory(output_dir.c_str());

        if (fs->has_semi_structured_mesh()) {
            m->write((output_dir + "/coarse_mesh").c_str());
            fs->semi_structured_mesh().export_as_standard((output_dir + "/mesh").c_str());
        } else {
            m->write((output_dir + "/mesh").c_str());
        }

        auto output = f->output();
        output->enable_AoS_to_SoA(fs->block_size() > 1);
        output->set_output_dir(output_dir.c_str());

#ifdef SFEM_ENABLE_CUDA
        if (x->mem_space() == sfem::MEMORY_SPACE_DEVICE) {
            output->write("x", sfem::to_host(x)->data());
            output->write("rhs", sfem::to_host(rhs)->data());
        } else
#endif
        {
            output->write("x", x->data());
            output->write("rhs", rhs->data());
        }
    }

    return SFEM_SUCCESS;
}

int solve_poisson_problem(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    SFEM_TRACE_SCOPE("solve_poisson_problem");
    
    auto es                        = sfem::Env::read("SFEM_EXECUTION_SPACE", sfem::EXECUTION_SPACE_HOST);
    auto SFEM_OPERATOR             = sfem::Env::read_string("SFEM_OPERATOR", "Laplacian");
    int  SFEM_ELEMENT_REFINE_LEVEL = sfem::Env::read("SFEM_ELEMENT_REFINE_LEVEL", 0);
    int  SFEM_BASE_RESOLUTION      = sfem::Env::read<int>("SFEM_BASE_RESOLUTION", 20);
    auto SFEM_ELEM_TYPE            = type_from_string(sfem::Env::read_string("SFEM_ELEM_TYPE", "HEX8").c_str());

    auto m = sfem::Mesh::create_cube(
            comm, SFEM_ELEM_TYPE, SFEM_BASE_RESOLUTION, SFEM_BASE_RESOLUTION, SFEM_BASE_RESOLUTION, 0, 0, 0, 4, 4, 4);

    // Important for packed elements
    auto sfc = sfem::SFC::create_from_env();
    sfc->reorder(*m);

    if(sfem::Env::read("SFEM_PROMOTE_TO_P2", false)) {
        m = sfem::convert_p1_mesh_to_p2(m);
    }

    auto fs = sfem::FunctionSpace::create(m, 1);
    fs->initialize_packed_mesh();

    auto op = sfem::create_op(fs, SFEM_OPERATOR, es);
    op->initialize();

    auto sideset0 = sfem::Sideset::create_from_selector(
            m, [](const geom_t x, const geom_t y, const geom_t z) -> bool { return z > 3.999; });

    auto sideset1 = sfem::Sideset::create_from_selector(
            m, [](const geom_t x, const geom_t y, const geom_t z) -> bool { return z < 0.001; });

    std::vector<sfem::DirichletConditions::Condition> boundary_conditions = {{.sidesets = sideset1, .value = -1, .component = 0},
                                                                             {.sidesets = sideset0, .value = 1, .component = 0}};

    if (SFEM_ELEMENT_REFINE_LEVEL > 1) {
        fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
    }

    auto f     = sfem::Function::create(fs);
    auto conds = sfem::create_dirichlet_conditions(fs, boundary_conditions, op->execution_space());
    f->add_constraint(conds);
    f->add_operator(op);

    return lsolve(f, "output_poisson");
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return solve_poisson_problem(ctx->communicator(), argc, argv);
}
