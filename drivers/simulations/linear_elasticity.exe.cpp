#include <stdio.h>

#include "sfem_defs.hpp"

#include "sfem_API.hpp"
#include "smesh_env.hpp"
#include "sfem_Function.hpp"
#include "smesh_mesh_reorder.hpp"


int lsolve(const std::shared_ptr<sfem::Function> &f, const std::string &output_dir) {
    auto es        = f->execution_space();
    auto fs        = f->space();
    auto m         = fs->mesh_ptr();
    auto linear_op = sfem::create_linear_operator(MATRIX_FREE, f, nullptr, es);
    auto cg        = sfem::create_cg<real_t>(linear_op, es);

    int    SFEM_MAX_IT             = smesh::Env::read<int>("SFEM_MAX_IT", 20000);
    bool   SFEM_USE_PRECONDITIONER = smesh::Env::read<bool>("SFEM_USE_PRECONDITIONER", false);
    bool   SFEM_VERBOSE            = smesh::Env::read<bool>("SFEM_VERBOSE", true);
    real_t SFEM_RTOL               = smesh::Env::read<real_t>("SFEM_RTOL", 1e-6);

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

    bool SFEM_ENABLE_OUTPUT = smesh::Env::read<bool>("SFEM_ENABLE_OUTPUT", true);

    if (SFEM_ENABLE_OUTPUT) {
        if (SFEM_VERBOSE) {
            printf("Writing output in %s\n", output_dir.c_str());
        }

        smesh::create_directory(output_dir.c_str());

        if (fs->has_semi_structured_mesh()) {
            m->write(smesh::Path((output_dir + "/coarse_mesh")));
            sfem::semi_structured_export_as_standard(fs->mesh_ptr(), (output_dir + "/mesh").c_str());
        } else {
            m->write(smesh::Path((output_dir + "/mesh")));
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

int solve_linear_elasticity(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    SFEM_TRACE_SCOPE("solve_linear_elasticity");

    if(argc != 4) {
        fprintf(stderr, "usage %s <mesh> <dirichlet.yaml> <output>\n", argv[0]);
        return SFEM_FAILURE;
    }
    
    auto es                        = smesh::Env::read("SFEM_EXECUTION_SPACE", sfem::EXECUTION_SPACE_HOST);
    auto SFEM_OPERATOR             = smesh::Env::read_string("SFEM_OPERATOR", "LinearElasticity");
    int  SFEM_ELEMENT_REFINE_LEVEL = smesh::Env::read("SFEM_ELEMENT_REFINE_LEVEL", 0);

    auto m = sfem::Mesh::create_from_file(comm, smesh::Path(argv[1]));

    // Important for packed elements
    auto sfc = smesh::SFC::create_from_env();
    sfc->reorder(*m);

    if(smesh::Env::read("SFEM_PROMOTE_TO_P2", false)) {
        m = smesh::promote_to(smesh::TET10, m);
    } else if (SFEM_ELEMENT_REFINE_LEVEL > 0) {
        m = smesh::to_semistructured(SFEM_ELEMENT_REFINE_LEVEL, m, true, false);
    }

    auto fs = sfem::FunctionSpace::create(m, m->spatial_dimension());
    fs->initialize_packed_mesh();

    auto op = sfem::create_op(fs, SFEM_OPERATOR, es);
    op->initialize();

    auto f     = sfem::Function::create(fs);
    auto conds = sfem::DirichletConditions::create_from_file(fs, argv[2]);
    f->add_constraint(conds);
    f->add_operator(op);

    return lsolve(f, argv[3]);
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return solve_linear_elasticity(ctx->communicator(), argc, argv);
}
