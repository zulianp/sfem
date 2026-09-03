#include <stdio.h>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "sfem_defs.hpp"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_NeumannConditions.hpp"
#include "smesh_env.hpp"
#include "smesh_mesh_reorder.hpp"

int lsolve(const std::shared_ptr<sfem::Function>          &f,
           const std::shared_ptr<sfem::Op>                &material_op,
           const std::shared_ptr<sfem::NeumannConditions> &neumann_conditions,
           const smesh::Path                              &output_dir) {
    auto es        = f->execution_space();
    auto fs        = f->space();
    auto m         = fs->mesh_ptr();
    auto linear_op = sfem::create_linear_operator(sfem::op_type::MATRIX_FREE, f, nullptr, es);
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

    auto blas = sfem::blas<real_t>(es);
    blas->zeros(fs->n_dofs(), x->data());
    blas->zeros(fs->n_dofs(), rhs->data());
    if (neumann_conditions) {
        if (!neumann_conditions->is_linear()) {
            SFEM_ERROR("linear_elasticity does not support follower pressure; use a nonlinear driver\n");
            return SFEM_FAILURE;
        }
        if (neumann_conditions->gradient(x->data(), rhs->data()) != SFEM_SUCCESS) return SFEM_FAILURE;
        blas->scal(fs->n_dofs(), -1, rhs->data());
    }

    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());

    double    tick         = MPI_Wtime();
    const int solve_status = cg->apply(rhs->data(), x->data());
    double    tock         = MPI_Wtime();

    if (SFEM_VERBOSE) {
        printf("---------------------\n");
        printf("%s #dofs %ld (%g seconds)\n", output_dir.c_str(), fs->n_dofs(), tock - tick);
        printf("---------------------\n");
    }

    bool SFEM_ENABLE_OUTPUT = smesh::Env::read<bool>("SFEM_ENABLE_OUTPUT", true);

    if (solve_status != SFEM_SUCCESS) return solve_status;

    if (SFEM_ENABLE_OUTPUT) {
        if (SFEM_VERBOSE) {
            printf("Writing output in %s\n", output_dir.c_str());
        }

        smesh::create_directory(output_dir);

        if (fs->has_semi_structured_mesh()) {
            m->write(output_dir / "coarse_mesh");
            smesh::semistructured_export_as_standard(fs->mesh_ptr(), output_dir / "mesh");
        } else {
            m->write(output_dir / "mesh");
        }

        auto output = f->output();
        output->enable_AoS_to_SoA(fs->block_size() > 1);
        output->set_output_dir(output_dir);

        auto material_reaction = sfem::create_buffer<real_t>(fs->n_dofs(), es);
        blas->zeros(fs->n_dofs(), material_reaction->data());
        if (material_op->gradient(x->data(), material_reaction->data()) != SFEM_SUCCESS) return SFEM_FAILURE;

        real_t material_objective = 0;
        if (material_op->value(x->data(), &material_objective) != SFEM_SUCCESS) return SFEM_FAILURE;

#ifdef SFEM_ENABLE_CUDA
        if (x->mem_space() == sfem::MEMORY_SPACE_DEVICE) {
            output->write("x", smesh::to_host(x)->data());
            output->write("rhs", smesh::to_host(rhs)->data());
            output->write("material_reaction", smesh::to_host(material_reaction)->data());
        } else
#endif
        {
            output->write("x", x->data());
            output->write("rhs", rhs->data());
            output->write("material_reaction", material_reaction->data());
        }

        if (!m->comm()->rank()) {
            std::ofstream quantities((output_dir / "quantities.yaml").c_str());
            quantities << std::setprecision(17);
            quantities << "material_objective: " << material_objective << '\n';
        }
    }

    return SFEM_SUCCESS;
}

int solve_linear_elasticity(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    SFEM_TRACE_SCOPE("solve_linear_elasticity");

    if (argc < 4 || argc > 6) {
        fprintf(stderr, "usage %s <mesh> <dirichlet.yaml|NONE> [neumann.yaml|NONE] [operator.yaml|NONE] <output_dir>\n", argv[0]);
        return SFEM_FAILURE;
    }

    auto es                        = smesh::Env::read("SFEM_EXECUTION_SPACE", sfem::EXECUTION_SPACE_HOST);
    auto SFEM_OPERATOR             = smesh::Env::read_string("SFEM_OPERATOR", "LinearElasticity");
    int  SFEM_ELEMENT_REFINE_LEVEL = smesh::Env::read("SFEM_ELEMENT_REFINE_LEVEL", 0);

    smesh::Path mesh_path{argv[1]};
    smesh::Path dirichlet_path{argv[2]};
    smesh::Path neumann_path{argc >= 5 ? argv[3] : "NONE"};
    smesh::Path operator_path{argc == 6 ? argv[4] : smesh::Env::read_string("SFEM_OPERATOR_CONFIG", "NONE")};
    smesh::Path output_dir{argv[argc - 1]};

    auto m = sfem::Mesh::create_from_file(comm, mesh_path);

    // External node and side sets use the numbering stored on disk. Preserve it
    // for the YAML form unless callers explicitly request the legacy reordering.
    if (smesh::Env::read("SFEM_REORDER", argc == 4)) {
        auto sfc = smesh::SFC::create_from_env();
        if (sfc->reorder(*m) != SFEM_SUCCESS) return SFEM_FAILURE;
    }

    if (smesh::Env::read("SFEM_PROMOTE_TO_P2", false)) {
        m = smesh::promote_to(smesh::TET10, m);
    } else if (SFEM_ELEMENT_REFINE_LEVEL > 0) {
        m = smesh::to_semistructured(SFEM_ELEMENT_REFINE_LEVEL, m, true, false);
    }

    auto fs = sfem::FunctionSpace::create(m, m->spatial_dimension());
    fs->initialize_packed_mesh();

    std::shared_ptr<sfem::Op> op;
    if (operator_path.to_string() != "NONE") {
#ifdef SFEM_ENABLE_RYAML
        std::ifstream stream(operator_path.c_str());
        if (!stream.good()) {
            SFEM_ERROR("Unable to read operator configuration %s\n", operator_path.c_str());
            return SFEM_FAILURE;
        }
        std::ostringstream contents;
        contents << stream.rdbuf();
        op = sfem::create_op_from_yaml(fs, contents.str(), es);
#else
        SFEM_ERROR("Operator YAML requires SFEM_ENABLE_RYAML\n");
        return SFEM_FAILURE;
#endif
    } else {
        op = sfem::create_op(fs, SFEM_OPERATOR, es);
        if (op && op->initialize() != SFEM_SUCCESS) return SFEM_FAILURE;
    }

    if (!op) return SFEM_FAILURE;

    auto f = sfem::Function::create(fs);
    if (dirichlet_path.to_string() != "NONE") {
        auto conds = sfem::DirichletConditions::create_from_file(fs, dirichlet_path);
        if (!conds) return SFEM_FAILURE;
        f->add_constraint(conds);
    }
    f->add_operator(op);

    std::shared_ptr<sfem::NeumannConditions> neumann_conditions;
    if (neumann_path.to_string() != "NONE") {
        neumann_conditions = sfem::NeumannConditions::create_from_file(fs, neumann_path);
        if (!neumann_conditions) return SFEM_FAILURE;
        f->add_operator(neumann_conditions);
    }

    return lsolve(f, op, neumann_conditions, output_dir);
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return solve_linear_elasticity(ctx->communicator(), argc, argv);
}
