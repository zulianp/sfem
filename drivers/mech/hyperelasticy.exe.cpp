#include <memory>

#include "sfem_test.h"

#include "sfem_Function.hpp"

#include "sfem_Buffer.hpp"
#include "sfem_base.h"
#include "sfem_crs_SpMV.hpp"
#include "spmv.h"

#include "matrixio_array.h"

#include "sfem_API.hpp"
#include "sfem_DirichletConditions.hpp"
#include "sfem_Env.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_solver.hpp"
#endif

#include "sfem_ssmgc.hpp"

int solve_hyperelasticity(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    SFEM_TRACE_SCOPE("solve_hyperelasticity");

    if (argc != 4) {
        fprintf(stderr, "usage: %s <mesh> <dirichlet_conditions> <output>\n", argv[0]);
        return SFEM_FAILURE;
    }

    if(comm->size() > 1) {
        SFEM_ERROR("MPI runtimes are not supported!\n");
    }

    const char *mesh_path      = argv[1];
    const char *dirichlet_path = argv[2];
    std::string output_path    = argv[3];

    int         SFEM_ELEMENT_REFINE_LEVEL = sfem::Env::read("SFEM_ELEMENT_REFINE_LEVEL", 0);
    const char *SFEM_OPERATOR             = "NeoHookeanOgden";
    SFEM_READ_ENV(SFEM_OPERATOR, );

    const bool   SFEM_VERBOSE     = sfem::Env::read("SFEM_VERBOSE", 0);
    const real_t SFEM_LSOLVE_RTOL = sfem::Env::read("SFEM_LSOLVE_RTOL", 1e-3);
    const real_t SFEM_NL_TOL      = sfem::Env::read("SFEM_NL_TOL", 1e-9);

    sfem::ExecutionSpace es = sfem::EXECUTION_SPACE_HOST;
    {
        const char *SFEM_EXECUTION_SPACE{nullptr};
        SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );
        if (SFEM_EXECUTION_SPACE) {
            es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
        }
    }

    auto      mesh       = sfem::Mesh::create_from_file(comm, mesh_path);
    const int block_size = mesh->spatial_dimension();
    auto      fs         = sfem::FunctionSpace::create(mesh, block_size);

    if (SFEM_ELEMENT_REFINE_LEVEL > 1) {
        fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
    }

    auto dirichlet_conditions = sfem::DirichletConditions::create_from_file(fs, dirichlet_path);

// FIXME
#ifdef SFEM_ENABLE_CUDA
    {
        auto elements = fs->device_elements();
        if (!elements) {
            elements = create_device_elements(fs, fs->element_type());
            fs->set_device_elements(elements);
        }
    }
#endif

    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, SFEM_OPERATOR, es);
    op->initialize();
    f->add_operator(op);
    f->add_constraint(dirichlet_conditions);

    const ptrdiff_t ndofs        = fs->n_dofs();
    auto            displacement = sfem::create_buffer<real_t>(ndofs, es);
    auto            increment    = sfem::create_buffer<real_t>(ndofs, es);
    auto            rhs          = sfem::create_buffer<real_t>(ndofs, es);

    auto linear_op = sfem::create_linear_operator("MF", f, displacement, es);
    auto cg        = sfem::create_cg<real_t>(linear_op, es);
    cg->verbose    = SFEM_VERBOSE;
    cg->set_max_it(20000);
    cg->set_op(linear_op);
    cg->set_rtol(SFEM_LSOLVE_RTOL);
    cg->set_atol(1e-11);

    // Newton iteration
    int    nl_max_it          = sfem::Env::read("SFEM_NL_MAX_IT", 30);
    real_t alpha              = sfem::Env::read("SFEM_NL_ALPHA", 1.0);
    bool   enable_line_search = sfem::Env::read("SFEM_ENABLE_LINE_SEARCH", true);
    auto   blas               = sfem::blas<real_t>(es);

    printf("Solving hyperelasticity: #%ld dofs\n", (long)fs->n_dofs());

    if (sfem::Env::read("SFEM_USE_GRADIENT_DESCENT", false)) {
        for (int i = 0; i < nl_max_it; i++) {
            blas->zeros(ndofs, rhs->data());
            f->gradient(displacement->data(), rhs->data());

            const real_t gnorm = blas->norm2(ndofs, rhs->data());
            printf("%d) gnorm = %g\n", i, gnorm);

            if (gnorm < SFEM_NL_TOL) break;

            blas->axpy(ndofs, -alpha, rhs->data(), displacement->data());
        }
    } else {
        real_t energy         = 0;
        real_t selected_alpha = 0;
        f->value(displacement->data(), &energy);

        // Newton solver with line search
        printf("%-10s %-14s %-14s %-14s\n", "Iteration", "gnorm", "energy", "alpha");
        printf("-------------------------------------------------------------\n");

        for (int i = 0; i < nl_max_it; i++) {
            f->update(displacement->data());
            blas->zeros(ndofs, rhs->data());
            f->gradient(displacement->data(), rhs->data());

            const real_t gnorm = blas->norm2(ndofs, rhs->data());
            printf("%-10d %-14.4e %-14.4e %-14.4f\n", i, gnorm, energy, -selected_alpha);

            if (gnorm < SFEM_NL_TOL) break;

            blas->zeros(ndofs, increment->data());
            f->copy_constrained_dofs(rhs->data(), increment->data());
            cg->apply(rhs->data(), increment->data());

            if (enable_line_search) {
                std::vector<real_t> alphas{-2 * alpha,
                                           -alpha,
                                           -(real_t)0.9 * alpha,
                                           2 * alpha / 3,
                                           -alpha / 2,
                                           -alpha / 4,
                                           -alpha / 8,
                                           -alpha / 32,
                                           -alpha / 128};
                std::vector<real_t> energies(alphas.size(), 0);

                f->value_steps(displacement->data(), increment->data(), alphas.size(), alphas.data(), energies.data());
                const int min_energy_index = std::distance(energies.begin(), std::min_element(energies.begin(), energies.end()));
                selected_alpha             = alphas[min_energy_index];
                energy                     = energies[min_energy_index];
                blas->axpy(ndofs, selected_alpha, increment->data(), displacement->data());
            } else {
                selected_alpha = -alpha;
                blas->axpy(ndofs, selected_alpha, increment->data(), displacement->data());
                f->value(displacement->data(), &energy);
            }
        }
    }
    

    // Output to disk
    sfem::create_directory(output_path.c_str());
    if (fs->has_semi_structured_mesh()) {
        fs->semi_structured_mesh().export_as_standard((output_path + "/mesh").c_str());
        fs->mesh_ptr()->write((output_path + "/coarse_mesh").c_str());
    } else {
        fs->mesh_ptr()->write((output_path + "/mesh").c_str());
    }

    auto out = f->output();

    out->set_output_dir((output_path + "/out").c_str());
    out->enable_AoS_to_SoA(true);

    out->write("rhs", sfem::to_host(rhs)->data());
    out->write("disp", sfem::to_host(displacement)->data());

    return SFEM_SUCCESS;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return solve_hyperelasticity(ctx->communicator(), argc, argv);
}
