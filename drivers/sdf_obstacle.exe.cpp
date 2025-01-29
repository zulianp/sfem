#include "sfem_Function.hpp"

#include "sfem_ShiftedPenalty.hpp"
#include "sfem_ShiftedPenaltyMultigrid.hpp"
#include "sfem_base.h"
#include "sfem_bcgs.hpp"
#include "sfem_cg.hpp"
#include "sfem_mprgp.hpp"

#include <sys/stat.h>
#include <cstdio>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_hex8_mesh_graph.h"
#include "sshex8.h"
#include "sshex8_laplacian.h"
#include "sshex8_linear_elasticity.h"

#include "boundary_condition.h"
#include "boundary_condition_io.h"
#include "dirichlet.h"

#include "matrixio_array.h"

#include "sfem_SSMultigrid.hpp"
#include "sfem_ssmgc.hpp"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 3) {
        fprintf(stderr, "usage: %s <folder> <output>\n", argv[0]);
        return EXIT_FAILURE;
    }

#ifdef SFEM_ENABLE_CUDA
    sfem::register_device_ops();
#endif

    // -------------------------------
    // Read inputs
    // -------------------------------

    const char *output_path = argv[2];

    int SFEM_ELEMENT_REFINE_LEVEL = 0;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    int SFEM_USE_PROJECTED_CG = 0;
    SFEM_READ_ENV(SFEM_USE_PROJECTED_CG, atoi);

    int SFEM_TEST_AGAINST_LINEAR = 0;
    SFEM_READ_ENV(SFEM_TEST_AGAINST_LINEAR, atoi);

    int SFEM_MAX_IT = 20;
    SFEM_READ_ENV(SFEM_MAX_IT, atoi);

    int SFEM_CONTACT_LINEARIZATIONS = 1;
    SFEM_READ_ENV(SFEM_CONTACT_LINEARIZATIONS, atoi);

    bool SFEM_USE_GPU = true;
    SFEM_READ_ENV(SFEM_USE_GPU, atoi);

    int SFEM_EXPORT_CRS_MATRIX = false;
    SFEM_READ_ENV(SFEM_EXPORT_CRS_MATRIX, atoi);

    int SFEM_MATRIX_FREE = 1;
    SFEM_READ_ENV(SFEM_MATRIX_FREE, atoi);

    int SFEM_USE_BSR_MATRIX = 1;
    SFEM_READ_ENV(SFEM_USE_BSR_MATRIX, atoi);

    int SFEM_USE_SHIFTED_PENALTY = 0;
    SFEM_READ_ENV(SFEM_USE_SHIFTED_PENALTY, atoi);

    const char *SFEM_CONTACT_CONDITIONS = nullptr;
    SFEM_READ_ENV(SFEM_CONTACT_CONDITIONS, );

    float SFEM_DAMPING = 1;
    SFEM_READ_ENV(SFEM_DAMPING, atof);

    if (!SFEM_CONTACT_CONDITIONS) {
        assert(false);
        fprintf(stderr, "SFEM_CONTACT_CONDITIONS must be defined!\n");
        return EXIT_FAILURE;
    }

    sfem::ExecutionSpace es = sfem::EXECUTION_SPACE_HOST;

    if (SFEM_USE_GPU) {
        es = sfem::EXECUTION_SPACE_DEVICE;
    }

    sfem::create_directory(output_path);

    double tick = MPI_Wtime();

    const char *folder     = argv[1];
    auto        m          = sfem::Mesh::create_from_file(comm, folder);
    const int   block_size = 3;

    auto fs = sfem::FunctionSpace::create(m, block_size);

    if (SFEM_ELEMENT_REFINE_LEVEL > 0) {
        fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
        fs->semi_structured_mesh().apply_hierarchical_renumbering();
    }

#ifdef SFEM_ENABLE_CUDA
    {
        auto elements = fs->device_elements();
        if (!elements) {
            elements = create_device_elements(fs, fs->element_type());
            fs->set_device_elements(elements);
        }
    }
#endif

    auto conds = sfem::create_dirichlet_conditions_from_env(fs, es);
    auto f     = sfem::Function::create(fs);
    auto op    = sfem::create_op(fs, "LinearElasticity", es);
    op->initialize();
    f->add_constraint(conds);
    f->add_operator(op);

    auto contact_conds = sfem::ContactConditions::create_from_file(fs, SFEM_CONTACT_CONDITIONS);

    ptrdiff_t ndofs = fs->n_dofs();
    auto      x     = sfem::create_buffer<real_t>(ndofs, es);
    auto      rhs   = sfem::create_buffer<real_t>(ndofs, es);

    std::shared_ptr<sfem::Operator<real_t>> linear_op;
    if (SFEM_MATRIX_FREE) {
        linear_op = sfem::make_linear_op(f);
    } else {
        if (SFEM_USE_BSR_MATRIX && fs->block_size() != 1) {
            linear_op = sfem::hessian_bsr(f, x, es);
        } else {
            linear_op = sfem::hessian_crs(f, x, es);
        }
    }

    contact_conds->update(x->data());

    std::shared_ptr<sfem::Operator<real_t>> solver;
    if (SFEM_ELEMENT_REFINE_LEVEL > 0 && !SFEM_USE_SHIFTED_PENALTY) {
        printf("Using Shifted-Penalty Multigrid\n");
        solver = sfem::create_ssmgc(f, contact_conds, es, nullptr);
    } else {
        printf("Using Shifted-Penalty\n");
        int SFEM_USE_STEEPEST_DESCENT = 0;
        SFEM_READ_ENV(SFEM_USE_STEEPEST_DESCENT, atoi);
        auto cc_op       = contact_conds->linear_constraints_op();
        auto cc_op_t     = contact_conds->linear_constraints_op_transpose();
        auto upper_bound = sfem::create_buffer<real_t>(contact_conds->n_constrained_dofs(), es);
        contact_conds->signed_distance(x->data(), upper_bound->data());

        int  sym_block_size = (block_size == 3 ? 6 : 3);
        auto normal_prod    = sfem::create_buffer<real_t>(sym_block_size * contact_conds->n_constrained_dofs(), es);
        contact_conds->hessian_block_diag_sym(x->data(), normal_prod->data());
        auto sbv = sfem::create_sparse_block_vector(contact_conds->node_mapping(), normal_prod);

        auto sp = std::make_shared<sfem::ShiftedPenalty<real_t>>();
        sp->set_op(linear_op);
        sp->default_init();

        sp->set_atol(1e-12);
        sp->set_max_it(SFEM_MAX_IT);
        sp->set_max_inner_it(30);
        sp->set_damping(SFEM_DAMPING);
        sp->set_penalty_param(10);

        auto cg     = sfem::create_cg(linear_op, es);
        cg->verbose = false;
        auto diag   = sfem::create_buffer<real_t>((fs->n_dofs() / block_size) * (block_size == 3 ? 6 : 3), es);
        auto mask   = sfem::create_buffer<mask_t>(mask_count(fs->n_dofs()), es);
        f->hessian_block_diag_sym(nullptr, diag->data());
        f->constaints_mask(mask->data());

        auto sj = sfem::h_shiftable_block_sym_jacobi(diag, mask);
        cg->set_preconditioner_op(sj);

        cg->set_atol(1e-12);
        cg->set_rtol(1e-4);
        cg->set_max_it(20000);

        sp->linear_solver_ = cg;
        sp->enable_steepest_descent(SFEM_USE_STEEPEST_DESCENT);

        sp->verbose = true;

        sp->set_upper_bound(upper_bound);
        sp->set_constraints_op(cc_op, cc_op_t, sbv);
        solver = sp;
    }

    // auto upper_bound_increment = sfem::create_buffer<real_t>(contact_conds->n_constrained_dofs(), es);

    double solve_tick = MPI_Wtime();

    // auto blas = sfem::blas<real_t>(es);

    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());

    solver->apply(rhs->data(), x->data());

    // for (int k = 0; k < SFEM_CONTACT_LINEARIZATIONS; k++) {
    //     printf("---------------------------\n");
    //     printf("Contact linerization %d\n", k);
    //     printf("---------------------------\n");

    //     solver->apply(rhs->data(), x->data());

    //     if (k + 1 < SFEM_CONTACT_LINEARIZATIONS) {
    //         // 1) Project x onto the current constraints space and store it as a translation
    //         blas->zeros(contact_conds->n_constrained_dofs(), upper_bound->data());
    //         cc_op->apply(x->data(), upper_bound->data());

    //         // 2) Update the normal fields with respect to x
    //         contact_conds->update(x->data());
    //         blas->zeros(normal_prod->size(), normal_prod->data());
    //         contact_conds->hessian_block_diag_sym(x->data(), normal_prod->data());

    //         // 3) Compute the distance increment
    //         blas->zeros(upper_bound_increment->size(), upper_bound_increment->data());
    //         contact_conds->signed_distance(x->data(), upper_bound_increment->data());

    //         // 4) Add increment to the upper-bound
    //         blas->axpy(upper_bound_increment->size(), 1, upper_bound_increment->data(), upper_bound->data());
    //         solver->set_upper_bound(upper_bound);
    //     }
    // }

    double solve_tock = MPI_Wtime();

#ifdef SFEM_ENABLE_CUDA
    auto h_x   = sfem::to_host(x);
    auto h_rhs = sfem::to_host(rhs);
#else
    auto h_x   = x;
    auto h_rhs = rhs;
#endif

    auto upper_bound_viz = sfem::create_buffer<real_t>(ndofs, sfem::MEMORY_SPACE_HOST);
    contact_conds->signed_distance_for_mesh_viz(x->data(), upper_bound_viz->data());

    if (fs->has_semi_structured_mesh()) {
        std::string path = output_path;
        path += "/ssmesh";
        fs->semi_structured_mesh().export_as_standard(path.c_str());
    }

    auto output = f->output();
    output->set_output_dir(output_path);
    output->enable_AoS_to_SoA(true);
    output->write("disp", h_x->data());
    output->write("rhs", h_rhs->data());
    output->write("gap", upper_bound_viz->data());

    double tock = MPI_Wtime();

    ptrdiff_t nelements = m->n_elements();
    ptrdiff_t nnodes    = fs->has_semi_structured_mesh() ? fs->semi_structured_mesh().n_nodes() : m->n_nodes();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("%s (%s):\n", argv[0], type_to_string(fs->element_type()));
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #dofs %ld\n", (long)nelements, (long)nnodes, (long)ndofs);
        printf("TTS:\t\t%g [s], solve: %g [s])\n", tock - tick, solve_tock - solve_tick);
        printf("----------------------------------------\n");
    }

    return MPI_Finalize();
}
