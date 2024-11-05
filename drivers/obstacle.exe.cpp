#include "sfem_Function.hpp"

#include "sfem_base.h"
#include "sfem_bcgs.hpp"
#include "sfem_cg.hpp"
#include "sfem_mprgp.hpp"

#include <sys/stat.h>
#include <cstdio>
#include <vector>

#include "proteus_hex8.h"
#include "proteus_hex8_laplacian.h"
#include "proteus_hex8_linear_elasticity.h"
#include "sfem_API.hpp"
#include "sfem_hex8_mesh_graph.h"

#include "boundary_condition.h"
#include "boundary_condition_io.h"
#include "dirichlet.h"

#include "matrixio_array.h"

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

    int SFEM_USE_ELASTICITY = 1;
    SFEM_READ_ENV(SFEM_USE_ELASTICITY, atoi);

    int SFEM_USE_PROJECTED_CG = 0;
    SFEM_READ_ENV(SFEM_USE_PROJECTED_CG, atoi);

    int SFEM_TEST_AGAINST_LINEAR = 0;
    SFEM_READ_ENV(SFEM_TEST_AGAINST_LINEAR, atoi);

    int SFEM_MAX_IT = 4000;
    SFEM_READ_ENV(SFEM_MAX_IT, atoi);

    bool SFEM_USE_GPU = true;
    SFEM_READ_ENV(SFEM_USE_GPU, atoi);

    int SFEM_EXPORT_CRS_MATRIX = false;
    SFEM_READ_ENV(SFEM_EXPORT_CRS_MATRIX, atoi);

    int SFEM_MATRIX_FREE = 1;
    SFEM_READ_ENV(SFEM_MATRIX_FREE, atoi);

    sfem::ExecutionSpace es = sfem::EXECUTION_SPACE_HOST;

    if (SFEM_USE_GPU) {
        es = sfem::EXECUTION_SPACE_DEVICE;
    }

    struct stat st = {0};
    if (stat(output_path, &st) == -1) {
        mkdir(output_path, 0700);
    }

    double tick = MPI_Wtime();

    const char *folder = argv[1];
    auto m = sfem::Mesh::create_from_file(comm, folder);
    int block_size = SFEM_USE_ELASTICITY ? m->spatial_dimension() : 1;

    auto fs = sfem::FunctionSpace::create(m, block_size);

    if (SFEM_ELEMENT_REFINE_LEVEL > 0) {
        fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
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
    auto f = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, SFEM_USE_ELASTICITY ? "LinearElasticity" : "Laplacian", es);
    op->initialize();
    f->add_constraint(conds);
    f->add_operator(op);

    auto contact_conds = sfem::create_contact_conditions_from_env(fs, es);

    ptrdiff_t ndofs = fs->n_dofs();
    auto x = sfem::create_buffer<real_t>(ndofs, es);
    auto rhs = sfem::create_buffer<real_t>(ndofs, es);

    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());

    std::shared_ptr<sfem::Operator<real_t>> linear_op;

    if (SFEM_MATRIX_FREE) {
        linear_op = sfem::make_linear_op(f);
    } else {
        linear_op = sfem::crs_hessian(f, x, es);
    }

    if (SFEM_EXPORT_CRS_MATRIX) {
        auto crs_graph = f->crs_graph();

        auto values = sfem::create_buffer<real_t>(crs_graph->colidx()->size(), es);

        f->hessian_crs(x->data(),
                       crs_graph->rowptr()->data(),
                       crs_graph->colidx()->data(),
                       values->data());

        char path[2048];
        sprintf(path, "%s/crs_matrix", output_path);
        write_crs(path, *crs_graph, *values);
    }

    std::shared_ptr<sfem::MatrixFreeLinearSolver<real_t>> solver;
    {
        auto mprgp = sfem::create_mprgp(linear_op, es);

        if (SFEM_USE_PROJECTED_CG) {
            mprgp->set_expansion_type(sfem::MPRGP<real_t>::EXPANSION_TYPE_PROJECTED_CG);
        }

        mprgp->set_op(linear_op);

        auto h_upper_bound = sfem::create_buffer<real_t>(ndofs, sfem::MEMORY_SPACE_HOST);

        {  // Fill default upper-bound value
            auto ub = h_upper_bound->data();
            for (ptrdiff_t i = 0; i < ndofs; i++) {
                ub[i] = 1000;
            }
        }

#ifdef SFEM_ENABLE_CUDA
        auto upper_bound = sfem::to_device(h_upper_bound);
#else
        auto upper_bound = h_upper_bound;
#endif

        contact_conds->apply(upper_bound->data());

#ifdef SFEM_ENABLE_CUDA
        h_upper_bound = sfem::to_host(upper_bound);
#endif

        char path[2048];
        sprintf(path, "%s/upper_bound.raw", output_path);
        if (array_write(comm, path, SFEM_MPI_REAL_T, (void *)h_upper_bound->data(), ndofs, ndofs)) {
            return SFEM_FAILURE;
        }

        mprgp->verbose = true;
        mprgp->set_max_it(SFEM_MAX_IT);
        mprgp->set_rtol(1e-12);
        mprgp->set_atol(1e-8);
        mprgp->set_upper_bound(upper_bound);
        solver = mprgp;
    }

    double solve_tick = MPI_Wtime();
    solver->apply(rhs->data(), x->data());
    double solve_tock = MPI_Wtime();

    char path[2048];
    sprintf(path, "%s/u.raw", output_path);
    if (array_write(comm, path, SFEM_MPI_REAL_T, (void *)x->data(), ndofs, ndofs)) {
        return SFEM_FAILURE;
    }

    sprintf(path, "%s/rhs.raw", output_path);
    if (array_write(comm, path, SFEM_MPI_REAL_T, (void *)rhs->data(), ndofs, ndofs)) {
        return SFEM_FAILURE;
    }

    double tock = MPI_Wtime();

    ptrdiff_t nelements = m->n_elements();
    ptrdiff_t nnodes =
            fs->has_semi_structured_mesh() ? fs->semi_structured_mesh().n_nodes() : m->n_nodes();

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
