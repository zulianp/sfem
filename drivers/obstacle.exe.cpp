#include "sfem_Function.hpp"

#include "sfem_base.h"
#include "sfem_bcgs.hpp"
#include "sfem_cg.hpp"
#include "sfem_mprgp.hpp"

#include <sys/stat.h>
#include <cstdio>
#include <vector>

#include "proteus_hex8_laplacian.h"
#include "sfem_API.hpp"
#include "sfem_hex8_mesh_graph.h"

#include "boundary_condition.h"
#include "boundary_condition_io.h"

#include "matrixio_array.h"

namespace sfem {
    class SemiStructuredMesh {
    public:
        std::shared_ptr<Mesh> macro_mesh_;
        int level_;

        idx_t **elements_{nullptr};
        ptrdiff_t n_unique_nodes_{-1}, interior_start_{-1};

        idx_t **element_data() { return elements_; }
        geom_t **point_data() { return ((mesh_t *)macro_mesh_->impl_mesh())->points; }
        ptrdiff_t interior_start() const { return interior_start_; }

        SemiStructuredMesh(const std::shared_ptr<Mesh> macro_mesh, const int level)
            : macro_mesh_(macro_mesh), level_(level) {
            const int nxe = proteus_hex8_nxe(level);
            elements_ = (idx_t **)malloc(nxe * sizeof(idx_t *));
            for (int d = 0; d < nxe; d++) {
                elements_[d] = (idx_t *)malloc(macro_mesh_->n_elements() * sizeof(idx_t));
            }

#ifndef NDEBUG
            for (int d = 0; d < nxe; d++) {
                for (ptrdiff_t i = 0; i < macro_mesh_->n_elements(); i++) {
                    elements_[d][i] = -1;
                }
            }
#endif

            proteus_hex8_create_full_idx(level,
                                         (mesh_t *)macro_mesh_->impl_mesh(),
                                         elements_,
                                         &n_unique_nodes_,
                                         &interior_start_);
        }

        ptrdiff_t n_nodes() const { return n_unique_nodes_; }
        int level() const { return level_; }
        ptrdiff_t n_elements() const { return macro_mesh_->n_elements(); }

        ~SemiStructuredMesh() {
            const int nxe = proteus_hex8_nxe(level_);

            for (int d = 0; d < nxe; d++) {
                free(elements_[d]);
            }

            free(elements_);
        }

        static std::shared_ptr<SemiStructuredMesh> create(const std::shared_ptr<Mesh> macro_mesh,
                                                          const int level) {
            return std::make_shared<SemiStructuredMesh>(macro_mesh, level);
        }
    };
}  // namespace sfem

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

    const char *output_path = argv[2];

    int SFEM_ELEMENT_REFINE_LEVEL = 0;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    if (!SFEM_ELEMENT_REFINE_LEVEL) {
        fprintf(stderr, "[Error] SFEM_ELEMENT_REFINE_LEVEL must be defined >= 2\n");
        return EXIT_FAILURE;
    }

    double tick = MPI_Wtime();

    // -------------------------------
    // Read inputs
    // -------------------------------

    const char *folder = argv[1];
    auto m = sfem::Mesh::create_from_file(comm, folder);
    // int block_size = m->spatial_dimension(); // Linear elasticity
    int block_size = 1;  // Laplace operator

    auto ssm = sfem::SemiStructuredMesh::create(m, SFEM_ELEMENT_REFINE_LEVEL);
    ptrdiff_t ndofs = ssm->n_nodes() * block_size;

    char *SFEM_DIRICHLET_NODESET = 0;
    char *SFEM_DIRICHLET_VALUE = 0;
    char *SFEM_DIRICHLET_COMPONENT = 0;
    SFEM_READ_ENV(SFEM_DIRICHLET_NODESET, );
    SFEM_READ_ENV(SFEM_DIRICHLET_VALUE, );
    SFEM_READ_ENV(SFEM_DIRICHLET_COMPONENT, );

    auto mesh = (mesh_t *)m->impl_mesh();

    int n_dirichlet_conditions{0};
    boundary_condition_t *dirichlet_conditions{nullptr};

    read_dirichlet_conditions(mesh,
                              SFEM_DIRICHLET_NODESET,
                              SFEM_DIRICHLET_VALUE,
                              SFEM_DIRICHLET_COMPONENT,
                              &dirichlet_conditions,
                              &n_dirichlet_conditions);

    printf("n_dirichlet_conditions = %d\n", n_dirichlet_conditions);

    auto op = sfem::make_op<real_t>(
            ndofs,
            ndofs,
            [=](const real_t *const x, real_t *const y) {
                // Apply operator
                proteus_affine_hex8_laplacian_apply //
                // proteus_hex8_laplacian_apply  //
                        (ssm->level(),
                         ssm->n_elements(),
                         ssm->interior_start(),
                         ssm->element_data(),
                         ssm->point_data(),
                         x,
                         y);
                        
                // Copy constrained nodes
                copy_at_dirichlet_nodes_vec(
                        n_dirichlet_conditions, dirichlet_conditions, block_size, x, y);
            },
            sfem::EXECUTION_SPACE_HOST);

    std::shared_ptr<sfem::MatrixFreeLinearSolver<real_t>> solver;
    {
        auto mprgp = std::make_shared<sfem::MPRGP<real_t>>();
        auto cg = sfem::h_cg<real_t>();
        cg->verbose = true;
        cg->set_op(op);
        cg->set_max_it(10000);
        cg->default_init();
        cg->set_rtol(1e-7);
        cg->set_atol(1e-8);
        solver = cg;
    }

    auto x = sfem::create_buffer<real_t>(ndofs, sfem::MEMORY_SPACE_HOST);
    auto rhs = sfem::create_buffer<real_t>(ndofs, sfem::MEMORY_SPACE_HOST);

    apply_dirichlet_condition_vec(
            n_dirichlet_conditions, dirichlet_conditions, block_size, x->data());

    apply_dirichlet_condition_vec(
            n_dirichlet_conditions, dirichlet_conditions, block_size, rhs->data());

    double solve_tick = MPI_Wtime();
    solver->apply(rhs->data(), x->data());
    double solve_tock = MPI_Wtime();

    struct stat st = {0};
    if (stat(output_path, &st) == -1) {
        mkdir(output_path, 0700);
    }

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


    if (!rank) {
        printf("----------------------------------------\n");
        printf("%s (%s):\n", argv[0], "PROTEUS_HEX8");
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #dofs %ld\n",
               (long)ssm->n_elements(),
               (long)ssm->n_nodes(),
               (long)ndofs);
        printf("TTS:\t\t%g [s], solve: %g [s])\n",
               tock - tick,
               solve_tock - solve_tick);
        printf("----------------------------------------\n");
    }

    return MPI_Finalize();
}
