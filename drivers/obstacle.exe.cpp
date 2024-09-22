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

    int SFEM_USE_ELASTICITY = 0;
    SFEM_READ_ENV(SFEM_USE_ELASTICITY, atoi);
    int block_size = SFEM_USE_ELASTICITY ? m->spatial_dimension() : 1;

    auto ssm = sfem::SemiStructuredMesh::create(m, SFEM_ELEMENT_REFINE_LEVEL);
    ptrdiff_t ndofs = ssm->n_nodes() * block_size;

    char *SFEM_DIRICHLET_NODESET = 0;
    char *SFEM_DIRICHLET_VALUE = 0;
    char *SFEM_DIRICHLET_COMPONENT = 0;
    SFEM_READ_ENV(SFEM_DIRICHLET_NODESET, );
    SFEM_READ_ENV(SFEM_DIRICHLET_VALUE, );
    SFEM_READ_ENV(SFEM_DIRICHLET_COMPONENT, );

    char *SFEM_CONTACT_NODESET = 0;
    char *SFEM_CONTACT_VALUE = 0;
    char *SFEM_CONTACT_COMPONENT = 0;

    SFEM_READ_ENV(SFEM_CONTACT_NODESET, );
    SFEM_READ_ENV(SFEM_CONTACT_VALUE, );
    SFEM_READ_ENV(SFEM_CONTACT_COMPONENT, );

    int SFEM_USE_PROJECTED_CG = 0;
    SFEM_READ_ENV(SFEM_USE_PROJECTED_CG, atoi);

    real_t SFEM_SHEAR_MODULUS = 1;
    real_t SFEM_FIRST_LAME_PARAMETER = 1;

    auto mesh = (mesh_t *)m->impl_mesh();

    int n_dirichlet_conditions{0};
    boundary_condition_t *dirichlet_conditions{nullptr};

    read_dirichlet_conditions(mesh,
                              SFEM_DIRICHLET_NODESET,
                              SFEM_DIRICHLET_VALUE,
                              SFEM_DIRICHLET_COMPONENT,
                              &dirichlet_conditions,
                              &n_dirichlet_conditions);

    int n_contact_conditions{0};
    boundary_condition_t *contact_conditions{nullptr};
    read_dirichlet_conditions(mesh,
                              SFEM_CONTACT_NODESET,
                              SFEM_CONTACT_VALUE,
                              SFEM_CONTACT_COMPONENT,
                              &contact_conditions,
                              &n_contact_conditions);

    printf("n_dirichlet_conditions = %d\n", n_dirichlet_conditions);
    printf("n_contact_conditions = %d\n", n_contact_conditions);

    //     auto inv_diag = sfem::create_buffer<real_t>(ndofs, sfem::MEMORY_SPACE_HOST);
    //     {  // Create diagonal operator

    //         if (SFEM_USE_ELASTICITY) {
    //             assert(false);
    //         } else {
    //             proteus_affine_hex8_laplacian_diag(ssm->level(),
    //                                                ssm->n_elements(),
    //                                                ssm->interior_start(),
    //                                                ssm->element_data(),
    //                                                ssm->point_data(),
    //                                                inv_diag->data());
    //         }

    //         {
    //             auto d = inv_diag->data();
    // #pragma omp parallel for
    //             for (ptrdiff_t i = 0; i < ndofs; i++) {
    //                 d[i] = 1 / d[i];
    //             }

    //             // Identity at eq-contraint nodes
    //             for (int i = 0; i < n_dirichlet_conditions; i++) {
    //                 constraint_nodes_to_value_vec(dirichlet_conditions[i].local_size,
    //                                               dirichlet_conditions[i].idx,
    //                                               block_size,
    //                                               dirichlet_conditions[i].component,
    //                                               1,
    //                                               d);
    //             }
    //         }
    // }

    auto op = sfem::make_op<real_t>(
            ndofs,
            ndofs,
            [=](const real_t *const x, real_t *const y) {
                // Apply operator
                if (SFEM_USE_ELASTICITY) {
                    // proteus_affine_hex8_linear_elasticity_apply //
                    proteus_hex8_linear_elasticity_apply  //
                            (ssm->level(),
                             ssm->n_elements(),
                             ssm->interior_start(),
                             ssm->element_data(),
                             ssm->point_data(),
                             SFEM_SHEAR_MODULUS,
                             SFEM_FIRST_LAME_PARAMETER,
                             3,
                             &x[0],
                             &x[1],
                             &x[2],
                             3,
                             &y[0],
                             &y[1],
                             &y[2]);
                } else {
                    proteus_affine_hex8_laplacian_apply  //
                                                         // proteus_hex8_laplacian_apply  //
                            (ssm->level(),
                             ssm->n_elements(),
                             ssm->interior_start(),
                             ssm->element_data(),
                             ssm->point_data(),
                             x,
                             y);
                }

                // Copy constrained nodes
                copy_at_dirichlet_nodes_vec(
                        n_dirichlet_conditions, dirichlet_conditions, block_size, x, y);
            },
            sfem::EXECUTION_SPACE_HOST);

    auto x = sfem::create_buffer<real_t>(ndofs, sfem::MEMORY_SPACE_HOST);
    auto rhs = sfem::create_buffer<real_t>(ndofs, sfem::MEMORY_SPACE_HOST);

    apply_dirichlet_condition_vec(
            n_dirichlet_conditions, dirichlet_conditions, block_size, x->data());

    apply_dirichlet_condition_vec(
            n_dirichlet_conditions, dirichlet_conditions, block_size, rhs->data());

    std::shared_ptr<sfem::MatrixFreeLinearSolver<real_t>> solver;

    if (n_contact_conditions) {
        auto mprgp = std::make_shared<sfem::MPRGP<real_t>>();

        if (SFEM_USE_PROJECTED_CG) {
            mprgp->set_expansion_type(sfem::MPRGP<real_t>::EXPANSION_TYPE_PROJECTED_CG);
        }

        mprgp->set_op(op);

        auto upper_bound = sfem::create_buffer<real_t>(ndofs, sfem::MEMORY_SPACE_HOST);

        {  // Fill default upper-bound value
            auto ub = upper_bound->data();
            for (ptrdiff_t i = 0; i < ndofs; i++) {
                ub[i] = 1000;
            }
        }

        apply_dirichlet_condition_vec(
                n_contact_conditions, contact_conditions, block_size, upper_bound->data());

        char path[2048];
        sprintf(path, "%s/upper_bound.raw", output_path);
        if (array_write(comm, path, SFEM_MPI_REAL_T, (void *)upper_bound->data(), ndofs, ndofs)) {
            return SFEM_FAILURE;
        }

        mprgp->verbose = true;
        mprgp->set_max_it(10000);
        mprgp->set_rtol(1e-12);
        mprgp->set_atol(1e-8);
        mprgp->set_upper_bound(upper_bound);
        // mprgp->set_max_eig(1);
        mprgp->default_init();
        solver = mprgp;
    } else {
        //         auto preconditioner = sfem::make_op<real_t>(
        //                 ndofs,
        //                 ndofs,
        //                 [=](const real_t *const x, real_t *const y) {
        //                     auto d = inv_diag->data();

        // #pragma omp parallel for
        //                     for (ptrdiff_t i = 0; i < ndofs; i++) {
        //                         y[i] = d[i] * x[i];
        //                     }
        //                 },
        //                 sfem::EXECUTION_SPACE_HOST);

        auto cg = sfem::h_cg<real_t>();
        cg->verbose = true;
        cg->set_op(op);
        // cg->set_preconditioner_op(preconditioner); // CHECK if diag code is correct!
        cg->set_max_it(10000);
        cg->set_rtol(1e-12);
        cg->set_atol(1e-8);
        cg->default_init();
        solver = cg;
    }

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

    {  // Residual test
        if (n_contact_conditions) {
            apply_dirichlet_condition_vec(
                    n_contact_conditions, contact_conditions, block_size, rhs->data());
        }

        auto linear_op = sfem::make_op<real_t>(
                ndofs,
                ndofs,
                [=](const real_t *const x, real_t *const y) {
                    // Apply operator

                    if (SFEM_USE_ELASTICITY) {
                        // proteus_affine_hex8_linear_elasticity_apply //
                        proteus_hex8_linear_elasticity_apply  //
                                (ssm->level(),
                                 ssm->n_elements(),
                                 ssm->interior_start(),
                                 ssm->element_data(),
                                 ssm->point_data(),
                                 SFEM_SHEAR_MODULUS,
                                 SFEM_FIRST_LAME_PARAMETER,
                                 3,
                                 &x[0],
                                 &x[1],
                                 &x[2],
                                 3,
                                 &y[0],
                                 &y[1],
                                 &y[2]);
                    } else {
                        // proteus_affine_hex8_laplacian_apply  //
                        proteus_hex8_laplacian_apply  //
                                (ssm->level(),
                                 ssm->n_elements(),
                                 ssm->interior_start(),
                                 ssm->element_data(),
                                 ssm->point_data(),
                                 x,
                                 y);
                    }

                    // Copy constrained nodes
                    copy_at_dirichlet_nodes_vec(
                            n_dirichlet_conditions, dirichlet_conditions, block_size, x, y);

                    if (n_contact_conditions) {
                        copy_at_dirichlet_nodes_vec(
                                n_contact_conditions, contact_conditions, block_size, x, y);
                    }
                },
                sfem::EXECUTION_SPACE_HOST);

        auto r = sfem::create_buffer<real_t>(ndofs, sfem::MEMORY_SPACE_HOST);
        real_t rtr = residual(*linear_op, rhs->data(), x->data(), r->data());

        printf("Linear residual: %g\n", rtr);
    }

    destroy_conditions(n_contact_conditions, contact_conditions);
    destroy_conditions(n_dirichlet_conditions, dirichlet_conditions);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("%s (%s):\n", argv[0], "PROTEUS_HEX8");
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #dofs %ld\n",
               (long)ssm->n_elements(),
               (long)ssm->n_nodes(),
               (long)ndofs);
        printf("TTS:\t\t%g [s], solve: %g [s])\n", tock - tick, solve_tock - solve_tick);
        printf("----------------------------------------\n");
    }

    return MPI_Finalize();
}
