#include <memory>

#include "sfem_test.hpp"

#include "sfem_Function.hpp"

#include "sfem_CRS.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"


#include "sfem_API.hpp"

#include <vector>

#define OP_HEADERS()                                                            \
    do {                                                                        \
        printf("Op,\t\tTTS [s],\tRTP [MDOF/s],\tBW [MDOF/s],\t(rows, cols)\n"); \
    } while (0)

#define OP_TIME(op, x, y)                                      \
    do {                                                       \
        if (SFEM_REPEAT > 1) {                                 \
            op->apply(x, y);                                   \
        }                                                      \
        sfem::device_synchronize();                            \
        double start = smesh::time_seconds();                            \
        for (int r = 0; r < SFEM_REPEAT; r++) {                \
            op->apply(x, y);                                   \
            sfem::device_synchronize();                        \
        }                                                      \
        double stop    = smesh::time_seconds();                          \
        double elapsed = (stop - start) / SFEM_REPEAT;         \
        printf("%s,\t%.5f,\t%.1f,\t\t%.1f,\t\t(%ld, %ld)\n",   \
               #op,                                            \
               elapsed,                                        \
               1e-6 * (op)->rows() / elapsed,                  \
               1e-6 * ((op)->rows() + (op)->cols()) / elapsed, \
               (op)->rows(),                                   \
               (op)->cols());                                  \
        fflush(stdout);                                        \
    } while (0)

template <typename Pred>
sfem::SharedBuffer<idx_t> nodeset_from_selector(const std::shared_ptr<smesh::Mesh> &mesh, Pred pred) {
    auto      pts = mesh->points()->data();
    ptrdiff_t n   = 0;
    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        if (pred(pts[0][i], pts[1][i], pts[2][i])) {
            ++n;
        }
    }

    auto ret = sfem::create_host_buffer<idx_t>(n);
    n        = 0;
    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        if (pred(pts[0][i], pts[1][i], pts[2][i])) {
            ret->data()[n++] = static_cast<idx_t>(i);
        }
    }

    return ret;
}

static int test_expanded_tet4_laplacian(const std::shared_ptr<sfem::Mesh> &ss_mesh,
                                        const char *const                  op_name,
                                        const sfem::ExecutionSpace         es,
                                        const int                          block_size) {
    SFEM_TEST_ASSERT(es == sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(smesh::is_tet_ss_family(ss_mesh->element_type(0)));

    auto tet4_mesh = smesh::convert_to(smesh::TET4, ss_mesh);
    SFEM_TEST_ASSERT(tet4_mesh != nullptr);
    SFEM_TEST_ASSERT(tet4_mesh->n_nodes() == ss_mesh->n_nodes());

    auto ss_fs   = sfem::FunctionSpace::create(ss_mesh, block_size);
    auto tet4_fs = sfem::FunctionSpace::create(tet4_mesh, block_size);

    auto ss_fun   = sfem::Function::create(ss_fs);
    auto tet4_fun = sfem::Function::create(tet4_fs);

    auto ss_op   = sfem::create_op(ss_fs, op_name, es);
    auto tet4_op = sfem::create_op(tet4_fs, op_name, es);
    SFEM_TEST_ASSERT(ss_op != nullptr);
    SFEM_TEST_ASSERT(tet4_op != nullptr);
    SFEM_TEST_ASSERT(ss_op->initialize() == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(tet4_op->initialize() == SFEM_SUCCESS);
    ss_fun->add_operator(ss_op);
    tet4_fun->add_operator(tet4_op);

    auto ss_linear   = sfem::create_linear_operator(sfem::op_type::MATRIX_FREE, ss_fun, nullptr, es);
    auto tet4_linear = sfem::create_linear_operator(sfem::op_type::MATRIX_FREE, tet4_fun, nullptr, es);

    const ptrdiff_t n_dofs = ss_fs->n_dofs();
    SFEM_TEST_ASSERT(tet4_fs->n_dofs() == n_dofs);

    auto input     = sfem::create_buffer<real_t>(n_dofs, es);
    auto ss_out    = sfem::create_buffer<real_t>(n_dofs, es);
    auto tet4_out  = sfem::create_buffer<real_t>(n_dofs, es);
    auto ss_diag   = sfem::create_buffer<real_t>(n_dofs, es);
    auto tet4_diag = sfem::create_buffer<real_t>(n_dofs, es);

    auto points = ss_mesh->points()->data();
    for (ptrdiff_t i = 0; i < ss_mesh->n_nodes(); ++i) {
        const real_t x = points[0][i];
        const real_t y = points[1][i];
        const real_t z = points[2][i];
        for (int b = 0; b < block_size; ++b) {
            input->data()[i * block_size + b] =
                    (b + 1) * (0.37 + x * x + 0.5 * y + 0.25 * z * z + 0.125 * x * y);
        }
    }

    SFEM_TEST_ASSERT(ss_linear->apply(input->data(), ss_out->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(tet4_linear->apply(input->data(), tet4_out->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(ss_fun->hessian_diag(nullptr, ss_diag->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(tet4_fun->hessian_diag(nullptr, tet4_diag->data()) == SFEM_SUCCESS);

    real_t    apply_largest_diff = 0;
    real_t    diag_largest_diff  = 0;
    ptrdiff_t apply_arg          = SFEM_PTRDIFF_INVALID;
    ptrdiff_t diag_arg           = SFEM_PTRDIFF_INVALID;

    for (ptrdiff_t i = 0; i < n_dofs; ++i) {
        const real_t apply_diff = fabs(ss_out->data()[i] - tet4_out->data()[i]);
        const real_t diag_diff  = fabs(ss_diag->data()[i] - tet4_diag->data()[i]);

        if (apply_diff > apply_largest_diff || apply_diff != apply_diff) {
            apply_largest_diff = apply_diff;
            apply_arg          = i;
        }

        if (diag_diff > diag_largest_diff || diag_diff != diag_diff) {
            diag_largest_diff = diag_diff;
            diag_arg          = i;
        }
    }

    printf("expanded TET4 check apply_largest_diff(%ld) = %g diag_largest_diff(%ld) = %g\n",
           apply_arg,
           (double)apply_largest_diff,
           diag_arg,
           (double)diag_largest_diff);
    SFEM_TEST_ASSERT(apply_largest_diff < 1e-7);
    SFEM_TEST_ASSERT(diag_largest_diff < 1e-8);

    return SFEM_TEST_SUCCESS;
}

static int test_hierarchical_transfer_scaling(
        const std::shared_ptr<sfem::FunctionSpace>    &fine_space,
        const std::shared_ptr<sfem::FunctionSpace>    &coarse_space,
        const std::shared_ptr<sfem::Operator<real_t>> &restriction,
        const std::shared_ptr<sfem::Operator<real_t>> &prolongation,
        const sfem::ExecutionSpace                     es) {
    SFEM_TEST_ASSERT(es == sfem::EXECUTION_SPACE_HOST);

    const ptrdiff_t fine_n_dofs   = fine_space->n_dofs();
    const ptrdiff_t coarse_n_dofs = coarse_space->n_dofs();
    const ptrdiff_t fine_n_nodes  = fine_space->mesh_ptr()->n_nodes();
    const ptrdiff_t coarse_n_nodes = coarse_space->mesh_ptr()->n_nodes();
    const int       block_size    = fine_space->block_size();

    SFEM_TEST_ASSERT(block_size == coarse_space->block_size());
    SFEM_TEST_ASSERT(fine_n_dofs == fine_n_nodes * block_size);
    SFEM_TEST_ASSERT(coarse_n_dofs == coarse_n_nodes * block_size);

    auto coarse = sfem::create_buffer<real_t>(coarse_n_dofs, es);
    auto fine   = sfem::create_buffer<real_t>(fine_n_dofs, es);

    for (ptrdiff_t i = 0; i < coarse_n_dofs; ++i) {
        coarse->data()[i] = 1;
    }

    for (ptrdiff_t i = 0; i < fine_n_dofs; ++i) {
        fine->data()[i] = 0;
    }

    SFEM_TEST_ASSERT(prolongation->apply(coarse->data(), fine->data()) == SFEM_SUCCESS);

    real_t    constant_largest_diff = 0;
    ptrdiff_t constant_arg          = SFEM_PTRDIFF_INVALID;
    for (ptrdiff_t i = 0; i < fine_n_dofs; ++i) {
        const real_t diff = fabs(fine->data()[i] - 1);
        if (diff > constant_largest_diff || diff != diff) {
            constant_largest_diff = diff;
            constant_arg          = i;
        }
    }

    geom_t **fine_points   = fine_space->mesh_ptr()->points()->data();
    geom_t **coarse_points = coarse_space->mesh_ptr()->points()->data();

    for (ptrdiff_t i = 0; i < coarse_n_nodes; ++i) {
        const real_t x = coarse_points[0][i];
        const real_t y = coarse_points[1][i];
        const real_t z = coarse_points[2][i];
        for (int b = 0; b < block_size; ++b) {
            coarse->data()[i * block_size + b] = (b + 1) * (0.25 + x + 0.5 * y - 0.125 * z);
        }
    }

    for (ptrdiff_t i = 0; i < fine_n_dofs; ++i) {
        fine->data()[i] = 0;
    }

    SFEM_TEST_ASSERT(prolongation->apply(coarse->data(), fine->data()) == SFEM_SUCCESS);

    real_t    affine_largest_diff = 0;
    ptrdiff_t affine_arg          = SFEM_PTRDIFF_INVALID;
    for (ptrdiff_t i = 0; i < fine_n_nodes; ++i) {
        const real_t x = fine_points[0][i];
        const real_t y = fine_points[1][i];
        const real_t z = fine_points[2][i];
        for (int b = 0; b < block_size; ++b) {
            const real_t expected = (b + 1) * (0.25 + x + 0.5 * y - 0.125 * z);
            const real_t diff     = fabs(fine->data()[i * block_size + b] - expected);
            if (diff > affine_largest_diff || diff != diff) {
                affine_largest_diff = diff;
                affine_arg          = i * block_size + b;
            }
        }
    }

    auto xf  = sfem::create_buffer<real_t>(fine_n_dofs, es);
    auto yc  = sfem::create_buffer<real_t>(coarse_n_dofs, es);
    auto rxf = sfem::create_buffer<real_t>(coarse_n_dofs, es);
    auto pyc = sfem::create_buffer<real_t>(fine_n_dofs, es);

    for (ptrdiff_t i = 0; i < fine_n_dofs; ++i) {
        xf->data()[i]  = 0.31 + real_t((i * 17) % 23) * 0.07 + real_t((i * 5) % 11) * 0.013;
        pyc->data()[i] = 0;
    }

    for (ptrdiff_t i = 0; i < coarse_n_dofs; ++i) {
        yc->data()[i]  = -0.21 + real_t((i * 13) % 19) * 0.05 + real_t((i * 7) % 5) * 0.017;
        rxf->data()[i] = 0;
    }

    SFEM_TEST_ASSERT(restriction->apply(xf->data(), rxf->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(prolongation->apply(yc->data(), pyc->data()) == SFEM_SUCCESS);

    real_t lhs = 0;
    real_t rhs = 0;
    for (ptrdiff_t i = 0; i < coarse_n_dofs; ++i) {
        lhs += rxf->data()[i] * yc->data()[i];
    }

    for (ptrdiff_t i = 0; i < fine_n_dofs; ++i) {
        rhs += xf->data()[i] * pyc->data()[i];
    }

    const real_t adjoint_abs_diff = fabs(lhs - rhs);
    const real_t adjoint_rel_diff = adjoint_abs_diff / (fabs(lhs) + fabs(rhs) + 1);

    printf("transfer scaling check constant_largest_diff(%ld) = %g affine_largest_diff(%ld) = %g "
           "adjoint_abs_diff = %g adjoint_rel_diff = %g\n",
           constant_arg,
           (double)constant_largest_diff,
           affine_arg,
           (double)affine_largest_diff,
           (double)adjoint_abs_diff,
           (double)adjoint_rel_diff);

    SFEM_TEST_ASSERT(constant_largest_diff < 1e-12);
    SFEM_TEST_ASSERT(affine_largest_diff < 1e-12);
    SFEM_TEST_ASSERT(adjoint_abs_diff < 1e-10 || adjoint_rel_diff < 1e-12);

    return SFEM_TEST_SUCCESS;
}

int test_cube() {
    auto comm = sfem::Communicator::world();
    auto es   = sfem::EXECUTION_SPACE_HOST;

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );

    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    const char *SFEM_OPERATOR = "Laplacian";
    SFEM_READ_ENV(SFEM_OPERATOR, );

    const char *SFEM_FINE_OP_TYPE = sfem::op_type::MATRIX_FREE;
    SFEM_READ_ENV(SFEM_FINE_OP_TYPE, );

    const char *SFEM_COARSE_OP_TYPE = sfem::op_type::MATRIX_FREE;
    SFEM_READ_ENV(SFEM_COARSE_OP_TYPE, );

    int SFEM_ELEMENT_REFINE_LEVEL = 4;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    int SFEM_BASE_RESOLUTION = 1;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    int SFEM_ELEMENT_DEREFINE = 1;
    SFEM_READ_ENV(SFEM_ELEMENT_DEREFINE, atoi);

    int SFEM_DEBUG_EXPORT = 0;
    SFEM_READ_ENV(SFEM_DEBUG_EXPORT, atoi);

    int SFEM_DEBUG_PRINT = 0;
    SFEM_READ_ENV(SFEM_DEBUG_PRINT, atoi);

    int SFEM_FULL_GALERKIN_CHECK = 0;
    SFEM_READ_ENV(SFEM_FULL_GALERKIN_CHECK, atoi);

    int SFEM_EXPANDED_TET_CHECK = 0;
    SFEM_READ_ENV(SFEM_EXPANDED_TET_CHECK, atoi);

    int SFEM_TRANSFER_SCALING_CHECK = 0;
    SFEM_READ_ENV(SFEM_TRANSFER_SCALING_CHECK, atoi);

    int SFEM_APPLY_TEST_CONSTRAINTS = 0;
    SFEM_READ_ENV(SFEM_APPLY_TEST_CONSTRAINTS, atoi);

    int SFEM_REPEAT = 1;
    SFEM_READ_ENV(SFEM_REPEAT, atoi);

    int SFEM_HIERARCHICAL_RENUMBERING = 1;
    SFEM_READ_ENV(SFEM_HIERARCHICAL_RENUMBERING, atoi);

    int SFEM_BLOCK_SIZE = 1;
    SFEM_READ_ENV(SFEM_BLOCK_SIZE, atoi);

    const char *SFEM_BASE_ELEMENT = "HEX8";
    SFEM_READ_ENV(SFEM_BASE_ELEMENT, );

    std::shared_ptr<sfem::Mesh> m;
    if (!strcmp(SFEM_BASE_ELEMENT, "TET4")) {
        m = sfem::Mesh::create_tet4_cube(comm,
                                         SFEM_BASE_RESOLUTION,
                                         SFEM_BASE_RESOLUTION,
                                         SFEM_BASE_RESOLUTION,
                                         0,
                                         0,
                                         0,
                                         1,
                                         1,
                                         1);
    } else {
        m = sfem::Mesh::create_hex8_cube(comm,
                                         SFEM_BASE_RESOLUTION,
                                         SFEM_BASE_RESOLUTION,
                                         SFEM_BASE_RESOLUTION,
                                         0,
                                         0,
                                         0,
                                         1,
                                         1,
                                         1);
    }

    if (SFEM_ELEMENT_REFINE_LEVEL > 0) {
        m = smesh::to_semistructured(SFEM_ELEMENT_REFINE_LEVEL, m, true, false);
    }

    auto fs = sfem::FunctionSpace::create(m, SFEM_BLOCK_SIZE);

    // if (es == sfem::EXECUTION_SPACE_DEVICE) {
    //     auto elements = fs->device_elements();
    //     if (!elements) {
    //         elements = create_device_elements(fs, fs->element_type());
    //         fs->set_device_elements(elements);
    //     }
    // }

    auto f  = sfem::Function::create(fs);
    auto x  = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto op = sfem::create_op(fs, SFEM_OPERATOR, es);

    op->initialize();
    f->add_operator(op);

    if (SFEM_EXPANDED_TET_CHECK) {
        SFEM_TEST_ASSERT(!strcmp(SFEM_BASE_ELEMENT, "TET4"));
        SFEM_TEST_ASSERT(test_expanded_tet4_laplacian(m, SFEM_OPERATOR, es, SFEM_BLOCK_SIZE) == SFEM_TEST_SUCCESS);
    }

    if (SFEM_APPLY_TEST_CONSTRAINTS) {
        auto bottom_ns = nodeset_from_selector(
                m, [](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool { return y > -1e-5 && y < 1e-5; });
        auto right_ns = nodeset_from_selector(
                m, [](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool { return x > 1 - 1e-5 && x < 1 + 1e-5; });

        sfem::DirichletConditions::Condition bottom{.nodeset = bottom_ns, .value = -1, .component = 0};
        sfem::DirichletConditions::Condition right{.nodeset = right_ns, .value = 1, .component = 0};
        f->add_constraint(sfem::create_dirichlet_conditions(fs, {bottom, right}, sfem::EXECUTION_SPACE_HOST));
    }

    std::shared_ptr<sfem::Operator<real_t>> fine_op, coarse_op;

    printf("Fine op (%d,%s):\t%s\n", SFEM_ELEMENT_REFINE_LEVEL, type_to_string(fs->element_type()), SFEM_FINE_OP_TYPE);
    fine_op = sfem::create_linear_operator(SFEM_FINE_OP_TYPE, f, nullptr, es);

    auto levels    = smesh::derefinement_levels(fs->mesh());
    SFEM_TEST_ASSERT(SFEM_ELEMENT_DEREFINE >= 0);
    SFEM_TEST_ASSERT(SFEM_ELEMENT_DEREFINE < static_cast<int>(levels.size()));
    auto fs_coarse = fs->derefine(levels[SFEM_ELEMENT_DEREFINE]);
    auto f_coarse  = f->derefine(fs_coarse, true);

    printf("Coarse op (%d,%s):\t%s\n",
           levels[SFEM_ELEMENT_DEREFINE],
           type_to_string(fs_coarse->element_type()),
           SFEM_COARSE_OP_TYPE);
    coarse_op = sfem::create_linear_operator(SFEM_COARSE_OP_TYPE, f_coarse, nullptr, es);

    auto restriction_unconstr = sfem::create_hierarchical_restriction(fs, fs_coarse, es);
    auto restriction          = sfem::make_op<real_t>(
            restriction_unconstr->rows(),
            restriction_unconstr->cols(),
            [=](const real_t *const from, real_t *const to) {
                restriction_unconstr->apply(from, to);
                f_coarse->apply_zero_constraints(to);
            },
            es);
    auto prolong_unconstr = sfem::create_hierarchical_prolongation(fs_coarse, fs, es);
    auto prolongation     = sfem::make_op<real_t>(
            prolong_unconstr->rows(),
            prolong_unconstr->cols(),
            [=](const real_t *const from, real_t *const to) {
                prolong_unconstr->apply(from, to);
                f->apply_zero_constraints(to);
            },
            es);

    if (SFEM_TRANSFER_SCALING_CHECK) {
        SFEM_TEST_ASSERT(test_hierarchical_transfer_scaling(fs, fs_coarse, restriction_unconstr, prolong_unconstr, es) ==
                         SFEM_TEST_SUCCESS);
    }

    auto h_input = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), sfem::MEMORY_SPACE_HOST);

    {
        geom_t **points{nullptr};
        if (fs_coarse->has_semi_structured_mesh()) {
            points = fs_coarse->mesh().points()->data();
        } else {
            points = fs_coarse->mesh_ptr()->points()->data();
        }

        ptrdiff_t n    = fs_coarse->mesh_ptr()->n_nodes();
        auto      data = h_input->data();
        for (ptrdiff_t i = 0; i < n; i++) {
            for (int b = 0; b < SFEM_BLOCK_SIZE; b++) {
                data[i * SFEM_BLOCK_SIZE + b] = (b + 1) * points[0][i] * points[0][i];
            }
        }
    }

    std::shared_ptr<sfem::Buffer<real_t>> input;

    if (es == sfem::EXECUTION_SPACE_DEVICE) {
        input = smesh::to_device(h_input);
    } else {
        input = h_input;
    }

    auto prolongated = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto Ax_fine     = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto restricted  = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), es);
    auto Ax_coarse   = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), es);

    double tick = smesh::time_seconds();

    OP_HEADERS();
    OP_TIME(coarse_op, input->data(), Ax_coarse->data());
    OP_TIME(prolongation, input->data(), prolongated->data());
    OP_TIME(fine_op, prolongated->data(), Ax_fine->data());
    OP_TIME(restriction, Ax_fine->data(), restricted->data());

    double tock = smesh::time_seconds();

    printf("#elements %ld #ndofs fine %ld coarse %ld\nTTS: %g [s]\n",
           m->n_elements(),
           fs->n_dofs(),
           fs_coarse->n_dofs(),
           tock - tick);

    if (SFEM_REPEAT == 1) {
        auto error = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), sfem::MEMORY_SPACE_HOST);

        // Compare two results

        auto h_restricted  = smesh::to_host(restricted);
        auto h_Ax_coarse   = smesh::to_host(Ax_coarse);
        auto h_prolongated = smesh::to_host(prolongated);

        {
            auto      err      = error->data();
            ptrdiff_t n        = fs_coarse->n_dofs();
            auto      actual   = h_restricted->data();
            auto      expected = h_Ax_coarse->data();

            real_t    largest_diff        = 0;
            real_t    largest_diff_factor = 0;
            ptrdiff_t arg_largest_diff    = SFEM_PTRDIFF_INVALID;
            for (ptrdiff_t i = 0; i < n; i++) {
                // actual: is composition of operators
                // expected: is application of coarse operator
                real_t diff = fabs(actual[i] - expected[i]);
                err[i]      = diff;
                if (diff > 1e-8 || diff != diff) {
                    printf("%ld) %g != %g (%g, %g)\n",
                           i,
                           (double)actual[i],
                           (double)expected[i],
                           (double)diff,
                           (double)actual[i] / expected[i]);
                }

                if (diff > largest_diff) {
                    largest_diff        = diff;
                    arg_largest_diff    = i;
                    largest_diff_factor = actual[i] / expected[i];
                }
            }

            if (SFEM_DEBUG_PRINT) {
                std::cout << "--------------\n";
                std::cout << "Prolongated\n";
                std::cout << "--------------\n";

                h_prolongated->print(std::cout);

                std::cout << "--------------\n";
                std::cout << "Actual\n";
                std::cout << "--------------\n";
                h_restricted->print(std::cout);

                std::cout << "--------------\n";
                std::cout << "Expected\n";
                std::cout << "--------------\n";
                h_Ax_coarse->print(std::cout);

                std::cout << "--------------\n";
            }

            if (SFEM_DEBUG_EXPORT) {
                smesh::create_directory(smesh::Path("galerkin"));
                smesh::create_directory(smesh::Path("galerkin/fields"));

                {  // COARSE
                    SFEM_TEST_ASSERT(smesh::semistructured_export_as_standard(fs_coarse->mesh_ptr(), smesh::Path("galerkin")) ==
                                     SFEM_SUCCESS);

                    sfem::Output out(fs_coarse);
                    out.enable_AoS_to_SoA(SFEM_BLOCK_SIZE > 1);

                    out.set_output_dir(smesh::Path("galerkin/fields"));
                    SFEM_TEST_ASSERT(out.write("R", h_restricted->data()) == SFEM_SUCCESS);
                    SFEM_TEST_ASSERT(out.write("u", h_input->data()) == SFEM_SUCCESS);
                    SFEM_TEST_ASSERT(out.write("Ax_coarse", h_Ax_coarse->data()) == SFEM_SUCCESS);
                    SFEM_TEST_ASSERT(out.write("err", error->data()) == SFEM_SUCCESS);
                }

                {  // FINE
                    smesh::create_directory("galerkin_fine");
                    smesh::create_directory("galerkin_fine/fields");
                    SFEM_TEST_ASSERT(smesh::semistructured_export_as_standard(fs->mesh_ptr(), smesh::Path("galerkin_fine")) ==
                                     SFEM_SUCCESS);

                    sfem::Output out(fs);
                    out.set_output_dir(smesh::Path("galerkin_fine/fields"));

                    SFEM_TEST_ASSERT(out.write("P", h_prolongated->data()) == SFEM_SUCCESS);
                }
            }

            if (arg_largest_diff != -1) {
                fflush(stdout);
                printf("largest_diff(%ld) = %g, %g\n", arg_largest_diff, largest_diff, largest_diff_factor);
                SFEM_TEST_ASSERT(largest_diff < 1e-7);
            }
        }

        if (SFEM_FULL_GALERKIN_CHECK) {
            SFEM_TEST_ASSERT(es == sfem::EXECUTION_SPACE_HOST);

            const ptrdiff_t n_coarse = fs_coarse->n_dofs();
            auto            basis    = sfem::create_buffer<real_t>(n_coarse, es);
            auto            lhs      = sfem::create_buffer<real_t>(n_coarse, es);
            auto            rhs      = sfem::create_buffer<real_t>(n_coarse, es);

            real_t    largest_diff     = 0;
            ptrdiff_t arg_largest_row  = SFEM_PTRDIFF_INVALID;
            ptrdiff_t arg_largest_col  = SFEM_PTRDIFF_INVALID;

            for (ptrdiff_t j = 0; j < n_coarse; ++j) {
                for (ptrdiff_t i = 0; i < n_coarse; ++i) {
                    basis->data()[i] = 0;
                    lhs->data()[i]   = 0;
                    rhs->data()[i]   = 0;
                }
                for (ptrdiff_t i = 0; i < fs->n_dofs(); ++i) {
                    prolongated->data()[i] = 0;
                    Ax_fine->data()[i]     = 0;
                }

                basis->data()[j] = 1;
                SFEM_TEST_ASSERT(coarse_op->apply(basis->data(), rhs->data()) == SFEM_SUCCESS);
                SFEM_TEST_ASSERT(prolongation->apply(basis->data(), prolongated->data()) == SFEM_SUCCESS);
                SFEM_TEST_ASSERT(fine_op->apply(prolongated->data(), Ax_fine->data()) == SFEM_SUCCESS);
                SFEM_TEST_ASSERT(restriction->apply(Ax_fine->data(), lhs->data()) == SFEM_SUCCESS);

                for (ptrdiff_t i = 0; i < n_coarse; ++i) {
                    const real_t diff = fabs(lhs->data()[i] - rhs->data()[i]);
                    if (diff > largest_diff || diff != diff) {
                        largest_diff    = diff;
                        arg_largest_row = i;
                        arg_largest_col = j;
                    }
                }
            }

            printf("full Galerkin largest_diff(%ld,%ld) = %g\n",
                   arg_largest_row,
                   arg_largest_col,
                   (double)largest_diff);
            SFEM_TEST_ASSERT(largest_diff < 1e-7);
        }
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_cube);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
