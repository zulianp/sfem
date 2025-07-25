#include <memory>

#include "sfem_test.h"

#include "cu_ssquad4_interpolate.h"
#include "matrixio_array.h"
#include "sfem_base.h"
#include "ssquad4_interpolate.h"

#include "sfem_API.hpp"
#include "sfem_Buffer.hpp"
#include "sfem_Function.hpp"
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_ShiftedPenaltyMultigrid.hpp"
#include "sfem_crs_SpMV.hpp"
#include "sfem_cuda_ShiftedPenalty_impl.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_solver.hpp"
#include "spmv.h"

using namespace sfem;

std::shared_ptr<sfem::ContactConditions> build_cuboid_sphere_contact(const std::shared_ptr<sfem::Function> &f,
                                                                     const int                              base_resolution) {
    auto fs   = f->space();
    auto m    = fs->mesh_ptr();
    auto comm = m->comm();
    auto es   = f->execution_space();

    auto top_ss = sfem::Sideset::create_from_selector(
            m, [=](const geom_t /*x*/, const geom_t y, const geom_t z) -> bool { return y > (1 - 1e-5) && y < (1 + 1e-5); });

    const int n = base_resolution * (fs->has_semi_structured_mesh() ? fs->semi_structured_mesh().level() : 1);

    sfem::DirichletConditions::Condition xtop{.sidesets = {top_ss}, .value = 0, .component = 0};
    sfem::DirichletConditions::Condition ytop{.sidesets = {top_ss}, .value = -0.05, .component = 1};
    sfem::DirichletConditions::Condition ztop{.sidesets = {top_ss}, .value = 0, .component = 2};

    auto conds = sfem::create_dirichlet_conditions(fs, {xtop, ytop, ztop}, es);
    f->add_constraint(conds);

    auto bottom_ss = sfem::Sideset::create_from_selector(
            m, [=](const geom_t /*x*/, const geom_t y, const geom_t z) -> bool { return y > -1e-5 && y < 1e-5; });

    auto sdf = sfem::create_sdf(comm,
                                n * 2,
                                n * 2,
                                n * 2,
                                -0.1,
                                -0.1,
                                -0.1,
                                1.1,
                                1.1,
                                1.1,
                                [](const geom_t x, const geom_t y, const geom_t z) -> geom_t {
                                    // Half-sphere
                                    geom_t cx = 0.5, cy = -0.5, cz = 0.5;
                                    geom_t radius = 0.5;

                                    geom_t dx = cx - x;
                                    geom_t dy = cy - y;
                                    geom_t dz = cz - z;

                                    geom_t dd = radius - sqrt(dx * dx + dy * dy + dz * dz);
                                    return dd;
                                });

    sdf->to_file("test_contact/sdf");
    auto contact_conds = sfem::ContactConditions::create(fs, sdf, bottom_ss, es);
    return contact_conds;
}

struct TestOutput {
    bool                            is_ml{false};
    std::shared_ptr<Buffer<real_t>> x;
    std::shared_ptr<Buffer<real_t>> rhs;
    std::shared_ptr<Buffer<real_t>> g;
    std::shared_ptr<Buffer<real_t>> diag;
    std::shared_ptr<Buffer<mask_t>> mask;
    std::shared_ptr<Buffer<real_t>> normal_prod;
    std::shared_ptr<Buffer<real_t>> cc_op_x;
    std::shared_ptr<Buffer<real_t>> cc_op_t_r;
    std::shared_ptr<Buffer<real_t>> rpen;
    std::shared_ptr<Buffer<real_t>> Jpen;
    std::shared_ptr<Buffer<real_t>> restricted;
    real_t                          e_pen;
    std::shared_ptr<Buffer<real_t>> lagr_ub;

    int compare(const struct TestOutput &other, const real_t tol = 1e-7) const {
        SFEM_ASSERT_ARRAY_APPROX_EQ(x->size(), x->data(), other.x->data(), tol);
        SFEM_ASSERT_ARRAY_APPROX_EQ(rhs->size(), rhs->data(), other.rhs->data(), tol);
        SFEM_ASSERT_ARRAY_APPROX_EQ(g->size(), g->data(), other.g->data(), tol);
        SFEM_ASSERT_ARRAY_APPROX_EQ(diag->size(), diag->data(), other.diag->data(), tol);
        SFEM_ASSERT_ARRAY_EQ(mask->size(), mask->data(), other.mask->data());
        SFEM_ASSERT_ARRAY_APPROX_EQ(normal_prod->size(), normal_prod->data(), other.normal_prod->data(), tol);
        SFEM_ASSERT_ARRAY_APPROX_EQ(cc_op_x->size(), cc_op_x->data(), other.cc_op_x->data(), tol);
        SFEM_ASSERT_ARRAY_APPROX_EQ(cc_op_t_r->size(), cc_op_t_r->data(), other.cc_op_t_r->data(), tol);
        SFEM_ASSERT_ARRAY_APPROX_EQ(rpen->size(), rpen->data(), other.rpen->data(), tol);
        SFEM_ASSERT_ARRAY_APPROX_EQ(Jpen->size(), Jpen->data(), other.Jpen->data(), tol);
        SFEM_TEST_APPROXEQ(e_pen, other.e_pen, tol);

        if (is_ml) {
            SFEM_ASSERT_ARRAY_APPROX_EQ(restricted->size(), restricted->data(), other.restricted->data(), tol);
        }

        return SFEM_TEST_SUCCESS;
    }

    struct TestOutput to_host() {
        TestOutput to{.x   = sfem::to_host(x),
                      .rhs = sfem::to_host(rhs),
                      .g   = sfem::to_host(g),
                      // .upper_bound = sfem::to_host(upper_bound),
                      .diag        = sfem::to_host(diag),
                      .mask        = sfem::to_host(mask),
                      .normal_prod = sfem::to_host(normal_prod),
                      .cc_op_x     = sfem::to_host(cc_op_x),
                      .cc_op_t_r   = sfem::to_host(cc_op_t_r),
                      .rpen        = sfem::to_host(rpen),
                      .Jpen        = sfem::to_host(Jpen),
                      .e_pen       = e_pen,
                      .lagr_ub     = sfem::to_host(lagr_ub)};

        if (is_ml) {
            to.restricted = sfem::to_host(restricted);
        }

        return to;
    }
};

struct TestOutput gen_test_data(enum ExecutionSpace es) {
    MPI_Comm comm = MPI_COMM_WORLD;

    int SFEM_BASE_RESOLUTION = 1;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    auto m = sfem::Mesh::create_hex8_cube(
            sfem::Communicator::wrap(comm), SFEM_BASE_RESOLUTION, SFEM_BASE_RESOLUTION, SFEM_BASE_RESOLUTION, 0, 0, 0, 1, 1, 1);

    const int block_size = m->spatial_dimension();

    auto fs = sfem::FunctionSpace::create(m, block_size);

    int SFEM_ELEMENT_REFINE_LEVEL = 2;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    if (SFEM_ELEMENT_REFINE_LEVEL > 1) {
        fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
        fs->semi_structured_mesh().apply_hierarchical_renumbering();
    }

    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, "LinearElasticity", es);
    op->initialize();

    f->add_operator(op);
    auto            contact_conds = build_cuboid_sphere_contact(f, SFEM_BASE_RESOLUTION);
    const ptrdiff_t ndofs         = fs->n_dofs();
    auto            x             = sfem::create_buffer<real_t>(ndofs, es);
    auto            rhs           = sfem::create_buffer<real_t>(ndofs, es);
    auto            g             = sfem::create_buffer<real_t>(ndofs, es);

    f->apply_constraints(rhs->data());
    f->apply_constraints(x->data());
    f->gradient(x->data(), g->data());

    const int sym_block_size = 6;
    auto      diag           = sfem::create_buffer<real_t>(fs->n_dofs() / fs->block_size() * sym_block_size, es);
    auto      mask           = sfem::create_buffer<mask_t>(mask_count(fs->n_dofs()), es);

    f->hessian_block_diag_sym(nullptr, diag->data());
    f->constaints_mask(mask->data());

    auto linear_op = sfem::create_linear_operator(MATRIX_FREE, f, nullptr, es);
    auto cg        = sfem::create_cg(linear_op, es);
    cg->apply(rhs->data(), x->data());

    contact_conds->init();

    auto upper_bound = sfem::create_buffer<real_t>(contact_conds->n_constrained_dofs(), es);
    auto normal_prod = sfem::create_buffer<real_t>(sym_block_size * contact_conds->n_constrained_dofs(), es);
    contact_conds->hessian_block_diag_sym(nullptr, normal_prod->data());

    auto cc_op   = contact_conds->linear_constraints_op();
    auto cc_op_t = contact_conds->linear_constraints_op_transpose();

    auto cc_op_x = sfem::create_buffer<real_t>(contact_conds->n_constrained_dofs(), es);
    cc_op->apply(x->data(), cc_op_x->data());

    auto cc_op_t_r = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    cc_op_t->apply(cc_op_x->data(), cc_op_t_r->data());

    ShiftedPenalty_Tpl<real_t> impl;
    if (EXECUTION_SPACE_DEVICE == es) {
        CUDA_ShiftedPenalty<real_t>::build(impl);
    } else {
        OpenMP_ShiftedPenalty<real_t>::build(impl);
    }

    auto lagrange_ub = sfem::create_buffer<real_t>(contact_conds->n_constrained_dofs(), es);
    auto rpen        = sfem::create_buffer<real_t>(contact_conds->n_constrained_dofs(), es);
    impl.calc_r_pen(contact_conds->n_constrained_dofs(),
                    x->data(),
                    10,
                    nullptr,
                    upper_bound->data(),
                    nullptr,
                    lagrange_ub->data(),
                    rpen->data());

    auto Jpen = sfem::create_buffer<real_t>(contact_conds->n_constrained_dofs(), es);
    impl.calc_J_pen(contact_conds->n_constrained_dofs(),
                    rhs->data(),
                    10,
                    nullptr,
                    upper_bound->data(),
                    nullptr,
                    lagrange_ub->data(),
                    Jpen->data());

    auto e_pen = impl.sq_norm_ramp_p(contact_conds->n_constrained_dofs(), x->data(), upper_bound->data());

    auto lagr_ub = sfem::create_buffer<real_t>(contact_conds->n_constrained_dofs(), es);
    impl.update_lagr_p(contact_conds->n_constrained_dofs(), 10, x->data(), upper_bound->data(), lagr_ub->data());

    const bool is_ml = fs->has_semi_structured_mesh();

    TestOutput to{.is_ml = is_ml,
                  .x     = x,
                  .rhs   = rhs,
                  .g     = g,
                  /*.upper_bound = upper_bound,*/
                  .diag        = diag,
                  .mask        = mask,
                  .normal_prod = normal_prod,
                  .cc_op_x     = cc_op_x,
                  .cc_op_t_r   = cc_op_t_r,
                  .rpen        = rpen,
                  .Jpen        = Jpen,
                  .e_pen       = e_pen,
                  .lagr_ub      = lagr_ub};

    if (is_ml) {
        int coarse_level = SFEM_ELEMENT_REFINE_LEVEL / 2;

        auto coarse_fs = fs->derefine(coarse_level);

        auto &ssmesh       = fs->semi_structured_mesh();
        auto  fine_sides   = contact_conds->ss_sides();
        auto  coarse_sides = sfem::ssquad4_derefine_element_connectivity(ssmesh.level(), coarse_level, to_host(fine_sides));

        auto fine_mapping = contact_conds->node_mapping();
        auto count        = sfem::create_host_buffer<uint16_t>(fine_mapping->size());
        ssquad4_element_node_incidence_count(
                ssmesh.level(), 1, fine_sides->extent(1), to_host(fine_sides)->data(), count->data());

        const ptrdiff_t n_coarse_contact_nodes = sfem::ss_elements_max_node_id(coarse_sides) + 1;

        auto input = sfem::create_host_buffer<real_t>(contact_conds->n_constrained_dofs());
        {
            auto in = input->data();
            for (ptrdiff_t i = 0; i < contact_conds->n_constrained_dofs(); i++) {
                in[i] = i;
            }
        }

        auto restricted = sfem::create_buffer<real_t>(n_coarse_contact_nodes, es);

        if (es == EXECUTION_SPACE_DEVICE) {
            fine_sides   = to_device(fine_sides);
            coarse_sides = to_device(coarse_sides);
            count        = to_device(count);

            cu_ssquad4_restrict(fine_sides->extent(1),
                                ssmesh.level(),
                                1,
                                fine_sides->data(),
                                count->data(),
                                coarse_level,
                                1,
                                coarse_sides->data(),
                                1,
                                SFEM_REAL_DEFAULT,
                                1,
                                input->data(),
                                SFEM_REAL_DEFAULT,
                                1,
                                restricted->data(),
                                SFEM_DEFAULT_STREAM);
        } else {
            ssquad4_restrict(fine_sides->extent(1),
                             ssmesh.level(),
                             1,
                             fine_sides->data(),
                             count->data(),
                             coarse_level,
                             1,
                             coarse_sides->data(),
                             1,
                             input->data(),
                             restricted->data());
        }

        to.restricted = restricted;
    }

    return to;
}

int test_obstacle() {
    auto host   = gen_test_data(EXECUTION_SPACE_HOST);
    auto device = gen_test_data(EXECUTION_SPACE_DEVICE).to_host();

    return host.compare(device);
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_obstacle);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
