#include "sfem_ssmgc.hpp"

#include "sfem_API.hpp"
#include "ssquad4_interpolate.h"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_ShiftedPenalty_impl.hpp"
#endif

namespace sfem {
    std::shared_ptr<ShiftedPenalty<real_t>> create_shifted_penalty(const std::shared_ptr<Function>         &f,
                                                                   const std::shared_ptr<ContactConditions> contact_conds,
                                                                   const enum ExecutionSpace                es,
                                                                   const std::shared_ptr<Input>            &in) {
        auto      fs          = f->space();
        const int block_size  = fs->block_size();
        auto      cc_op       = contact_conds->linear_constraints_op();
        auto      cc_op_t     = contact_conds->linear_constraints_op_transpose();
        auto      upper_bound = sfem::create_buffer<real_t>(contact_conds->n_constrained_dofs(), es);
        contact_conds->signed_distance(upper_bound->data());

        int  sym_block_size = (block_size == 3 ? 6 : 3);
        auto normal_prod    = sfem::create_buffer<real_t>(sym_block_size * contact_conds->n_constrained_dofs(), es);
        contact_conds->hessian_block_diag_sym(nullptr, normal_prod->data());
        auto sbv = sfem::create_sparse_block_vector(contact_conds->node_mapping(), normal_prod);

        auto sp = std::make_shared<sfem::ShiftedPenalty<real_t>>();

        auto linear_op = sfem::create_linear_operator("MF", f, nullptr, es);
        sp->set_op(linear_op);
        sp->default_init();

        sp->set_atol(1e-12);
        sp->set_max_it(20);
        sp->set_max_inner_it(30);
        sp->set_damping(1);
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
        // sp->enable_steepest_descent(SFEM_USE_STEEPEST_DESCENT);

        sp->verbose = true;

        sp->set_upper_bound(upper_bound);
        sp->set_constraints_op(cc_op, cc_op_t, sbv);
        return sp;
    }

    std::shared_ptr<ShiftedPenaltyMultigrid<real_t>> create_ssmgc(const std::shared_ptr<Function>         &f,
                                                                  const std::shared_ptr<ContactConditions> contact_conds,
                                                                  const enum ExecutionSpace                es,
                                                                  const std::shared_ptr<Input>            &in) {
        if (!f->space()->has_semi_structured_mesh()) {
            SFEM_ERROR("create_ssmgc cannot build MG without a semistructured mesh");
        }

        auto fs = f->space();

        ////////////////////////////////////////////////////////////////////////////////////
        // Default/read Input parameters
        ////////////////////////////////////////////////////////////////////////////////////
        int         nlsmooth_steps                     = 5;
        bool        project_coarse_correction          = false;
        bool        enable_line_search                 = true;
        std::string fine_op_type                       = "MF";
        std::string coarse_op_type                     = "MF";
        int         linear_smoothing_steps             = 3;
        bool        enable_coarse_space_preconditioner = true;
        bool        coarse_solver_verbose              = false;

        if (in) {
            in->get("nlsmooth_steps", nlsmooth_steps);
            in->get("project_coarse_correction", project_coarse_correction);
            in->get("enable_line_search", enable_line_search);
            in->get("fine_op_type", fine_op_type);
            in->get("coarse_op_type", coarse_op_type);
            in->get("linear_smoothing_steps", linear_smoothing_steps);
            in->get("enable_coarse_space_preconditioner", enable_coarse_space_preconditioner);
            in->get("coarse_solver_verbose", coarse_solver_verbose);
        }

        ////////////////////////////////////////////////////////////////////////////////////

        auto fs_coarse = fs->derefine();
        auto f_coarse  = f->derefine(fs_coarse, true);

        auto linear_op        = sfem::create_linear_operator(fine_op_type, f, nullptr, es);
        auto linear_op_coarse = sfem::create_linear_operator(coarse_op_type, f_coarse, nullptr, es);

        auto diag = sfem::create_buffer<real_t>(fs->n_dofs() / fs->block_size() * (fs->block_size() == 3 ? 6 : 3), es);
        auto mask = sfem::create_buffer<mask_t>(mask_count(fs->n_dofs()), es);

        f->constaints_mask(mask->data());
        f->hessian_block_diag_sym(nullptr, diag->data());

        auto sj                  = sfem::h_shiftable_block_sym_jacobi(diag, mask);
        sj->relaxation_parameter = 1. / fs->block_size();
        auto smoother            = sfem::create_stationary<real_t>(linear_op, sj, es);
        smoother->set_max_it(linear_smoothing_steps);

        auto restriction_unconstr = sfem::create_hierarchical_restriction(fs, fs_coarse, es);
        auto prolong_unconstr     = sfem::create_hierarchical_prolongation(fs_coarse, fs, es);
        auto prolongation         = sfem::make_op<real_t>(
                prolong_unconstr->rows(),
                prolong_unconstr->cols(),
                [=](const real_t *const from, real_t *const to) {
                    prolong_unconstr->apply(from, to);
                    f->apply_zero_constraints(to);
                },
                es);

        auto restriction = sfem::make_op<real_t>(
                restriction_unconstr->rows(),
                restriction_unconstr->cols(),
                [=](const real_t *const from, real_t *const to) {
                    restriction_unconstr->apply(from, to);
                    f_coarse->apply_zero_constraints(to);
                },
                es);

        auto mg = std::make_shared<ShiftedPenaltyMultigrid<real_t>>();
        mg->set_nlsmooth_steps(nlsmooth_steps);
        mg->set_project_coarse_space_correction(project_coarse_correction);
        mg->enable_line_search(enable_line_search);
        mg->set_max_it(20);

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            // FIXME this should not be here!
            CUDA_BLAS<real_t>::build_blas(mg->blas());
            CUDA_ShiftedPenalty<real_t>::build(mg->impl());
            mg->execution_space_ = EXECUTION_SPACE_DEVICE;
        } else
#endif
        {
            mg->default_init();
        }

        mg->add_level(linear_op, smoother, nullptr, restriction);

        auto solver_coarse     = sfem::create_cg<real_t>(linear_op_coarse, es);
        solver_coarse->verbose = coarse_solver_verbose;
        solver_coarse->set_rtol(1e-8);

        if (enable_coarse_space_preconditioner) {
            auto diag = sfem::create_buffer<real_t>(
                    fs_coarse->n_dofs() / fs_coarse->block_size() * (fs_coarse->block_size() == 3 ? 6 : 3), es);
            f_coarse->hessian_block_diag_sym(nullptr, diag->data());

            auto mask = sfem::create_buffer<mask_t>(mask_count(fs_coarse->n_dofs()), es);
            f_coarse->constaints_mask(mask->data());

            auto sj_coarse                  = sfem::h_shiftable_block_sym_jacobi(diag, mask);
            sj_coarse->relaxation_parameter = 1. / fs_coarse->block_size();
            solver_coarse->set_preconditioner_op(sj_coarse);
        }

        mg->add_level(linear_op_coarse, solver_coarse, prolongation, nullptr);

        ////////////////////////////////////////////////////////////////////////////////////
        // Contact
        ////////////////////////////////////////////////////////////////////////////////////

        contact_conds->init();
        auto cc_op       = contact_conds->linear_constraints_op();
        auto cc_op_t     = contact_conds->linear_constraints_op_transpose();
        auto upper_bound = sfem::create_buffer<real_t>(contact_conds->n_constrained_dofs(), es);
        contact_conds->signed_distance(upper_bound->data());

        // Top-level only
        mg->set_upper_bound(upper_bound);
        mg->set_constraints_op(cc_op, cc_op_t);

        // All levels
        // Add transformation matrices
        int  sym_block_size = (fs->block_size() == 3 ? 6 : 3);
        auto normal_prod    = sfem::create_buffer<real_t>(sym_block_size * contact_conds->n_constrained_dofs(), es);
        contact_conds->hessian_block_diag_sym(nullptr, normal_prod->data());

        auto fine_sbv = sfem::create_sparse_block_vector(contact_conds->node_mapping(), normal_prod);

        auto &&fine_ssmesh = fs->semi_structured_mesh();
        auto   fine_sides  = contact_conds->ss_sides();

        auto coarse_sides = sfem::ssquad4_derefine_element_connectivity(fine_ssmesh.level(), 1, fine_sides);
        const ptrdiff_t n_coarse_contact_nodes = sfem::ss_elements_max_node_id(coarse_sides) + 1;
        auto coarse_node_mapping = sfem::view(contact_conds->node_mapping(), 0, n_coarse_contact_nodes);

        auto coarse_normal_prod  = sfem::create_buffer<real_t>(sym_block_size * coarse_node_mapping->size(), es);
        auto coarse_sbv = sfem::create_sparse_block_vector(coarse_node_mapping, coarse_normal_prod);
        auto count = sfem::create_host_buffer<uint16_t>(contact_conds->n_constrained_dofs());

        ssquad4_element_node_incidence_count(fine_ssmesh.level(), 1, fine_sides->extent(1), fine_sides->data(), count->data());
        
        // FIXME When SPMG does not converge this may be the reason (this restriction is not variationally consistent)
        ssquad4_restrict(fine_sides->extent(1),  // nelements
                         fine_ssmesh.level(),    // from_level
                         1,                      // from_level_stride
                         fine_sides->data(),     // from_elements
                         count->data(),          // from_element_to_node_incidence_count
                         1,                      // to_level
                         1,                      // to_level_stride
                         coarse_sides->data(),   // to_elements
                         6,                      // vec_size
                         fine_sbv->data()->data(),
                         coarse_sbv->data()->data());

        auto c_restriction = sfem::make_op<real_t>(
                coarse_node_mapping->size(),
                contact_conds->node_mapping()->size(),
                [=](const real_t *const from, real_t *const to) {
                    auto &fine_ssmesh = fs->semi_structured_mesh();
                    ssquad4_restrict(fine_sides->extent(1),  // nelements
                                     fine_ssmesh.level(),    // from_level
                                     1,                      // from_level_stride
                                     fine_sides->data(),     // from_elements
                                     count->data(),          // from_element_to_node_incidence_count
                                     1,                      // to_level
                                     1,                      // to_level_stride
                                     coarse_sides->data(),   // to_elements
                                     1,                      // vec_size
                                     from,
                                     to);
                },
                es);

        mg->add_level_constraint_op_x_op(fine_sbv);
        mg->add_constraints_restriction(c_restriction);
        mg->add_level_constraint_op_x_op(coarse_sbv);

        ////////////////////////////////////////////////////////////////////////////////////
        mg->debug = true;
        // mg->skip_coarse = true;
        return mg;
    }

}  // namespace sfem
