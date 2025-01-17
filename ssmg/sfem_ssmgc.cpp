#include "sfem_ssmgc.hpp"

#include "sfem_API.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_ShiftedPenalty_impl.hpp"
#endif

namespace sfem {
    // TODO improve contact integration in SSMG
    // auto coarse_contact_conditions = contact_conds->derefine(fs_coarse, true);
    // auto coarse_contact_mask = sfem::create_buffer<mask_t>(fs_coarse->n_dofs(), es);
    // coarse_contact_conditions->mask(coarse_contact_mask->data());

    std::shared_ptr<ShiftedPenaltyMultigrid<real_t>> create_ssmgc(const std::shared_ptr<Function>         &f,
                                                                  const std::shared_ptr<ContactConditions> contact_conds,
                                                                  const enum ExecutionSpace                es,
                                                                  std::shared_ptr<Input>                  &in) {
        if (!f->space()->has_semi_structured_mesh()) {
            SFEM_ERROR("create_ssmgc cannot build MG without a semistructured mesh");
        }

        auto fs        = f->space();

        ////////////////////////////////////////////////////////////////////////////////////
        // Default/read Input parameters
        ////////////////////////////////////////////////////////////////////////////////////
        bool nlsmooth_steps = 10;
        in->get("nlsmooth_steps", nlsmooth_steps);

        bool project_coarse_correction = false;
        in->get("project_coarse_correction", project_coarse_correction);

        bool enable_line_search = false;
        in->get("enable_line_search", enable_line_search);

        std::string fine_op_type = "MF";
        in->get("fine_op_type", fine_op_type);

        std::string coarse_op_type = fs->block_size() == 1 ? "COO_SYM" : "BCRS_SYM";
        in->get("coarse_op_type", coarse_op_type);

        int linear_smoothing_steps = 3;
        in->get("linear_smoothing_steps", linear_smoothing_steps);

        bool enable_coarse_space_preconditioner = false;
        in->get("enable_coarse_space_preconditioner", enable_coarse_space_preconditioner);

        bool coarse_solver_verbose = false;
        in->get("coarse_solver_verbose", coarse_solver_verbose);

        ////////////////////////////////////////////////////////////////////////////////////

        auto fs_coarse = fs->derefine();
        auto f_coarse  = f->derefine(fs_coarse, true);

        auto linear_op        = sfem::create_linear_operator(fine_op_type, f, nullptr, es);
        auto linear_op_coarse = sfem::create_linear_operator(coarse_op_type, f_coarse, nullptr, es);

        auto diag = sfem::create_buffer<real_t>(fs->mesh_ptr()->n_nodes() * (fs->block_size() == 3 ? 6 : 3), es);
        auto mask = sfem::create_buffer<mask_t>(mask_count(fs->n_dofs()), es);
        f->hessian_block_diag_sym(nullptr, diag->data());

        auto sj                  = sfem::h_shiftable_block_sym_jacobi(diag, mask);
        sj->relaxation_parameter = 1. / fs->block_size();
        auto smoother            = sfem::create_stationary<real_t>(linear_op, sj, es);
        smoother->set_max_it(linear_smoothing_steps);

        auto restriction      = sfem::create_hierarchical_restriction(fs, fs_coarse, es);
        auto prolong_unconstr = sfem::create_hierarchical_prolongation(fs_coarse, fs, es);
        auto prolongation     = sfem::make_op<real_t>(
                prolong_unconstr->rows(),
                prolong_unconstr->cols(),
                [=](const real_t *const from, real_t *const to) {
                    prolong_unconstr->apply(from, to);
                    f->apply_zero_constraints(to);
                },
                es);

        auto mg = std::make_shared<ShiftedPenaltyMultigrid<real_t>>();
        mg->set_nlsmooth_steps(nlsmooth_steps);
        mg->set_project_coarse_space_correction(project_coarse_correction);
        mg->enable_line_search(enable_line_search);

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

        if (enable_coarse_space_preconditioner) {
            auto diag = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), es);
            f_coarse->hessian_diag(nullptr, diag->data());
            auto sj_coarse = sfem::create_shiftable_jacobi(diag, es);
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


        int  sym_block_size = (fs->block_size() == 3 ? 6 : 3);
        auto normal_prod    = sfem::create_buffer<real_t>(sym_block_size * contact_conds->n_constrained_dofs(), es);
        contact_conds->hessian_block_diag_sym(nullptr, normal_prod->data());
        auto sbv = sfem::create_sparse_block_vector(contact_conds->node_mapping(), normal_prod);


        mg->set_upper_bound(upper_bound);
        mg->set_constraints_op(cc_op, cc_op_t, sbv);

        ////////////////////////////////////////////////////////////////////////////////////

        return mg;
    }

}  // namespace sfem
