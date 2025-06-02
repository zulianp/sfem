#include "sfem_ssmgc.hpp"

#include "sfem_API.hpp"
#include "ssquad4_interpolate.h"

#include "lumped_ptdp.h"

#ifdef SFEM_ENABLE_CUDA
#include "cu_ssquad4_interpolate.h"
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_ShiftedPenalty_impl.hpp"
#endif

namespace sfem {
    std::shared_ptr<ShiftedPenalty<real_t>> create_shifted_penalty(const std::shared_ptr<Function>         &f,
                                                                   const std::shared_ptr<ContactConditions> contact_conds,

                                                                   const std::shared_ptr<Input> &in) {
        const enum ExecutionSpace es          = f->execution_space();
        auto                      fs          = f->space();
        const int                 block_size  = fs->block_size();
        auto                      cc_op       = contact_conds->linear_constraints_op();
        auto                      cc_op_t     = contact_conds->linear_constraints_op_transpose();
        auto                      upper_bound = sfem::create_buffer<real_t>(contact_conds->n_constrained_dofs(), es);
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

        auto sj = sfem::create_shiftable_block_sym_jacobi(fs->block_size(), diag, mask, es);
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

    template <typename T>
    class SSMGC<T>::Impl {
    public:
        std::shared_ptr<ContactConditions> contact_conds;
        std::shared_ptr<Operator<T>>       linear_constraints_op;
        std::shared_ptr<Operator<T>>       linear_constraints_op_transpose;

        class Level_t {
        public:
            std::shared_ptr<SparseBlockVector<T>> sbv;
            std::shared_ptr<Buffer<uint16_t>>     count;
            std::shared_ptr<Buffer<idx_t *>>      sides;
            std::shared_ptr<Buffer<idx_t>>        mapping;
            std::shared_ptr<Function>             function;

            void print(std::ostream &os = std::cout) {
                os << "-----------------------\n";
                os << "SBV\n";
                os << "- block_size: " << sbv->block_size() << "\n";
                os << "- n_blocks:   " << sbv->n_blocks() << "\n";

                if (count)
                    os << "count:    " << count->size() << "\n";
                else
                    os << "no count\n";

                os << "sides:        " << sides->extent(0) << " x " << sides->extent(1) << "\n";
                os << "mapping:      " << mapping->size() << "\n";
                os << "ndofs:        " << function->space()->n_dofs() << "\n";
                os << "-----------------------\n";
            }
        };

        std::vector<std::shared_ptr<Level_t>>     levels;
        std::vector<std::shared_ptr<Operator<T>>> restrict_sbv;
        std::vector<std::shared_ptr<Operator<T>>> restrict_penalization;

        std::shared_ptr<Buffer<T>>                  upper_bound;
        std::shared_ptr<ShiftedPenaltyMultigrid<T>> mg;

        void check() {
            for (auto &l : levels) {
                if (l->mapping->size() != l->count->size()) {
                    SFEM_ERROR("Mapping and count inconsistent sizes! %ld != %ld\n", l->mapping->size(), l->count->size());
                }
            }

            const int nlevels = levels.size();
            for (int l = 0; l < nlevels - 1; l++) {
                if (levels[l]->mapping->size() != restrict_penalization[l]->cols()) {
                    SFEM_ERROR("Mapping and restrict_penalization inconsistent sizes! %ld != %ld\n",
                               levels[l]->mapping->size(),
                               restrict_penalization[l]->cols());
                }
            }
        }

        void restrict_contact_constraints() {
            SFEM_TRACE_SCOPE("SPMG::restrict_contact_constraints");
            const int nlevels = levels.size();
            for (int i = 1; i < nlevels; i++) {
                auto fine   = levels[i - 1];
                auto coarse = levels[i];
                restrict_sbv[i - 1]->apply(fine->sbv->data()->data(), coarse->sbv->data()->data());
            }
        }

        void init_discretization(const std::shared_ptr<Function> f) {
            SFEM_TRACE_SCOPE("SPMG::init_discretization");

            auto            fs             = f->space();
            auto           &ssmesh         = f->space()->semi_structured_mesh();
            const int       L              = ssmesh.level();
            const ptrdiff_t sym_block_size = (fs->block_size() == 3 ? 6 : 3);

            std::vector<int> levels = ssmesh.derefinement_levels();
            this->levels.clear();

            // Order from finest to coarsest!
            std::reverse(levels.begin(), levels.end());
            const int nlevels = levels.size();

            auto l0      = std::make_shared<Level_t>();
            l0->function = f;
            this->levels.push_back(l0);

            for (int l = 1; l < nlevels; l++) {
                auto f_prev  = this->levels.back()->function;
                auto fs_prev = f_prev->space();
                auto fs_next = fs_prev->derefine(levels[l]);

                auto li      = std::make_shared<Level_t>();
                li->function = f_prev->derefine(fs_next, true);
                this->levels.push_back(li);
            }
        }

        void update_contact(const T *const disp) {
            SFEM_TRACE_SCOPE("SPMG::update_contact");
#ifdef SFEM_ENABLE_CUDA
            if (EXECUTION_SPACE_DEVICE == es) {
                auto upper_bound = sfem::to_host(upper_bound);
                upper_bound      = sfem::to_device(upper_bound);
                // TODO cc_op and cc_op_t to device
            } else
#endif
            {
                contact_conds->update(disp);
                contact_conds->update_signed_distance(disp, upper_bound->data());
            }
        }

        void init_contact() {
            SFEM_TRACE_SCOPE("SPMG::init_contact");
            contact_conds->init();
            linear_constraints_op           = contact_conds->linear_constraints_op();
            linear_constraints_op_transpose = contact_conds->linear_constraints_op_transpose();
            upper_bound = sfem::create_buffer<T>(contact_conds->n_constrained_dofs(), sfem::MEMORY_SPACE_HOST);

            contact_conds->signed_distance(upper_bound->data());
            const ExecutionSpace es             = levels[0]->function->execution_space();
            const int            block_size     = levels[0]->function->space()->block_size();
            const int            sym_block_size = (block_size == 3 ? 6 : 3);
            const int            nlevels        = levels.size();

#ifdef SFEM_ENABLE_CUDA
            if (EXECUTION_SPACE_DEVICE == es) {
                upper_bound = sfem::to_device(upper_bound);
            }
#endif

            std::vector<std::shared_ptr<Buffer<idx_t *>>> host_sides;
            {
                auto normal_prod = sfem::create_buffer<T>(sym_block_size * contact_conds->n_constrained_dofs(), es);
                contact_conds->hessian_block_diag_sym(nullptr, normal_prod->data());

                auto fine = levels[0];
                fine->sbv = sfem::create_sparse_block_vector(contact_conds->node_mapping(), normal_prod);

                fine->sides   = contact_conds->ss_sides();
                fine->mapping = contact_conds->node_mapping();

                host_sides.push_back(fine->sides);

#ifdef SFEM_ENABLE_CUDA
                if (EXECUTION_SPACE_DEVICE == es) {
                    fine->sbv   = sfem::to_device(fine->sbv);
                    fine->sides = sfem::to_device(fine->sides);
                }
#endif
            }

            for (int i = 1; i < nlevels; i++) {
                auto fine   = levels[i - 1];
                auto coarse = levels[i];

                auto      fine_space   = fine->function->space();
                const int level        = fine_space->semi_structured_mesh().level();
                auto      coarse_space = coarse->function->space();
                const int coarse_level =
                        coarse_space->has_semi_structured_mesh() ? coarse_space->semi_structured_mesh().level() : 1;

                auto coarse_sides = sfem::ssquad4_derefine_element_connectivity(level, coarse_level, host_sides[i - 1]);
                coarse->sides     = coarse_sides;
                host_sides.push_back(coarse->sides);

                const ptrdiff_t n_coarse_contact_nodes = sfem::ss_elements_max_node_id(coarse_sides) + 1;
                coarse->mapping                        = sfem::view(fine->mapping, 0, n_coarse_contact_nodes);

                auto coarse_normal_prod =
                        sfem::create_buffer<T>(sym_block_size * coarse->mapping->size(), sfem::MEMORY_SPACE_HOST);
                coarse->sbv = sfem::create_sparse_block_vector(coarse->mapping, coarse_normal_prod);

                fine->count = sfem::create_host_buffer<uint16_t>(fine->mapping->size());
                ssquad4_element_node_incidence_count(
                        level, 1, fine->sides->extent(1), host_sides[i - 1]->data(), fine->count->data());

#ifdef SFEM_ENABLE_CUDA
                if (es == EXECUTION_SPACE_DEVICE) {
                    assert(fine->sides->mem_space() == MEMORY_SPACE_DEVICE);

                    fine->count   = sfem::to_device(fine->count);
                    coarse->sides = sfem::to_device(coarse->sides);
                    coarse->sbv   = sfem::to_device(coarse->sbv);

                    auto op = sfem::make_op<T>(
                                 coarse_node_mapping->size() * sym_block_size,
                                 fine_mapping->size() * sym_block_size,
                                 [coarse, fine, sym_block_size, level, coarse_level](const T *const from, T *const to) {
                                     SFEM_TRACE_SCOPE("ssquad4_restrict(sbv)");

                                     cu_ssquad4_restrict(fine->sides->extent(1),
                                                         level,
                                                         1,
                                                         fine->sides->data(),
                                                         fine->count->data(),
                                                         coarse_level,
                                                         1,
                                                         coarse->sides->data(),
                                                         sym_block_size,
                                                         SFEM_REAL_DEFAULT,
                                                         1,
                                                         fine->sbv->data()->data(),
                                                         SFEM_REAL_DEFAULT,
                                                         1,
                                                         coarse->sbv->data()->data(),
                                                         SFEM_DEFAULT_STREAM);
                                 }),
                         es;
                    restrict_sbv.push_back(op);

                } else
#endif
                {
                    auto op = sfem::make_op<T>(
                            coarse->mapping->size() * sym_block_size,
                            fine->mapping->size() * sym_block_size,
                            [coarse, fine, sym_block_size, level, coarse_level](const T *const from, T *const to) {
                                SFEM_TRACE_SCOPE("ssquad4_restrict(sbv)");

                                ssquad4_restrict(fine->sides->extent(1),  // nelements
                                                 level,                   // from_level
                                                 1,                       // from_level_stride
                                                 fine->sides->data(),     // from_elements
                                                 fine->count->data(),     // from_element_to_node_incidence_count
                                                 coarse_level,            // to_level
                                                 1,                       // to_level_stride
                                                 coarse->sides->data(),   // to_elements
                                                 sym_block_size,          // vec_size
                                                 fine->sbv->data()->data(),
                                                 coarse->sbv->data()->data());
                            },
                            es);
                    restrict_sbv.push_back(op);
                }

                auto c_restriction = sfem::make_op<T>(
                        coarse->mapping->size(),
                        fine->mapping->size(),
                        [fine, coarse, level, coarse_level](const T *const from, T *const to) {
                            SFEM_TRACE_SCOPE("ssquad4_restrict");

// TODO check (cu_)ssquad4_restrict implementations
#ifdef SFEM_ENABLE_CUDA
                            if (es == EXECUTION_SPACE_DEVICE) {
                                cu_ssquad4_restrict(fine->sides->extent(1),
                                                    level,
                                                    1,
                                                    fine->sides->data(),
                                                    fine->count->data(),
                                                    coarse_level,
                                                    1,
                                                    coarse->sides->data(),
                                                    1,
                                                    SFEM_REAL_DEFAULT,
                                                    1,
                                                    from,
                                                    SFEM_REAL_DEFAULT,
                                                    1,
                                                    to,
                                                    SFEM_DEFAULT_STREAM);
                                return;
                            }
#endif
                            ssquad4_restrict(fine->sides->extent(1),  // nelements
                                             level,                   // from_level
                                             1,                       // from_level_stride
                                             fine->sides->data(),     // from_elements
                                             fine->count->data(),     // from_element_to_node_incidence_count
                                             coarse_level,            // to_level
                                             1,                       // to_level_stride
                                             coarse->sides->data(),   // to_elements
                                             1,                       // vec_size
                                             from,
                                             to);
                        },
                        es);

                restrict_penalization.push_back(c_restriction);
            }
        }

        void init_solver(const std::shared_ptr<Input> &in) {
            SFEM_TRACE_SCOPE("SPMG::init_solver");

            auto                      f  = levels[0]->function;
            const enum ExecutionSpace es = f->execution_space();

            ////////////////////////////////////////////////////////////////////////////////////
            // Default/read Input parameters
            ////////////////////////////////////////////////////////////////////////////////////

            bool enable_coarse_space_preconditioner = true;
            bool enable_mixed_precision             = true;

            bool collect_energy_norm_correction = true;
            bool coarse_solver_verbose          = false;
            bool debug                          = false;
            bool enable_shift                   = true;
            bool enable_line_search             = false;
            bool project_coarse_correction      = false;
            bool enable_nl_obstacle             = true;

            int coarse_linear_smoothing_steps = 10;
            int linear_smoothing_steps        = 1;
            int max_inner_it                  = 40;
            int max_it                        = 10;
            int nlsmooth_steps                = 10;
            int max_coarse_it                 = 40000;

            static constexpr bool is_double = std::is_same<real_t, double>::value;

            real_t atol                   = is_double ? 1e-9 : 5e-7;
            real_t max_penalty_param      = enable_mixed_precision ? (is_double ? 1e5 : 1e4) : (is_double ? 1e6 : 1e4);
            real_t penalty_param          = 1e4;
            real_t penalty_param_increase = 10;
            real_t coarse_rtol            = 1e-6;

            std::string coarse_op_type = es == EXECUTION_SPACE_HOST ? "BSR" : "MF";
            std::string debug_folder   = "debug_ssmgc";
            std::string fine_op_type   = "MF";

            if (in) {
                printf("SPMG: Reading Input\n");
                in->get("atol", atol);
                in->get("coarse_linear_smoothing_steps", coarse_linear_smoothing_steps);
                in->get("coarse_op_type", coarse_op_type);
                in->get("coarse_solver_verbose", coarse_solver_verbose);
                in->get("collect_energy_norm_correction", collect_energy_norm_correction);
                in->get("debug", debug);
                in->get("debug_folder", debug_folder);
                in->get("enable_coarse_space_preconditioner", enable_coarse_space_preconditioner);
                in->get("enable_line_search", enable_line_search);
                in->get("enable_mixed_precision", enable_mixed_precision);
                in->get("enable_shift", enable_shift);
                in->get("enable_nl_obstacle", enable_nl_obstacle);
                in->get("fine_op_type", fine_op_type);
                in->get("linear_smoothing_steps", linear_smoothing_steps);
                in->get("max_inner_it", max_inner_it);
                in->get("max_it", max_it);
                in->get("max_penalty_param", max_penalty_param);
                in->get("nlsmooth_steps", nlsmooth_steps);
                in->get("penalty_param", penalty_param);
                in->get("penalty_param_increase", penalty_param_increase);
                in->get("project_coarse_correction", project_coarse_correction);
                in->get("max_coarse_it", max_coarse_it);
                in->get("coarse_rtol", coarse_rtol);
            }

            ////////////////////////////////////////////////////////////////////////////////////

            auto      fs             = f->space();
            const int nlevels        = levels.size();
            const int block_size     = fs->block_size();
            const int sym_block_size = (block_size == 3 ? 6 : 3);

            std::vector<std::shared_ptr<Operator<real_t>>>               operators;
            std::vector<std::shared_ptr<MatrixFreeLinearSolver<real_t>>> smoothers_or_solver;

            for (int i = 0; i < nlevels - 1; i++) {
                auto fi        = levels[i]->function;
                auto fsi       = fi->space();
                auto linear_op = sfem::create_linear_operator(fine_op_type.c_str(), fi, nullptr, es);
                operators.push_back(linear_op);

                auto diag = sfem::create_buffer<real_t>(fsi->n_dofs() / fsi->block_size() * sym_block_size, es);
                auto mask = sfem::create_buffer<mask_t>(mask_count(fsi->n_dofs()), es);

                fi->constaints_mask(mask->data());
                fi->hessian_block_diag_sym(nullptr, diag->data());

                std::shared_ptr<sfem::Operator<real_t>> sj;

                if (enable_mixed_precision) {
                    sj = sfem::create_mixed_precision_shiftable_block_sym_jacobi<real_t, float>(
                            fsi->block_size(), diag, mask, es);
                } else {
                    sj = sfem::create_shiftable_block_sym_jacobi(fsi->block_size(), diag, mask, es);
                }

                auto smoother = sfem::create_stationary<real_t>(linear_op, sj, es);

                if (i == 0) {
                    smoother->set_max_it(linear_smoothing_steps);

                    // Avoid recomputing the residual and just apply preconditioner
                    smoother->use_arg_as_first_residual = true;
                } else {
                    smoother->set_max_it(coarse_linear_smoothing_steps);
                }

                smoothers_or_solver.push_back(smoother);
            }

            // ----------------------------------
            // Coarse level
            // ----------------------------------

            auto f_coarse  = levels.back()->function;
            auto linear_op = sfem::create_linear_operator(coarse_op_type.c_str(), f_coarse, nullptr, es);
            operators.push_back(linear_op);

            // Coarse-grid solver
            auto coarse_solver = sfem::create_cg<real_t>(operators.back(), es);
            coarse_solver->set_max_it(max_coarse_it);
            coarse_solver->verbose = coarse_solver_verbose;
            coarse_solver->set_rtol(coarse_rtol);

            if (enable_coarse_space_preconditioner) {
                auto f_coarse  = levels.back()->function;
                auto fs_coarse = f_coarse->space();
                auto diag      = sfem::create_buffer<real_t>(fs_coarse->n_dofs() / fs_coarse->block_size() * sym_block_size, es);
                f_coarse->hessian_block_diag_sym(nullptr, diag->data());

                auto mask = sfem::create_buffer<mask_t>(mask_count(fs_coarse->n_dofs()), es);
                f_coarse->constaints_mask(mask->data());

                std::shared_ptr<sfem::Operator<real_t>> sj_coarse;
                if (enable_mixed_precision) {
                    sj_coarse = sfem::create_mixed_precision_shiftable_block_sym_jacobi<real_t, float>(
                            fs_coarse->block_size(), diag, mask, es);
                } else {
                    sj_coarse = sfem::create_shiftable_block_sym_jacobi(fs_coarse->block_size(), diag, mask, es);
                }

                coarse_solver->set_preconditioner_op(sj_coarse);
            }

            smoothers_or_solver.push_back(coarse_solver);

            for (int i = 0; i < nlevels; i++) {
                auto s = levels[i]->function->space();
                printf("%d) \tL=%d\n", i, s->has_semi_structured_mesh() ? s->semi_structured_mesh().level() : 1);

                levels[i]->function->describe(std::cout);
            }

            mg = std::make_shared<ShiftedPenaltyMultigrid<real_t>>();

            {
                SFEM_TRACE_SCOPE("create_ssgmg::construct");
                // Construct actual multigrid

                auto restriction_unconstr =
                        sfem::create_hierarchical_restriction(levels[0]->function->space(), levels[1]->function->space(), es);
                auto f_coarse    = levels[1]->function;
                auto restriction = sfem::make_op<real_t>(
                        restriction_unconstr->rows(),
                        restriction_unconstr->cols(),
                        [=](const real_t *const from, real_t *const to) {
                            restriction_unconstr->apply(from, to);
                            f_coarse->apply_zero_constraints(to);
                        },
                        es);

                mg->add_level(operators[0], smoothers_or_solver[0], nullptr, restriction);

                // ----------------------------------
                // Intermediate levels
                // ----------------------------------
                for (int i = 1; i < nlevels - 1; i++) {
                    auto restriction_unconstr = sfem::create_hierarchical_restriction(
                            levels[i]->function->space(), levels[i + 1]->function->space(), es);
                    auto f_coarse    = levels[i + 1]->function;
                    auto restriction = sfem::make_op<real_t>(
                            restriction_unconstr->rows(),
                            restriction_unconstr->cols(),
                            [=](const real_t *const from, real_t *const to) {
                                restriction_unconstr->apply(from, to);
                                f_coarse->apply_zero_constraints(to);
                            },
                            es);

                    auto prolong_unconstr = sfem::create_hierarchical_prolongation(
                            levels[i]->function->space(), levels[i - 1]->function->space(), es);

                    auto prolongation = sfem::make_op<real_t>(
                            prolong_unconstr->rows(),
                            prolong_unconstr->cols(),
                            [prolong_unconstr, f = levels[i - 1]->function](const real_t *const from, real_t *const to) {
                                prolong_unconstr->apply(from, to);
                                f->apply_zero_constraints(to);
                            },
                            es);

                    mg->add_level(operators[i], smoothers_or_solver[i], prolongation, restriction);
                }

                auto prolong_unconstr = sfem::create_hierarchical_prolongation(
                        levels.back()->function->space(), levels[nlevels - 2]->function->space(), es);

                auto prolongation = sfem::make_op<real_t>(
                        prolong_unconstr->rows(),
                        prolong_unconstr->cols(),
                        [prolong_unconstr, f = levels.back()->function](const real_t *const from, real_t *const to) {
                            prolong_unconstr->apply(from, to);
                            f->apply_zero_constraints(to);
                        },
                        es);

                // ----------------------------------
                // Coarse level
                // ----------------------------------
                mg->add_level(operators.back(), smoothers_or_solver.back(), prolongation, nullptr);
            }

#ifdef SFEM_ENABLE_CUDA
            if (es == EXECUTION_SPACE_DEVICE) {
                // FIXME this should not be here!
                CUDA_BLAS<real_t>::build_blas(mg->blas());
                CUDA_ShiftedPenalty<real_t>::build(mg->impl());
                mg->set_execution_space(EXECUTION_SPACE_DEVICE);
            } else
#endif
            {
                mg->default_init();
            }

            // ----------------------------------
            // Contact
            // ----------------------------------

            mg->set_upper_bound(upper_bound);
            mg->set_constraints_op(linear_constraints_op, linear_constraints_op_transpose);

            auto fine = levels[0];
            mg->add_level_constraint_op_x_op(fine->sbv);

            auto fine_sides   = contact_conds->ss_sides();
            auto fine_mapping = contact_conds->node_mapping();

            for (int i = 1; i < nlevels; i++) {
                auto fine   = levels[i - 1]->function;
                auto coarse = levels[i];

                mg->add_level_constraint_op_x_op(coarse->sbv);

                assert(size_t(i - 1) < restrict_penalization.size());
                mg->add_constraints_restriction(restrict_penalization[i - 1]);
            }

            ////////////////////////////////////////////////////////////////////////////////////
            mg->set_debug(true);
            mg->enable_line_search(enable_line_search);
            mg->set_max_it(max_it);
            mg->set_max_inner_it(max_inner_it);
            mg->set_max_penalty_param(max_penalty_param);
            mg->set_penalty_param(penalty_param);
            mg->set_atol(atol);
            mg->collect_energy_norm_correction(collect_energy_norm_correction);
            mg->set_enable_shift(enable_shift);
            mg->set_penalty_param_increase(penalty_param_increase);
            mg->set_nlsmooth_steps(nlsmooth_steps);

            if (enable_nl_obstacle) {
                mg->set_update_constraints([that = this](const T *const disp) { that->update_contact(disp); });
            }
        }

        void init(const std::shared_ptr<Function>         &f,
                  const std::shared_ptr<ContactConditions> contact_conds,
                  const std::shared_ptr<Input>            &in) {
            this->contact_conds = contact_conds;
            init_discretization(f);
            init_contact();
            restrict_contact_constraints();
            init_solver(in);
        }
    };

    template <typename T>
    SSMGC<T>::~SSMGC() = default;

    template <typename T>
    SSMGC<T>::SSMGC() : impl_(std::make_unique<Impl>()) {}

    template <typename T>
    std::shared_ptr<SSMGC<T>> SSMGC<T>::create(const std::shared_ptr<Function>         &f,
                                               const std::shared_ptr<ContactConditions> contact_conds,
                                               const std::shared_ptr<Input>            &in) {
        auto ret = std::make_shared<SSMGC<T>>();
        ret->impl_->init(f, contact_conds, in);
        int ll = 0;
        for (auto l : ret->impl_->levels) {
            std::cout << "-----------------------\n";
            std::cout << "level " << ll << "\n";
            l->print();
        }
        // ret->impl_->mg = create_ssmgc(f, contact_conds, in);
        return ret;
    }

    template <typename T>
    int SSMGC<T>::apply(const T *const rhs, T *const x) {
        return impl_->mg->apply(rhs, x);
    }

    template <typename T>
    std::ptrdiff_t SSMGC<T>::rows() const {
        return impl_->mg->rows();
    }

    template <typename T>
    std::ptrdiff_t SSMGC<T>::cols() const {
        return impl_->mg->cols();
    }

    template <typename T>
    ExecutionSpace SSMGC<T>::execution_space() const {
        return impl_->mg->execution_space();
    }

    template class SSMGC<real_t>;

    std::shared_ptr<SSMGC<real_t>> create_ssmgc(const std::shared_ptr<Function>         &f,
                                                const std::shared_ptr<ContactConditions> contact_conds,
                                                const std::shared_ptr<Input>            &in) {
        return SSMGC<real_t>::create(f, contact_conds, in);
    }
}  // namespace sfem
