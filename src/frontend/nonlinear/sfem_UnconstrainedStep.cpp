#include "sfem_UnconstrainedStep.hpp"

#include "sfem_BSR.hpp"
#include "sfem_BSRTangent.hpp"
#include "sfem_DIA.hpp"
#include "sfem_Function.hpp"
#include "sfem_LinearOperatorFactory.hpp"
#include "sfem_ShiftableJacobi.hpp"
#include "sfem_bcgs.hpp"
#include "sfem_blas_factory.hpp"
#include "sfem_cg.hpp"
#include "sfem_ShiftableJacobi.hpp"
#include "sfem_bcgs.hpp"
#include "sfem_cg.hpp"

namespace sfem {

    class UnconstrainedStep::Impl {
    public:
        std::shared_ptr<Function>     f;
        ExecutionSpace                es{EXECUTION_SPACE_HOST};
        LinearSolverOptions           options;
        std::shared_ptr<BLAS<real_t>> blas;
        ptrdiff_t                     ndofs{0};

        std::shared_ptr<BSRTangent>                       tangent;  // format BSR
        SharedBuffer<real_t>                              state;    // the iterate the formats that read one see
        std::shared_ptr<Operator<real_t>>                 op;
        std::shared_ptr<MatrixFreeLinearSolver<real_t>>   solver;
        SharedBuffer<real_t>                              diag;
        std::shared_ptr<ShiftableJacobi<real_t>>          jacobi;
        SharedBuffer<real_t>                              rhs;
        int                                               iterations{0};

        bool is_bsr() const { return options.format == op_type::BSR && f->space()->block_size() > 1; }

        int create_solver() {
            if (options.type == "cg") {
                auto cg     = create_cg<real_t>(op, es);
                cg->verbose = false;
                // The tangent away from the solution can be indefinite: the step is then the
                // descent direction CG has built before it met negative curvature.
                cg->set_stop_on_negative_curvature(options.stop_on_negative_curvature);
                cg->set_rtol(options.rtol);
                cg->set_atol(options.atol);
                solver = cg;
            } else if (options.type == "bcgs") {
                auto bcgs     = create_bcgs<real_t>(op, es);
                bcgs->verbose = false;
                bcgs->set_rtol(options.rtol);
                bcgs->set_atol(options.atol);
                solver = bcgs;
            } else {
                fprintf(stderr, "UnconstrainedStep: unknown linear solver \"%s\" (cg or bcgs)\n", options.type.c_str());
                return SFEM_FAILURE;
            }

            solver->set_n_dofs(ndofs);
            solver->set_max_it(options.max_it);
            if (jacobi) {
                solver->set_preconditioner_op(jacobi);
            }
            return SFEM_SUCCESS;
        }
    };

    UnconstrainedStep::UnconstrainedStep() : impl_(std::make_unique<Impl>()) {}
    UnconstrainedStep::~UnconstrainedStep() = default;

    std::shared_ptr<UnconstrainedStep> UnconstrainedStep::create(const std::shared_ptr<Function> &f,
                                                                 const ExecutionSpace             es,
                                                                 const LinearSolverOptions       &options) {
        std::shared_ptr<UnconstrainedStep> ret(new UnconstrainedStep());
        auto                              &impl = *ret->impl_;
        impl.f                                  = f;
        impl.es                                 = es;
        impl.options                            = options;
        impl.blas                               = blas<real_t>(es);
        impl.ndofs                              = f->space()->n_dofs();
        impl.state                              = create_buffer<real_t>(impl.ndofs, es);
        impl.rhs                                = create_buffer<real_t>(impl.ndofs, es);

        // create_linear_operator turns a name it does not know into BSR_SYM, which an operator
        // without a symmetric block assembly aborts on: a misspelt format has to stop here.
        {
            const OperatorType formats[] = {op_type::MATRIX_FREE,
                                            op_type::CRS,
                                            op_type::SPLITCRS,
                                            op_type::ALIGNEDCRS,
                                            op_type::CRS_SYM,
                                            op_type::BSR,
                                            op_type::BSR_SYM,
                                            op_type::DIA,
                                            op_type::COO_SYM,
                                            op_type::SPLITDACRS,
                                            op_type::SELL,
                                            op_type::INEXACT};
            bool known = false;
            std::string names;
            for (auto name : formats) {
                known = known || options.format == name;
                names += (names.empty() ? "" : ", ") + std::string(name);
            }
            if (!known) {
                fprintf(stderr, "UnconstrainedStep: unknown format \"%s\" (%s)\n", options.format.c_str(), names.c_str());
                return nullptr;
            }
        }

        if (options.preconditioner == "jacobi") {
            impl.diag   = create_buffer<real_t>(impl.ndofs, es);
            impl.jacobi = create_shiftable_jacobi(impl.diag, es);
        } else if (options.preconditioner != "none") {
            fprintf(stderr, "UnconstrainedStep: unknown preconditioner \"%s\" (jacobi or none)\n", options.preconditioner.c_str());
            return nullptr;
        }

        if (impl.is_bsr()) {
            impl.tangent = BSRTangent::create(f, es, options.storage_bytes);
            if (!impl.tangent) return nullptr;
        } else if (options.format == op_type::MATRIX_FREE) {
            // Reads the iterate the Newton loop handed to `Function::update`; built once.
            impl.op = create_linear_operator(op_type::MATRIX_FREE, f, impl.state, es);
        }
        // INEXACT is built at the first linearisation: an operator that stores a tangent may join
        // the function after the step is created.

        return ret;
    }

    StepMeasure UnconstrainedStep::measure(const real_t *const, const real_t *const r) {
        StepMeasure m;
        m.value     = impl_->blas->norm2(impl_->ndofs, r);
        m.reference = m.value;
        return m;
    }

    int UnconstrainedStep::linearize(const real_t *const x, const real_t *const) {
        SFEM_TRACE_SCOPE("UnconstrainedStep::linearize");

        auto &impl = *impl_;
        impl.blas->copy(impl.ndofs, x, impl.state->data());

        if (impl.tangent) {
            if (impl.tangent->assemble(x) != SFEM_SUCCESS) return SFEM_FAILURE;
            impl.op = impl.tangent->op();
            if (impl.jacobi && impl.tangent->diagonal(impl.diag->data()) != SFEM_SUCCESS) return SFEM_FAILURE;
        } else {
            if (impl.options.format == op_type::INEXACT) {
                if (impl.f->inexact_supported() && impl.f->inexact_update(x) != SFEM_SUCCESS) return SFEM_FAILURE;
                // Matrix-free when no operator stores a tangent (create_linear_operator says so).
                if (!impl.op) impl.op = create_linear_operator(op_type::INEXACT, impl.f, impl.state, impl.es);
            } else if (impl.options.format != op_type::MATRIX_FREE && impl.options.format != op_type::INEXACT) {
                impl.op = create_linear_operator(impl.options.format, impl.f, impl.state, impl.es);
                if (!impl.op) return SFEM_FAILURE;
            }

            if (impl.jacobi) {
                impl.blas->zeros(impl.ndofs, impl.diag->data());
                if (impl.f->hessian_diag(x, impl.diag->data()) != SFEM_SUCCESS) {
                    fprintf(stderr,
                            "UnconstrainedStep: the Jacobi preconditioner needs Function::hessian_diag, which an "
                            "operator here does not provide; use format BSR or preconditioner none\n");
                    return SFEM_FAILURE;
                }
            }
        }

        if (impl.jacobi) {
            // The constrained rows of the operator are the identity (compose_constraints_op).
            impl.f->set_value_to_constrained_dofs(1, impl.diag->data());
            impl.jacobi->set_diag(impl.diag);
        }

        if (!impl.solver) return impl.create_solver();
        impl.solver->set_op(impl.op);
        return SFEM_SUCCESS;
    }

    int UnconstrainedStep::solve(const real_t *const, const real_t *const r, real_t *const dx) {
        SFEM_TRACE_SCOPE("UnconstrainedStep::solve");

        auto &impl = *impl_;
        // The constrained rows are the identity: their right-hand side is the increment dx
        // arrives with, and a Krylov start that already holds it keeps the iteration on the
        // interior, which follows the increment through the tangent's coupling.
        impl.blas->zaxpby(impl.ndofs, -1, r, 0, r, impl.rhs->data());
        impl.f->copy_constrained_dofs(dx, impl.rhs->data());
        impl.blas->zeros(impl.ndofs, dx);
        impl.f->copy_constrained_dofs(impl.rhs->data(), dx);
        const int status = impl.solver->apply(impl.rhs->data(), dx);
        impl.iterations  = impl.solver->iterations();
        return status;
    }

    int UnconstrainedStep::iterations() const { return impl_->iterations; }

    void UnconstrainedStep::set_rtol(const real_t rtol) {
        impl_->options.rtol = rtol;
        if (!impl_->solver) return;
        if (auto cg = std::dynamic_pointer_cast<ConjugateGradient<real_t>>(impl_->solver)) cg->set_rtol(rtol);
        if (auto bcgs = std::dynamic_pointer_cast<BiCGStab<real_t>>(impl_->solver)) bcgs->set_rtol(rtol);
    }

    const LinearSolverOptions &UnconstrainedStep::options() const { return impl_->options; }

}  // namespace sfem
