#ifndef SFEM_STATIONARY_HPP
#define SFEM_STATIONARY_HPP

#include <cstddef>
#include <functional>
#include <memory>

#include "sfem_Buffer.hpp"
#include "sfem_LpSmoother.hpp"
#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_openmp_blas.hpp"

namespace sfem {

    template <typename T>
    static std::shared_ptr<Operator<T>> diag_op(const SharedBuffer<T> &diagonal_scaling, const ExecutionSpace es);

    
    template <typename T>
    class StationaryIteration final : public MatrixFreeLinearSolver<T> {
    public:
        ExecutionSpace               execution_space_{EXECUTION_SPACE_INVALID};
        ptrdiff_t                    n_dofs{SFEM_PTRDIFF_INVALID};
        int                          max_it{3};
        SharedBuffer<T>              workspace;
        std::shared_ptr<Operator<T>> op;
        std::shared_ptr<Operator<T>> preconditioner;
        bool                         verbose{false};
        bool                         use_arg_as_first_residual{false};
        BLAS_Tpl<T>                  blas;

        int iterations_{0};

        std::function<void(T*)> interceptor;

        int iterations() const override { return iterations_; }

        void ensure_workspace() {
            if (!workspace || workspace->size() != n_dofs) {
                workspace = Buffer<T>::own(n_dofs, blas.allocate(n_dofs), this->blas.destroy);
            }
        }

        void default_init() {
            OpenMP_BLAS<T>::build_blas(blas);
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        /* Operator */
        int apply(const T* const b, T* const x) override {
            SFEM_TRACE_SCOPE("StationaryIteration::apply");

            iterations_ = 0;
            if(use_arg_as_first_residual) {
                iterations_ = 1;
                preconditioner->apply(b, x);
            }

            if(iterations_ >= max_it) return SFEM_SUCCESS;

            ensure_workspace();

            T* r = workspace->data();
            for (; iterations_ < max_it; iterations_++) {
                blas.zeros(workspace->size(), r);
                op->apply(x, r);
                blas.axpby(n_dofs, 1.0, b, -1.0, r);
                if (verbose) {
                    T norm_residual = this->blas.norm2(workspace->size(), r);
                    printf("%d : %f\n", iterations_, (double)norm_residual);
                }
                preconditioner->apply(r, x);

                if(interceptor) {
                    interceptor(x);
                }
            }

            return SFEM_SUCCESS;
        }
        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }
        ExecutionSpace        execution_space() const override { return execution_space_; }

        /* MatrixFreeLinearSolver */
        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            this->op = op;
            n_dofs   = op->rows();
        }
        void set_preconditioner_op(const std::shared_ptr<Operator<T>>& op) override { this->preconditioner = op; }
        void set_max_it(const int it) override { max_it = it; }
        void set_n_dofs(const ptrdiff_t n) override { this->n_dofs = n; }

        int set_op_and_diag_shift(const std::shared_ptr<Operator<T>>& op, const SharedBuffer<T>& diag) override {
            this->op       = op + sfem::diag_op(diag, execution_space());
            auto shiftable = std::dynamic_pointer_cast<ShiftableOperator<T>>(preconditioner);
            if (shiftable) {
                return shiftable->shift(diag);
            } else {
                SFEM_ERROR("Tried to call shift on object that is not subclass of ShiftableOperator!\n");
                return SFEM_FAILURE;
            }
        }

        int set_op_and_diag_shift(const std::shared_ptr<Operator<T>>&          op,
                                  const std::shared_ptr<SparseBlockVector<T>>& sbv,
                                  const SharedBuffer<T>&            diag) override {
            assert(sbv->n_blocks() == diag->size());
            
            this->op = op + sfem::create_sparse_block_vector_mult(op->rows(), sbv, diag);

            auto shiftable = std::dynamic_pointer_cast<ShiftableOperator<T>>(preconditioner);
            if (shiftable) {
                return shiftable->shift(sbv, diag);
            } else {
                SFEM_ERROR("Tried to call shift on object that is not subclass of ShiftableOperator!\n");
                return SFEM_FAILURE;
            }

            return SFEM_FAILURE;
        }
    };

    template <typename T>
    std::shared_ptr<StationaryIteration<T>> h_stationary(const std::shared_ptr<Operator<T>>& op,
                                                         const std::shared_ptr<Operator<T>>& preconditioner) {
        auto solver            = std::make_shared<StationaryIteration<T>>();
        solver->op             = op;
        solver->preconditioner = preconditioner;
        solver->n_dofs         = op->cols();
        solver->default_init();
        return solver;
    }
}  // namespace sfem

#endif  // SFEM_STATIONARY_HPP
