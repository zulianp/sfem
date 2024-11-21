#ifndef SFEM_PWC_INTERPOLATOR_HPP
#define SFEM_PWC_INTERPOLATOR_HPP

#include <cstddef>
#include <memory>

#include "sfem_Buffer.hpp"
#include "sfem_CooSym.hpp"
#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_openmp_blas.hpp"

// This class might be better off as just a sparse matrix, but the coarsen method is an optimized
// version of the matrix triple product ptap and transposing is basically a NOP
namespace sfem {
    template <typename R, typename T>
    class PiecewiseConstantInterpolator final : public Operator<T> {
    public:
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};
        ptrdiff_t fine_dim{-1};
        ptrdiff_t coarse_dim{-1};
        bool transposed{false};
        BLAS_Tpl<T> blas;

        void default_init() {
            OpenMP_BLAS<T>::build_blas(blas);
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        void transpose() { transposed = !transposed; }
        bool is_transposed() { return transposed; }
        void set_weights(const std::shared_ptr<Buffer<T>>& weights) { weights_ = weights; }
        void set_partition(const std::shared_ptr<Buffer<R>>& partition) { partition_ = partition; }

        // Internally allocates a workspace with same memory requirement as `a`
        // (this could be passed in as arg...)
        std::shared_ptr<CooSymSpMV<R, T>> coarsen(const std::shared_ptr<CooSymSpMV<R, T>>& a);

        void pwc_interpolate(const T* const v_coarse, T* const v) {
            R* partition = partition_->data();
            T* weights = weights_->data();
            // Only OMP impl for now
#pragma omp parallel for
            for (R k = 0; k < fine_dim; k++) {
                R coarse_idx = partition[k];
                if (coarse_idx >= 0) {
                    v[k] = v_coarse[coarse_idx] * weights[k];
                }
            }
        }

        void pwc_restrict(const T* const v, T* const v_coarse) {
            R* partition = partition_->data();
            T* weights = weights_->data();
            // Only OMP impl for now
#pragma omp parallel for
            for (R k = 0; k < coarse_dim; k++) {
                v_coarse[k] = 0.0;
            }

            for (R k = 0; k < fine_dim; k++) {
                R coarse_idx = partition[k];
                if (coarse_idx >= 0) {
                    v_coarse[coarse_idx] += v[k] * weights[k];
                }
            }
        }

        /* Operator */
        int apply(const T* const b, T* const x) override {
            transposed ? pwc_restrict(b, x) : pwc_interpolate(b, x);
            return 0;
        }
        inline std::ptrdiff_t rows() const override { return transposed ? fine_dim : coarse_dim; }
        inline std::ptrdiff_t cols() const override { return transposed ? coarse_dim : fine_dim; }
        ExecutionSpace execution_space() const override { return execution_space_; }

    private:
        // Length of `fine_dim` and values indicate coarse grid indices
        std::shared_ptr<Buffer<R>> partition_;
        // Length of `fine_dim` and values weight the PWC gridfunction
        std::shared_ptr<Buffer<T>> weights_;
    };

    template <typename R, typename T>
    std::shared_ptr<PiecewiseConstantInterpolator<R, T>> h_pwc_interp(
            const std::shared_ptr<Buffer<T>>& weights,
            const std::shared_ptr<Buffer<R>>& partition,
            const ptrdiff_t coarse_dim) {
        auto ret = std::make_shared<PiecewiseConstantInterpolator<R, T>>();
        ret->coarse_dim = coarse_dim;
        ret->fine_dim = weights->size();
        ret->set_weights(weights);
        ret->set_partition(partition);
        ret->default_init();
        return ret;
    }
}  // namespace sfem

#endif  // SFEM_PWC_INTERPOLATOR_HPP
