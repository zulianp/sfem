#ifndef SFEM_BSR_BLOCK_GAUSS_SEIDEL_HPP
#define SFEM_BSR_BLOCK_GAUSS_SEIDEL_HPP

#include "sfem_BSR.hpp"
#include "sfem_openmp_blas.hpp"

#include <cassert>
#include <memory>

namespace sfem {

    namespace bsr_bgs_private_ {

        template <typename T>
        static SFEM_INLINE void invert1(const T a00, T* const out) {
            assert(a00 != T(0));
            out[0] = T(1) / a00;
        }

        template <typename T>
        static SFEM_INLINE void invert2(const T* const SFEM_RESTRICT a, T* const SFEM_RESTRICT out) {
            const T det = a[0] * a[3] - a[1] * a[2];
            assert(det != T(0));
            const T idet = T(1) / det;
            out[0]       = a[3] * idet;
            out[1]       = -a[1] * idet;
            out[2]       = -a[2] * idet;
            out[3]       = a[0] * idet;
        }

        template <typename T>
        static SFEM_INLINE void invert3(const T* const SFEM_RESTRICT a, T* const SFEM_RESTRICT out) {
            const T x0 = a[4] * a[8];
            const T x1 = a[5] * a[7];
            const T x2 = a[1] * a[5];
            const T x3 = a[1] * a[8];
            const T x4 = a[2] * a[4];
            const T x5 = T(1) / (a[0] * x0 - a[0] * x1 + a[2] * a[3] * a[7] - a[3] * x3 + a[6] * x2 - a[6] * x4);
            assert(x5 == x5);
            out[0] = x5 * (x0 - x1);
            out[1] = x5 * (a[2] * a[7] - x3);
            out[2] = x5 * (x2 - x4);
            out[3] = x5 * (-a[3] * a[8] + a[5] * a[6]);
            out[4] = x5 * (a[0] * a[8] - a[2] * a[6]);
            out[5] = x5 * (-a[0] * a[5] + a[2] * a[3]);
            out[6] = x5 * (a[3] * a[7] - a[4] * a[6]);
            out[7] = x5 * (-a[0] * a[7] + a[1] * a[6]);
            out[8] = x5 * (a[0] * a[4] - a[1] * a[3]);
        }

        template <typename T>
        static SFEM_INLINE void invert_block(const int bs, const T* const SFEM_RESTRICT a, T* const SFEM_RESTRICT out) {
            switch (bs) {
                case 1:
                    invert1(a[0], out);
                    break;
                case 2:
                    invert2(a, out);
                    break;
                case 3:
                    invert3(a, out);
                    break;
                default:
                    SFEM_ERROR("BSRBlockGaussSeidel: block size %d not supported\n", bs);
                    break;
            }
        }

        template <typename T>
        static SFEM_INLINE void matvec_set(const int                     bs,
                                           const T* const SFEM_RESTRICT  a,
                                           const T* const SFEM_RESTRICT  x,
                                           T* const SFEM_RESTRICT        y) {
            for (int d1 = 0; d1 < bs; d1++) {
                T acc = 0;
                for (int d2 = 0; d2 < bs; d2++) {
                    acc += a[d1 * bs + d2] * x[d2];
                }
                y[d1] = acc;
            }
        }

    }  // namespace bsr_bgs_private_

    /**
     * Host block Gauss–Seidel smoother for square BSR matrices.
     *
     * Operator semantics match ShiftableJacobi: apply(b, x) computes a correction
     * delta ≈ A^{-1} b via GS sweeps (from zero) and accumulates x += delta.
     */
    template <typename R = count_t, typename C = idx_t, typename T = real_t>
    class BSRBlockGaussSeidel final : public Operator<T> {
    public:
        using Matrix = BSR<R, C, T, T>;

        static constexpr int MAX_BLOCK_SIZE = 4;

        explicit BSRBlockGaussSeidel(const std::shared_ptr<Matrix>& bsr) : bsr_(bsr) {
            assert(bsr_);
            assert(bsr_->execution_space() == EXECUTION_SPACE_HOST);
            assert(bsr_->row_block_size() == bsr_->col_block_size());
            assert(bsr_->rows() == bsr_->cols());
            assert(bsr_->block_size() >= 1 && bsr_->block_size() <= MAX_BLOCK_SIZE);

            blas_ = make_openmp_blas<T>();
            build_inv_diag();
        }

        void set_max_it(const int it) { max_it_ = it; }
        void set_symmetric(const bool s) { symmetric_ = s; }
        void set_relaxation(const T omega) {
            relaxation_ = omega;
            build_inv_diag();
        }

        int apply(const T* const b, T* const x) override {
            SFEM_TRACE_SCOPE("BSRBlockGaussSeidel::apply");

            ensure_workspace();

            const ptrdiff_t n     = workspace_->size();
            T* const        delta = workspace_->data();
            blas_->zeros(n, delta);

            for (int it = 0; it < max_it_; it++) {
                forward_sweep(b, delta);
                if (symmetric_) {
                    backward_sweep(b, delta);
                }
            }

            blas_->axpy(n, T(1), delta, x);
            return SFEM_SUCCESS;
        }

        std::ptrdiff_t rows() const override { return bsr_->rows(); }
        std::ptrdiff_t cols() const override { return bsr_->cols(); }
        ExecutionSpace execution_space() const override { return EXECUTION_SPACE_HOST; }

    private:
        void ensure_workspace() {
            const ptrdiff_t n = bsr_->rows();
            if (!workspace_ || workspace_->size() != static_cast<size_t>(n)) {
                workspace_ = create_host_buffer<T>(n);
            }
        }

        void build_inv_diag() {
            SFEM_TRACE_SCOPE("BSRBlockGaussSeidel::build_inv_diag");

            const int       bs       = bsr_->block_size();
            const ptrdiff_t n_blocks = bsr_->row_ptr->size() - 1;
            const int       bmat     = bs * bs;

            inv_diag_ = create_host_buffer<T>(n_blocks * bmat);

            const R* const rowptr = bsr_->row_ptr->data();
            const C* const colidx = bsr_->col_idx->data();
            const T* const values = bsr_->values->data();
            T* const       invd   = inv_diag_->data();

            for (ptrdiff_t i = 0; i < n_blocks; i++) {
                const R begin = rowptr[i];
                const R end   = rowptr[i + 1];

                R diag_k = static_cast<R>(-1);
                for (R k = begin; k < end; k++) {
                    if (static_cast<ptrdiff_t>(colidx[k]) == i) {
                        diag_k = k;
                        break;
                    }
                }

                if (diag_k < 0) {
                    SFEM_ERROR("BSRBlockGaussSeidel: missing diagonal block at row %td\n", i);
                }

                T* const out = &invd[i * bmat];
                bsr_bgs_private_::invert_block(bs, &values[diag_k * bmat], out);
                for (int e = 0; e < bmat; e++) {
                    out[e] *= relaxation_;
                }
            }
        }

        void sweep_row(const ptrdiff_t i,
                       const int       bs,
                       const R         begin,
                       const R         end,
                       const C* const  colidx,
                       const T* const  values,
                       const T* const  invd,
                       const T* const  b,
                       T* const        x) const {
            T r[MAX_BLOCK_SIZE];
            for (int d = 0; d < bs; d++) {
                r[d] = b[i * bs + d];
            }

            for (R k = begin; k < end; k++) {
                const ptrdiff_t j = colidx[k];
                if (j == i) {
                    continue;
                }

                const T* const aij = &values[k * bs * bs];
                const T* const xj  = &x[j * bs];
                for (int d1 = 0; d1 < bs; d1++) {
                    for (int d2 = 0; d2 < bs; d2++) {
                        r[d1] -= aij[d1 * bs + d2] * xj[d2];
                    }
                }
            }

            bsr_bgs_private_::matvec_set(bs, &invd[i * bs * bs], r, &x[i * bs]);
        }

        void forward_sweep(const T* const b, T* const x) const {
            const int       bs       = bsr_->block_size();
            const ptrdiff_t n_blocks = bsr_->row_ptr->size() - 1;
            const R* const  rowptr   = bsr_->row_ptr->data();
            const C* const  colidx   = bsr_->col_idx->data();
            const T* const  values   = bsr_->values->data();
            const T* const  invd     = inv_diag_->data();

            for (ptrdiff_t i = 0; i < n_blocks; i++) {
                sweep_row(i, bs, rowptr[i], rowptr[i + 1], colidx, values, invd, b, x);
            }
        }

        void backward_sweep(const T* const b, T* const x) const {
            const int       bs       = bsr_->block_size();
            const ptrdiff_t n_blocks = bsr_->row_ptr->size() - 1;
            const R* const  rowptr   = bsr_->row_ptr->data();
            const C* const  colidx   = bsr_->col_idx->data();
            const T* const  values   = bsr_->values->data();
            const T* const  invd     = inv_diag_->data();

            for (ptrdiff_t i = n_blocks - 1; i >= 0; i--) {
                sweep_row(i, bs, rowptr[i], rowptr[i + 1], colidx, values, invd, b, x);
            }
        }

        std::shared_ptr<Matrix>  bsr_;
        SharedBuffer<T>          inv_diag_;
        SharedBuffer<T>          workspace_;
        std::shared_ptr<BLAS<T>> blas_;
        int                      max_it_{1};
        bool                     symmetric_{false};
        T                        relaxation_{1};
    };

    template <typename R = count_t, typename C = idx_t, typename T = real_t>
    std::shared_ptr<BSRBlockGaussSeidel<R, C, T>> h_bsr_block_gauss_seidel(
            const std::shared_ptr<BSR<R, C, T, T>>& bsr) {
        return std::make_shared<BSRBlockGaussSeidel<R, C, T>>(bsr);
    }

}  // namespace sfem

#endif  // SFEM_BSR_BLOCK_GAUSS_SEIDEL_HPP

