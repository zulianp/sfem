#ifndef SFEM_SHIFABLE_JACOBI_HPP
#define SFEM_SHIFABLE_JACOBI_HPP

#include <cstddef>
#include <memory>

#include "sfem_Buffer.hpp"
#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_openmp_blas.hpp"

#include "sfem_Buffer.hpp"

namespace sfem {
    template <typename T>
    static std::shared_ptr<Buffer<T>> create_buffer(const std::ptrdiff_t n, const MemorySpace es);

    template <typename T>
    class ShiftableJacobi final : public ShiftableOperator<T> {
    public:
        ExecutionSpace             execution_space_{EXECUTION_SPACE_INVALID};
        BLAS_Tpl<T>                blas;
        std::shared_ptr<Buffer<T>> diag;
        std::shared_ptr<Buffer<T>> inv_diag;
        T                          relaxation_parameter{0.3};

        void default_init() {
            OpenMP_BLAS<T>::build_blas(blas);
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        void set_diag(const std::shared_ptr<Buffer<T>>& d) {
            diag     = d;
            inv_diag = create_buffer<T>(d->size(), execution_space());
            blas.copy(diag->size(), diag->data(), inv_diag->data());
            blas.reciprocal(inv_diag->size(), relaxation_parameter, inv_diag->data());
        }

        int shift(const std::shared_ptr<Buffer<T>>& d) override {
            assert(d->size() == diag->size());

            blas.copy(diag->size(), diag->data(), inv_diag->data());
            blas.axpy(diag->size(), 1, d->data(), inv_diag->data());
            blas.reciprocal(inv_diag->size(), relaxation_parameter, inv_diag->data());
            return SFEM_SUCCESS;
        }

        /* Operator */
        int apply(const T* const b, T* const x) override {
            blas.xypaz(inv_diag->size(), inv_diag->data(), b, 1, x);
            return SFEM_SUCCESS;
        }

        inline std::ptrdiff_t rows() const override { return diag->size(); }
        inline std::ptrdiff_t cols() const override { return diag->size(); }
        ExecutionSpace        execution_space() const override { return execution_space_; }
    };

    template <typename T>
    std::shared_ptr<ShiftableJacobi<T>> h_shiftable_jacobi(const std::shared_ptr<Buffer<T>>& diag) {
        auto ret = std::make_shared<ShiftableJacobi<T>>();
        ret->default_init();
        ret->set_diag(diag);
        return ret;
    }

    template <typename T>
    class ShiftableBlockSymJacobi final : public ShiftableOperator<T> {
    public:
        ExecutionSpace                  execution_space_{EXECUTION_SPACE_INVALID};
        BLAS_Tpl<T>                     blas;
        std::shared_ptr<Buffer<T>>      diag;
        std::shared_ptr<Buffer<T>>      inv_diag;
        std::shared_ptr<Buffer<mask_t>> boundary_mask;
        T                               relaxation_parameter{0.3};
        int                             block_size{3};
        bool                            is_symmetric{true};

        void default_init() {
            OpenMP_BLAS<T>::build_blas(blas);
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        void add_sym_diag_to_diag(const std::shared_ptr<Buffer<T>>& in, const std::shared_ptr<Buffer<T>>& out) {
            auto dd  = in->data();
            auto ivd = out->data();
            auto md  = boundary_mask->data();

            const ptrdiff_t nblocks = in->size() / 6;

#pragma omp parallel for
            for (ptrdiff_t b = 0; b < nblocks; b++) {
                auto ivi = &ivd[b * 9];
                auto di  = &dd[b * 6];

                // row 0
                ivi[0] += di[0];
                ivi[1] += di[1];
                ivi[2] += di[2];

                // row 1
                ivi[3] += di[1];
                ivi[4] += di[3];
                ivi[5] += di[4];

                // row 2
                ivi[6] += di[2];
                ivi[7] += di[4];
                ivi[8] += di[5];
            }
        }

        void sym_diag_to_diag(const std::shared_ptr<Buffer<T>>& in, const std::shared_ptr<Buffer<T>>& out) {
            auto dd  = in->data();
            auto ivd = out->data();
            auto md  = boundary_mask->data();

            const ptrdiff_t nblocks = in->size() / 6;

#pragma omp parallel for
            for (ptrdiff_t b = 0; b < nblocks; b++) {
                auto ivi = &ivd[b * 9];
                auto di  = &dd[b * 6];

                // row 0
                ivi[0] = di[0];
                ivi[1] = di[1];
                ivi[2] = di[2];

                // row 1
                ivi[3] = di[1];
                ivi[4] = di[3];
                ivi[5] = di[4];

                // row 2
                ivi[6] = di[2];
                ivi[7] = di[4];
                ivi[8] = di[5];

                if (mask_get(md, b * block_size + 0)) {
                    ivi[0] = 1;
                    ivi[1] = 0;
                    ivi[2] = 0;
                }

                if (mask_get(md, b * block_size + 1)) {
                    ivi[3] = 0;
                    ivi[4] = 1;
                    ivi[5] = 0;
                }

                if (mask_get(md, b * block_size + 2)) {
                    ivi[3] = 0;
                    ivi[4] = 0;
                    ivi[5] = 1;
                }
            }
        }

        static SFEM_INLINE void inverse3(
                // Input
                const T mat_0,
                const T mat_1,
                const T mat_2,
                const T mat_3,
                const T mat_4,
                const T mat_5,
                const T mat_6,
                const T mat_7,
                const T mat_8,
                // Output
                T* const SFEM_RESTRICT mat_inv_0,
                T* const SFEM_RESTRICT mat_inv_1,
                T* const SFEM_RESTRICT mat_inv_2,
                T* const SFEM_RESTRICT mat_inv_3,
                T* const SFEM_RESTRICT mat_inv_4,
                T* const SFEM_RESTRICT mat_inv_5,
                T* const SFEM_RESTRICT mat_inv_6,
                T* const SFEM_RESTRICT mat_inv_7,
                T* const SFEM_RESTRICT mat_inv_8) {
            const T x0 = mat_4 * mat_8;
            const T x1 = mat_5 * mat_7;
            const T x2 = mat_1 * mat_5;
            const T x3 = mat_1 * mat_8;
            const T x4 = mat_2 * mat_4;
            const T x5 = 1.0 / (mat_0 * x0 - mat_0 * x1 + mat_2 * mat_3 * mat_7 - mat_3 * x3 + mat_6 * x2 - mat_6 * x4);
            *mat_inv_0 = x5 * (x0 - x1);
            *mat_inv_1 = x5 * (mat_2 * mat_7 - x3);
            *mat_inv_2 = x5 * (x2 - x4);
            *mat_inv_3 = x5 * (-mat_3 * mat_8 + mat_5 * mat_6);
            *mat_inv_4 = x5 * (mat_0 * mat_8 - mat_2 * mat_6);
            *mat_inv_5 = x5 * (-mat_0 * mat_5 + mat_2 * mat_3);
            *mat_inv_6 = x5 * (mat_3 * mat_7 - mat_4 * mat_6);
            *mat_inv_7 = x5 * (-mat_0 * mat_7 + mat_1 * mat_6);
            *mat_inv_8 = x5 * (mat_0 * mat_4 - mat_1 * mat_3);
        }

        void inplace_invert(const std::shared_ptr<Buffer<T>>& inout) {
            const ptrdiff_t nblocks = inout->size() / 6;
            auto            dd      = inout->data();

#pragma omp parallel for
            for (ptrdiff_t b = 0; b < nblocks; b++) {
                auto ddi = &dd[b * 9];

                inverse3(ddi[0],
                         ddi[1],
                         ddi[2],
                         ddi[3],
                         ddi[4],
                         ddi[5],
                         ddi[6],
                         ddi[7],
                         ddi[8],
                         &ddi[0],
                         &ddi[1],
                         &ddi[2],
                         &ddi[3],
                         &ddi[4],
                         &ddi[5],
                         &ddi[6],
                         &ddi[7],
                         &ddi[8]);
            }
        }

        void set_diag(const std::shared_ptr<Buffer<T>>& d) {
            assert(block_size == 3);
            assert(is_symmetric);
            assert(execution_space_ == EXECUTION_SPACE_HOST);
            assert(boundary_mask);

            const ptrdiff_t nblocks = d->size() / 6;
            diag                    = d;
            inv_diag                = create_buffer<T>(nblocks * block_size * block_size, execution_space());
            sym_diag_to_diag(diag, inv_diag);
            inplace_invert(inv_diag);
            blas.scal(inv_diag->size(), relaxation_parameter, inv_diag->data());
        }

        // TODO shift by sparse block vector
        int shift(const std::shared_ptr<Buffer<T>>& d) override {
            assert(d->size() == diag->size());

            sym_diag_to_diag(diag, inv_diag);
            add_sym_diag_to_diag(d, inv_diag);
            inplace_invert(inv_diag);
            blas.scal(inv_diag->size(), relaxation_parameter, inv_diag->data());
            return SFEM_SUCCESS;
        }

        /* Operator */
        int apply(const T* const x, T* const y) override {
            const ptrdiff_t nblocks = inv_diag->size() / (block_size * block_size);

            const T * const dd = inv_diag->data();
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < nblocks; i++) {
                const T* const xi = &x[i * block_size];
                T* const       yi = &y[i * block_size];
                const T* const di = dd[i * block_size * block_size];

                for (int d1 = 0; d1 > block_size; d1++) {
                    for (int d2 = 0; d2 > block_size; d2++) {
                        yi[d1] += di[d1 * block_size + d2] * xi[d2];
                    }
                }
            }

            return SFEM_SUCCESS;
        }

        inline std::ptrdiff_t rows() const override { return diag->size(); }
        inline std::ptrdiff_t cols() const override { return diag->size(); }
        ExecutionSpace        execution_space() const override { return execution_space_; }
    };
}  // namespace sfem

#endif  // SFEM_SHIFABLE_JACOBI_HPP
