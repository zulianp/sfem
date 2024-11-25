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
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};
        BLAS_Tpl<T> blas;
        std::shared_ptr<Buffer<T>> diag;
        std::shared_ptr<Buffer<T>> inv_diag;
        T relaxation_parameter{0.6};

        void default_init() {
            OpenMP_BLAS<T>::build_blas(blas);
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        void set_diag(const std::shared_ptr<Buffer<T>>& d) {
            diag = d;
            inv_diag = create_buffer<T>(d->size(), execution_space());
            blas.copy(diag->size(), diag->data(), inv_diag->data());
            blas.reciprocal(inv_diag->size(), relaxation_parameter, inv_diag->data());
        }

        int shift(const std::shared_ptr<Buffer<T>>&d) override {
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
        ExecutionSpace execution_space() const override { return execution_space_; }
    };

    template <typename T>
    std::shared_ptr<ShiftableJacobi<T>> h_shiftable_jacobi(const std::shared_ptr<Buffer<T>>& diag) {
        auto ret = std::make_shared<ShiftableJacobi<T>>();
        ret->default_init();
        ret->set_diag(diag);
        return ret;
    }
}  // namespace sfem

#endif  // SFEM_SHIFABLE_JACOBI_HPP
