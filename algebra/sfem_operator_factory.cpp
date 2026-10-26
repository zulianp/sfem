#include "sfem_operator_factory.hpp"

#include "sfem_blas_factory.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_BLAS_cuda.hpp"
#include "smesh_device_buffer.hpp"
#endif

namespace sfem {

    template <typename T>
    std::shared_ptr<Operator<T>> diag_op(const SharedBuffer<T> &diagonal_scaling, const ExecutionContext &context) {
        const std::ptrdiff_t n = diagonal_scaling->size();

        // // FIXME make simpler version
        auto impl = sfem::blas<T>(context);
        return sfem::make_op<T>(
                n,
                n,
                [n, diagonal_scaling, impl](const T *const x, T *const y) {
                    auto d = diagonal_scaling->data();
                    impl->xypaz(n, x, d, 1, y);
                },
                context);
    }

    template <typename T>
    std::shared_ptr<Operator<T>> create_inverse_diagonal_scaling(const SharedBuffer<T> &diag, const ExecutionContext &context) {
        SFEM_UNUSED(context);
#ifdef SFEM_ENABLE_CUDA
        if (context.space() == EXECUTION_SPACE_DEVICE) {
            auto d_diag = smesh::to_device(diag);
            return sfem::make_op<T>(
                    d_diag->size(),
                    d_diag->size(),
                    [=](const T *const x, T *const y) {
                        auto d = d_diag->data();
                        // FIXME (only supports real_t)
                        d_ediv(d_diag->size(), x, d, y, context.stream());
                    },
                    context);
        }
#endif  // SFEM_ENABLE_CUDA

        return sfem::make_op<T>(
                diag->size(),
                diag->size(),
                [=](const T *const x, T *const y) {
                    auto d = diag->data();

#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < (ptrdiff_t)diag->size(); ++i) {
                        y[i] = x[i] / d[i];
                    }
                },
                EXECUTION_SPACE_HOST);
    }

    real_t residual(Operator<real_t> &op, const real_t *const rhs, const real_t *const x, real_t *const r) {
        auto b = sfem::blas<real_t>(op.execution_context());
        b->zeros(op.rows(), r);
        op.apply(x, r);
        b->axpby(op.rows(), 1, rhs, -1, r);
        return b->norm2(op.rows(), r);
    }

    template std::shared_ptr<Operator<float>>  diag_op<float>(const SharedBuffer<float> &, const ExecutionContext &);
    template std::shared_ptr<Operator<double>> diag_op<double>(const SharedBuffer<double> &, const ExecutionContext &);

    template std::shared_ptr<Operator<real_t>> create_inverse_diagonal_scaling<real_t>(const SharedBuffer<real_t> &,
                                                                                       const ExecutionContext &);

}  // namespace sfem
