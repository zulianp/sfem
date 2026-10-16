#include "sfem_BSRTangent.hpp"

#include "sfem_API.hpp"
#include "sfem_BSR.hpp"
#include "sfem_Function.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_BSR.hpp"
#endif

namespace sfem {

    class BSRTangent::Impl {
    public:
        std::shared_ptr<Function> f;
        ExecutionSpace            es{EXECUTION_SPACE_HOST};
        std::shared_ptr<CRSGraph> graph;  // in `es`
        int                       block_size{0};
        SharedBuffer<real_t>      values;
    };

    BSRTangent::BSRTangent() : impl_(std::make_unique<Impl>()) {}
    BSRTangent::~BSRTangent() = default;

    std::shared_ptr<BSRTangent> BSRTangent::create(const std::shared_ptr<Function> &f, const ExecutionSpace es) {
        SFEM_TRACE_SCOPE("BSRTangent::create");

        std::shared_ptr<BSRTangent> ret(new BSRTangent());
        auto                       &impl = *ret->impl_;
        impl.f                           = f;
        impl.es                          = es;
        impl.block_size                  = f->space()->block_size();
        impl.graph                       = f->space()->node_to_node_graph();
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            impl.graph = smesh::to_device(impl.graph);
        }
#endif
        impl.values = create_buffer<real_t>(impl.graph->nnz() * impl.block_size * impl.block_size, es);
        return ret;
    }

    int BSRTangent::assemble(const real_t *const x) {
        SFEM_TRACE_SCOPE("BSRTangent::assemble");

        auto &impl = *impl_;
        blas<real_t>(impl.es)->zeros(impl.values->size(), impl.values->data());
        return impl.f->hessian_bsr(x, impl.graph->rowptr()->data(), impl.graph->colidx()->data(), impl.values->data());
    }

    std::shared_ptr<Operator<real_t>> BSRTangent::op(const int storage_bytes) const {
        auto           &impl  = *impl_;
        const ptrdiff_t nrows = impl.graph->n_nodes();

#ifdef SFEM_ENABLE_CUDA
        if (impl.es == EXECUTION_SPACE_DEVICE) {
            if (storage_bytes != (int)sizeof(real_t)) {
                SFEM_ERROR("BSRTangent::op: reduced-precision storage is host only\n");
                return nullptr;
            }
            return compose_constraints_op(
                    impl.f,
                    d_bsr_spmv(nrows, nrows, impl.block_size, impl.graph->rowptr(), impl.graph->colidx(), impl.values, (real_t)1));
        }
#endif

        switch (storage_bytes) {
            case 2:
                return compose_constraints_op(impl.f,
                                              h_bsr_spmv<count_t, idx_t, half_t, real_t>(nrows,
                                                                                         nrows,
                                                                                         impl.block_size,
                                                                                         impl.graph->rowptr(),
                                                                                         impl.graph->colidx(),
                                                                                         astype<half_t>(impl.values),
                                                                                         (real_t)1));
#if !SFEM_REAL_T_IS_FLOAT32
            case 4:
                return compose_constraints_op(impl.f,
                                              h_bsr_spmv<count_t, idx_t, float, real_t>(nrows,
                                                                                        nrows,
                                                                                        impl.block_size,
                                                                                        impl.graph->rowptr(),
                                                                                        impl.graph->colidx(),
                                                                                        astype<float>(impl.values),
                                                                                        (real_t)1));
#endif
            default:
                return compose_constraints_op(
                        impl.f,
                        h_bsr_spmv<count_t, idx_t, real_t>(
                                nrows, nrows, impl.block_size, impl.graph->rowptr(), impl.graph->colidx(), impl.values, (real_t)1));
        }
    }

    int BSRTangent::diagonal(real_t *const out) const {
        SFEM_TRACE_SCOPE("BSRTangent::diagonal");

        auto &impl = *impl_;
#ifdef SFEM_ENABLE_CUDA
        if (impl.es == EXECUTION_SPACE_DEVICE) {
            return d_bsr_extract_diagonal(block_rows(), impl.block_size, rowptr(), colidx(), impl.values->data(), out);
        }
#endif
        bsr_extract_diagonal(block_rows(), impl.block_size, rowptr(), colidx(), impl.values->data(), out);
        return SFEM_SUCCESS;
    }

    real_t BSRTangent::max_abs_row_sum() const {
        SFEM_TRACE_SCOPE("BSRTangent::max_abs_row_sum");

        auto &impl = *impl_;
        if (impl.es != EXECUTION_SPACE_HOST) {
            SFEM_ERROR("BSRTangent::max_abs_row_sum is host only\n");
            return 0;
        }
        return bsr_max_abs_row_sum<count_t, idx_t, real_t, real_t>(block_rows(), impl.block_size, rowptr(), colidx(), impl.values->data());
    }

    int BSRTangent::shift_diagonal(const real_t factor, const real_t *const reference) {
        SFEM_TRACE_SCOPE("BSRTangent::shift_diagonal");

        auto &impl = *impl_;
        if (impl.es != EXECUTION_SPACE_HOST) {
            SFEM_ERROR("BSRTangent::shift_diagonal is host only\n");
            return SFEM_FAILURE;
        }
        bsr_shift_diagonal(block_rows(), impl.block_size, rowptr(), colidx(), factor, reference, impl.values->data());
        return SFEM_SUCCESS;
    }

    ptrdiff_t      BSRTangent::block_rows() const { return impl_->graph->n_nodes(); }
    int            BSRTangent::block_size() const { return impl_->block_size; }
    const count_t *BSRTangent::rowptr() const { return impl_->graph->rowptr()->data(); }
    const idx_t   *BSRTangent::colidx() const { return impl_->graph->colidx()->data(); }
    real_t        *BSRTangent::values() { return impl_->values->data(); }
    ExecutionSpace BSRTangent::execution_space() const { return impl_->es; }

}  // namespace sfem
