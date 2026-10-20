#include "sfem_BSRTangent.hpp"

#include "sfem_API.hpp"
#include "sfem_BSR.hpp"
#include "sfem_Function.hpp"
#include "sfem_ParallelOperator.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_BSR.hpp"
#endif

#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sfem {

    class BSRTangent::Impl {
    public:
        std::shared_ptr<Function> f;
        ExecutionSpace            es{EXECUTION_SPACE_HOST};
        std::shared_ptr<CRSGraph> graph;  // in `es`
        int                       block_size{0};
        SharedBuffer<real_t>      values;
        // Rows of the nodes this rank owns, a prefix of the graph's: under MPI the only complete
        // ones (the assembly runs over the aura) and the only ones the product computes.
        ptrdiff_t                 owned_rows{0};
        bool                      distributed{false};
        // Under MPI, where each owned row's entries of owned columns end: the row's local diagonal
        // block is [rowptr, split), its block of ghost and aura columns [split, rowptr + 1). The
        // two views share the assembled values, which the obstacle and the shifts change in place.
        SharedBuffer<count_t>     split;  // in `es`
#ifdef SFEM_ENABLE_CUDA
        // The device's local diagonal block runs here, beside MPI's copies of the ghost values.
        std::shared_ptr<DeviceSideStream> side;
#endif
    };

    // Where each of the first `rows` rows' owned columns (< rows) end; null if a row has an owned
    // column after a non-owned one, which would make its owned entries more than one range.
    static SharedBuffer<count_t> owned_column_split(const ptrdiff_t                  rows,
                                                    const count_t *const SFEM_RESTRICT rowptr,
                                                    const idx_t *const SFEM_RESTRICT   colidx) {
        auto     split = create_host_buffer<count_t>(rows);
        count_t *s     = split->data();
        bool     ok    = true;
#pragma omp parallel for reduction(&& : ok)
        for (ptrdiff_t i = 0; i < rows; ++i) {
            count_t k = rowptr[i];
            while (k < rowptr[i + 1] && colidx[k] < rows) ++k;
            s[i] = k;
            for (count_t j = k; j < rowptr[i + 1]; ++j) ok = ok && colidx[j] >= rows;
        }
        return ok ? split : nullptr;
    }

    BSRTangent::BSRTangent() : impl_(std::make_unique<Impl>()) {}
    BSRTangent::~BSRTangent() = default;

    std::shared_ptr<BSRTangent> BSRTangent::create(const std::shared_ptr<Function> &f, const ExecutionSpace es) {
        SFEM_TRACE_SCOPE("BSRTangent::create");

        std::shared_ptr<BSRTangent> ret(new BSRTangent());
        auto                       &impl = *ret->impl_;
        impl.f                           = f;
        impl.es                          = es;
        impl.block_size                  = f->space()->block_size();
        impl.graph                       = f->space()->element_node_graph();
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            impl.graph = smesh::to_device(impl.graph);
        }
#endif
        impl.values = create_buffer<real_t>(impl.graph->nnz() * impl.block_size * impl.block_size, es);
        impl.owned_rows  = f->space()->n_owned_dofs() / impl.block_size;
        impl.distributed = f->space()->mesh_ptr()->comm()->size() > 1;
        if (impl.distributed) {
            auto host  = f->space()->element_node_graph();
            impl.split = owned_column_split(impl.owned_rows, host->rowptr()->data(), host->colidx()->data());
            if (!impl.split) {
                SFEM_ERROR("BSRTangent: a row's owned columns are not one range (unsorted graph)\n");
                return nullptr;
            }
#ifdef SFEM_ENABLE_CUDA
            if (es == EXECUTION_SPACE_DEVICE) impl.split = smesh::to_device(impl.split);
#endif
        }
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
        if (impl.es == EXECUTION_SPACE_DEVICE && storage_bytes != (int)sizeof(real_t)) {
            SFEM_ERROR("BSRTangent::op: reduced-precision storage is host only\n");
            return nullptr;
        }

        // The product of `rows` block rows against every local column.
        auto product = [&](const ptrdiff_t rows) -> std::shared_ptr<Operator<real_t>> {
#ifdef SFEM_ENABLE_CUDA
            if (impl.es == EXECUTION_SPACE_DEVICE) {
                return compose_constraints_op(
                        impl.f,
                        d_bsr_spmv(rows, nrows, impl.block_size, impl.graph->rowptr(), impl.graph->colidx(), impl.values, (real_t)1));
            }
#endif
            return h_bsr_spmv<count_t, idx_t, real_t>(
                    rows, nrows, impl.block_size, impl.graph->rowptr(), impl.graph->colidx(), impl.values, (real_t)1);
        };

        if (impl.distributed) {
            // The owned rows against every local column, y += A x, as two products: the local
            // diagonal block (owned columns) while x's ghost and aura values travel from their
            // owners, then the block of those columns. x and y are space-sized; the owned prefix
            // of y is the result.
            if (storage_bytes != (int)sizeof(real_t)) {
                SFEM_ERROR("BSRTangent::op: a distributed tangent stores full precision\n");
                return nullptr;
            }
            auto                 space  = impl.f->space();
            auto                 f      = impl.f;
            const ptrdiff_t      owned  = impl.owned_rows * impl.block_size;
            const ptrdiff_t      rows   = impl.owned_rows;
            const int            bs     = impl.block_size;
            const ExecutionSpace es     = impl.es;
            auto                 rowptr = impl.graph->rowptr();
            auto                 colidx = impl.graph->colidx();
            auto                 values = impl.values;
            auto                 split  = impl.split;
#ifdef SFEM_ENABLE_CUDA
            if (es == EXECUTION_SPACE_DEVICE && !impl.side) impl.side = DeviceSideStream::create();
            auto side = impl.side;
#endif
            return make_parallel_op<real_t>(
                    space->mesh_ptr()->comm(),
                    owned,
                    owned,
                    space->n_dofs(),
                    space->n_dofs(),
                    [=](const real_t *const x, real_t *const y) {
                        real_t *const  xs = const_cast<real_t *>(x);
                        const count_t *rp = rowptr->data(), *sp = split->data();
                        const idx_t   *ci = colidx->data();
                        const real_t  *v  = values->data();
                        if (space->synchronize_begin(xs, es) != SFEM_SUCCESS) return;
#ifdef SFEM_ENABLE_CUDA
                        if (es == EXECUTION_SPACE_DEVICE) {
                            // Launched before the wait, on a stream of its own: on the default
                            // stream it would hold back MPI's copies until it ended (measured on
                            // GH200, Cray MPICH). It reads only owned columns, which the unpack of
                            // the aura does not touch.
                            side->wait_for_default();
                            d_bsr_spmv_ranges<count_t, idx_t, real_t, real_t>(rows, bs, rp, sp, ci, v, 1, x, y, side->stream());
                            space->synchronize_end(es);
                            side->join_default();
                            d_bsr_spmv_ranges<count_t, idx_t, real_t, real_t>(rows, bs, sp, rp + 1, ci, v, 1, x, y);
                            f->copy_constrained_dofs(x, y);
                            return;
                        }
#endif
                        // One thread waits for the messages while the others take the diagonal
                        // block, row ranges of it each.
                        int threads = 1;
#ifdef _OPENMP
                        threads = omp_get_max_threads();
#endif
                        if (threads > 1) {
#pragma omp parallel num_threads(threads)
                            {
#ifdef _OPENMP
                                const int tid = omp_get_thread_num(), workers = omp_get_num_threads() - 1;
#else
                                const int tid = 0, workers = 0;
#endif
                                if (tid == 0) {
                                    space->synchronize_end(es);
                                } else {
                                    const ptrdiff_t chunk = (rows + workers - 1) / workers;
                                    const ptrdiff_t begin = std::min(rows, (tid - 1) * chunk);
                                    const ptrdiff_t end   = std::min(rows, begin + chunk);
                                    bsr_spmv_ranges<count_t, idx_t, real_t, real_t>(
                                            end - begin, bs, bs, rp + begin, sp + begin, ci, v, 1, x, y + begin * bs);
                                }
                            }
                        } else {
                            bsr_spmv_ranges<count_t, idx_t, real_t, real_t>(rows, bs, bs, rp, sp, ci, v, 1, x, y);
                            space->synchronize_end(es);
                        }
                        bsr_spmv_ranges<count_t, idx_t, real_t, real_t>(rows, bs, bs, sp, rp + 1, ci, v, 1, x, y);
                    },
                    impl.es);
        }
        if (impl.es == EXECUTION_SPACE_DEVICE) return product(nrows);

        // On the host the constraints write identity rows into the assembled values
        // (Function::hessian_bsr), so the product is the constrained operator as it stands:
        // replacing the constrained entries after each product, as the device does, would only
        // repeat them, once per Krylov iteration.
        switch (storage_bytes) {
            case 2:
                return h_bsr_spmv<count_t, idx_t, half_t, real_t>(
                        nrows, nrows, impl.block_size, impl.graph->rowptr(), impl.graph->colidx(), astype<half_t>(impl.values), (real_t)1);
#if !SFEM_REAL_T_IS_FLOAT32
            case 4:
                return h_bsr_spmv<count_t, idx_t, float, real_t>(
                        nrows, nrows, impl.block_size, impl.graph->rowptr(), impl.graph->colidx(), astype<float>(impl.values), (real_t)1);
#endif
            default:
                return product(nrows);
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
        // Over the owned rows, then the ranks, when distributed.
        const ptrdiff_t rows = impl.distributed ? impl.owned_rows : block_rows();
        real_t          local;
#ifdef SFEM_ENABLE_CUDA
        if (impl.es == EXECUTION_SPACE_DEVICE) {
            local = d_bsr_max_abs_row_sum<count_t, idx_t, real_t, real_t>(rows, impl.block_size, rowptr(), colidx(), impl.values->data());
        } else
#endif
        {
            local = bsr_max_abs_row_sum<count_t, idx_t, real_t, real_t>(rows, impl.block_size, rowptr(), colidx(), impl.values->data());
        }
        return impl.distributed ? impl.f->space()->mesh_ptr()->comm()->max(local) : local;
    }

    int BSRTangent::shift_diagonal(const real_t factor, const real_t *const reference) {
        SFEM_TRACE_SCOPE("BSRTangent::shift_diagonal");

        auto &impl = *impl_;
#ifdef SFEM_ENABLE_CUDA
        if (impl.es == EXECUTION_SPACE_DEVICE) {
            return d_bsr_shift_diagonal<count_t, idx_t, real_t>(block_rows(), impl.block_size, rowptr(), colidx(), factor, reference, impl.values->data());
        }
#endif
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
