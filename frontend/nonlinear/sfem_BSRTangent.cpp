#include "sfem_BSRTangent.hpp"

#include "sfem_API.hpp"
#include "sfem_BSR.hpp"
#include "sfem_Function.hpp"
#include "sfem_ParallelOperator.hpp"
#include "sfem_TangentValues.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_BSR.hpp"
#endif

#include <algorithm>
#include <cassert>

namespace sfem {

    class BSRTangent::Impl {
    public:
        std::shared_ptr<Function> f;
        ExecutionContext          context;
        std::shared_ptr<CRSGraph> graph;  // in `context`
        int                       block_size{0};
        // The values: in real_t, or in float with storage_bytes 4 (one of the two is set).
        SharedBuffer<real_t>      values;
        SharedBuffer<float>       values32;
        int                       storage_bytes{(int)sizeof(real_t)};
        // Rows of the nodes this rank owns, a prefix of the graph's: under MPI the only complete
        // ones (the assembly runs over the aura) and the only ones the product computes.
        ptrdiff_t                 owned_rows{0};
        bool                      distributed{false};
        // Under MPI, where each owned row's entries of owned columns end: the row's local diagonal
        // block is [rowptr, split), its block of ghost and aura columns [split, rowptr + 1). The
        // two views share the assembled values, which the obstacle and the shifts change in place.
        SharedBuffer<count_t> split;  // in `context`
#ifdef SFEM_ENABLE_CUDA
        // The device's local diagonal block runs here, beside MPI's copies of the ghost values.
        std::shared_ptr<DeviceSideStream> side;
#endif

        // f(the values' buffer), in the type they are stored in.
        template <typename F>
        auto visit(F &&f) const {
            return values32 ? f(values32) : f(values);
        }
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

    std::shared_ptr<BSRTangent> BSRTangent::create(const std::shared_ptr<Function> &f,
                                                   const ExecutionContext          &context,
                                                   const int                        storage_bytes) {
        SFEM_TRACE_SCOPE("BSRTangent::create");

        const ExecutionSpace es          = context.space();
        const bool distributed = f->space()->mesh_ptr()->comm()->size() > 1;
        if (storage_bytes != (int)sizeof(real_t) && storage_bytes != 4 && storage_bytes != 2) {
            SFEM_ERROR("BSRTangent: storage_bytes %d (2, 4 or %d)\n", storage_bytes, (int)sizeof(real_t));
            return nullptr;
        }
        if (storage_bytes == 2 && (es == EXECUTION_SPACE_DEVICE || distributed)) {
            SFEM_ERROR("BSRTangent: half-precision storage is host only, on one process\n");
            return nullptr;
        }

        std::shared_ptr<BSRTangent> ret(new BSRTangent());
        auto                       &impl = *ret->impl_;
        impl.f                           = f;
        impl.context                     = context;
        impl.block_size                  = f->space()->block_size();
        impl.storage_bytes               = storage_bytes;
        impl.graph                       = f->space()->element_node_graph();
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            impl.graph = smesh::to_device(impl.graph);
        }
#endif
        const ptrdiff_t nvalues = impl.graph->nnz() * impl.block_size * impl.block_size;
        if (storage_bytes == 4 && storage_bytes != (int)sizeof(real_t)) {
            impl.values32 = create_buffer<float>(nvalues, es);
        } else {
            impl.values = create_buffer<real_t>(nvalues, es);
        }
        impl.owned_rows  = f->space()->n_owned_dofs() / impl.block_size;
        impl.distributed = distributed;
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
        impl.visit([&](const auto &values) {
            using V = std::remove_pointer_t<decltype(values->data())>;
            blas<V>(impl.context)->zeros(values->size(), values->data());
            return SFEM_SUCCESS;
        });
        return impl.f->hessian_bsr(x, impl.graph->rowptr()->data(), impl.graph->colidx()->data(), values_type(), values());
    }

    std::shared_ptr<Operator<real_t>> BSRTangent::op(const real_t scale_output) const {
        return impl_->visit([&](const auto &stored) { return op(stored, scale_output); });
    }

    template <typename V>
    std::shared_ptr<Operator<real_t>> BSRTangent::op(const SharedBuffer<V> &stored, const real_t scale_output) const {
        auto           &impl  = *impl_;
        const ptrdiff_t nrows = impl.graph->n_nodes();

#ifdef SFEM_ENABLE_CUDA
        if (impl.context.space() == EXECUTION_SPACE_DEVICE) {
            if (impl.distributed) {
                // The owned rows against every local column, y += A x, as two products: the local
                // diagonal block (owned columns) while x's ghost and aura values travel from their
                // owners, then the block of those columns. x and y are space-sized; the owned prefix
                // of y is the result.
                auto            space  = impl.f->space();
                auto            f      = impl.f;
                const ptrdiff_t owned  = impl.owned_rows * impl.block_size;
                const ptrdiff_t rows   = impl.owned_rows;
                const int       bs     = impl.block_size;
                auto            rowptr = impl.graph->rowptr();
                auto            colidx = impl.graph->colidx();
                auto            values = stored;
                auto            split  = impl.split;
                if (!impl.side) impl.side = DeviceSideStream::create();
                auto side    = impl.side;
                auto context = impl.context;
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
                            const V       *v  = values->data();
                            // Launched before the wait, on a stream of its own: on the default
                            // stream it would hold back MPI's copies until it ended (measured on
                            // GH200, Cray MPICH), and follow NCCL's messages, queued there. It
                            // reads only owned columns, which the unpack of the aura does not
                            // touch, so it waits only for what came before the gather.
                            // All of it on the tangent's stream, the exchange's copies included.
                            void *const main = context.stream();
                            side->fork(main);
                            if (space->synchronize_begin(xs, context) != SFEM_SUCCESS) return;
                            d_bsr_spmv_ranges<count_t, idx_t, V, real_t>(rows, bs, rp, sp, ci, v, scale_output, x, y, side->stream());
                            space->synchronize_end(context);
                            side->join(main);
                            d_bsr_spmv_ranges<count_t, idx_t, V, real_t>(rows, bs, sp, rp + 1, ci, v, 1, x, y, main);
                            f->copy_constrained_dofs(x, y);
                        },
                        context);
            }
            if constexpr (std::is_same_v<V, real_t>) {
                return compose_constraints_op(impl.f,
                                              d_bsr_spmv(nrows,
                                                         nrows,
                                                         impl.block_size,
                                                         impl.graph->rowptr(),
                                                         impl.graph->colidx(),
                                                         stored,
                                                         scale_output,
                                                         impl.context));
            } else {
                // cuSPARSE multiplies no single-precision matrix into real_t vectors: the
                // library's own kernel, as the distributed product uses.
                auto      rowptr = impl.graph->rowptr();
                auto      colidx = impl.graph->colidx();
                const int bs     = impl.block_size;
                void     *stream = impl.context.stream();
                return compose_constraints_op(impl.f,
                                              make_op<real_t>(
                                                      nrows * bs,
                                                      nrows * bs,
                                                      [=](const real_t *const x, real_t *const y) {
                                                          d_bsr_spmv_native<count_t, idx_t, V, real_t>(nrows,
                                                                                                       bs,
                                                                                                       rowptr->data(),
                                                                                                       colidx->data(),
                                                                                                       stored->data(),
                                                                                                       scale_output,
                                                                                                       x,
                                                                                                       y,
                                                                                                       stream);
                                                      },
                                                      impl.context));
            }
        }
#endif

        // On the host one region, each thread the block rows of its share (TeamParallelOperator),
        // y = scale_output y + A x from `values` (the stored ones, or a copy). Under MPI the owned
        // rows against every local column, as two products: the owned columns while x's ghost and
        // aura values travel from their owners (thread 0 sends, takes its rows, then waits), then,
        // once every thread sees them, the columns of those. x must be complete when the threads
        // enter (the caller's barrier): thread 0 sends from it.
        // The constraints write identity rows into the assembled values (Function::hessian_bsr),
        // so the product is the constrained operator as it stands: replacing the constrained
        // entries after each product, as the device does, would only repeat them, once per Krylov
        // iteration.
        auto host_op = [&](const auto &values) -> std::shared_ptr<Operator<real_t>> {
            auto            space  = impl.f->space();
            const int       bs     = impl.block_size;
            const bool      halo   = impl.distributed;
            const ptrdiff_t rows   = halo ? impl.owned_rows : nrows;
            auto            rowptr = impl.graph->rowptr();
            auto            colidx = impl.graph->colidx();
            auto            split  = impl.split;
            return make_team_op<real_t>(space->mesh_ptr()->comm(),
                                        rows * bs,
                                        rows * bs,
                                        halo ? space->n_dofs() : rows * bs,
                                        halo ? space->n_dofs() : nrows * bs,
                                        false,
                                        [=](const Team &team, const real_t *const x, real_t *const y) {
                                            const count_t  *rp    = rowptr->data();
                                            const idx_t    *ci    = colidx->data();
                                            const auto     *v     = values->data();
                                            assert(team.begin % bs == 0 && team.end % bs == 0);
                                            const ptrdiff_t first = team.begin / bs, end = team.end / bs;
                                            if (!halo) {
                                                bsr_spmv_rows(first, end, bs, bs, rp, rp + 1, ci, v, scale_output, x, y);
                                                return;
                                            }
                                            const count_t *sp = split->data();
                                            if (team.tid == 0 &&
                                                space->synchronize_begin(const_cast<real_t *>(x), EXECUTION_SPACE_HOST) != SFEM_SUCCESS) {
                                                SFEM_ERROR("BSRTangent: the gather of x did not start\n");
                                            }
                                            bsr_spmv_rows(first, end, bs, bs, rp, sp, ci, v, scale_output, x, y);
                                            if (team.tid == 0 && space->synchronize_end(EXECUTION_SPACE_HOST) != SFEM_SUCCESS) {
                                                SFEM_ERROR("BSRTangent: the gather of x did not complete\n");
                                            }
                                            team.barrier();
                                            bsr_spmv_rows(first, end, bs, bs, sp, rp + 1, ci, v, real_t(1), x, y);
                                        });
        };
        if (impl.storage_bytes == 2) return host_op(astype<half_t>(stored));
        return host_op(stored);
    }

    int BSRTangent::diagonal(real_t *const out) const {
        SFEM_TRACE_SCOPE("BSRTangent::diagonal");

        auto &impl = *impl_;
        return impl.visit([&](const auto &values) {
#ifdef SFEM_ENABLE_CUDA
            if (impl.context.space() == EXECUTION_SPACE_DEVICE) {
                return d_bsr_extract_diagonal(block_rows(), impl.block_size, rowptr(), colidx(), values->data(), out);
            }
#endif
            bsr_extract_diagonal(block_rows(), impl.block_size, rowptr(), colidx(), values->data(), out);
            return SFEM_SUCCESS;
        });
    }

    real_t BSRTangent::max_abs_row_sum(const real_t *const scale) const {
        SFEM_TRACE_SCOPE("BSRTangent::max_abs_row_sum");

        auto &impl = *impl_;
        // Over the owned rows, then the ranks, when distributed.
        const ptrdiff_t rows  = impl.distributed ? impl.owned_rows : block_rows();
        const real_t    local = impl.visit([&](const auto &values) -> real_t {
            using V = std::remove_pointer_t<decltype(values->data())>;
#ifdef SFEM_ENABLE_CUDA
            if (impl.context.space() == EXECUTION_SPACE_DEVICE) {
                return d_bsr_max_abs_row_sum<count_t, idx_t, V, real_t>(rows, impl.block_size, rowptr(), colidx(), values->data(), scale);
            }
#endif
            return bsr_max_abs_row_sum<count_t, idx_t, V, real_t>(rows, impl.block_size, rowptr(), colidx(), values->data(), scale);
        });
        return impl.distributed ? impl.f->space()->mesh_ptr()->comm()->max(local) : local;
    }

    int BSRTangent::shift_diagonal(const real_t factor, const real_t *const reference) {
        SFEM_TRACE_SCOPE("BSRTangent::shift_diagonal");

        auto &impl = *impl_;
        return impl.visit([&](const auto &values) {
#ifdef SFEM_ENABLE_CUDA
            if (impl.context.space() == EXECUTION_SPACE_DEVICE) {
                return d_bsr_shift_diagonal(block_rows(), impl.block_size, rowptr(), colidx(), factor, reference, values->data());
            }
#endif
            bsr_shift_diagonal(block_rows(), impl.block_size, rowptr(), colidx(), factor, reference, values->data());
            return SFEM_SUCCESS;
        });
    }

    ptrdiff_t      BSRTangent::block_rows() const { return impl_->graph->n_nodes(); }
    int            BSRTangent::block_size() const { return impl_->block_size; }
    const count_t *BSRTangent::rowptr() const { return impl_->graph->rowptr()->data(); }
    const idx_t   *BSRTangent::colidx() const { return impl_->graph->colidx()->data(); }
    void *BSRTangent::values() {
        return impl_->visit([](const auto &values) -> void * { return values->data(); });
    }
    smesh::PrimitiveType BSRTangent::values_type() const {
        return impl_->visit([](const auto &values) {
            return smesh::TypeToEnum<std::remove_pointer_t<decltype(values->data())>>::value();
        });
    }
    ExecutionSpace BSRTangent::execution_space() const { return impl_->context.space(); }

}  // namespace sfem
