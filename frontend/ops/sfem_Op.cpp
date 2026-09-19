#include "sfem_Op.hpp"

#include "sfem_API.hpp"

#include <algorithm>
#include <vector>

namespace sfem {

    std::shared_ptr<Op> no_op() {
        return std::make_shared<NoOp>();
    }

    real_t squared_residual_norm(const ptrdiff_t ndofs, const int block_size, const real_t *const values) {
        // Deterministic by construction, and that is the point rather than a
        // detail: a line search compares these numbers across runs, and a flat
        // `reduction(+ : acc)` over the dofs would add them in whatever order
        // the threads happened to finish.  Nodes are grouped into fixed chunks,
        // each chunk is summed by one thread, and the chunk partials are added
        // in index order -- so the answer does not depend on the thread count.
        const bool      blocked    = block_size > 0 && (ndofs % block_size) == 0;
        const ptrdiff_t nnodes     = blocked ? ndofs / block_size : ndofs;
        const int       components = blocked ? block_size : 1;

        static constexpr ptrdiff_t nodes_per_chunk = 4096;
        const ptrdiff_t            n_chunks = (nnodes + nodes_per_chunk - 1) / nodes_per_chunk;
        std::vector<real_t>        partial(n_chunks > 0 ? n_chunks : 1, 0);

#pragma omp parallel for schedule(static)
        for (ptrdiff_t chunk = 0; chunk < n_chunks; ++chunk) {
            const ptrdiff_t begin = chunk * nodes_per_chunk;
            const ptrdiff_t end   = std::min(begin + nodes_per_chunk, nnodes);
            real_t          sum   = 0;
            for (ptrdiff_t node = begin; node < end; ++node) {
                for (int component = 0; component < components; ++component) {
                    const real_t r = values[node * components + component];
                    sum += r * r;
                }
            }
            partial[chunk] = sum;
        }

        real_t acc = 0;
        for (ptrdiff_t chunk = 0; chunk < n_chunks; ++chunk) {
            acc += partial[chunk];
        }
        return acc;
    }

    int Op::residual_merit_steps(const real_t *const x,
                                 const real_t *const h,
                                 const int           nsteps,
                                 const real_t *const steps,
                                 const real_t *const accumulator,
                                 real_t *const       out) {
        // The shape this interface exists to replace, kept so an operator that
        // has no sampling kernel yet still answers.  It assembles the whole
        // residual once per trial step, which is what the interface is meant to
        // stop: an implementation worth having does one pass over the mesh with
        // the steps as an inner loop, and is told so here rather than in a
        // commit message.
        const ptrdiff_t ndofs = n_dofs_image();
        if (ndofs <= 0) {
            SFEM_ERROR("%s cannot contract the residual merit: it reports no image\n", name());
            return SFEM_FAILURE;
        }

        auto       blas     = sfem::blas<real_t>(execution_space());
        auto       residual = create_buffer<real_t>(ndofs, execution_space());
        auto       stepped  = create_buffer<real_t>(ndofs, execution_space());
        const auto block    = 1;

        for (int step = 0; step < nsteps; ++step) {
            blas->zeros(ndofs, residual->data());
            blas->copy(ndofs, accumulator, residual->data());
            blas->zaxpby(ndofs, 1, x, steps[step], h, stepped->data());
            if (gradient(stepped->data(), residual->data()) != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }
            if (execution_space() == EXECUTION_SPACE_DEVICE) {
                // `norm2`, not `dot(r, r)`: passing one pointer as both cuBLAS
                // operands aliases the operands, which cuBLAS does not support.
                sfem::device_synchronize();
                const real_t norm = blas->norm2(ndofs, residual->data());
                out[step] += real_t(0.5) * norm * norm;
            } else {
                out[step] += real_t(0.5) * squared_residual_norm(ndofs, block, residual->data());
            }
        }
        return SFEM_SUCCESS;
    }

} // namespace sfem
