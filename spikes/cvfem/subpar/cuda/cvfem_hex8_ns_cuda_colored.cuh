#ifndef CVFEM_HEX8_NS_CUDA_COLORED_CUH
#define CVFEM_HEX8_NS_CUDA_COLORED_CUH

// Pack-coloured assembly on the device. REMOVED FROM THE DEFAULT BUILD.
//
// Why it is here: pack colouring removes the race between packs, and on the CPU that is
// enough, because there a pack is one thread. On the device a pack is a whole block, so
// the race *within* the pack survives and the kernel is only correct with
// blockDim.x == 1 -- one thread per block, 1.2 MDOF/s, roughly 200x slower than the
// atomic variant it was meant to beat. Measured: blockDim.x > 1 gives a relative error
// of 0.54; blockDim.x == 1 agrees to 7.7e-16.
//
// It is kept, and kept compiling, because it is the executable demonstration of that
// distinction -- the CPU intuition "colouring removes the atomics" carried onto a device
// where it does not hold. Element colouring (cvfem_element_coloring.hpp) is the form that
// does work here, and stays in the main build.
//
// Included by cuda/cvfem_hex8_ns_cuda.cu only under -DCVFEM_ENABLE_SUBPAR. It is not
// self-contained: it expects the enclosing translation unit's anonymous namespace, the
// cvfem_cuda_ctx definition and CVFEM_CUDA_CHECK.

// One block per pack of the current colour, accumulating without atomics.
//
// !! CORRECT ONLY WITH blockDim.x == 1. !!
//
// Pack colouring removes *inter*-pack races: two packs of the same colour share no
// nodes, so they cannot write the same BSR block. It does nothing about *intra*-pack
// races, and on a GPU a pack is a whole block of threads -- two threads working on two
// elements of the same pack do share nodes, and do write the same block.
//
// On the CPU this distinction does not exist because a pack is one thread, which is why
// the colored layout is correct there. Measured here: with blockDim.x > 1 the result is
// wrong by a relative 0.54; with blockDim.x == 1 it agrees to 7.7e-16 and runs at
// 1.2 MDOF/s, roughly 200x slower than the atomic variants.
//
// So this kernel does not test the "is assembly atomic-bound?" question. Testing that
// needs an *element*-level colouring, which is a different and much larger colouring
// problem than the pack colouring the code already builds. Kept as an executable
// demonstration of the distinction, not as a performance path.
template <bool USE_SYMPY>
__global__ void cvfem_hex8_assemble_colored_kernel(
        const ptrdiff_t nelements, const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t color_begin, const double rho, const double mu,
        const ptrdiff_t *const __restrict__ pack_order,
        const int32_t *const __restrict__ elements,
        const int32_t *const __restrict__ slots,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        double *const __restrict__ values) {
    const ptrdiff_t p       = pack_order[color_begin + blockIdx.x];
    const ptrdiff_t e_start = p * n_elements_per_pack;
    const ptrdiff_t e_end   = min(nelements, (p + 1) * n_elements_per_pack);

    for (ptrdiff_t e = e_start + threadIdx.x; e < e_end; e += blockDim.x) {
        double ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES];
        double uz[CVFEM_HEX8_N_NODES], pe[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const ptrdiff_t g     = elements[(ptrdiff_t)a * nelements + e];
            const double *const n = &u[g * CVFEM_CUDA_NF];
            ux[a] = n[0]; uy[a] = n[1]; uz[a] = n[2]; pe[a] = n[3];
        }
        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];

        const int32_t *const es = &slots[e * 64];
        if constexpr (USE_SYMPY)
            cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        else
            cvfem_hex8_ns_upwind_jacobian_add_slots<false>(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        (void)pe;
    }
}

template <bool USE_SYMPY>
int launch_colored_v(cvfem_cuda_ctx *ctx, double rho, double mu, int block, cudaStream_t s) {
    for (int c = 0; c < ctx->n_colors; ++c) {
        const ptrdiff_t b = ctx->h_color_ptr[c], e = ctx->h_color_ptr[c + 1];
        const int       n = (int)(e - b);
        if (n <= 0) continue;
        cvfem_hex8_assemble_colored_kernel<USE_SYMPY><<<n, block, 0, s>>>(
                ctx->nelements, ctx->n_elements_per_pack, b, rho, mu, ctx->pack_order,
                ctx->elements_global, ctx->element_slots, ctx->adj, ctx->det,
                ctx->u, ctx->values);
        CVFEM_CUDA_CHECK(cudaGetLastError());
    }
    return 0;
}

int launch_colored(cvfem_cuda_ctx *ctx, double rho, double mu, int use_sympy,
                   int block_size, cudaStream_t s) {
    if (!ctx->values || !ctx->pack_order) return 1;
    const int block = block_size > 0 ? block_size : 128;
    CVFEM_CUDA_CHECK(cudaMemsetAsync(ctx->values, 0,
                                     (size_t)ctx->nnz * 16 * sizeof(double), s));
    return use_sympy ? launch_colored_v<true>(ctx, rho, mu, block, s)
                     : launch_colored_v<false>(ctx, rho, mu, block, s);
}

extern "C" int cvfem_cuda_coloring_attach(cvfem_cuda_ctx *ctx, int n_colors,
                                          const ptrdiff_t *pack_order,
                                          const ptrdiff_t *color_ptr) {
    ctx->n_colors = n_colors;
    ctx->h_color_ptr.assign(color_ptr, color_ptr + n_colors + 1);
    if (device_dup(&ctx->pack_order, pack_order, (size_t)ctx->n_packs) ||
        device_dup(&ctx->color_ptr, color_ptr, (size_t)n_colors + 1))
        return 1;
    return 0;
}

extern "C" int cvfem_cuda_assemble_colored(cvfem_cuda_ctx *ctx, double rho, double mu,
                                           int use_sympy, int block_size, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);
    return launch_colored(ctx, rho, mu, use_sympy, block_size, s);
}

extern "C" double cvfem_cuda_time_assemble_colored(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                   int use_sympy, int block_size, int repeat) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.0;
    if (launch_colored(ctx, rho, mu, use_sympy, block_size, 0) != 0) return -1.0;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEventRecord(a);
    for (int i = 0; i < repeat; ++i)
        if (launch_colored(ctx, rho, mu, use_sympy, block_size, 0) != 0) return -1.0;
    cudaEventRecord(b);
    if (cudaEventSynchronize(b) != cudaSuccess) return -1.0;
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / 1000.0 / (repeat > 0 ? repeat : 1);
}


#endif  // CVFEM_HEX8_NS_CUDA_COLORED_CUH
