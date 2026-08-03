#include "bench_packed_laplacian_cuda.hpp"

#include "cu_hex8_laplacian_inline.hpp"
#include "cu_tet10_laplacian_inline.hpp"
#include "cu_tet4_laplacian_inline.hpp"
#include "sfem_cuda_base.hpp"
#include "smesh_elem_type.hpp"

#include <algorithm>
#include <cstdio>

namespace {

    using pack_idx_t = bench_pack_idx_t;

    static ptrdiff_t ceil_div(const ptrdiff_t n, const ptrdiff_t d) { return (n + d - 1) / d; }

    struct DeviceBuffers {
        idx_t            **elements;
        bench_pack_idx_t **packed_elements;
        const ptrdiff_t   *owned_nodes_ptr;
        const ptrdiff_t   *n_shared;
        const ptrdiff_t   *ghost_ptr;
        const idx_t       *ghost_idx;
        const ptrdiff_t   *ghost_reduce_ptr;
        const ptrdiff_t   *ghost_reduce_idx;
        const idx_t       *ghost_reduce_dest;
        const jacobian_t  *fff;
        const real_t      *x;
        real_t            *y_packed_atomic;
        real_t            *y_packed_two_pass;
        real_t            *ghost_buf;
    };

    template <typename T>
    struct Tet4Element {
        static constexpr int   NXE = 4;
        static __device__ void apply(const T *const SFEM_RESTRICT fff,
                                     const T *const SFEM_RESTRICT element_u,
                                     T *const SFEM_RESTRICT       element_out) {
            cu_tet4_laplacian_apply_fff(fff, 1, element_u, element_out);
        }
    };

    template <typename T>
    struct Tet10Element {
        static constexpr int   NXE = 10;
        static __device__ void apply(const T *const SFEM_RESTRICT fff,
                                     const T *const SFEM_RESTRICT element_u,
                                     T *const SFEM_RESTRICT       element_out) {
            cu_tet10_laplacian_apply_fff(fff, 1, element_u, element_out);
        }
    };

    template <typename T>
    struct Hex8Element {
        static constexpr int   NXE = 8;
        static __device__ void apply(const T *const SFEM_RESTRICT fff,
                                     const T *const SFEM_RESTRICT element_u,
                                     T *const SFEM_RESTRICT       element_out) {
            cu_hex8_laplacian_apply_fff_integral(fff, element_u, element_out);
        }
    };

    template <typename T>
    struct SharedMemoryAccessor {
        T *data;

        __device__ explicit SharedMemoryAccessor(T *const data) : data(data) {}

        static __host__ __device__ size_t bytes(const ptrdiff_t logical_size) {
            return static_cast<size_t>(logical_size) * sizeof(T);
        }

        static __device__ SharedMemoryAccessor make(unsigned char *const data, const ptrdiff_t) {
            return SharedMemoryAccessor(reinterpret_cast<T *>(data));
        }

        __device__ T load(const ptrdiff_t i) const { return data[i]; }

        __device__ void store(const ptrdiff_t i, const T value) { data[i] = value; }
    };

    template <>
    struct SharedMemoryAccessor<double> {
        unsigned int *lo;
        unsigned int *hi;

        __device__ SharedMemoryAccessor(unsigned int *const lo, unsigned int *const hi) : lo(lo), hi(hi) {}

        static __host__ __device__ size_t bytes(const ptrdiff_t logical_size) {
            return 2 * static_cast<size_t>(logical_size) * sizeof(unsigned int);
        }

        static __device__ SharedMemoryAccessor make(unsigned char *const data, const ptrdiff_t logical_size) {
            unsigned int *const words = reinterpret_cast<unsigned int *>(data);
            return SharedMemoryAccessor(words, words + logical_size);
        }

        __device__ double load(const ptrdiff_t i) const {
            const unsigned long long bits =
                    static_cast<unsigned long long>(lo[i]) | (static_cast<unsigned long long>(hi[i]) << 32);
            return __longlong_as_double(static_cast<long long>(bits));
        }

        __device__ void store(const ptrdiff_t i, const double value) {
            const unsigned long long bits = static_cast<unsigned long long>(__double_as_longlong(value));
            lo[i]                         = static_cast<unsigned int>(bits);
            hi[i]                         = static_cast<unsigned int>(bits >> 32);
        }
    };

    template <typename T>
    struct SharedMemoryAtomicAccessor {
        T *data;

        __device__ explicit SharedMemoryAtomicAccessor(T *const data) : data(data) {}

        static __host__ __device__ ptrdiff_t physical_size(const ptrdiff_t logical_size) { return logical_size; }

        static __host__ __device__ size_t bytes(const ptrdiff_t logical_size) {
            return static_cast<size_t>(physical_size(logical_size)) * sizeof(T);
        }

        static __device__ SharedMemoryAtomicAccessor make(unsigned char *const data, const ptrdiff_t) {
            return SharedMemoryAtomicAccessor(reinterpret_cast<T *>(data));
        }

        __device__ T load(const ptrdiff_t i) const { return data[i]; }

        __device__ void store(const ptrdiff_t i, const T value) { data[i] = value; }

        __device__ T *ptr(const ptrdiff_t i) { return &data[i]; }
    };

    // template <>
    // struct SharedMemoryAtomicAccessor<double> {
    //     double *data;

    //     __device__ explicit SharedMemoryAtomicAccessor(double *const data) : data(data) {}

    //     // Skew double arrays so common power-of-two pack strides do not alias the same shared-memory banks.
    //     static constexpr ptrdiff_t padding_interval = 16;

    //     static __host__ __device__ ptrdiff_t physical_index(const ptrdiff_t i) {
    //         return i + i / padding_interval;
    //     }

    //     static __host__ __device__ ptrdiff_t physical_size(const ptrdiff_t logical_size) {
    //         return logical_size > 0 ? logical_size + (logical_size - 1) / padding_interval : 0;
    //     }

    //     static __host__ __device__ size_t bytes(const ptrdiff_t logical_size) {
    //         return static_cast<size_t>(physical_size(logical_size)) * sizeof(double);
    //     }

    //     static __device__ SharedMemoryAtomicAccessor make(unsigned char *const data, const ptrdiff_t) {
    //         return SharedMemoryAtomicAccessor(reinterpret_cast<double *>(data));
    //     }

    //     __device__ double load(const ptrdiff_t i) const { return data[physical_index(i)]; }

    //     __device__ void store(const ptrdiff_t i, const double value) { data[physical_index(i)] = value; }

    //     __device__ double *ptr(const ptrdiff_t i) { return &data[physical_index(i)]; }
    // };

    template <typename T>
    static size_t shared_workspace_bytes(const ptrdiff_t max_pack_nodes) {
        return SharedMemoryAccessor<T>::bytes(max_pack_nodes) + SharedMemoryAtomicAccessor<T>::bytes(max_pack_nodes);
    }

    template <typename T, typename Element>
    __global__ void packed_laplacian_atomic_kernel(const ptrdiff_t                          n_elements_per_pack,
                                                   const ptrdiff_t                          n_elements,
                                                   const ptrdiff_t                          max_pack_nodes,
                                                   pack_idx_t **const SFEM_RESTRICT         elements,
                                                   const ptrdiff_t *const SFEM_RESTRICT     owned_nodes_ptr,
                                                   const ptrdiff_t *const SFEM_RESTRICT     n_shared_nodes,
                                                   const ptrdiff_t *const SFEM_RESTRICT     ghost_ptr,
                                                   const idx_t *const SFEM_RESTRICT         ghost_idx,
                                                   const cu_jacobian_t *const SFEM_RESTRICT fff,
                                                   const ptrdiff_t                          fff_stride,
                                                   const T *const SFEM_RESTRICT             u,
                                                   T *const SFEM_RESTRICT                   values) {
        extern __shared__ unsigned char shared_mem[];
        using Shared               = SharedMemoryAccessor<T>;
        using SharedAtomic         = SharedMemoryAtomicAccessor<T>;
        Shared               s_u   = Shared::make(shared_mem, max_pack_nodes);
        SharedAtomic         s_out = SharedAtomic::make(shared_mem + Shared::bytes(max_pack_nodes), max_pack_nodes);
        static constexpr int NXE   = Element::NXE;

        const ptrdiff_t p            = blockIdx.x;
        const ptrdiff_t e_start      = p * n_elements_per_pack;
        const ptrdiff_t e_end        = n_elements < (p + 1) * n_elements_per_pack ? n_elements : (p + 1) * n_elements_per_pack;
        const ptrdiff_t owned_begin  = owned_nodes_ptr[p];
        const ptrdiff_t n_contiguous = owned_nodes_ptr[p + 1] - owned_begin;
        const ptrdiff_t ghost_begin  = ghost_ptr[p];
        const ptrdiff_t n_ghost      = ghost_ptr[p + 1] - ghost_begin;
        const ptrdiff_t total_nodes  = n_contiguous + n_ghost;

        for (ptrdiff_t i = threadIdx.x; i < total_nodes; i += blockDim.x) {
            s_out.store(i, 0);
            s_u.store(i, i < n_contiguous ? u[owned_begin + i] : u[ghost_idx[ghost_begin + i - n_contiguous]]);
        }
        __syncthreads();

        for (ptrdiff_t e = e_start + threadIdx.x; e < e_end; e += blockDim.x) {
            pack_idx_t ev[NXE];
            T          element_u[NXE];
            T          element_out[NXE];
            T          fffe[6];

#pragma unroll
            for (int v = 0; v < NXE; ++v) {
                ev[v]          = elements[v][e];
                element_u[v]   = s_u.load(ev[v]);
                element_out[v] = 0;
            }

#pragma unroll
            for (int d = 0; d < 6; ++d) {
                fffe[d] = static_cast<T>(fff[d * fff_stride + e]);
            }

            Element::apply(fffe, element_u, element_out);

#pragma unroll
            for (int v = 0; v < NXE; ++v) {
                atomicAdd(s_out.ptr(ev[v]), element_out[v]);
            }
        }
        __syncthreads();

        const ptrdiff_t n_not_shared = n_contiguous - n_shared_nodes[p];
        for (ptrdiff_t i = threadIdx.x; i < n_not_shared; i += blockDim.x) {
            values[owned_begin + i] += s_out.load(i);
        }
        for (ptrdiff_t i = n_not_shared + threadIdx.x; i < n_contiguous; i += blockDim.x) {
            atomicAdd(&values[owned_begin + i], s_out.load(i));
        }
        for (ptrdiff_t i = threadIdx.x; i < n_ghost; i += blockDim.x) {
            atomicAdd(&values[ghost_idx[ghost_begin + i]], s_out.load(n_contiguous + i));
        }
    }

    template <typename T, typename Element>
    __global__ void packed_laplacian_two_pass_pack_kernel(const ptrdiff_t                          n_elements_per_pack,
                                                          const ptrdiff_t                          n_elements,
                                                          const ptrdiff_t                          max_pack_nodes,
                                                          pack_idx_t **const SFEM_RESTRICT         elements,
                                                          const ptrdiff_t *const SFEM_RESTRICT     owned_nodes_ptr,
                                                          const ptrdiff_t *const SFEM_RESTRICT     ghost_ptr,
                                                          const idx_t *const SFEM_RESTRICT         ghost_idx,
                                                          const cu_jacobian_t *const SFEM_RESTRICT fff,
                                                          const ptrdiff_t                          fff_stride,
                                                          const T *const SFEM_RESTRICT             u,
                                                          T *const SFEM_RESTRICT                   values,
                                                          T *const SFEM_RESTRICT                   ghost_buf) {
        extern __shared__ unsigned char shared_mem[];
        using Shared               = SharedMemoryAccessor<T>;
        using SharedAtomic         = SharedMemoryAtomicAccessor<T>;
        Shared               s_u   = Shared::make(shared_mem, max_pack_nodes);
        SharedAtomic         s_out = SharedAtomic::make(shared_mem + Shared::bytes(max_pack_nodes), max_pack_nodes);
        static constexpr int NXE   = Element::NXE;

        const ptrdiff_t p            = blockIdx.x;
        const ptrdiff_t e_start      = p * n_elements_per_pack;
        const ptrdiff_t e_end        = n_elements < (p + 1) * n_elements_per_pack ? n_elements : (p + 1) * n_elements_per_pack;
        const ptrdiff_t owned_begin  = owned_nodes_ptr[p];
        const ptrdiff_t n_contiguous = owned_nodes_ptr[p + 1] - owned_begin;
        const ptrdiff_t ghost_begin  = ghost_ptr[p];
        const ptrdiff_t n_ghost      = ghost_ptr[p + 1] - ghost_begin;
        const ptrdiff_t total_nodes  = n_contiguous + n_ghost;

        for (ptrdiff_t i = threadIdx.x; i < total_nodes; i += blockDim.x) {
            s_out.store(i, 0);
            s_u.store(i, i < n_contiguous ? u[owned_begin + i] : u[ghost_idx[ghost_begin + i - n_contiguous]]);
        }
        __syncthreads();

        for (ptrdiff_t e = e_start + threadIdx.x; e < e_end; e += blockDim.x) {
            pack_idx_t ev[NXE];
            T          element_u[NXE];
            T          element_out[NXE];
            T          fffe[6];

#pragma unroll
            for (int v = 0; v < NXE; ++v) {
                ev[v]          = elements[v][e];
                element_u[v]   = s_u.load(ev[v]);
                element_out[v] = 0;
            }

#pragma unroll
            for (int d = 0; d < 6; ++d) {
                fffe[d] = static_cast<T>(fff[d * fff_stride + e]);
            }

            Element::apply(fffe, element_u, element_out);

#pragma unroll
            for (int v = 0; v < NXE; ++v) {
                atomicAdd(s_out.ptr(ev[v]), element_out[v]);
            }
        }
        __syncthreads();

        for (ptrdiff_t i = threadIdx.x; i < n_contiguous; i += blockDim.x) {
            values[owned_begin + i] += s_out.load(i);
        }
        for (ptrdiff_t i = threadIdx.x; i < n_ghost; i += blockDim.x) {
            ghost_buf[ghost_begin + i] = s_out.load(n_contiguous + i);
        }
    }

    template <typename T>
    __global__ void packed_laplacian_two_pass_reduce_kernel(const ptrdiff_t                      n_rows,
                                                            const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
                                                            const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
                                                            const idx_t *const SFEM_RESTRICT     ghost_reduce_dest,
                                                            const T *const SFEM_RESTRICT         ghost_buf,
                                                            T *const SFEM_RESTRICT               values) {
        for (ptrdiff_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n_rows; row += blockDim.x * gridDim.x) {
            T sum = 0;
            for (ptrdiff_t j = ghost_reduce_ptr[row]; j < ghost_reduce_ptr[row + 1]; ++j) {
                sum += ghost_buf[ghost_reduce_idx[j]];
            }
            values[ghost_reduce_dest[row]] += sum;
        }
    }

    template <typename T>
    static void set_dynamic_shared_memory_limit(const smesh::ElemType element_type, const size_t shmem_size) {
        if (element_type == smesh::TET4) {
            SFEM_CUDA_CHECK(cudaFuncSetAttribute(
                    packed_laplacian_atomic_kernel<T, Tet4Element<T>>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
            SFEM_CUDA_CHECK(cudaFuncSetAttribute(packed_laplacian_two_pass_pack_kernel<T, Tet4Element<T>>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 shmem_size));
        } else if (element_type == smesh::TET10) {
            SFEM_CUDA_CHECK(cudaFuncSetAttribute(
                    packed_laplacian_atomic_kernel<T, Tet10Element<T>>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
            SFEM_CUDA_CHECK(cudaFuncSetAttribute(packed_laplacian_two_pass_pack_kernel<T, Tet10Element<T>>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 shmem_size));
        } else if (element_type == smesh::HEX8) {
            SFEM_CUDA_CHECK(cudaFuncSetAttribute(
                    packed_laplacian_atomic_kernel<T, Hex8Element<T>>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
            SFEM_CUDA_CHECK(cudaFuncSetAttribute(packed_laplacian_two_pass_pack_kernel<T, Hex8Element<T>>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 shmem_size));
        }
    }

    template <typename T>
    static void launch_atomic(const smesh::ElemType element_type,
                              const ptrdiff_t       n_packs,
                              const ptrdiff_t       n_elements_per_pack,
                              const ptrdiff_t       n_elements,
                              const ptrdiff_t       max_pack_nodes,
                              const size_t          shmem_size,
                              const int             block_size,
                              const DeviceBuffers  &d,
                              cudaStream_t          stream) {
        const dim3 block(block_size, 1, 1);
        if (element_type == smesh::TET4) {
            packed_laplacian_atomic_kernel<T, Tet4Element<T>>
                    <<<n_packs, block, shmem_size, stream>>>(n_elements_per_pack,
                                                             n_elements,
                                                             max_pack_nodes,
                                                             d.packed_elements,
                                                             d.owned_nodes_ptr,
                                                             d.n_shared,
                                                             d.ghost_ptr,
                                                             d.ghost_idx,
                                                             reinterpret_cast<const cu_jacobian_t *>(d.fff),
                                                             n_elements,
                                                             d.x,
                                                             d.y_packed_atomic);
        } else if (element_type == smesh::TET10) {
            packed_laplacian_atomic_kernel<T, Tet10Element<T>>
                    <<<n_packs, block, shmem_size, stream>>>(n_elements_per_pack,
                                                             n_elements,
                                                             max_pack_nodes,
                                                             d.packed_elements,
                                                             d.owned_nodes_ptr,
                                                             d.n_shared,
                                                             d.ghost_ptr,
                                                             d.ghost_idx,
                                                             reinterpret_cast<const cu_jacobian_t *>(d.fff),
                                                             n_elements,
                                                             d.x,
                                                             d.y_packed_atomic);
        } else if (element_type == smesh::HEX8) {
            packed_laplacian_atomic_kernel<T, Hex8Element<T>>
                    <<<n_packs, block, shmem_size, stream>>>(n_elements_per_pack,
                                                             n_elements,
                                                             max_pack_nodes,
                                                             d.packed_elements,
                                                             d.owned_nodes_ptr,
                                                             d.n_shared,
                                                             d.ghost_ptr,
                                                             d.ghost_idx,
                                                             reinterpret_cast<const cu_jacobian_t *>(d.fff),
                                                             n_elements,
                                                             d.x,
                                                             d.y_packed_atomic);
        }
    }

    template <typename T>
    static void launch_two_pass(const smesh::ElemType element_type,
                                const ptrdiff_t       n_packs,
                                const ptrdiff_t       n_elements_per_pack,
                                const ptrdiff_t       n_elements,
                                const ptrdiff_t       n_ghost_reduce_rows,
                                const ptrdiff_t       max_pack_nodes,
                                const size_t          shmem_size,
                                const int             block_size,
                                const DeviceBuffers  &d,
                                cudaStream_t          stream) {
        const dim3 block(block_size, 1, 1);
        if (element_type == smesh::TET4) {
            packed_laplacian_two_pass_pack_kernel<T, Tet4Element<T>>
                    <<<n_packs, block, shmem_size, stream>>>(n_elements_per_pack,
                                                             n_elements,
                                                             max_pack_nodes,
                                                             d.packed_elements,
                                                             d.owned_nodes_ptr,
                                                             d.ghost_ptr,
                                                             d.ghost_idx,
                                                             reinterpret_cast<const cu_jacobian_t *>(d.fff),
                                                             n_elements,
                                                             d.x,
                                                             d.y_packed_two_pass,
                                                             d.ghost_buf);
        } else if (element_type == smesh::TET10) {
            packed_laplacian_two_pass_pack_kernel<T, Tet10Element<T>>
                    <<<n_packs, block, shmem_size, stream>>>(n_elements_per_pack,
                                                             n_elements,
                                                             max_pack_nodes,
                                                             d.packed_elements,
                                                             d.owned_nodes_ptr,
                                                             d.ghost_ptr,
                                                             d.ghost_idx,
                                                             reinterpret_cast<const cu_jacobian_t *>(d.fff),
                                                             n_elements,
                                                             d.x,
                                                             d.y_packed_two_pass,
                                                             d.ghost_buf);
        } else if (element_type == smesh::HEX8) {
            packed_laplacian_two_pass_pack_kernel<T, Hex8Element<T>>
                    <<<n_packs, block, shmem_size, stream>>>(n_elements_per_pack,
                                                             n_elements,
                                                             max_pack_nodes,
                                                             d.packed_elements,
                                                             d.owned_nodes_ptr,
                                                             d.ghost_ptr,
                                                             d.ghost_idx,
                                                             reinterpret_cast<const cu_jacobian_t *>(d.fff),
                                                             n_elements,
                                                             d.x,
                                                             d.y_packed_two_pass,
                                                             d.ghost_buf);
        }

        const int reduce_block_size = 256;
        const int reduce_blocks     = static_cast<int>(std::max<ptrdiff_t>(1, ceil_div(n_ghost_reduce_rows, reduce_block_size)));
        packed_laplacian_two_pass_reduce_kernel<T><<<reduce_blocks, reduce_block_size, 0, stream>>>(n_ghost_reduce_rows,
                                                                                                    d.ghost_reduce_ptr,
                                                                                                    d.ghost_reduce_idx,
                                                                                                    d.ghost_reduce_dest,
                                                                                                    d.ghost_buf,
                                                                                                    d.y_packed_two_pass);
    }

    static cudaStream_t stream_from_void(void *const stream) {
        return stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(nullptr);
    }

}  // namespace

extern "C" int bench_cuda_copy_device_to_host(void *const dst, const void *const src, const size_t nbytes) {
    SFEM_CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost));
    return SFEM_SUCCESS;
}

extern "C" int bench_cuda_copy_host_to_device(void *const dst, const void *const src, const size_t nbytes) {
    SFEM_CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice));
    return SFEM_SUCCESS;
}

extern "C" int bench_cuda_memset(void *const ptr, const int value, const size_t nbytes) {
    SFEM_CUDA_CHECK(cudaMemset(ptr, value, nbytes));
    return SFEM_SUCCESS;
}

extern "C" int bench_cuda_device_synchronize() {
    SFEM_CUDA_CHECK(cudaDeviceSynchronize());
    return SFEM_SUCCESS;
}

extern "C" int bench_cuda_peek_at_last_error() {
    SFEM_CUDA_CHECK(cudaPeekAtLastError());
    return SFEM_SUCCESS;
}

extern "C" int bench_cuda_shared_memory_limits(int *const max_shmem, int *const max_optin_shmem) {
    int device = 0;
    SFEM_CUDA_CHECK(cudaGetDevice(&device));
    SFEM_CUDA_CHECK(cudaDeviceGetAttribute(max_shmem, cudaDevAttrMaxSharedMemoryPerBlock, device));
    SFEM_CUDA_CHECK(cudaDeviceGetAttribute(max_optin_shmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    return SFEM_SUCCESS;
}

extern "C" double bench_cuda_time(const int repeat, void (*callback)(void *), void *const ctx) {
    cudaEvent_t start;
    cudaEvent_t stop;
    SFEM_CUDA_CHECK(cudaEventCreate(&start));
    SFEM_CUDA_CHECK(cudaEventCreate(&stop));
    SFEM_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        callback(ctx);
    }
    SFEM_CUDA_CHECK(cudaEventRecord(stop));
    SFEM_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    SFEM_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    SFEM_CUDA_CHECK(cudaEventDestroy(start));
    SFEM_CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(elapsed_ms) * 1e-3 / repeat;
}

extern "C" size_t bench_packed_laplacian_shared_workspace_bytes(const ptrdiff_t max_pack_nodes) {
    return shared_workspace_bytes<real_t>(max_pack_nodes);
}

extern "C" ptrdiff_t bench_packed_laplacian_atomic_physical_size(const ptrdiff_t max_pack_nodes) {
    return SharedMemoryAtomicAccessor<real_t>::physical_size(max_pack_nodes);
}

extern "C" int bench_packed_laplacian_set_dynamic_shared_memory_limit(const int element_type, const size_t shmem_size) {
    set_dynamic_shared_memory_limit<real_t>(static_cast<smesh::ElemType>(element_type), shmem_size);
    return SFEM_SUCCESS;
}

extern "C" int bench_packed_laplacian_launch_atomic(const int                element_type,
                                                    const ptrdiff_t          n_packs,
                                                    const ptrdiff_t          n_elements_per_pack,
                                                    const ptrdiff_t          n_elements,
                                                    const ptrdiff_t          max_pack_nodes,
                                                    const size_t             shmem_size,
                                                    const int                block_size,
                                                    bench_pack_idx_t **const packed_elements,
                                                    const ptrdiff_t *const   owned_nodes_ptr,
                                                    const ptrdiff_t *const   n_shared,
                                                    const ptrdiff_t *const   ghost_ptr,
                                                    const idx_t *const       ghost_idx,
                                                    const jacobian_t *const  fff,
                                                    const real_t *const      x,
                                                    real_t *const            y,
                                                    void *const              stream) {
    DeviceBuffers d;
    d.elements          = nullptr;
    d.packed_elements   = packed_elements;
    d.owned_nodes_ptr   = owned_nodes_ptr;
    d.n_shared          = n_shared;
    d.ghost_ptr         = ghost_ptr;
    d.ghost_idx         = ghost_idx;
    d.ghost_reduce_ptr  = nullptr;
    d.ghost_reduce_idx  = nullptr;
    d.ghost_reduce_dest = nullptr;
    d.fff               = fff;
    d.x                 = x;
    d.y_packed_atomic   = y;
    d.y_packed_two_pass = nullptr;
    d.ghost_buf         = nullptr;
    launch_atomic<real_t>(static_cast<smesh::ElemType>(element_type),
                          n_packs,
                          n_elements_per_pack,
                          n_elements,
                          max_pack_nodes,
                          shmem_size,
                          block_size,
                          d,
                          stream_from_void(stream));
    return SFEM_SUCCESS;
}

extern "C" int bench_packed_laplacian_launch_two_pass(const int                element_type,
                                                      const ptrdiff_t          n_packs,
                                                      const ptrdiff_t          n_elements_per_pack,
                                                      const ptrdiff_t          n_elements,
                                                      const ptrdiff_t          n_ghost_reduce_rows,
                                                      const ptrdiff_t          max_pack_nodes,
                                                      const size_t             shmem_size,
                                                      const int                block_size,
                                                      bench_pack_idx_t **const packed_elements,
                                                      const ptrdiff_t *const   owned_nodes_ptr,
                                                      const ptrdiff_t *const   ghost_ptr,
                                                      const idx_t *const       ghost_idx,
                                                      const ptrdiff_t *const   ghost_reduce_ptr,
                                                      const ptrdiff_t *const   ghost_reduce_idx,
                                                      const idx_t *const       ghost_reduce_dest,
                                                      const jacobian_t *const  fff,
                                                      const real_t *const      x,
                                                      real_t *const            y,
                                                      real_t *const            ghost_buf,
                                                      void *const              stream) {
    DeviceBuffers d;
    d.elements          = nullptr;
    d.packed_elements   = packed_elements;
    d.owned_nodes_ptr   = owned_nodes_ptr;
    d.n_shared          = nullptr;
    d.ghost_ptr         = ghost_ptr;
    d.ghost_idx         = ghost_idx;
    d.ghost_reduce_ptr  = ghost_reduce_ptr;
    d.ghost_reduce_idx  = ghost_reduce_idx;
    d.ghost_reduce_dest = ghost_reduce_dest;
    d.fff               = fff;
    d.x                 = x;
    d.y_packed_atomic   = nullptr;
    d.y_packed_two_pass = y;
    d.ghost_buf         = ghost_buf;
    launch_two_pass<real_t>(static_cast<smesh::ElemType>(element_type),
                            n_packs,
                            n_elements_per_pack,
                            n_elements,
                            n_ghost_reduce_rows,
                            max_pack_nodes,
                            shmem_size,
                            block_size,
                            d,
                            stream_from_void(stream));
    return SFEM_SUCCESS;
}
