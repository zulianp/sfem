#include "bench_packed_laplacian_cuda.hpp"

#include "sfem_API.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_PackedLaplacian.hpp"
#include "smesh_base.hpp"
#include "smesh_device_buffer.hpp"
#include "smesh_env.hpp"
#include "smesh_kernel_data.hpp"
#include "smesh_mesh_reorder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <vector>

extern "C" {
int cu_tet4_laplacian_apply(const ptrdiff_t                 nelements,
                            idx_t **const SFEM_RESTRICT     elements,
                            const ptrdiff_t                 fff_stride,
                            const void *const SFEM_RESTRICT fff,
                            const enum smesh::PrimitiveType real_type_xy,
                            const void *const SFEM_RESTRICT x,
                            void *const SFEM_RESTRICT       y,
                            void                           *stream);

int cu_tet10_laplacian_apply(const ptrdiff_t                 nelements,
                             idx_t **const SFEM_RESTRICT     elements,
                             const ptrdiff_t                 fff_stride,
                             const void *const SFEM_RESTRICT fff,
                             const enum smesh::PrimitiveType real_type_xy,
                             const void *const SFEM_RESTRICT x,
                             void *const SFEM_RESTRICT       y,
                             void                           *stream);

int cu_affine_hex8_laplacian_apply(const ptrdiff_t                 nelements,
                                   idx_t **const SFEM_RESTRICT     elements,
                                   const ptrdiff_t                 fff_stride,
                                   const void *const SFEM_RESTRICT fff,
                                   const enum smesh::PrimitiveType real_type_xy,
                                   const void *const SFEM_RESTRICT x,
                                   void *const SFEM_RESTRICT       y,
                                   void                           *stream);
}

namespace {

    using pack_idx_t = sfem::FunctionSpace::PackedIdxType;
    static_assert(std::is_same<pack_idx_t, bench_pack_idx_t>::value, "Packed index ABI mismatch");

    struct DeviceBuffers {
        smesh::SharedBuffer<idx_t *>      elements;
        smesh::SharedBuffer<pack_idx_t *> packed_elements;
        smesh::SharedBuffer<ptrdiff_t>    owned_nodes_ptr;
        smesh::SharedBuffer<ptrdiff_t>    n_shared;
        smesh::SharedBuffer<ptrdiff_t>    ghost_ptr;
        smesh::SharedBuffer<idx_t>        ghost_idx;
        smesh::SharedBuffer<ptrdiff_t>    ghost_reduce_ptr;
        smesh::SharedBuffer<ptrdiff_t>    ghost_reduce_idx;
        smesh::SharedBuffer<idx_t>        ghost_reduce_dest;
        smesh::SharedBuffer<jacobian_t>   fff;
        smesh::SharedBuffer<real_t>       x;
        smesh::SharedBuffer<real_t>       y_cuda;
        smesh::SharedBuffer<real_t>       y_packed_atomic;
        smesh::SharedBuffer<real_t>       y_packed_two_pass;
        smesh::SharedBuffer<real_t>       ghost_buf;
    };

    static double mdofs_per_second(const double seconds, const ptrdiff_t ndofs) {
        return seconds > 0 ? 1e-6 * static_cast<double>(ndofs) / seconds : 0;
    }

    static double melems_per_second(const double seconds, const ptrdiff_t nelements) {
        return seconds > 0 ? 1e-6 * static_cast<double>(nelements) / seconds : 0;
    }

    static void require_success(const int err, const char *const label) {
        if (err != SFEM_SUCCESS) {
            SFEM_ERROR("%s failed with code %d\n", label, err);
        }
    }

    static std::shared_ptr<sfem::Mesh> create_mesh(const std::shared_ptr<sfem::Communicator> &comm,
                                                   const smesh::ElemType                      element_type,
                                                   const int                                  resolution) {
        switch (element_type) {
            case smesh::TET4:
            case smesh::TET10:
            case smesh::HEX8:
                return sfem::Mesh::create_cube(comm, element_type, resolution, resolution, resolution, 0, 0, 0, 1, 1, 1);
            default:
                return nullptr;
        }
    }

    static real_t max_abs_diff(const std::vector<real_t> &left, const std::vector<real_t> &right) {
        real_t ret = 0;
        const ptrdiff_t n = left.size();
#pragma omp parallel for reduction(max : ret)
        for (ptrdiff_t i = 0; i < n; ++i) {
            ret = std::max(ret, static_cast<real_t>(std::abs(left[i] - right[i])));
        }
        return ret;
    }

    static ptrdiff_t max_actual_pack_nodes(const ptrdiff_t                  n_packs,
                                           const ptrdiff_t *const           owned_nodes_ptr,
                                           const ptrdiff_t *const           ghost_ptr) {
        ptrdiff_t ret = 0;
        for (ptrdiff_t p = 0; p < n_packs; ++p) {
            const ptrdiff_t n_contiguous = owned_nodes_ptr[p + 1] - owned_nodes_ptr[p];
            const ptrdiff_t n_ghost      = ghost_ptr[p + 1] - ghost_ptr[p];
            ret                          = std::max(ret, n_contiguous + n_ghost);
        }
        return ret;
    }

    static smesh::SharedBuffer<jacobian_t> create_flat_device_fff(const std::shared_ptr<sfem::Mesh> &mesh) {
        constexpr ptrdiff_t fff_size = 6;
        auto                fff_src  = smesh::FFF::create_SoA(mesh, smesh::MEMORY_SPACE_HOST, 0);
        if (!fff_src || !fff_src->fff_SoA()) {
            SFEM_ERROR("Unable to create Laplacian FFF data\n");
        }

        const ptrdiff_t nelements = mesh->n_elements(0);
        auto            flat_fff  = smesh::create_host_buffer<jacobian_t>(fff_size * nelements);

        auto *const       dst = flat_fff->data();
        const auto *const src = fff_src->fff_SoA()->data();
        for (ptrdiff_t d = 0; d < fff_size; ++d) {
            memcpy(&dst[d * nelements], src[d], nelements * sizeof(jacobian_t));
        }

        return smesh::to_device(flat_fff);
    }

    template <typename T>
    static void copy_to_host(const smesh::SharedBuffer<T> &device, std::vector<T> &host) {
        host.resize(device->size());
        require_success(bench_cuda_copy_device_to_host(host.data(), device->data(), device->nbytes()), "cudaMemcpyDeviceToHost");
    }

    template <typename T>
    static void copy_to_device(const std::vector<T> &host, const smesh::SharedBuffer<T> &device) {
        require_success(bench_cuda_copy_host_to_device(device->data(), host.data(), host.size() * sizeof(T)), "cudaMemcpyHostToDevice");
    }

    template <typename T>
    static smesh::SharedBuffer<T> to_device_or_dummy(const smesh::SharedBuffer<T> &host) {
        if (host->size() == 0) {
            return smesh::create_device_buffer<T>(1);
        }

        return smesh::to_device(host);
    }

    template <typename T>
    static void zero_device_buffer(const smesh::SharedBuffer<T> &device) {
        require_success(bench_cuda_memset(device->data(), 0, device->nbytes()), "cudaMemset");
    }

    template <typename T>
    static void print_rate(const char *const name, const double seconds, const ptrdiff_t nelements, const ptrdiff_t ndofs) {
        printf("%-32s %12.6e %16.3f %13.3f\n",
               name,
               seconds,
               melems_per_second(seconds, nelements),
               mdofs_per_second(seconds, ndofs));
    }

    struct RunContext {
        smesh::ElemType elem_type;
        ptrdiff_t       n_packs;
        ptrdiff_t       n_elements_per_pack;
        ptrdiff_t       nelements;
        ptrdiff_t       n_ghost_reduce_rows;
        ptrdiff_t       actual_max_pack_nodes;
        size_t          shmem_size;
        int             block_size;
        DeviceBuffers  *d;
        void           *stream;
    };

    static void run_cuda_baseline_callback(void *const opaque) {
        auto *const c = static_cast<RunContext *>(opaque);
        auto &      d = *c->d;
        int         err = SFEM_FAILURE;
        if (c->elem_type == smesh::TET4) {
            err = cu_tet4_laplacian_apply(c->nelements,
                                          d.elements->data(),
                                          c->nelements,
                                          d.fff->data(),
                                          smesh::SMESH_DEFAULT,
                                          d.x->data(),
                                          d.y_cuda->data(),
                                          c->stream);
        } else if (c->elem_type == smesh::TET10) {
            err = cu_tet10_laplacian_apply(c->nelements,
                                           d.elements->data(),
                                           c->nelements,
                                           d.fff->data(),
                                           smesh::SMESH_DEFAULT,
                                           d.x->data(),
                                           d.y_cuda->data(),
                                           c->stream);
        } else if (c->elem_type == smesh::HEX8) {
            err = cu_affine_hex8_laplacian_apply(c->nelements,
                                                 d.elements->data(),
                                                 c->nelements,
                                                 d.fff->data(),
                                                 smesh::SMESH_DEFAULT,
                                                 d.x->data(),
                                                 d.y_cuda->data(),
                                                 c->stream);
        }
        require_success(err, "CUDA baseline Laplacian apply");
    }

    static void run_packed_atomic_callback(void *const opaque) {
        auto *const c = static_cast<RunContext *>(opaque);
        auto &      d = *c->d;
        require_success(bench_packed_laplacian_launch_atomic(static_cast<int>(c->elem_type),
                                                             c->n_packs,
                                                             c->n_elements_per_pack,
                                                             c->nelements,
                                                             c->actual_max_pack_nodes,
                                                             c->shmem_size,
                                                             c->block_size,
                                                             d.packed_elements->data(),
                                                             d.owned_nodes_ptr->data(),
                                                             d.n_shared->data(),
                                                             d.ghost_ptr->data(),
                                                             d.ghost_idx->data(),
                                                             d.fff->data(),
                                                             d.x->data(),
                                                             d.y_packed_atomic->data(),
                                                             c->stream),
                        "packed atomic launch");
        require_success(bench_cuda_peek_at_last_error(), "cudaPeekAtLastError");
    }

    static void run_packed_two_pass_callback(void *const opaque) {
        auto *const c = static_cast<RunContext *>(opaque);
        auto &      d = *c->d;
        require_success(bench_packed_laplacian_launch_two_pass(static_cast<int>(c->elem_type),
                                                               c->n_packs,
                                                               c->n_elements_per_pack,
                                                               c->nelements,
                                                               c->n_ghost_reduce_rows,
                                                               c->actual_max_pack_nodes,
                                                               c->shmem_size,
                                                               c->block_size,
                                                               d.packed_elements->data(),
                                                               d.owned_nodes_ptr->data(),
                                                               d.ghost_ptr->data(),
                                                               d.ghost_idx->data(),
                                                               d.ghost_reduce_ptr->data(),
                                                               d.ghost_reduce_idx->data(),
                                                               d.ghost_reduce_dest->data(),
                                                               d.fff->data(),
                                                               d.x->data(),
                                                               d.y_packed_two_pass->data(),
                                                               d.ghost_buf->data(),
                                                               c->stream),
                        "packed two-pass launch");
        require_success(bench_cuda_peek_at_last_error(), "cudaPeekAtLastError");
    }

}  // namespace

int main(int argc, char *argv[]) {
    sfem::Context context(argc, argv);
    auto          comm = context.communicator();

    if (comm->size() != 1) {
        SFEM_ERROR("bench_packed_laplacian_cuda supports one MPI rank\n");
    }

    if (!getenv("SMESH_ELEMENTS_PER_PACK")) {
        setenv("SMESH_ELEMENTS_PER_PACK", "256", 0);
    }

    const int         resolution = smesh::Env::read("SFEM_BASE_RESOLUTION", 32);
    const int         warmup     = smesh::Env::read("SFEM_WARMUP", 3);
    const int         repeat     = smesh::Env::read("SFEM_REPEAT", 20);
    const int         block_size = smesh::Env::read("SFEM_PACKED_CUDA_BLOCK_SIZE", 256);
    const std::string elem_name  = smesh::Env::read_string("SFEM_ELEM_TYPE", "TET4");
    const auto        elem_type  = static_cast<smesh::ElemType>(smesh::type_from_string(elem_name.c_str()));

    if (elem_type != smesh::TET4 && elem_type != smesh::TET10 && elem_type != smesh::HEX8) {
        SFEM_ERROR("SFEM_ELEM_TYPE must be TET4, TET10, or HEX8\n");
    }
    if (block_size <= 0 || block_size > 1024) {
        SFEM_ERROR("SFEM_PACKED_CUDA_BLOCK_SIZE must be in [1, 1024]\n");
    }

    auto mesh = create_mesh(comm, elem_type, resolution);
    if (!mesh) {
        SFEM_ERROR("Unable to create benchmark mesh for %s\n", elem_name.c_str());
    }

    auto sfc = smesh::SFC::create_from_env();
    sfc->reorder(*mesh);

    auto packed_mesh = sfem::FunctionSpace::PackedMesh::create(mesh, {}, true);
    auto fs          = sfem::FunctionSpace::create(mesh, 1);
    auto packed_fs   = sfem::FunctionSpace::create(packed_mesh, 1);

    auto host_laplacian = sfem::create_op(fs, "Laplacian", sfem::EXECUTION_SPACE_HOST);
    if (!host_laplacian) {
        SFEM_ERROR("Unable to create host Laplacian\n");
    }
    require_success(host_laplacian->initialize(), "host Laplacian initialize");

    auto host_packed_laplacian = sfem::create_op(packed_fs, "PackedLaplacian", sfem::EXECUTION_SPACE_HOST);
    if (!host_packed_laplacian) {
        SFEM_ERROR("Unable to create host PackedLaplacian\n");
    }
    require_success(host_packed_laplacian->initialize(), "host PackedLaplacian initialize");

    const ptrdiff_t nelements            = mesh->n_elements(0);
    const ptrdiff_t ndofs                = fs->n_dofs();
    const ptrdiff_t n_packs              = packed_mesh->n_packs(0);
    const ptrdiff_t n_elements_per_pack  = packed_mesh->n_elements_per_pack(0);
    const ptrdiff_t n_ghost_entries      = packed_mesh->n_ghost_entries(0);
    const ptrdiff_t n_ghost_reduce_rows  = packed_mesh->n_ghost_reduce_rows(0);
    const ptrdiff_t actual_max_pack_nodes = max_actual_pack_nodes(
            n_packs, packed_mesh->owned_nodes_ptr(0)->data(), packed_mesh->ghost_ptr(0)->data());
    const ptrdiff_t atomic_physical_max_pack_nodes =
            bench_packed_laplacian_atomic_physical_size(actual_max_pack_nodes);
    const size_t shmem_size = bench_packed_laplacian_shared_workspace_bytes(actual_max_pack_nodes);

    int max_shmem = 0;
    int max_optin_shmem = 0;
    require_success(bench_cuda_shared_memory_limits(&max_shmem, &max_optin_shmem), "cudaDeviceGetAttribute");
    const int allowed_shmem = std::max(max_shmem, max_optin_shmem);
    if (shmem_size > static_cast<size_t>(allowed_shmem)) {
        SFEM_ERROR("Packed CUDA Laplacian requires %zu bytes of shared memory per block, device allows %d. "
                   "Lower SMESH_ELEMENTS_PER_PACK.\n",
                   shmem_size,
                   allowed_shmem);
    }
    require_success(bench_packed_laplacian_set_dynamic_shared_memory_limit(static_cast<int>(elem_type), shmem_size),
                    "set packed dynamic shared memory limit");

    std::vector<real_t> h_x(ndofs);
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        h_x[i] = static_cast<real_t>(((i * 7) % 101) + 1) / 101;
    }

    std::vector<real_t> h_ref(ndofs, 0);
    std::vector<real_t> h_packed_ref(ndofs, 0);
    require_success(host_laplacian->apply(nullptr, h_x.data(), h_ref.data()), "host Laplacian apply");
    require_success(host_packed_laplacian->apply(nullptr, h_x.data(), h_packed_ref.data()), "host PackedLaplacian apply");

    DeviceBuffers d;
    d.elements               = smesh::to_device(mesh->elements(0));
    d.packed_elements        = smesh::to_device(packed_mesh->elements(0));
    d.owned_nodes_ptr        = smesh::to_device(packed_mesh->owned_nodes_ptr(0));
    d.n_shared               = smesh::to_device(packed_mesh->n_shared(0));
    d.ghost_ptr              = smesh::to_device(packed_mesh->ghost_ptr(0));
    d.ghost_idx              = to_device_or_dummy(packed_mesh->ghost_idx(0));
    d.ghost_reduce_ptr       = smesh::to_device(packed_mesh->ghost_reduce_ptr(0));
    d.ghost_reduce_idx       = to_device_or_dummy(packed_mesh->ghost_reduce_idx(0));
    d.ghost_reduce_dest      = to_device_or_dummy(packed_mesh->ghost_reduce_dest(0));
    d.fff                   = create_flat_device_fff(mesh);
    d.x                     = smesh::create_device_buffer<real_t>(ndofs);
    d.y_cuda                = smesh::create_device_buffer<real_t>(ndofs);
    d.y_packed_atomic       = smesh::create_device_buffer<real_t>(ndofs);
    d.y_packed_two_pass     = smesh::create_device_buffer<real_t>(ndofs);
    d.ghost_buf             = smesh::create_device_buffer<real_t>(std::max<ptrdiff_t>(1, n_ghost_entries));
    copy_to_device(h_x, d.x);

    void *stream = nullptr;
    RunContext run_context{elem_type,
                           n_packs,
                           n_elements_per_pack,
                           nelements,
                           n_ghost_reduce_rows,
                           actual_max_pack_nodes,
                           shmem_size,
                           block_size,
                           &d,
                           stream};

    auto run_cuda_baseline = [&]() {
        run_cuda_baseline_callback(&run_context);
    };
    auto run_packed_atomic = [&]() {
        run_packed_atomic_callback(&run_context);
    };
    auto run_packed_two_pass = [&]() {
        run_packed_two_pass_callback(&run_context);
    };

    for (int i = 0; i < warmup; ++i) {
        zero_device_buffer(d.y_cuda);
        run_cuda_baseline();
        zero_device_buffer(d.y_packed_atomic);
        run_packed_atomic();
        zero_device_buffer(d.y_packed_two_pass);
        zero_device_buffer(d.ghost_buf);
        run_packed_two_pass();
    }
    require_success(bench_cuda_device_synchronize(), "cudaDeviceSynchronize");

    zero_device_buffer(d.y_cuda);
    run_cuda_baseline();
    zero_device_buffer(d.y_packed_atomic);
    run_packed_atomic();
    zero_device_buffer(d.y_packed_two_pass);
    zero_device_buffer(d.ghost_buf);
    run_packed_two_pass();
    require_success(bench_cuda_device_synchronize(), "cudaDeviceSynchronize");

    std::vector<real_t> h_cuda;
    std::vector<real_t> h_packed_atomic;
    std::vector<real_t> h_packed_two_pass;
    copy_to_host(d.y_cuda, h_cuda);
    copy_to_host(d.y_packed_atomic, h_packed_atomic);
    copy_to_host(d.y_packed_two_pass, h_packed_two_pass);

    zero_device_buffer(d.y_cuda);
    const double cuda_seconds = bench_cuda_time(repeat, run_cuda_baseline_callback, &run_context);
    zero_device_buffer(d.y_packed_atomic);
    const double packed_atomic_seconds = bench_cuda_time(repeat, run_packed_atomic_callback, &run_context);
    zero_device_buffer(d.y_packed_two_pass);
    zero_device_buffer(d.ghost_buf);
    const double packed_two_pass_seconds = bench_cuda_time(repeat, run_packed_two_pass_callback, &run_context);

    printf("element_type %s\n", type_to_string(elem_type));
    printf("#elements %ld\n", static_cast<long>(nelements));
    printf("#nodes %ld\n", static_cast<long>(mesh->n_nodes()));
    printf("#dofs %ld\n", static_cast<long>(ndofs));
    printf("#packs %ld\n", static_cast<long>(n_packs));
    printf("elements_per_pack %ld\n", static_cast<long>(n_elements_per_pack));
    printf("ghost_entries %ld\n", static_cast<long>(n_ghost_entries));
    printf("ghost_reduce_rows %ld\n", static_cast<long>(n_ghost_reduce_rows));
    printf("actual_max_pack_nodes %ld\n", static_cast<long>(actual_max_pack_nodes));
    printf("atomic_physical_max_pack_nodes %ld\n", static_cast<long>(atomic_physical_max_pack_nodes));
    printf("shared_memory_bytes %zu\n", shmem_size);
    printf("block_size %d\n", block_size);
    printf("cuda_vs_host_max_abs_diff %g\n", static_cast<double>(max_abs_diff(h_cuda, h_ref)));
    printf("host_packed_vs_host_max_abs_diff %g\n", static_cast<double>(max_abs_diff(h_packed_ref, h_ref)));
    printf("packed_atomic_vs_cuda_max_abs_diff %g\n", static_cast<double>(max_abs_diff(h_packed_atomic, h_cuda)));
    printf("packed_two_pass_vs_cuda_max_abs_diff %g\n", static_cast<double>(max_abs_diff(h_packed_two_pass, h_cuda)));
    printf("packed_two_pass_vs_atomic_max_abs_diff %g\n", static_cast<double>(max_abs_diff(h_packed_two_pass, h_packed_atomic)));
    printf("\n%-32s %12s %16s %13s\n", "Operation", "Time [s]", "Rate [MElem/s]", "Rate [MDOF/s]");
    printf("-------------------------------------------------------------------------------\n");
    print_rate<real_t>("cuda_laplacian_apply", cuda_seconds, nelements, ndofs);
    print_rate<real_t>("packed_atomic_apply", packed_atomic_seconds, nelements, ndofs);
    print_rate<real_t>("packed_two_pass_apply", packed_two_pass_seconds, nelements, ndofs);
    printf("packed_atomic_speedup_vs_cuda %g\n", cuda_seconds / packed_atomic_seconds);
    printf("packed_two_pass_speedup_vs_cuda %g\n", cuda_seconds / packed_two_pass_seconds);
    printf("packed_two_pass_speedup_vs_atomic %g\n", packed_atomic_seconds / packed_two_pass_seconds);

    return SFEM_SUCCESS;
}
