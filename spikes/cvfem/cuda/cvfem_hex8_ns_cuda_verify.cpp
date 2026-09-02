// Verification and timing driver for the packed CVFEM HEX8 residual on CUDA.
//
// Kept separate from cvfem_hex8_ns_upwind_bench.cpp on purpose: the CPU benchmark is a
// measured baseline and threading CUDA through it would perturb its build. This driver
// reuses the same layout headers, so both paths run the identical host reference.
//
// Usage: cvfem_hex8_ns_cuda_verify [--n N] [--pack-size N] [--block-size N]
//                                  [--repeat N] [--no-sfc]

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <mpi.h>

#include "cvfem_hex8_layout_common.hpp"
#include "cvfem_hex8_layout_atomic.hpp"
#include "cvfem_hex8_layout_packed.hpp"
#include "cvfem_pack_coloring.hpp"
#include "cvfem_hex8_layout_store.hpp"

#include "cvfem_hex8_ns_cuda.hpp"

namespace {

double max_abs_diff(const std::vector<double> &a, const std::vector<double> &b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
    return m;
}

double max_abs(const std::vector<double> &a) {
    double m = 0;
    for (double v : a) m = std::fmax(m, std::fabs(v));
    return m;
}

// The device works in interleaved [node * 4 + field] order throughout: 32 contiguous
// bytes per node is one coalesced access, where the host residual keeps four separate
// SoA arrays. These two helpers are the only place the two layouts meet.
void soa_to_interleaved(const MeshData &d, std::vector<double> &out) {
    out.resize((size_t)d.nnodes * 4);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        out[i * 4 + 0] = d.ux[i]; out[i * 4 + 1] = d.uy[i];
        out[i * 4 + 2] = d.uz[i]; out[i * 4 + 3] = d.p[i];
    }
}

void residual_soa_to_interleaved(const MeshData &d, std::vector<double> &out) {
    out.resize((size_t)d.nnodes * 4);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        out[i * 4 + 0] = d.rx[i]; out[i * 4 + 1] = d.ry[i];
        out[i * 4 + 2] = d.rz[i]; out[i * 4 + 3] = d.rc[i];
    }
}

}  // namespace

int main(int argc, char **argv) {
    int mpi_ready = 0;
    MPI_Initialized(&mpi_ready);
    bool own_mpi = false;
    if (!mpi_ready) { MPI_Init(&argc, &argv); own_mpi = true; }

    int  n = 32, pack_size = 512, block_size = 128, repeat = 20;
    bool use_sfc = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--n" && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (a == "--pack-size" && i + 1 < argc) pack_size = std::atoi(argv[++i]);
        else if (a == "--block-size" && i + 1 < argc) block_size = std::atoi(argv[++i]);
        else if (a == "--repeat" && i + 1 < argc) repeat = std::atoi(argv[++i]);
        else if (a == "--no-sfc") use_sfc = false;
    }

    int sm = 0, shmem = 0, optin = 0, warp = 0;
    if (cvfem_cuda_device_info(&sm, &shmem, &optin, &warp) != 0) {
        std::fprintf(stderr, "no CUDA device\n");
        return 1;
    }

    for (int i = 0; i < 64; ++i) g_identity_slots[i] = i;

    MeshData d;
    d.mesh = smesh::Mesh::create_hex8_cube(smesh::Communicator::self(), n, n, n, 0, 0, 0, 1, 1, 1);
    if (!d.mesh) { std::fprintf(stderr, "mesh creation failed\n"); return 1; }
    if (use_sfc) { auto sfc = smesh::SFC::create_from_env(); sfc->reorder(*d.mesh); }

    PackedData packed = make_packed(d.mesh, pack_size);

    d.nnodes    = d.mesh->n_nodes();
    d.nelements = d.mesh->n_elements(0);
    d.elems     = d.mesh->elements(0)->data();
    d.points    = d.mesh->points()->data();
    fill_fields(d);
    precompute_affine_geometry(d);

    const size_t shmem_need = cvfem_cuda_residual_shmem_bytes(packed.max_actual_nodes_per_pack);
    std::printf("mesh   n=%d  nodes=%td  elements=%td  dofs=%td\n",
                n, d.nnodes, d.nelements, d.nnodes * 4);
    std::printf("packs  n_packs=%td  elements/pack=%td  max_nodes/pack=%td\n",
                packed.n_packs, packed.n_elements_per_pack, packed.max_actual_nodes_per_pack);
    std::printf("device %d SMs, shared/block %d B, opt-in max %d B; kernel needs %zu B",
                sm, shmem, optin, shmem_need);
    if (shmem_need > (size_t)optin) {
        std::printf("  -- EXCEEDS OPT-IN LIMIT, reduce --pack-size\n");
        return 1;
    }
    std::printf("  (%.0f%% of opt-in)\n", 100.0 * (double)shmem_need / (double)optin);

    // ---- host reference ------------------------------------------------------
    const scalar_t rho = 1.0, mu = 0.01;
    apply_residual_atomic(d, rho, mu);
    std::vector<double> ref;
    residual_soa_to_interleaved(d, ref);

    // ---- device --------------------------------------------------------------
    std::vector<uint16_t> elems_flat((size_t)8 * d.nelements);
    for (int v = 0; v < 8; ++v)
        for (ptrdiff_t e = 0; e < d.nelements; ++e)
            elems_flat[(size_t)v * d.nelements + e] = packed.elems[v][e];

    std::vector<double> adj_flat((size_t)9 * d.nelements);
    for (int c = 0; c < 9; ++c)
        std::memcpy(&adj_flat[(size_t)c * d.nelements], d.jacobian_adjugate[c].data(),
                    (size_t)d.nelements * sizeof(double));

    cvfem_cuda_ctx *ctx = nullptr;
    if (cvfem_cuda_create(&ctx, d.nnodes, d.nelements, packed.n_packs,
                          packed.n_elements_per_pack, packed.max_actual_nodes_per_pack,
                          packed.n_ghost_entries, packed.n_ghost_reduce_rows,
                          elems_flat.data(), packed.owned_nodes_ptr, packed.n_shared,
                          packed.ghost_ptr, packed.ghost_idx, packed.ghost_reduce_ptr,
                          packed.ghost_reduce_idx, packed.ghost_reduce_dest,
                          adj_flat.data(), d.jacobian_determinant.data()) != 0) {
        std::fprintf(stderr, "cvfem_cuda_create failed\n");
        return 1;
    }

    std::vector<double> u_int;
    soa_to_interleaved(d, u_int);
    if (cvfem_cuda_upload_u(ctx, u_int.data()) != 0) return 1;

    const double refmax = max_abs(ref);
    int          fail   = 0;
    std::vector<double> dev(ref.size()), dev2(ref.size());

    struct { int mode; const char *name; } modes[] = {
        {CVFEM_CUDA_FLUSH_TWO_PASS, "two_pass"},
        {CVFEM_CUDA_FLUSH_ATOMIC,   "atomic"},
    };

    for (auto &m : modes) {
        if (cvfem_cuda_residual(ctx, rho, mu, m.mode, block_size, nullptr) != 0) return 1;
        if (cvfem_cuda_synchronize() != 0) return 1;
        if (cvfem_cuda_download_r(ctx, dev.data()) != 0) return 1;

        const double diff = max_abs_diff(ref, dev);
        const double rel  = refmax > 0 ? diff / refmax : diff;
        // The device sums an element's contributions in a different order than the host
        // sweep, so exact equality is not expected; this is at rounding level.
        const bool ok = rel <= 1e-12;
        fail |= !ok;
        std::printf("residual %-9s vs apply_residual_atomic: max|diff| = %.3e  rel = %.3e  %s\n",
                    m.name, diff, rel, ok ? "OK" : "FAIL");

        // Repeat, to see whether this mode is reproducible run to run.
        //
        // Neither mode is, and the reason is worth stating: the two-pass flush removes
        // atomics from the *global* reduction, but both modes still accumulate a pack's
        // element contributions with atomicAdd into shared memory, and those fix no
        // order. Reproducibility would need the in-pack accumulation ordered too -- a
        // per-node gather rather than a per-element scatter.
        if (cvfem_cuda_residual(ctx, rho, mu, m.mode, block_size, nullptr) != 0) return 1;
        if (cvfem_cuda_synchronize() != 0) return 1;
        if (cvfem_cuda_download_r(ctx, dev2.data()) != 0) return 1;
        const bool bitwise = std::memcmp(dev.data(), dev2.data(),
                                         dev.size() * sizeof(double)) == 0;
        std::printf("         %-9s run-to-run: %s\n", m.name,
                    bitwise ? "bit-identical"
                            : "differs (shared-memory atomics in the element loop)");
    }

    // ---- timing --------------------------------------------------------------
    std::printf("\n%-10s %12s %14s %14s\n", "flush", "s/apply", "MDOF/s", "MELEM/s");
    for (auto &m : modes) {
        const double s = cvfem_cuda_time_residual(ctx, rho, mu, m.mode, block_size, repeat);
        if (s <= 0) { std::printf("%-10s timing failed\n", m.name); continue; }
        std::printf("%-10s %12.3e %14.1f %14.1f\n", m.name, s,
                    (double)(d.nnodes * 4) / s * 1e-6, (double)d.nelements / s * 1e-6);
    }

    // ---- assembled BSR Jacobian ----------------------------------------------
    std::printf("\n=== assembled BSR Jacobian ===\n");
    BSR4 bsr = make_bsr4(d.mesh);
    precompute_element_bsr_slots(d, bsr);
    std::printf("bsr    nnz=%td  values=%.1f MiB\n", bsr.nnz,
                (double)bsr.nnz * 16 * sizeof(double) / (1024.0 * 1024.0));

    // Host reference: the same kernel the device runs, Atomic=true.
    assemble_jacobian_atomic_sumfact(d, bsr, rho, mu);
    std::vector<double> href((size_t)bsr.nnz * 16);
    std::memcpy(href.data(), bsr.values->data(), href.size() * sizeof(double));
    double hmax = 0;
    for (double v : href) hmax = std::fmax(hmax, std::fabs(v));

    std::vector<int32_t> eg((size_t)8 * d.nelements);
    for (int v = 0; v < 8; ++v)
        for (ptrdiff_t e = 0; e < d.nelements; ++e)
            eg[(size_t)v * d.nelements + e] = d.elems[v][e];

    if (cvfem_cuda_bsr_attach(ctx, bsr.nnz, eg.data(), bsr.element_slots.data()) != 0) {
        std::fprintf(stderr, "bsr_attach failed\n");
        return 1;
    }

    std::vector<double> dvals(href.size());
    std::printf("%-13s %12s %12s %14s %12s\n", "variant", "max|diff|", "rel", "s/assemble", "MDOF/s");
    for (int v = 0; v < CVFEM_CUDA_JAC_N_VARIANTS; ++v) {
        if (cvfem_cuda_assemble(ctx, rho, mu, v, block_size, nullptr) != 0 ||
            cvfem_cuda_synchronize() != 0) {
            std::printf("%-13s launch failed\n", cvfem_cuda_jac_variant_name(v));
            fail = 1;
            continue;
        }
        if (cvfem_cuda_download_values(ctx, dvals.data()) != 0) return 1;
        double dmax = 0;
        for (size_t i = 0; i < href.size(); ++i)
            dmax = std::fmax(dmax, std::fabs(href[i] - dvals[i]));
        const double rel = hmax > 0 ? dmax / hmax : dmax;
        const bool   ok  = rel <= 1e-12;
        fail |= !ok;
        const double t = cvfem_cuda_time_assemble(ctx, rho, mu, v, block_size, 10);
        std::printf("%-13s %12.3e %12.3e %14.3e %12.1f  %s\n",
                    cvfem_cuda_jac_variant_name(v), dmax, rel, t,
                    t > 0 ? (double)(d.nnodes * 4) / t * 1e-6 : 0.0, ok ? "OK" : "FAIL");
    }

    cvfem_cuda_destroy(ctx);
    if (own_mpi) MPI_Finalize();
    std::printf("\n%s\n", fail ? "VERIFICATION FAILED" : "verification passed");
    return fail;
}
