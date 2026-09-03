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

#include <unistd.h>

#include <mpi.h>

#include "cvfem_hex8_layout_common.hpp"
#include "cvfem_hex8_layout_atomic.hpp"
#include "cvfem_pack_coloring.hpp"
#include "cvfem_hex8_boundary_scs.hpp"
#include "cvfem_element_coloring.hpp"

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

// ---------------------------------------------------------------- CSV output
//
// Emits the same schema cvfem_hex8_ns_upwind_bench.cpp writes, so GPU rows land in the
// files report_cvfem_bench.py and plot_cvfem_bench.py already read, with no change to
// either. Two columns are appended -- `device` and `block_size` -- which those scripts
// ignore, because they read with csv.DictReader and .get() with defaults.
//
// `threads` is left at 0 rather than being filled with the CUDA block size: it drives
// the CPU thread-scaling sweep, and a block size is not a thread count in that sense.
struct CudaCsvRow {
    const char *operation{""};
    const char *layout{""};
    const char *kernel{""};
    ptrdiff_t   pack_size{0}, cube_n{0}, nodes{0}, elements{0}, dofs{0}, bsr_nnz{0};
    double      bsr_values_MiB{0};
    int         repeat{0}, block_size{0};
    double      seconds_per_call{0}, MDOF_s{0}, MDOF_s_element_visits{0}, MELEM_s{0};
};

void csv_write(const char *path, const char *tag, const char *device,
               const std::vector<CudaCsvRow> &rows) {
    if (!path || !*path) return;
    bool empty = true;
    if (FILE *probe = std::fopen(path, "rb")) {
        std::fseek(probe, 0, SEEK_END);
        empty = std::ftell(probe) == 0;
        std::fclose(probe);
    }
    FILE *f = std::fopen(path, "ab");
    if (!f) { std::fprintf(stderr, "could not open %s\n", path); return; }
    if (empty)
        std::fprintf(f,
                     "tag,host,element,operation,layout,kernel,geom,warp,threads,pack_size,"
                     "cube_n,nodes,elements,dofs,bsr_nnz,bsr_values_MiB,repeat,"
                     "seconds_per_call,MDOF_s,MDOF_s_element_visits,MELEM_s,GFLOP_s_model,"
                     "n_colors,packs_per_color_min,packs_per_color_max,checksum,"
                     "ms_zero_global,ms_zero_local,ms_gather_u,ms_element_kernel,"
                     "ms_local_to_global,ms_ghost_reduce,device,block_size\n");
    char host[256] = {0};
    if (gethostname(host, sizeof(host) - 1) != 0) std::snprintf(host, sizeof(host), "unknown");
    for (const CudaCsvRow &r : rows)
        std::fprintf(f,
                     "%s,%s,hex8,%s,%s,%s,affine,0,0,%td,%td,%td,%td,%td,%td,%.3f,%d,"
                     "%.9e,%.6f,%.6f,%.6f,,,,,,,,,,,,%s,%d\n",
                     tag, host, r.operation, r.layout, r.kernel, r.pack_size, r.cube_n,
                     r.nodes, r.elements, r.dofs, r.bsr_nnz, r.bsr_values_MiB, r.repeat,
                     r.seconds_per_call, r.MDOF_s, r.MDOF_s_element_visits, r.MELEM_s,
                     device, r.block_size);
    std::fclose(f);
    std::printf("\nwrote %zu rows to %s\n", rows.size(), path);
}

}  // namespace

int main(int argc, char **argv) {
    double warp_amp = 0.05;  // sinusoidal shear; makes the isoparametric path non-trivial
    int    time_only = 0;    // skip host references; for the large sizes of a saturation sweep
    int mpi_ready = 0;
    MPI_Initialized(&mpi_ready);
    bool own_mpi = false;
    if (!mpi_ready) { MPI_Init(&argc, &argv); own_mpi = true; }

    int         n = 32, pack_size = 512, block_size = 128, repeat = 20;
    bool        use_sfc = true;
    std::string csv_path, csv_tag = "gpu";
    std::vector<CudaCsvRow> csv_rows;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--n" && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (a == "--pack-size" && i + 1 < argc) pack_size = std::atoi(argv[++i]);
        else if (a == "--block-size" && i + 1 < argc) block_size = std::atoi(argv[++i]);
        else if (a == "--repeat" && i + 1 < argc) repeat = std::atoi(argv[++i]);
        else if (a == "--warp" && i + 1 < argc) warp_amp = std::atof(argv[++i]);
        else if (a == "--time-only") time_only = 1;
        else if (a == "--no-sfc") use_sfc = false;
        else if (a == "--csv" && i + 1 < argc) csv_path = argv[++i];
        else if (a == "--tag" && i + 1 < argc) csv_tag = argv[++i];
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

    // Shear the mesh before the geometry is precomputed. On a perfect box the
    // isoparametric Jacobian is constant within an element and equal to the affine one,
    // so an unwarped mesh would verify the isoparametric path against a degenerate case
    // and catch nothing. This is the same warp the CPU benchmark applies.
    if (warp_amp > 0) {
        const scalar_t pi = std::acos(scalar_t(-1));
        for (ptrdiff_t i = 0; i < d.nnodes; ++i)
            d.points[0][i] += smesh::geom_t(warp_amp * std::sin(pi * scalar_t(d.points[1][i])));
    }

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

    auto mkrow = [&](const char *op, const char *layout, const char *kernel, double t) {
        CudaCsvRow r;
        r.operation = op; r.layout = layout; r.kernel = kernel;
        r.pack_size = packed.n_elements_per_pack; r.cube_n = n;
        r.nodes = d.nnodes; r.elements = d.nelements; r.dofs = d.nnodes * 4;
        r.repeat = repeat; r.block_size = block_size;
        r.seconds_per_call = t;
        if (t > 0) {
            r.MDOF_s = (double)(d.nnodes * 4) / t * 1e-6;
            r.MELEM_s = (double)d.nelements / t * 1e-6;
            r.MDOF_s_element_visits = (double)(d.nelements * 8 * 4) / t * 1e-6;
        }
        return r;
    };

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

    // Timing-only path. The host reference passes -- a full CPU assembly and residual --
    // dominate the run at the sizes needed to saturate the device, and they are not what
    // a saturation sweep is asking about. Everything here has already been verified at
    // the smaller sizes; this measures how throughput moves with the problem.
    if (time_only) {
        std::vector<double> tv((size_t)d.nnodes * 4);
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            tv[i * 4 + 0] = 0.7 * std::sin(0.011 * (double)i);
            tv[i * 4 + 1] = 0.3 * std::cos(0.017 * (double)i);
            tv[i * 4 + 2] = 0.5 * std::sin(0.023 * (double)i + 1.0);
            tv[i * 4 + 3] = 0.9 * std::cos(0.007 * (double)i + 2.0);
        }
        if (cvfem_cuda_upload_v(ctx, tv.data()) != 0) return 1;

        // The matrix is the memory limit at these sizes -- at n=192 it is ~25 GiB -- so
        // build it only while it still fits alongside everything else, and report the
        // matrix-free rows regardless.
        // The connectivity goes up unconditionally: the standard-mesh kernels need it and
        // do not need the matrix, so they must keep working past the size where the
        // matrix stops fitting.
        std::vector<int32_t> eg2((size_t)8 * d.nelements);
        for (int v = 0; v < 8; ++v)
            for (ptrdiff_t e = 0; e < d.nelements; ++e)
                eg2[(size_t)v * d.nelements + e] = d.elems[v][e];
        if (cvfem_cuda_attach_elements_global(ctx, eg2.data()) != 0) return 1;

        const double bsr_gib = (double)d.nnodes * 27.0 * 16.0 * sizeof(double) / (1 << 30);
        bool with_bsr = false;
        BSR4 tbsr;
        if (bsr_gib < 20.0) {
            tbsr = make_bsr4(d.mesh);
            precompute_element_bsr_slots(d, tbsr);
            with_bsr = (cvfem_cuda_bsr_attach(ctx, tbsr.nnz, eg2.data(),
                                              tbsr.element_slots.data(),
                                              tbsr.rowptr, tbsr.colidx) == 0);
        } else {
            std::printf("skipping assembled rows: the matrix would be %.1f GiB\n", bsr_gib);
        }

        const double dofs = (double)(d.nnodes * 4);
        auto row = [&](const char *what, double t) {
            std::printf("TIMING %-28s n=%-4d dofs=%-12td %10.4e s %10.1f MDOF/s\n",
                        what, n, d.nnodes * 4, t, t > 0 ? dofs / t * 1e-6 : 0.0);
        };
        row("residual packed",  cvfem_cuda_time_residual(ctx, rho, mu, CVFEM_CUDA_FLUSH_ATOMIC, block_size, repeat));
        row("residual standard", cvfem_cuda_time_residual_global(ctx, rho, mu, 0, block_size, repeat));
        row("jac_action packed", cvfem_cuda_time_jacobian_action(ctx, rho, mu, CVFEM_CUDA_FLUSH_ATOMIC, block_size, repeat));
        row("jac_action standard", cvfem_cuda_time_jacobian_action_global(ctx, rho, mu, 0, block_size, repeat));
        if (with_bsr) {
            row("assemble sympy", cvfem_cuda_time_assemble(ctx, rho, mu, CVFEM_CUDA_JAC_SYMPY, block_size, repeat));
            row("assemble diag",  cvfem_cuda_time_assemble_diag(ctx, rho, mu, block_size, repeat));
        }
        cvfem_cuda_destroy(ctx);
        if (own_mpi) MPI_Finalize();
        return 0;
    }

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
        csv_rows.push_back(mkrow("residual",
                                 m.mode == CVFEM_CUDA_FLUSH_TWO_PASS ? "cuda_two_pass" : "cuda_atomic",
                                 "current", s));
    }

    // ---- matrix-free Jacobian action -----------------------------------------
    std::printf("\n=== matrix-free Jacobian action, y = J(u) v ===\n");
    {
        const size_t jv_need = cvfem_cuda_jacobian_action_shmem_bytes(packed.max_actual_nodes_per_pack);
        std::printf("shared memory %zu B (%.0f%% of opt-in)\n", jv_need,
                    100.0 * (double)jv_need / (double)optin);
        if (jv_need > (size_t)optin) {
            std::printf("  EXCEEDS OPT-IN LIMIT, reduce --pack-size\n");
            fail = 1;
        } else {
            // A direction with structure, so cancellation does not hide a sign error.
            std::vector<double> vh((size_t)d.nnodes * 4), jv_h((size_t)d.nnodes * 4, 0.0);
            for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
                vh[i * 4 + 0] = 0.7 * std::sin(0.011 * (double)i);
                vh[i * 4 + 1] = 0.3 * std::cos(0.017 * (double)i);
                vh[i * 4 + 2] = 0.5 * std::sin(0.023 * (double)i + 1.0);
                vh[i * 4 + 3] = 0.9 * std::cos(0.007 * (double)i + 2.0);
            }
            apply_jacobian_action_atomic(d, rho, mu, vh.data(), jv_h.data());
            const double jvmax = max_abs(jv_h);

            if (cvfem_cuda_upload_v(ctx, vh.data()) != 0) return 1;
            for (auto &m : modes) {
                if (cvfem_cuda_jacobian_action(ctx, rho, mu, m.mode, block_size, nullptr) != 0 ||
                    cvfem_cuda_synchronize() != 0) { fail = 1; continue; }
                if (cvfem_cuda_download_r(ctx, dev.data()) != 0) return 1;
                const double diff = max_abs_diff(jv_h, dev);
                const double rel  = jvmax > 0 ? diff / jvmax : diff;
                const bool   ok   = rel <= 1e-12;
                fail |= !ok;
                const double t = cvfem_cuda_time_jacobian_action(ctx, rho, mu, m.mode,
                                                                 block_size, repeat);
                std::printf("J*v %-9s vs apply_jacobian_action_atomic: rel = %.3e  "
                            "%12.1f MDOF/s  %s\n",
                            m.name, rel, t > 0 ? (double)(d.nnodes * 4) / t * 1e-6 : 0.0,
                            ok ? "OK" : "FAIL");
                csv_rows.push_back(mkrow("jac_action",
                                         m.mode == CVFEM_CUDA_FLUSH_TWO_PASS ? "cuda_two_pass"
                                                                             : "cuda_atomic",
                                         "current", t));
            }
        }
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

    if (cvfem_cuda_bsr_attach(ctx, bsr.nnz, eg.data(), bsr.element_slots.data(),
                              bsr.rowptr, bsr.colidx) != 0) {
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
        CudaCsvRow row = mkrow("assemble", "cuda_atomic", cvfem_cuda_jac_variant_name(v), t);
        row.bsr_nnz = bsr.nnz;
        row.bsr_values_MiB = (double)bsr.nnz * 16 * sizeof(double) / (1024.0 * 1024.0);
        csv_rows.push_back(row);
    }

#ifdef CVFEM_ENABLE_SUBPAR
    // Pack colouring on the device: see subpar/cuda/cvfem_hex8_ns_cuda_colored.cuh.
    // ---- coloured assembly: does removing atomics help? ----------------------
    PackColoring colors = cvfem_build_pack_coloring(packed.n_packs, packed.owned_nodes_ptr,
                                                    packed.ghost_ptr, packed.ghost_idx);
    std::printf("\ncoloring n_colors=%d  packs/color min=%td max=%td\n",
                colors.n_colors, colors.min_packs_per_color, colors.max_packs_per_color);
    if (cvfem_cuda_coloring_attach(ctx, colors.n_colors, colors.pack_order.data(),
                                   colors.color_ptr.data()) != 0) {
        std::fprintf(stderr, "coloring_attach failed\n");
        return 1;
    }
    // Pack colouring only removes inter-pack races; with more than one thread per pack
    // the intra-pack race remains and the result is simply wrong. Run it only in the
    // configuration where it is defined, rather than reporting a meaningless FAIL.
    if (block_size != 1) {
        std::printf("col/*          skipped: pack colouring needs one thread per pack to be\n"
                    "               race-free on device (see cvfem_hex8_ns_cuda.cu). Re-run\n"
                    "               with --block-size 1 to check it, but it is ~200x slower.\n");
    }
    for (int us = 0; block_size == 1 && us < 2; ++us) {
        if (cvfem_cuda_assemble_colored(ctx, rho, mu, us, block_size, nullptr) != 0 ||
            cvfem_cuda_synchronize() != 0) {
            std::printf("%-13s launch failed\n", us ? "col/sympy" : "col/hand");
            fail = 1; continue;
        }
        if (cvfem_cuda_download_values(ctx, dvals.data()) != 0) return 1;
        double dmax = 0;
        for (size_t i = 0; i < href.size(); ++i)
            dmax = std::fmax(dmax, std::fabs(href[i] - dvals[i]));
        const double rel = hmax > 0 ? dmax / hmax : dmax;
        const bool   ok  = rel <= 1e-12;
        fail |= !ok;
        const double t = cvfem_cuda_time_assemble_colored(ctx, rho, mu, us, block_size, 10);
        std::printf("%-13s %12.3e %12.3e %14.3e %12.1f  %s\n",
                    us ? "col/sympy" : "col/hand", dmax, rel, t,
                    t > 0 ? (double)(d.nnodes * 4) / t * 1e-6 : 0.0, ok ? "OK" : "FAIL");
    }

#endif  // CVFEM_ENABLE_SUBPAR

    // ---- split assembly: do not rebuild the terms that did not change ---------
    std::printf("\n=== split assembly (linear part reused) ===\n");
    {
        // The reference is the full hand-written assembly: linear + nonlinear must
        // reproduce it, because they are the two halves of the same kernel.
        assemble_jacobian_atomic_sumfact(d, bsr, rho, mu);
        std::vector<double> ref2((size_t)bsr.nnz * 16);
        std::memcpy(ref2.data(), bsr.values->data(), ref2.size() * sizeof(double));
        double r2max = 0;
        for (double v : ref2) r2max = std::fmax(r2max, std::fabs(v));

        if (cvfem_cuda_assemble_linear(ctx, mu, block_size, nullptr) != 0 ||
            cvfem_cuda_assemble_nonlinear(ctx, rho, mu, block_size, nullptr) != 0 ||
            cvfem_cuda_synchronize() != 0) {
            std::printf("split assembly failed\n"); fail = 1;
        } else {
            if (cvfem_cuda_download_values(ctx, dvals.data()) != 0) return 1;
            double dm = 0;
            for (size_t i = 0; i < ref2.size(); ++i)
                dm = std::fmax(dm, std::fabs(ref2[i] - dvals[i]));
            const double rel = r2max > 0 ? dm / r2max : dm;
            const bool   ok  = rel <= 1e-12;
            fail |= !ok;

            const double t_full = cvfem_cuda_time_assemble(ctx, rho, mu,
                                        CVFEM_CUDA_JAC_HANDWRITTEN, block_size, 10);
            const double t_nl   = cvfem_cuda_time_assemble_nonlinear(ctx, rho, mu,
                                        block_size, 10);
            std::printf("linear+nonlinear vs full assembly: rel = %.3e  %s\n",
                        rel, ok ? "OK" : "FAIL");
            std::printf("%-34s %12s %12s\n", "", "s/assemble", "MDOF/s");
            std::printf("%-34s %12.3e %12.1f\n", "full assembly (every iteration)", t_full,
                        t_full > 0 ? (double)(d.nnodes * 4) / t_full * 1e-6 : 0.0);
            std::printf("%-34s %12.3e %12.1f\n", "restore linear + nonlinear only", t_nl,
                        t_nl > 0 ? (double)(d.nnodes * 4) / t_nl * 1e-6 : 0.0);
            // Which blocks does the nonlinear half write? Determined once, from two
            // different velocity fields, so a value that happens to vanish for one of
            // them does not drop a block from the list.
            std::vector<int32_t>  nl_blocks;
            std::vector<uint16_t> nl_masks;
            {
                std::vector<double> probe((size_t)bsr.nnz * 16, 0.0), acc((size_t)bsr.nnz * 16, 0.0);
                const std::vector<scalar_t> saved_ux = d.ux, saved_uy = d.uy, saved_uz = d.uz;
                for (int trial = 0; trial < 2; ++trial) {
                    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
                        const double t = 0.3 + 0.7 * std::sin(0.013 * (double)(i + 7 * trial));
                        d.ux[i] = t; d.uy[i] = 0.5 * t + 0.2 * trial; d.uz[i] = 0.25 - 0.4 * t;
                    }
                    assemble_jacobian_atomic_nonlinear(d, bsr, rho, mu, probe);  // probe is all zeros
                    const scalar_t *pv = bsr.values->data();
                    for (size_t i = 0; i < acc.size(); ++i) acc[i] += std::fabs(pv[i]);
                }
                d.ux = saved_ux; d.uy = saved_uy; d.uz = saved_uz;
                nl_masks.assign((size_t)bsr.nnz, 0);
                for (ptrdiff_t b = 0; b < bsr.nnz; ++b) {
                    uint16_t m = 0;
                    for (int k = 0; k < 16; ++k)
                        if (acc[(size_t)b * 16 + k] != 0.0) m |= (uint16_t)(1u << k);
                    nl_masks[(size_t)b] = m;
                    if (m) nl_blocks.push_back((int32_t)b);
                }
                size_t changed = 0;
                for (uint16_t m : nl_masks) changed += (size_t)__builtin_popcount(m);
                std::printf("entries that change per iteration: %zu of %td (%.1f%%)\n",
                            changed, bsr.nnz * 16,
                            100.0 * (double)changed / (double)(bsr.nnz * 16));
                std::printf("blocks written by the nonlinear half: %zu of %td (%.1f%%)\n",
                            nl_blocks.size(), bsr.nnz,
                            100.0 * (double)nl_blocks.size() / (double)bsr.nnz);
            }
            double t_sparse = -1.0;
            if (cvfem_cuda_nonlinear_blocks_attach(ctx, (ptrdiff_t)nl_blocks.size(),
                                                   nl_blocks.data(), nl_masks.data()) == 0) {
                if (cvfem_cuda_assemble_linear(ctx, mu, block_size, nullptr) == 0 &&
                    cvfem_cuda_assemble_nonlinear_sparse(ctx, rho, mu, block_size, nullptr) == 0 &&
                    cvfem_cuda_synchronize() == 0 &&
                    cvfem_cuda_download_values(ctx, dvals.data()) == 0) {
                    double dm2 = 0;
                    for (size_t i = 0; i < ref2.size(); ++i)
                        dm2 = std::fmax(dm2, std::fabs(ref2[i] - dvals[i]));
                    const double rel2 = r2max > 0 ? dm2 / r2max : dm2;
                    const bool ok2 = rel2 <= 1e-12;
                    fail |= !ok2;
                    t_sparse = cvfem_cuda_time_assemble_nonlinear_sparse(ctx, rho, mu, block_size, 10);
                    std::printf("sparse-restore vs full assembly: rel = %.3e  %s\n",
                                rel2, ok2 ? "OK" : "FAIL");
                }
            }
            const double t_restore = cvfem_cuda_time_restore_only(ctx, 10);
            const double t_nlonly   = cvfem_cuda_time_nonlinear_only(ctx, rho, mu, block_size, 10);
            std::printf("%-34s %12.3e %12.1f\n", "  ...of which: restore copy", t_restore,
                        t_restore > 0 ? (double)(d.nnodes * 4) / t_restore * 1e-6 : 0.0);
            std::printf("%-34s %12.3e %12.1f\n", "  ...of which: nonlinear kernel", t_nlonly,
                        t_nlonly > 0 ? (double)(d.nnodes * 4) / t_nlonly * 1e-6 : 0.0);
            if (t_nl > 0 && t_restore > 0)
                std::printf("restore is %.0f%% of the split's cost; without it the ceiling is %.1f MDOF/s\n",
                            100.0 * t_restore / t_nl,
                            t_nlonly > 0 ? (double)(d.nnodes * 4) / t_nlonly * 1e-6 : 0.0);
            // Single-matrix variant: no side buffer at all.
            double t_dyn = -1.0;
            if (cvfem_cuda_assemble_static(ctx, mu, block_size, nullptr) == 0 &&
                cvfem_cuda_assemble_dynamic(ctx, rho, mu, block_size, nullptr) == 0 &&
                cvfem_cuda_synchronize() == 0 &&
                cvfem_cuda_download_values(ctx, dvals.data()) == 0) {
                double dm3 = 0;
                for (size_t i = 0; i < ref2.size(); ++i)
                    dm3 = std::fmax(dm3, std::fabs(ref2[i] - dvals[i]));
                const double rel3 = r2max > 0 ? dm3 / r2max : dm3;
                const bool ok3 = rel3 <= 1e-12;
                fail |= !ok3;
                t_dyn = cvfem_cuda_time_assemble_dynamic(ctx, rho, mu, block_size, 10);
                std::printf("zero+recompute (one matrix) vs full: rel = %.3e  %s\n",
                            rel3, ok3 ? "OK" : "FAIL");
            }
            std::printf("side buffer held: %.1f MiB\n",
                        (double)cvfem_cuda_linear_side_bytes(ctx) / (1024.0 * 1024.0));
            if (t_dyn > 0)
                std::printf("%-34s %12.3e %12.1f\n", "zero + recompute, ONE matrix", t_dyn,
                            (double)(d.nnodes * 4) / t_dyn * 1e-6);
            if (t_sparse > 0)
                std::printf("%-34s %12.3e %12.1f\n", "restore touched blocks only", t_sparse,
                            (double)(d.nnodes * 4) / t_sparse * 1e-6);
            if (t_full > 0 && t_nl > 0)
                std::printf("per Newton iteration: full->split %.2fx", t_full / t_nl);
            if (t_full > 0 && t_sparse > 0)
                std::printf(", full->sparse-restore %.2fx", t_full / t_sparse);
            std::printf("\n");
        }
    }

    // ---- element-coloured assembly: remove the atomics ------------------------
    std::printf("\n=== element-coloured assembly (no atomics) ===\n");
    {
        ElementColoring ec = cvfem_build_element_coloring(d.nelements, d.nnodes, d.elems);
        std::printf("element colours: %d   elements/colour min=%td max=%td\n",
                    ec.n_colors, ec.min_per_color, ec.max_per_color);
        if (cvfem_cuda_element_coloring_attach(ctx, ec.n_colors, ec.element_order.data(),
                                               ec.color_ptr.data()) != 0) {
            std::printf("element_coloring_attach failed\n"); fail = 1;
        } else {
            std::printf("%-13s %12s %14s %12s %10s\n", "variant", "rel", "s/assemble",
                        "MDOF/s", "vs atomic");
            for (int v = 0; v < CVFEM_CUDA_JAC_N_VARIANTS; ++v) {
#ifndef CVFEM_ENABLE_SUBPAR
                // Colouring is only worth it for the fused kernels; see launch_ecolored.
                if (v == CVFEM_CUDA_JAC_HANDWRITTEN) continue;
#endif
                if (cvfem_cuda_assemble_ecolored(ctx, rho, mu, v, block_size, nullptr) != 0 ||
                    cvfem_cuda_synchronize() != 0) {
                    std::printf("%-13s launch failed\n", cvfem_cuda_jac_variant_name(v));
                    fail = 1; continue;
                }
                if (cvfem_cuda_download_values(ctx, dvals.data()) != 0) return 1;
                double dm = 0;
                for (size_t i = 0; i < href.size(); ++i)
                    dm = std::fmax(dm, std::fabs(href[i] - dvals[i]));
                const double rel = hmax > 0 ? dm / hmax : dm;
                const bool   ok  = rel <= 1e-12;
                fail |= !ok;
                const double t  = cvfem_cuda_time_assemble_ecolored(ctx, rho, mu, v, block_size, 10);
                const double ta = cvfem_cuda_time_assemble(ctx, rho, mu, v, block_size, 10);
                std::printf("%-13s %12.3e %14.3e %12.1f %9.2fx  %s\n",
                            cvfem_cuda_jac_variant_name(v), rel, t,
                            t > 0 ? (double)(d.nnodes * 4) / t * 1e-6 : 0.0,
                            (t > 0 && ta > 0) ? ta / t : 0.0, ok ? "OK" : "FAIL");
            }
        }
    }

    // ---- block diagonal for the block-Jacobi preconditioner -------------------
    std::printf("\n=== block diagonal (block-Jacobi preconditioner) ===\n");
    {
        // Reference: the diagonal blocks of the full assembly.
        assemble_jacobian_atomic_sumfact(d, bsr, rho, mu);
        const scalar_t *hv = bsr.values->data();
        std::vector<double> ref_diag((size_t)d.nnodes * 16, 0.0);
        for (ptrdiff_t r = 0; r < d.nnodes; ++r)
            for (smesh::count_t j = bsr.rowptr[r]; j < bsr.rowptr[r + 1]; ++j)
                if (bsr.colidx[j] == (smesh::idx_t)r)
                    std::memcpy(&ref_diag[(size_t)r * 16], &hv[(size_t)j * 16], 16 * sizeof(double));
        double dmax = 0;
        for (double v : ref_diag) dmax = std::fmax(dmax, std::fabs(v));

        std::vector<double> dev_diag((size_t)d.nnodes * 16);
        const double mib = (double)d.nnodes * 16 * sizeof(double) / (1024.0 * 1024.0);
        const double full_mib = (double)bsr.nnz * 16 * sizeof(double) / (1024.0 * 1024.0);
        std::printf("diagonal is %.1f MiB against the full matrix's %.1f MiB (%.0fx smaller)\n",
                    mib, full_mib, full_mib / mib);

        if (cvfem_cuda_assemble_diag(ctx, rho, mu, block_size, nullptr) == 0 &&
            cvfem_cuda_synchronize() == 0 &&
            cvfem_cuda_download_diag(ctx, dev_diag.data()) == 0) {
            const double rel = dmax > 0 ? max_abs_diff(ref_diag, dev_diag) / dmax : 0.0;
            const bool ok = rel <= 1e-12;
            fail |= !ok;
            const double t = cvfem_cuda_time_assemble_diag(ctx, rho, mu, block_size, 10);
            std::printf("%-30s rel = %.3e  %10.3e s  %10.1f MDOF/s  %s\n",
                        "diagonal, full rebuild", rel, t,
                        t > 0 ? (double)(d.nnodes * 4) / t * 1e-6 : 0.0, ok ? "OK" : "FAIL");
        }
        if (cvfem_cuda_assemble_diag_static(ctx, mu, block_size, nullptr) == 0 &&
            cvfem_cuda_assemble_diag_dynamic(ctx, rho, mu, block_size, nullptr) == 0 &&
            cvfem_cuda_synchronize() == 0 &&
            cvfem_cuda_download_diag(ctx, dev_diag.data()) == 0) {
            const double rel = dmax > 0 ? max_abs_diff(ref_diag, dev_diag) / dmax : 0.0;
            const bool ok = rel <= 1e-12;
            fail |= !ok;
            const double t = cvfem_cuda_time_assemble_diag_dynamic(ctx, rho, mu, block_size, 10);
            std::printf("%-30s rel = %.3e  %10.3e s  %10.1f MDOF/s  %s\n",
                        "diagonal, split rebuild", rel, t,
                        t > 0 ? (double)(d.nnodes * 4) / t * 1e-6 : 0.0, ok ? "OK" : "FAIL");
        }
        // The preconditioner block, against the same routine applied on the host.
        if (cvfem_cuda_assemble_diag(ctx, rho, mu, block_size, nullptr) == 0 &&
            cvfem_cuda_invert_diag(ctx, block_size, nullptr) == 0 &&
            cvfem_cuda_synchronize() == 0 &&
            cvfem_cuda_download_diag(ctx, dev_diag.data()) == 0) {
            std::vector<double> host_inv((size_t)d.nnodes * 16);
            for (ptrdiff_t n2 = 0; n2 < d.nnodes; ++n2)
                cvfem_hex8_block_jacobi_block(&ref_diag[(size_t)n2 * 16], (const unsigned char *)nullptr,
                                              &host_inv[(size_t)n2 * 16]);
            double hm = 0;
            for (double v : host_inv) hm = std::fmax(hm, std::fabs(v));
            const double rel = hm > 0 ? max_abs_diff(host_inv, dev_diag) / hm : 0.0;
            const bool ok = rel <= 1e-12;
            fail |= !ok;
            std::printf("%-30s rel = %.3e  %s   (3x3 velocity inverse + scalar pressure)\n",
                        "block-Jacobi block", rel, ok ? "OK" : "FAIL");
        }
    }

    // ---- cuSPARSE BSR SpMV vs the matrix-free Jacobian action ----------------
    //
    // The question this answers: on a GPU, is it worth applying J matrix-free at all,
    // or is assembling once and multiplying with a vendor kernel faster? On the CPU the
    // benchmark already asks this via --bsr-apply.
    std::printf("\n=== cuSPARSE BSR SpMV vs matrix-free J*v ===\n");
    {
        std::vector<double> vh((size_t)d.nnodes * 4), jv_h((size_t)d.nnodes * 4, 0.0);
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            vh[i * 4 + 0] = 0.7 * std::sin(0.011 * (double)i);
            vh[i * 4 + 1] = 0.3 * std::cos(0.017 * (double)i);
            vh[i * 4 + 2] = 0.5 * std::sin(0.023 * (double)i + 1.0);
            vh[i * 4 + 3] = 0.9 * std::cos(0.007 * (double)i + 2.0);
        }
        apply_jacobian_action_atomic(d, rho, mu, vh.data(), jv_h.data());
        const double jvmax = max_abs(jv_h);

        if (cvfem_cuda_upload_v(ctx, vh.data()) != 0) return 1;
        // Assemble first: SpMV multiplies whatever is in the values array.
        if (cvfem_cuda_assemble(ctx, rho, mu, CVFEM_CUDA_JAC_SYMPY, block_size, nullptr) != 0 ||
            cvfem_cuda_spmv(ctx, nullptr) != 0 || cvfem_cuda_synchronize() != 0) {
            std::printf("cuSPARSE SpMV failed\n"); fail = 1;
        } else {
            if (cvfem_cuda_download_r(ctx, dev.data()) != 0) return 1;
            const double rel = jvmax > 0 ? max_abs_diff(jv_h, dev) / jvmax : 0.0;
            const bool   ok  = rel <= 1e-12;
            fail |= !ok;
            const double t_spmv = cvfem_cuda_time_spmv(ctx, repeat);
            const double t_mf   = cvfem_cuda_time_jacobian_action(ctx, rho, mu,
                                        CVFEM_CUDA_FLUSH_ATOMIC, block_size, repeat);
            std::printf("cuSPARSE bsrmv vs host J*v: rel = %.3e  %s\n", rel, ok ? "OK" : "FAIL");
            std::printf("%-22s %12s %12s\n", "", "s/apply", "MDOF/s");
            std::printf("%-22s %12.3e %12.1f\n", "cuSPARSE BSR SpMV", t_spmv,
                        t_spmv > 0 ? (double)(d.nnodes * 4) / t_spmv * 1e-6 : 0.0);
            std::printf("%-22s %12.3e %12.1f\n", "matrix-free J*v", t_mf,
                        t_mf > 0 ? (double)(d.nnodes * 4) / t_mf * 1e-6 : 0.0);
            CudaCsvRow sp = mkrow("bsr_apply", "cuda_cusparse", "bsrmv", t_spmv);
            sp.bsr_nnz = bsr.nnz;
            sp.bsr_values_MiB = (double)bsr.nnz * 16 * sizeof(double) / (1024.0 * 1024.0);
            csv_rows.push_back(sp);
            if (t_spmv > 0 && t_mf > 0)
                std::printf("matrix-free is %.2fx %s than one SpMV; assembly costs %.1f SpMVs\n",
                            t_spmv > t_mf ? t_spmv / t_mf : t_mf / t_spmv,
                            t_spmv > t_mf ? "faster" : "slower",
                            cvfem_cuda_time_assemble(ctx, rho, mu, CVFEM_CUDA_JAC_SYMPY,
                                                     block_size, 5) / t_spmv);
        }
    }

    // ---- boundary sub-control-surface residual -------------------------------
    // ---- isoparametric geometry ---------------------------------------------
    //
    // The affine path uses one precomputed adjugate and determinant per element; the
    // isoparametric path rebuilds the trilinear Jacobian at each of the 12
    // sub-control-surface points. On a sheared mesh the two give genuinely different
    // answers, so each device kernel is checked against its own host counterpart rather
    // than against the affine result.
    // ---- packed mesh against standard mesh -----------------------------------
    //
    // Both compute the same operator; they differ only in how the mesh is addressed.
    // The packed form pays for a staging pass and ghost bookkeeping and gets locality
    // and a cheaper flush; the standard form does none of that and atomics straight to
    // global. This is the same question the CPU answers with `atomic` vs `packed`.
    // ---- bit-reproducible residual -------------------------------------------
    //
    // The open question this closes: neither existing flush mode gives the same bits
    // twice, because both accumulate a pack's elements with shared-memory atomicAdd.
    // This mode orders the accumulation instead -- store per element, then gather per
    // node in element order -- and pays for it in a scratch array.
    std::printf("\n=== bit-reproducible residual ===\n");
    {
        // node -> element CSR, entries in increasing element order so the gather's sum
        // has a fixed order. enc packs element * 8 + local index.
        std::vector<ptrdiff_t> n2e_ptr((size_t)d.nnodes + 1, 0);
        for (ptrdiff_t e = 0; e < d.nelements; ++e)
            for (int a = 0; a < 8; ++a) n2e_ptr[(size_t)d.elems[a][e] + 1]++;
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) n2e_ptr[i + 1] += n2e_ptr[i];
        std::vector<int32_t>   n2e_enc((size_t)n2e_ptr[d.nnodes]);
        std::vector<ptrdiff_t> cur(n2e_ptr.begin(), n2e_ptr.end() - 1);
        for (ptrdiff_t e = 0; e < d.nelements; ++e)
            for (int a = 0; a < 8; ++a)
                n2e_enc[(size_t)cur[d.elems[a][e]]++] = (int32_t)(e * 8 + a);

        if (cvfem_cuda_attach_node_to_element(ctx, n2e_ptr.data(), n2e_enc.data(),
                                              (ptrdiff_t)n2e_enc.size()) != 0) {
            std::fprintf(stderr, "attach_node_to_element failed\n");
            return 1;
        }
        std::printf("node->element CSR: %td entries, %.1f MiB scratch for the element store\n",
                    (ptrdiff_t)n2e_enc.size(),
                    (double)d.nelements * 32 * sizeof(double) / (1024.0 * 1024.0));

        std::vector<double> det1(ref.size()), det2(ref.size());
        if (cvfem_cuda_residual_deterministic(ctx, rho, mu, 0, block_size, nullptr) != 0 ||
            cvfem_cuda_synchronize() != 0 || cvfem_cuda_download_r(ctx, det1.data()) != 0)
            return 1;
        // Same call again, and at a different block size: neither may change a bit.
        if (cvfem_cuda_residual_deterministic(ctx, rho, mu, 0, block_size == 64 ? 256 : 64,
                                              nullptr) != 0 ||
            cvfem_cuda_synchronize() != 0 || cvfem_cuda_download_r(ctx, det2.data()) != 0)
            return 1;

        const bool bitwise = std::memcmp(det1.data(), det2.data(),
                                         det1.size() * sizeof(double)) == 0;
        const double rel = max_abs_diff(ref, det1) / (refmax > 0 ? refmax : 1.0);
        fail |= !(rel <= 1e-12) || !bitwise;
        std::printf("vs host: rel = %.3e  %s\n", rel, rel <= 1e-12 ? "OK" : "FAIL");
        std::printf("bit-identical across runs and block sizes: %s\n",
                    bitwise ? "YES" : "NO -- FAIL");

        const double td = cvfem_cuda_time_residual_deterministic(ctx, rho, mu, 0, block_size, repeat);
        // Compared against the STANDARD-mesh kernel, not the packed one. The
        // deterministic kernel is grid-stride over global ids, so that is its structural
        // twin; timing it against the packed form would flatter it by charging the packed
        // form's staging overhead to the baseline.
        const double tg = cvfem_cuda_time_residual_global(ctx, rho, mu, 0, block_size, repeat);
        std::printf("%-34s %10.4e s %9.1f MDOF/s\n", "deterministic", td,
                    td > 0 ? (double)(d.nnodes * 4) / td * 1e-6 : 0.0);
        std::printf("%-34s %10.4e s %9.1f MDOF/s   (reproducibility costs %.2fx)\n",
                    "standard mesh, same shape", tg,
                    tg > 0 ? (double)(d.nnodes * 4) / tg * 1e-6 : 0.0,
                    (td > 0 && tg > 0) ? td / tg : 0.0);
        csv_rows.push_back(mkrow("residual", "cuda_deterministic", "current", td));
    }

    std::printf("\n=== packed mesh vs standard mesh, matrix-free ===\n");
    {
        // Own direction and own host references, so this section does not depend on
        // buffers left behind by an earlier block.
        std::vector<double> gvh((size_t)d.nnodes * 4), jv_ref((size_t)d.nnodes * 4, 0.0);
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            gvh[i * 4 + 0] = 0.7 * std::sin(0.011 * (double)i);
            gvh[i * 4 + 1] = 0.3 * std::cos(0.017 * (double)i);
            gvh[i * 4 + 2] = 0.5 * std::sin(0.023 * (double)i + 1.0);
            gvh[i * 4 + 3] = 0.9 * std::cos(0.007 * (double)i + 2.0);
        }
        apply_jacobian_action_atomic(d, rho, mu, gvh.data(), jv_ref.data());
        if (cvfem_cuda_upload_v(ctx, gvh.data()) != 0) return 1;

        std::printf("%-38s %12s %12s %10s\n", "operator / mesh", "s/call", "MDOF/s", "rel");

        struct Row { const char *name; bool jv; bool packed; };
        const Row rows_mf[] = {
            {"residual, packed mesh",   false, true},
            {"residual, standard mesh", false, false},
            {"J*v, packed mesh",        true,  true},
            {"J*v, standard mesh",      true,  false},
        };
        double t_pack_r = 0, t_glob_r = 0, t_pack_j = 0, t_glob_j = 0;
        for (const auto &rw : rows_mf) {
            int rc;
            if (rw.packed)
                rc = rw.jv ? cvfem_cuda_jacobian_action(ctx, rho, mu, CVFEM_CUDA_FLUSH_ATOMIC,
                                                        block_size, nullptr)
                           : cvfem_cuda_residual(ctx, rho, mu, CVFEM_CUDA_FLUSH_ATOMIC,
                                                 block_size, nullptr);
            else
                rc = rw.jv ? cvfem_cuda_jacobian_action_global(ctx, rho, mu, 0, block_size, nullptr)
                           : cvfem_cuda_residual_global(ctx, rho, mu, 0, block_size, nullptr);
            if (rc != 0 || cvfem_cuda_synchronize() != 0) {
                std::printf("%-38s launch failed\n", rw.name); fail = 1; continue;
            }
            if (cvfem_cuda_download_r(ctx, dev.data()) != 0) return 1;

            // Against the host reference for the matching operator.
            const std::vector<double> &hostref = rw.jv ? jv_ref : ref;
            const double hmax = max_abs(hostref);
            const double rel  = max_abs_diff(hostref, dev) / (hmax > 0 ? hmax : 1.0);
            const bool   ok   = rel <= 1e-12;
            fail |= !ok;

            const double t = rw.packed
                    ? (rw.jv ? cvfem_cuda_time_jacobian_action(ctx, rho, mu, CVFEM_CUDA_FLUSH_ATOMIC,
                                                               block_size, repeat)
                             : cvfem_cuda_time_residual(ctx, rho, mu, CVFEM_CUDA_FLUSH_ATOMIC,
                                                        block_size, repeat))
                    : (rw.jv ? cvfem_cuda_time_jacobian_action_global(ctx, rho, mu, 0, block_size, repeat)
                             : cvfem_cuda_time_residual_global(ctx, rho, mu, 0, block_size, repeat));
            if (rw.jv && rw.packed)  t_pack_j = t;
            if (rw.jv && !rw.packed) t_glob_j = t;
            if (!rw.jv && rw.packed)  t_pack_r = t;
            if (!rw.jv && !rw.packed) t_glob_r = t;

            std::printf("%-38s %12.3e %12.1f %10.2e %s\n", rw.name, t,
                        t > 0 ? (double)(d.nnodes * 4) / t * 1e-6 : 0.0, rel, ok ? "OK" : "FAIL");
            csv_rows.push_back(mkrow(rw.jv ? "jac_action" : "residual",
                                     rw.packed ? "cuda_packed" : "cuda_standard", "current", t));
        }
        if (t_pack_r > 0 && t_glob_r > 0)
            std::printf("packed mesh is %.2fx the standard mesh on the residual, %.2fx on J*v\n",
                        t_glob_r / t_pack_r, (t_pack_j > 0 && t_glob_j > 0) ? t_glob_j / t_pack_j : 0.0);

        // ---- the same question for assembly ---------------------------------
        //
        // Assembly writes 64 blocks x 16 doubles per element and reads 32 doubles, so
        // the write side dominates and is identical in both forms. What is being
        // compared is purely the gather.
        assemble_jacobian_atomic_sumfact(d, bsr, rho, mu);
        std::vector<double> asm_ref(bsr.values->data(), bsr.values->data() + (size_t)bsr.nnz * 16);
        double amax = 0;
        for (double v : asm_ref) amax = std::fmax(amax, std::fabs(v));
        std::vector<double> asm_dev(asm_ref.size());

        struct ARow { const char *name; bool packed; int variant; };
        const ARow rows_asm[] = {
            {"assembly, standard mesh (handwritten)", false, CVFEM_CUDA_JAC_HANDWRITTEN},
            {"assembly, packed mesh   (handwritten)", true,  CVFEM_CUDA_JAC_HANDWRITTEN},
            {"assembly, standard mesh (sympy)",       false, CVFEM_CUDA_JAC_SYMPY},
            {"assembly, packed mesh   (sympy)",       true,  CVFEM_CUDA_JAC_SYMPY},
        };
        double t_asm[4] = {0, 0, 0, 0};
        int    ai       = 0;
        for (const auto &ar : rows_asm) {
            const int rc = ar.packed
                    ? cvfem_cuda_assemble_packed(ctx, rho, mu, ar.variant, 0, block_size, nullptr)
                    : cvfem_cuda_assemble(ctx, rho, mu, ar.variant, block_size, nullptr);
            if (rc != 0 || cvfem_cuda_synchronize() != 0) {
                std::printf("%-38s launch failed\n", ar.name); fail = 1; ++ai; continue;
            }
            if (cvfem_cuda_download_values(ctx, asm_dev.data()) != 0) return 1;
            const double rel = max_abs_diff(asm_ref, asm_dev) / (amax > 0 ? amax : 1.0);
            const bool   ok  = rel <= 1e-12;
            fail |= !ok;
            const double t = ar.packed
                    ? cvfem_cuda_time_assemble_packed(ctx, rho, mu, ar.variant, 0, block_size, repeat)
                    : cvfem_cuda_time_assemble(ctx, rho, mu, ar.variant, block_size, repeat);
            t_asm[ai++] = t;
            std::printf("%-38s %12.3e %12.1f %10.2e %s\n", ar.name, t,
                        t > 0 ? (double)(d.nnodes * 4) / t * 1e-6 : 0.0, rel, ok ? "OK" : "FAIL");
            csv_rows.push_back(mkrow("assemble", ar.packed ? "cuda_packed" : "cuda_standard",
                                     cvfem_cuda_jac_variant_name(ar.variant), t));
        }
        if (t_asm[0] > 0 && t_asm[1] > 0)
            std::printf("packed mesh is %.2fx the standard mesh on assembly (handwritten), "
                        "%.2fx (sympy)\n",
                        t_asm[0] / t_asm[1],
                        (t_asm[2] > 0 && t_asm[3] > 0) ? t_asm[2] / t_asm[3] : 0.0);
    }

    std::printf("\n=== isoparametric geometry (warp = %.3f) ===\n", warp_amp);
    {
        std::vector<double> cx((size_t)d.nnodes), cy((size_t)d.nnodes), cz((size_t)d.nnodes);
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            cx[i] = d.points[0][i]; cy[i] = d.points[1][i]; cz[i] = d.points[2][i];
        }
        if (cvfem_cuda_attach_coords(ctx, cx.data(), cy.data(), cz.data()) != 0) {
            std::fprintf(stderr, "attach_coords failed\n");
            return 1;
        }

        const size_t iso_r  = cvfem_cuda_residual_isoparam_shmem_bytes(packed.max_actual_nodes_per_pack);
        const size_t iso_jv = cvfem_cuda_jacobian_action_isoparam_shmem_bytes(packed.max_actual_nodes_per_pack);
        std::printf("shared memory: residual %zu B (%.0f%% of opt-in), J*v %zu B (%.0f%%)\n",
                    iso_r, 100.0 * (double)iso_r / (double)optin,
                    iso_jv, 100.0 * (double)iso_jv / (double)optin);
        if (iso_jv > (size_t)optin) {
            std::printf("  -- EXCEEDS OPT-IN LIMIT, reduce --pack-size\n");
            return 1;
        }

        // --- residual -------------------------------------------------------
        apply_residual_atomic_isoparam(d, rho, mu);
        std::vector<double> iref;
        residual_soa_to_interleaved(d, iref);
        const double irefmax = max_abs(iref);

        // Reported for information, not as an acceptance criterion. On an unwarped box
        // the two formulations agree to rounding (1e-18); under this shear they separate
        // to ~1e-9 and then stay there, essentially independent of the amplitude -- the
        // deformation is smooth on the scale of an element, so each hex stays very nearly
        // a parallelepiped however far the domain is sheared. Raising --warp does not
        // widen the gap; it was measured on the host at 0.1 and 0.4 and does not move.
        //
        // So the check that has teeth is device-against-host below, not this number. What
        // this number does confirm is that the isoparametric path is being taken at all:
        // it is 1e-18 when the mesh is a box and 1e-9 when it is not.
        const double vs_affine = max_abs_diff(ref, iref) / (refmax > 0 ? refmax : 1.0);
        std::printf("isoparam vs affine on this mesh: rel = %.3e  (%s)\n", vs_affine,
                    vs_affine > 1e-12 ? "isoparametric path active"
                                      : "degenerate -- mesh is affine, pass --warp");

        for (auto &m : modes) {
            if (cvfem_cuda_residual_isoparam(ctx, rho, mu, m.mode, block_size, nullptr) != 0) {
                std::fprintf(stderr, "residual_isoparam failed\n");
                return 1;
            }
            if (cvfem_cuda_synchronize() != 0) return 1;
            if (cvfem_cuda_download_r(ctx, dev.data()) != 0) return 1;
            const double diff = max_abs_diff(iref, dev);
            const double rel  = irefmax > 0 ? diff / irefmax : diff;
            const bool   ok   = rel <= 1e-12;
            fail |= !ok;
            std::printf("residual isoparam %-9s vs host: max|diff| = %.3e  rel = %.3e  %s\n",
                        m.name, diff, rel, ok ? "OK" : "FAIL");
        }

        // --- Jacobian action ------------------------------------------------
        // Same structured direction the affine check uses, rebuilt here because that
        // one is block-scoped.
        std::vector<double> ivh((size_t)d.nnodes * 4), ijv_h((size_t)d.nnodes * 4, 0.0);
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            ivh[i * 4 + 0] = 0.7 * std::sin(0.011 * (double)i);
            ivh[i * 4 + 1] = 0.3 * std::cos(0.017 * (double)i);
            ivh[i * 4 + 2] = 0.5 * std::sin(0.023 * (double)i + 1.0);
            ivh[i * 4 + 3] = 0.9 * std::cos(0.007 * (double)i + 2.0);
        }
        apply_jacobian_action_atomic_isoparam(d, rho, mu, ivh.data(), ijv_h.data());
        const double jrefmax = max_abs(ijv_h);
        if (cvfem_cuda_upload_v(ctx, ivh.data()) != 0) return 1;
        for (auto &m : modes) {
            if (cvfem_cuda_jacobian_action_isoparam(ctx, rho, mu, m.mode, block_size, nullptr) != 0) {
                std::fprintf(stderr, "jacobian_action_isoparam failed\n");
                return 1;
            }
            if (cvfem_cuda_synchronize() != 0) return 1;
            if (cvfem_cuda_download_r(ctx, dev.data()) != 0) return 1;
            const double rel = max_abs_diff(ijv_h, dev) / (jrefmax > 0 ? jrefmax : 1.0);
            const bool   ok  = rel <= 1e-12;
            fail |= !ok;
            std::printf("J*v isoparam %-9s vs host: rel = %.3e  %s\n",
                        m.name, rel, ok ? "OK" : "FAIL");
        }

        // --- assembled Jacobian ---------------------------------------------
        assemble_jacobian_atomic_isoparam(d, bsr, rho, mu);
        std::vector<double> iso_vals(bsr.values->data(), bsr.values->data() + (size_t)bsr.nnz * 16);
        const double        vmax = max_abs(iso_vals);
        if (cvfem_cuda_assemble_isoparam(ctx, rho, mu, block_size, nullptr) != 0) {
            std::fprintf(stderr, "assemble_isoparam failed\n");
            return 1;
        }
        if (cvfem_cuda_synchronize() != 0) return 1;
        std::vector<double> idvals(iso_vals.size());
        if (cvfem_cuda_download_values(ctx, idvals.data()) != 0) return 1;
        const double arel = max_abs_diff(iso_vals, idvals) / (vmax > 0 ? vmax : 1.0);
        const bool   aok  = arel <= 1e-12;
        fail |= !aok;
        std::printf("assemble isoparam vs host: rel = %.3e  %s\n", arel, aok ? "OK" : "FAIL");

        // --- the other BSR-assembly strategies, all against the same host matrix ---
        //
        // Every one of these must reproduce iso_vals exactly: element colouring only
        // changes the write order, and the split only changes when each half is
        // computed, not what it computes.
        struct { const char *name; int (*fn)(cvfem_cuda_ctx *, double, double, int, void *); } strat[] = {
            {"element-coloured", cvfem_cuda_assemble_ecolored_isoparam},
            {"sympy (generated)", cvfem_cuda_assemble_isoparam_sympy},
        };
        for (auto &st : strat) {
            if (st.fn(ctx, rho, mu, block_size, nullptr) != 0 || cvfem_cuda_synchronize() != 0) {
                std::fprintf(stderr, "%s isoparam failed\n", st.name);
                return 1;
            }
            if (cvfem_cuda_download_values(ctx, idvals.data()) != 0) return 1;
            const double r = max_abs_diff(iso_vals, idvals) / (vmax > 0 ? vmax : 1.0);
            fail |= !(r <= 1e-12);
            std::printf("assemble isoparam %-18s vs host: rel = %.3e  %s\n",
                        st.name, r, r <= 1e-12 ? "OK" : "FAIL");
        }

        // Split: build the constant half once, then the velocity-dependent half.
        if (cvfem_cuda_assemble_linear_isoparam(ctx, mu, block_size, nullptr) != 0 ||
            cvfem_cuda_assemble_nonlinear_isoparam(ctx, rho, mu, block_size, nullptr) != 0 ||
            cvfem_cuda_synchronize() != 0) {
            std::fprintf(stderr, "split isoparam failed\n");
            return 1;
        }
        if (cvfem_cuda_download_values(ctx, idvals.data()) != 0) return 1;
        {
            const double r = max_abs_diff(iso_vals, idvals) / (vmax > 0 ? vmax : 1.0);
            fail |= !(r <= 1e-12);
            std::printf("assemble isoparam %-18s vs host: rel = %.3e  %s\n",
                        "split", r, r <= 1e-12 ? "OK" : "FAIL");
        }

        // Block diagonal: the 4x4 diagonal blocks only, against the same host matrix
        // read at its diagonal.
        if (cvfem_cuda_assemble_diag_isoparam(ctx, rho, mu, block_size, nullptr) != 0 ||
            cvfem_cuda_synchronize() != 0) {
            std::fprintf(stderr, "diag isoparam failed\n");
            return 1;
        }
        {
            std::vector<double> ddiag((size_t)d.nnodes * 16);
            if (cvfem_cuda_download_diag(ctx, ddiag.data()) != 0) return 1;
            // Pull the diagonal blocks out of the host matrix for comparison.
            std::vector<double> hdiag((size_t)d.nnodes * 16, 0.0);
            for (ptrdiff_t r = 0; r < d.nnodes; ++r)
                for (smesh::count_t k = bsr.rowptr[r]; k < bsr.rowptr[r + 1]; ++k)
                    if (bsr.colidx[k] == (smesh::idx_t)r)
                        std::memcpy(&hdiag[(size_t)r * 16], &iso_vals[(size_t)k * 16],
                                    16 * sizeof(double));
            const double dmaxv = max_abs(hdiag);
            const double r = max_abs_diff(hdiag, ddiag) / (dmaxv > 0 ? dmaxv : 1.0);
            fail |= !(r <= 1e-12);
            std::printf("assemble isoparam %-18s vs host: rel = %.3e  %s\n",
                        "block diagonal", r, r <= 1e-12 ? "OK" : "FAIL");
        }

        // --- throughput ------------------------------------------------------
        std::printf("%-34s %12s %12s\n", "variant", "s/call", "MDOF/s");
        struct { const char *name; double t; } iso_rows[] = {
            {"residual isoparam two_pass",
             cvfem_cuda_time_residual_isoparam(ctx, rho, mu, CVFEM_CUDA_FLUSH_TWO_PASS, block_size, repeat)},
            {"residual isoparam atomic",
             cvfem_cuda_time_residual_isoparam(ctx, rho, mu, CVFEM_CUDA_FLUSH_ATOMIC, block_size, repeat)},
            {"J*v isoparam two_pass",
             cvfem_cuda_time_jacobian_action_isoparam(ctx, rho, mu, CVFEM_CUDA_FLUSH_TWO_PASS, block_size, repeat)},
            {"J*v isoparam atomic",
             cvfem_cuda_time_jacobian_action_isoparam(ctx, rho, mu, CVFEM_CUDA_FLUSH_ATOMIC, block_size, repeat)},
            {"assemble isoparam",
             cvfem_cuda_time_assemble_isoparam(ctx, rho, mu, block_size, repeat)},
            {"assemble isoparam ecoloured",
             cvfem_cuda_time_assemble_ecolored_isoparam(ctx, rho, mu, block_size, repeat)},
            {"assemble isoparam sympy",
             cvfem_cuda_time_assemble_isoparam_sympy(ctx, rho, mu, block_size, repeat)},
            {"assemble isoparam split",
             cvfem_cuda_time_assemble_nonlinear_isoparam(ctx, rho, mu, block_size, repeat)},
            {"assemble isoparam diagonal",
             cvfem_cuda_time_assemble_diag_isoparam(ctx, rho, mu, block_size, repeat)},
        };
        for (auto &r : iso_rows) {
            std::printf("%-34s %12.3e %12.1f\n", r.name, r.t,
                        r.t > 0 ? (double)(d.nnodes * 4) / r.t * 1e-6 : 0.0);
            const char *op = (r.name[0] == 'r') ? "residual"
                                                : (r.name[0] == 'J') ? "jac_action" : "assemble";
            csv_rows.push_back(mkrow(op, "cuda_isoparam", "isoparam", r.t));
        }

        // Leave the affine matrix in place for anything downstream.
        assemble_jacobian_atomic_sumfact(d, bsr, rho, mu);
    }

    std::printf("\n=== boundary sub-control-surface residual ===\n");
    const scalar_t Lx = 1, Ly = 1, Lz = 1;   // create_hex8_cube(0,0,0 -> 1,1,1)

    // Host: coordinates as double, and the list of elements with a face on the boundary.
    std::vector<double> px(d.nnodes), py(d.nnodes), pz(d.nnodes);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        px[i] = d.points[0][i]; py[i] = d.points[1][i]; pz[i] = d.points[2][i];
    }
    std::vector<int32_t> blist;
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ex[8], ey[8], ez[8];
        for (int a = 0; a < 8; ++a) {
            const auto g = d.elems[a][e];
            ex[a] = px[g]; ey[a] = py[g]; ez[a] = pz[g];
        }
        for (int f = 0; f < 6; ++f)
            if (hex8_face_on_domain(f, ex, ey, ez, Lx, Ly, Lz)) { blist.push_back((int32_t)e); break; }
    }
    std::printf("boundary elements: %zu of %td (%.1f%%)\n", blist.size(), d.nelements,
                100.0 * (double)blist.size() / (double)d.nelements);

    // Host reference: volume residual, then the boundary correction, mirroring
    // apply_boundary_scs_residual in cvfem_hex8_ns_steady.cpp.
    apply_residual_atomic(d, rho, mu);
    for (int32_t e : blist) {
        scalar_t ex[8], ey[8], ez[8], eu[8], ev_[8], ew[8], ep[8], re[CVFEM_HEX8_N_DOF] = {0};
        for (int a = 0; a < 8; ++a) {
            const auto g = d.elems[a][e];
            ex[a] = px[g]; ey[a] = py[g]; ez[a] = pz[g];
            eu[a] = d.ux[g]; ev_[a] = d.uy[g]; ew[a] = d.uz[g]; ep[a] = d.p[g];
        }
        scalar_t adj_e[9], det_e;
        load_hex8_adj(d, e, adj_e, &det_e);
        boundary_scs_add_residual(rho, mu, 0, adj_e, det_e, Lx, Ly, Lz,
                                  ex, ey, ez, eu, ev_, ew, ep, re);
        for (int a = 0; a < 8; ++a) {
            const auto g = d.elems[a][e];
            d.rx[g] += re[a * 4 + 0]; d.ry[g] += re[a * 4 + 1];
            d.rz[g] += re[a * 4 + 2]; d.rc[g] += re[a * 4 + 3];
        }
    }
    std::vector<double> bref;
    residual_soa_to_interleaved(d, bref);
    double bmax = max_abs(bref);

    if (cvfem_cuda_boundary_attach(ctx, (ptrdiff_t)blist.size(), blist.data(),
                                   px.data(), py.data(), pz.data(), Lx, Ly, Lz) != 0) {
        std::fprintf(stderr, "boundary_attach failed\n");
        return 1;
    }
    if (cvfem_cuda_residual(ctx, rho, mu, CVFEM_CUDA_FLUSH_ATOMIC, block_size, nullptr) != 0 ||
        cvfem_cuda_boundary_residual(ctx, rho, mu, block_size, nullptr) != 0 ||
        cvfem_cuda_synchronize() != 0) return 1;
    if (cvfem_cuda_download_r(ctx, dev.data()) != 0) return 1;
    {
        const double diff = max_abs_diff(bref, dev);
        const double rel  = bmax > 0 ? diff / bmax : diff;
        const bool   ok   = rel <= 1e-12;
        fail |= !ok;
        std::printf("volume+boundary vs host: max|diff| = %.3e  rel = %.3e  %s\n",
                    diff, rel, ok ? "OK" : "FAIL");
    }

    // ---- boundary Jacobian action and boundary assembly ----------------------
    std::printf("\n=== boundary Jacobian ===\n");
    {
        std::vector<double> vh((size_t)d.nnodes * 4), jv_h((size_t)d.nnodes * 4, 0.0);
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            vh[i * 4 + 0] = 0.7 * std::sin(0.011 * (double)i);
            vh[i * 4 + 1] = 0.3 * std::cos(0.017 * (double)i);
            vh[i * 4 + 2] = 0.5 * std::sin(0.023 * (double)i + 1.0);
            vh[i * 4 + 3] = 0.9 * std::cos(0.007 * (double)i + 2.0);
        }
        // Host: volume J*v, then the boundary correction.
        apply_jacobian_action_atomic(d, rho, mu, vh.data(), jv_h.data());
        for (int32_t e : blist) {
            scalar_t ex[8], ey[8], ez[8], eu[8], ev_[8], ew[8];
            scalar_t gx[8], gy[8], gz[8], gq[8], re[CVFEM_HEX8_N_DOF] = {0};
            for (int a = 0; a < 8; ++a) {
                const auto g = d.elems[a][e];
                ex[a] = px[g]; ey[a] = py[g]; ez[a] = pz[g];
                eu[a] = d.ux[g]; ev_[a] = d.uy[g]; ew[a] = d.uz[g];
                gx[a] = vh[(size_t)g * 4 + 0]; gy[a] = vh[(size_t)g * 4 + 1];
                gz[a] = vh[(size_t)g * 4 + 2]; gq[a] = vh[(size_t)g * 4 + 3];
            }
            scalar_t adj_e[9], det_e;
            load_hex8_adj(d, e, adj_e, &det_e);
            boundary_scs_add_jacobian_action(rho, mu, 0, adj_e, det_e, Lx, Ly, Lz,
                                             ex, ey, ez, eu, ev_, ew, gx, gy, gz, gq, re);
            for (int a = 0; a < 8; ++a) {
                const auto g = d.elems[a][e];
                for (int f = 0; f < 4; ++f) jv_h[(size_t)g * 4 + f] += re[a * 4 + f];
            }
        }
        const double jvmax = max_abs(jv_h);
        if (cvfem_cuda_upload_v(ctx, vh.data()) != 0) return 1;
        if (cvfem_cuda_jacobian_action(ctx, rho, mu, CVFEM_CUDA_FLUSH_ATOMIC, block_size, nullptr) != 0 ||
            cvfem_cuda_boundary_jacobian_action(ctx, rho, mu, block_size, nullptr) != 0 ||
            cvfem_cuda_synchronize() != 0) return 1;
        if (cvfem_cuda_download_r(ctx, dev.data()) != 0) return 1;
        const double rel = jvmax > 0 ? max_abs_diff(jv_h, dev) / jvmax : 0.0;
        const bool   ok  = rel <= 1e-12;
        fail |= !ok;
        std::printf("volume+boundary J*v vs host: rel = %.3e  %s\n", rel, ok ? "OK" : "FAIL");

        // Boundary assembly: host reference is the same kernel with Atomic=true.
        assemble_jacobian_atomic_sumfact(d, bsr, rho, mu);
        {
            scalar_t *const hv = bsr.values->data();
            for (int32_t e : blist) {
                scalar_t ex[8], ey[8], ez[8], eu[8], ev_[8], ew[8];
                for (int a = 0; a < 8; ++a) {
                    const auto g = d.elems[a][e];
                    ex[a] = px[g]; ey[a] = py[g]; ez[a] = pz[g];
                    eu[a] = d.ux[g]; ev_[a] = d.uy[g]; ew[a] = d.uz[g];
                }
                scalar_t adj_e[9], det_e;
                load_hex8_adj(d, e, adj_e, &det_e);
                boundary_scs_add_jacobian<true>(rho, mu, 0, adj_e, det_e, Lx, Ly, Lz,
                                                ex, ey, ez, eu, ev_, ew,
                                                bsr.element_slots.data() + (size_t)e * 64, hv);
            }
            std::vector<double> href2((size_t)bsr.nnz * 16);
            std::memcpy(href2.data(), hv, href2.size() * sizeof(double));
            double h2max = 0;
            for (double v : href2) h2max = std::fmax(h2max, std::fabs(v));

            if (cvfem_cuda_assemble(ctx, rho, mu, CVFEM_CUDA_JAC_HANDWRITTEN, block_size, nullptr) != 0 ||
                cvfem_cuda_boundary_assemble(ctx, rho, mu, block_size, nullptr) != 0 ||
                cvfem_cuda_synchronize() != 0) return 1;
            if (cvfem_cuda_download_values(ctx, dvals.data()) != 0) return 1;
            double dm = 0;
            for (size_t i = 0; i < href2.size(); ++i)
                dm = std::fmax(dm, std::fabs(href2[i] - dvals[i]));
            const double rel2 = h2max > 0 ? dm / h2max : dm;
            const bool   ok2  = rel2 <= 1e-12;
            fail |= !ok2;
            std::printf("volume+boundary BSR vs host: rel = %.3e  %s\n", rel2, ok2 ? "OK" : "FAIL");
        }
    }

    // ---- Rhie-Chow: nodal pressure gradient ----------------------------------
    std::printf("\n=== Rhie-Chow nodal pressure gradient ===\n");
    {
        // Host reference, mirroring assemble_nodal_p_grad in cvfem_hex8_ns_steady.cpp.
        std::vector<double> hx(d.nnodes, 0.0), hy(d.nnodes, 0.0), hz(d.nnodes, 0.0),
                            hw(d.nnodes, 0.0);
        for (ptrdiff_t e = 0; e < d.nelements; ++e) {
            scalar_t pe[8], adj_e[9], det_e, gx, gy, gz;
            for (int a = 0; a < 8; ++a) pe[a] = d.p[d.elems[a][e]];
            load_hex8_adj(d, e, adj_e, &det_e);
            const scalar_t vol = std::fabs(det_e);
            if (vol < scalar_t(1e-30)) continue;
            cvfem_hex8_grad_scalar(adj_e, det_e, pe, gx, gy, gz);
            for (int a = 0; a < 8; ++a) {
                const auto g = d.elems[a][e];
                hx[g] += vol * gx; hy[g] += vol * gy; hz[g] += vol * gz; hw[g] += vol;
            }
        }
        for (ptrdiff_t i = 0; i < d.nnodes; ++i)
            if (hw[i] > 0) { hx[i] /= hw[i]; hy[i] /= hw[i]; hz[i] /= hw[i]; }

        std::vector<double> gx_(d.nnodes), gy_(d.nnodes), gz_(d.nnodes);
        if (cvfem_cuda_nodal_p_grad(ctx, block_size, nullptr) != 0 ||
            cvfem_cuda_synchronize() != 0 ||
            cvfem_cuda_download_p_grad(ctx, gx_.data(), gy_.data(), gz_.data()) != 0) {
            std::printf("nodal p-grad failed\n"); fail = 1;
        } else {
            const double m = std::fmax(max_abs(hx), std::fmax(max_abs(hy), max_abs(hz)));
            const double diff = std::fmax(max_abs_diff(hx, gx_),
                                std::fmax(max_abs_diff(hy, gy_), max_abs_diff(hz, gz_)));
            const double rel = m > 0 ? diff / m : diff;
            const bool   ok  = rel <= 1e-12;
            fail |= !ok;
            std::printf("nodal grad(p) vs host: max|diff| = %.3e  rel = %.3e  %s\n",
                        diff, rel, ok ? "OK" : "FAIL");
        }
    }

    // ---- Rhie-Chow residual --------------------------------------------------
    {
        const size_t rc_need = cvfem_cuda_residual_rc_shmem_bytes(packed.max_actual_nodes_per_pack);
        std::printf("RC residual shared memory %zu B (%.0f%% of opt-in)\n", rc_need,
                    100.0 * (double)rc_need / (double)optin);
        if (rc_need > (size_t)optin) {
            std::printf("  EXCEEDS OPT-IN LIMIT, reduce --pack-size\n");
            fail = 1;
        } else {
            const scalar_t rc_scale = 1.0;
            // Host reference: same kernel, with rc pointing at element-local gathers.
            std::vector<double> hr((size_t)d.nnodes * 4, 0.0);
            std::vector<double> ghx(d.nnodes), ghy(d.nnodes), ghz(d.nnodes);
            cvfem_cuda_download_p_grad(ctx, ghx.data(), ghy.data(), ghz.data());
            for (ptrdiff_t e = 0; e < d.nelements; ++e) {
                scalar_t eu[8], ev_[8], ew[8], ep[8], re[CVFEM_HEX8_N_DOF];
                scalar_t rx[8], ry[8], rz[8], rgx[8], rgy[8], rgz[8];
                for (int a = 0; a < 8; ++a) {
                    const auto g = d.elems[a][e];
                    eu[a] = d.ux[g]; ev_[a] = d.uy[g]; ew[a] = d.uz[g]; ep[a] = d.p[g];
                    rx[a] = px[g]; ry[a] = py[g]; rz[a] = pz[g];
                    rgx[a] = ghx[g]; rgy[a] = ghy[g]; rgz[a] = ghz[g];
                }
                scalar_t adj_e[9], det_e;
                load_hex8_adj(d, e, adj_e, &det_e);
                Hex8RhieChowT<scalar_t> rc;
                rc.x = rx; rc.y = ry; rc.z = rz;
                rc.pgx = rgx; rc.pgy = rgy; rc.pgz = rgz; rc.scale = rc_scale;
                cvfem_hex8_ns_upwind_residual_sumfact(rho, mu, adj_e, det_e,
                                                      eu, ev_, ew, ep, re, rc);
                for (int a = 0; a < 8; ++a) {
                    const auto g = d.elems[a][e];
                    for (int f = 0; f < 4; ++f) hr[(size_t)g * 4 + f] += re[a * 4 + f];
                }
            }
            const double hm = max_abs(hr);
            if (cvfem_cuda_residual_rc(ctx, rho, mu, rc_scale, CVFEM_CUDA_FLUSH_ATOMIC,
                                       block_size, nullptr) != 0 ||
                cvfem_cuda_synchronize() != 0) { std::printf("RC residual failed\n"); fail = 1; }
            else {
                if (cvfem_cuda_download_r(ctx, dev.data()) != 0) return 1;
                const double rel = hm > 0 ? max_abs_diff(hr, dev) / hm : 0.0;
                const bool   ok  = rel <= 1e-12;
                fail |= !ok;
                std::printf("Rhie-Chow residual vs host: rel = %.3e  %s\n", rel, ok ? "OK" : "FAIL");
            }
        }
    }

    {
        // Name the device in the CSV so rows from different GPUs stay distinguishable.
        char devname[64];
        std::snprintf(devname, sizeof(devname), "sm90_%dSM", sm);
        csv_write(csv_path.c_str(), csv_tag.c_str(), devname, csv_rows);
    }

    cvfem_cuda_destroy(ctx);
    if (own_mpi) MPI_Finalize();
    std::printf("\n%s\n", fail ? "VERIFICATION FAILED" : "verification passed");
    return fail;
}
