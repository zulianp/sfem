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
#include "cvfem_hex8_layout_packed.hpp"
#include "cvfem_pack_coloring.hpp"
#include "cvfem_hex8_layout_store.hpp"
#include "cvfem_hex8_boundary_scs.hpp"

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
