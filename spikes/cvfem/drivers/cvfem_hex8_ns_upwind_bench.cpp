// HEX8 CVFEM Navier-Stokes benchmark driver.
//
// The operator implementations live in the per-layout headers below; this file
// holds the mesh/option setup, the verification harness and the timing loop.

#include "cvfem_hex8_layout_common.hpp"
#include "cvfem_hex8_layout_atomic.hpp"
#include "cvfem_hex8_layout_colored.hpp"
#include "cvfem_hex8_layout_packed.hpp"
#include "cvfem_hex8_layout_store.hpp"

// Consumes the churn's reduction under --live-vectors so the compiler cannot delete the
// memory traffic that option exists to create.
static volatile double g_churn_sink = 0;

static void pack_residual(const MeshData &d, std::vector<scalar_t> &r) {
    r.resize((size_t)d.nnodes * 4);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        r[(size_t)i * 4 + 0] = d.rx[i];
        r[(size_t)i * 4 + 1] = d.ry[i];
        r[(size_t)i * 4 + 2] = d.rz[i];
        r[(size_t)i * 4 + 3] = d.rc[i];
    }
}

static void bsr4_spmv(const BSR4 &b, const ptrdiff_t nnodes, const scalar_t *const x, scalar_t *const y) {
    std::fill(y, y + nnodes * 4, scalar_t(0));

#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < nnodes; ++row) {
        scalar_t acc[4] = {0, 0, 0, 0};
        for (smesh::count_t k = b.rowptr[row]; k < b.rowptr[row + 1]; ++k) {
            const scalar_t *const blk = b.values->data() + (ptrdiff_t)k * 16;
            const scalar_t *const xx  = x + (ptrdiff_t)b.colidx[k] * 4;
            acc[0] += blk[0] * xx[0] + blk[1] * xx[1] + blk[2] * xx[2] + blk[3] * xx[3];
            acc[1] += blk[4] * xx[0] + blk[5] * xx[1] + blk[6] * xx[2] + blk[7] * xx[3];
            acc[2] += blk[8] * xx[0] + blk[9] * xx[1] + blk[10] * xx[2] + blk[11] * xx[3];
            acc[3] += blk[12] * xx[0] + blk[13] * xx[1] + blk[14] * xx[2] + blk[15] * xx[3];
        }
        y[(ptrdiff_t)row * 4 + 0] = acc[0];
        y[(ptrdiff_t)row * 4 + 1] = acc[1];
        y[(ptrdiff_t)row * 4 + 2] = acc[2];
        y[(ptrdiff_t)row * 4 + 3] = acc[3];
    }
}

static scalar_t max_abs_diff(const scalar_t *const a, const scalar_t *const b, const ptrdiff_t n) {
    scalar_t m = 0;
    for (ptrdiff_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

static scalar_t verify_jacobian_fd(MeshData        &d,
                                   BSR4            &b,
                                   const scalar_t   rho,
                                   const scalar_t   mu,
                                   const GeomKind   geom_kind) {
    const ptrdiff_t ndof = d.nnodes * 4;
    std::vector<scalar_t> x0((size_t)ndof), dir((size_t)ndof), rm, rp, jv((size_t)ndof);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        x0[(size_t)i * 4 + 0] = d.ux[i];
        x0[(size_t)i * 4 + 1] = d.uy[i];
        x0[(size_t)i * 4 + 2] = d.uz[i];
        x0[(size_t)i * 4 + 3] = d.p[i];
    }
    std::fill(dir.begin(), dir.end(), scalar_t(0));
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) dir[(size_t)i * 4 + 3] = scalar_t(1);

    const scalar_t eps = scalar_t(1.0e-6);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        d.ux[i] = x0[(size_t)i * 4 + 0] - eps * dir[(size_t)i * 4 + 0];
        d.uy[i] = x0[(size_t)i * 4 + 1] - eps * dir[(size_t)i * 4 + 1];
        d.uz[i] = x0[(size_t)i * 4 + 2] - eps * dir[(size_t)i * 4 + 2];
        d.p[i]  = x0[(size_t)i * 4 + 3] - eps * dir[(size_t)i * 4 + 3];
    }
    if (geom_kind == GeomKind::Isoparam)
        apply_residual_atomic_isoparam(d, rho, mu);
    else
        apply_residual_atomic(d, rho, mu);
    pack_residual(d, rm);

    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        d.ux[i] = x0[(size_t)i * 4 + 0] + eps * dir[(size_t)i * 4 + 0];
        d.uy[i] = x0[(size_t)i * 4 + 1] + eps * dir[(size_t)i * 4 + 1];
        d.uz[i] = x0[(size_t)i * 4 + 2] + eps * dir[(size_t)i * 4 + 2];
        d.p[i]  = x0[(size_t)i * 4 + 3] + eps * dir[(size_t)i * 4 + 3];
    }
    if (geom_kind == GeomKind::Isoparam)
        apply_residual_atomic_isoparam(d, rho, mu);
    else
        apply_residual_atomic(d, rho, mu);
    pack_residual(d, rp);

    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        d.ux[i] = x0[(size_t)i * 4 + 0];
        d.uy[i] = x0[(size_t)i * 4 + 1];
        d.uz[i] = x0[(size_t)i * 4 + 2];
        d.p[i]  = x0[(size_t)i * 4 + 3];
    }

    bsr4_spmv(b, d.nnodes, dir.data(), jv.data());

    scalar_t max_fd = 0;
    scalar_t max_er = 0;
    for (ptrdiff_t i = 0; i < ndof; ++i) {
        const scalar_t fd = (rp[(size_t)i] - rm[(size_t)i]) / (2 * eps);
        max_fd            = std::max(max_fd, std::fabs(fd));
        max_er            = std::max(max_er, std::fabs(fd - jv[(size_t)i]));
    }
    return max_er / std::max(max_fd, scalar_t(1.0e-30));
}

// One machine-readable row per run. The header is written when the file is new,
// so a sweep can just append and the analysis scripts (plot_cvfem_bench.py,
// report_cvfem_bench.py) read whatever accumulated.
struct CsvRow {
    const char *tag;
    const char *operation;
    const char *layout;
    const char *kernel;
    const char *geom;
    int         threads;
    int         pack_size;
    int         cube_n;
    ptrdiff_t   nodes;
    ptrdiff_t   elements;
    ptrdiff_t   dofs;
    ptrdiff_t   bsr_nnz;
    double      bsr_values_mib;
    int         repeat;
    double      seconds_per_call;
    double      mdofs;
    double      mdofs_element_visits;
    double      melems;
    double      gflops_model;
    double      warp;
    int         n_colors;
    ptrdiff_t   packs_per_color_min;
    ptrdiff_t   packs_per_color_max;
    double      checksum;
    int         rhie_chow;   // 0, or the scale the Rhie-Chow term ran at
    double      rhie_chow_scale;
    int         boundary;    // 1 if the boundary control volumes were closed
    const double *phase;  // PH_N entries, thread-summed ms per call, or nullptr
};

static void csv_write(const std::string &path, const CsvRow &r) {
    if (path.empty()) return;

    // The column set changes when options are added, and this file is opened for APPEND.
    // Writing a new row shape under an old header produces a csv whose columns silently
    // shift partway down -- and report_cvfem_bench.py / plot_cvfem_bench.py read these by
    // name. So the existing header is compared against the one we would write, and a
    // mismatch is refused rather than appended to.
    std::string header =
            "tag,host,element,operation,layout,kernel,geom,warp,threads,pack_size,cube_n,"
            "nodes,elements,dofs,bsr_nnz,bsr_values_MiB,repeat,seconds_per_call,"
            "MDOF_s,MDOF_s_element_visits,MELEM_s,GFLOP_s_model,"
            "n_colors,packs_per_color_min,packs_per_color_max,checksum,"
            "rhie_chow,rhie_chow_scale,boundary";
    for (int i = 0; i < PH_N; ++i) header += std::string(",ms_") + g_phase_name[i];

    bool need_header = true;
    if (FILE *probe = std::fopen(path.c_str(), "r")) {
        std::fseek(probe, 0, SEEK_END);
        const long size = std::ftell(probe);
        need_header     = size == 0;
        if (size > 0) {
            std::rewind(probe);
            std::string first;
            for (int c = std::fgetc(probe); c != EOF && c != '\n'; c = std::fgetc(probe))
                first.push_back((char)c);
            if (first != header) {
                std::fclose(probe);
                std::fprintf(stderr,
                             "error: '%s' has a different column set than this build writes.\n"
                             "       Appending would misalign it. Write to a new file instead.\n",
                             path.c_str());
                return;
            }
        }
        std::fclose(probe);
    }

    FILE *f = std::fopen(path.c_str(), "a");
    if (!f) {
        std::fprintf(stderr, "warning: could not open csv '%s' for append\n", path.c_str());
        return;
    }

    char host[256] = "unknown";
    if (gethostname(host, sizeof(host) - 1) != 0) std::snprintf(host, sizeof(host), "unknown");
    host[sizeof(host) - 1] = '\0';

    if (need_header) std::fprintf(f, "%s\n", header.c_str());

    std::fprintf(f,
                 "%s,%s,hex8,%s,%s,%s,%s,%.6e,%d,%d,%d,%td,%td,%td,%td,%.4f,%d,%.9e,%.4f,%.4f,%.4f,%.4f,%d,%td,%td,%.12e",
                 r.tag, host, r.operation, r.layout, r.kernel, r.geom, r.warp, r.threads, r.pack_size, r.cube_n,
                 r.nodes, r.elements, r.dofs, r.bsr_nnz, r.bsr_values_mib, r.repeat, r.seconds_per_call,
                 r.mdofs, r.mdofs_element_visits, r.melems, r.gflops_model,
                 r.n_colors, r.packs_per_color_min, r.packs_per_color_max, r.checksum);
    std::fprintf(f, ",%d,%.6f,%d", r.rhie_chow, r.rhie_chow_scale, r.boundary);
    for (int i = 0; i < PH_N; ++i) {
        if (r.phase)
            std::fprintf(f, ",%.6f", 1000.0 * r.phase[i] / double(r.repeat));
        else
            std::fprintf(f, ",");
    }
    std::fprintf(f, "\n");
    std::fclose(f);
}

int main(int argc, char **argv) {
    int own_mpi = 0;
    MPI_Initialized(&own_mpi);
    own_mpi = !own_mpi;
    if (own_mpi) MPI_Init(&argc, &argv);

    int         n          = 8;
    int         repeat     = 10;
    int         warmup     = 2;
    int         assemble   = 0;
    int         jac_action = 0;
    int         bsr_apply  = 0;
    int         verify     = 0;
    int         verify_jac = 0;
    int         use_sfc    = 1;
    scalar_t    rho        = 1.0;
    scalar_t    mu         = 0.01;
    std::string layout     = "atomic";
    std::string kernel     = "sumfact";
    std::string geom       = "affine";
    std::string csv_path;
    std::string csv_tag    = "run";
    scalar_t    warp       = 0;
    int         pack_size  = 2048;
    int         assemble_diag = 0;
    // Both off by default. The benchmark's job is to isolate the element kernel, and the
    // recorded throughput baselines were measured without either term; turning one on
    // changes what is being measured, so it has to be asked for.
    int         rhie_chow  = 0;
    scalar_t    rc_scale   = 1;
    int         boundary   = 0;
    // Whether the nodal pressure gradient is rebuilt inside every apply or hoisted out of
    // the timed loop. This is the difference SFEM_PGRAD_CACHE makes in the solver, and it
    // is the other half of the cascade: the gradient is a full element sweep.
    int         pgrad_per_apply = 0;
    int         live_vectors    = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc)
            n = std::atoi(argv[++i]);
        else if (arg == "--repeat" && i + 1 < argc)
            repeat = std::atoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc)
            warmup = std::atoi(argv[++i]);
        else if (arg == "--rho" && i + 1 < argc)
            rho = std::atof(argv[++i]);
        else if (arg == "--mu" && i + 1 < argc)
            mu = std::atof(argv[++i]);
        else if (arg == "--kernel" && i + 1 < argc)
            kernel = argv[++i];
        else if (arg == "--geom" && i + 1 < argc)
            geom = argv[++i];
        else if (arg == "--warp" && i + 1 < argc)
            warp = std::atof(argv[++i]);
        else if (arg == "--layout" && i + 1 < argc)
            layout = argv[++i];
        else if (arg == "--pack-size" && i + 1 < argc)
            pack_size = std::atoi(argv[++i]);
        else if (arg == "--assemble")
            assemble = 1;
        else if (arg == "--assemble-diag")
            assemble_diag = 1;
        else if (arg == "--jac-action")
            jac_action = 1;
        else if (arg == "--bsr-apply")
            bsr_apply = 1;
        else if (arg == "--verify")
            verify = 1;
        else if (arg == "--verify-jac")
            verify_jac = 1;
        else if (arg == "--no-sfc")
            use_sfc = 0;
        else if (arg == "--breakdown")
            g_breakdown = 1;
        else if (arg == "--kernel-only")
            g_kernel_only = 1;
        else if (arg == "--dense-flush")
            g_dense_flush = 1;
        else if (arg == "--rhie-chow") {
            rhie_chow = 1;
            // Optional scale, so --rhie-chow 0.5 works and a bare --rhie-chow means 1.
            if (i + 1 < argc && argv[i + 1][0] != '-') rc_scale = (scalar_t)std::atof(argv[++i]);
        } else if (arg == "--boundary")
            boundary = 1;
        else if (arg == "--pgrad-per-apply")
            pgrad_per_apply = 1;
        else if (arg == "--live-vectors" && i + 1 < argc)
            live_vectors = std::atoi(argv[++i]);
        else if (arg == "--csv" && i + 1 < argc)
            csv_path = argv[++i];
        else if (arg == "--tag" && i + 1 < argc)
            csv_tag = argv[++i];
        else if (arg == "--help") {
            std::printf(
                    "usage: %s [--n N] [--repeat N] [--warmup N] [--assemble] [--jac-action] [--bsr-apply]\n"
                    "          [--verify] [--verify-jac] [--layout packed|atomic|colored|store]\n"
                    "          [--kernel sumfact|current|fd|sympy|sympy_block|sympy_row|sympy_face|split]\n"
                     "          [--assemble-diag]  block diagonal only, for block-Jacobi\n"
                    "          [--geom affine|isoparam] [--warp EPS] [--pack-size N] [--no-sfc]\n"
                    "          [--breakdown] [--kernel-only] [--dense-flush]\n"
                    "          [--csv FILE] [--tag NAME]\n"
                    "  --layout NAME  layout used by residual / jac-action / assemble (default atomic)\n"
                    "                 atomic  : flat element sweep, #pragma omp atomic per entry\n"
                    "                           (cvfem_hex8_layout_atomic.hpp)\n"
                    "                 packed  : pack-local buffer folded back into the global one,\n"
                    "                           ghost rows reduced after (cvfem_hex8_layout_packed.hpp)\n"
                    "                 colored : colored pack sweep, no reduction and no atomics\n"
                    "                           (cvfem_hex8_layout_colored.hpp). Best for\n"
                    "                           --assemble; for residual and --jac-action the\n"
                    "                           color barriers cost more than the ghost reduce\n"
                    "                           they replace, so prefer packed there\n"
                    "                 store   : packed assembly whose owned rows carry the global\n"
                    "                           pattern and are flushed with one streaming memcpy;\n"
                    "                           every block written once, no zeroing pass\n"
                    "                           (cvfem_hex8_layout_store.hpp; residual/jac-action\n"
                    "                           fall back to packed)\n"
                    "  --breakdown    per-phase timing of the assembly (thread-summed ms/call)\n"
                    "  --kernel-only  element kernel writes to a dense stack buffer (no scatter);\n"
                    "                 measures the arithmetic floor of the element kernel\n"
                    "  --dense-flush  sumfact only: stage ke densely, then flush 64 contiguous\n"
                    "                 blocks (measured slower than direct scatter on Apple M1)\n"
                    "  --csv FILE     append one machine-readable row per run (header written if\n"
                    "                 the file is new); pairs with report_cvfem_bench.py\n"
                    "  --tag NAME     free-form label carried into the csv (e.g. the machine)\n"
                    "  --kernel NAME  residual/Jacobian micro-kernel variant (default sumfact)\n"
                    "  --geom NAME    affine (constant J) or isoparam (12 SCS trilinear J)\n"
                    "  --warp EPS     x += EPS * sin(pi y) nodal perturbation\n"
                    "  --bsr-apply    assemble once, then time BSR SpMV y = J(u) v\n"
                    "  --rhie-chow [S]  include the Rhie-Chow pressure-velocity coupling at scale\n"
                    "                 S (default 1), and the nodal pressure gradient it needs.\n"
                    "                 sumfact only -- the hand-written and generated kernels carry\n"
                    "                 no Rhie-Chow term. Off by default: without it there is no\n"
                    "                 pressure-pressure coupling at all, which is a smaller and\n"
                    "                 faster operator than the one the solver runs.\n"
                    "  --live-vectors N   keep N extra vectors of the solution size resident and\n"
                    "                     churned between applies, and take the direction from them in\n"
                    "                     rotation. The default measures an apply replayed on a warm,\n"
                    "                     small working set; a Krylov iteration evaluates the same\n"
                    "                     kernel with its own vectors live and a different direction\n"
                    "                     every time. BiCGStab holds about 7, which is what makes N=7\n"
                    "                     the interesting value. Only the apply is timed -- the churn\n"
                    "                     runs between applies, outside the clock.\n"
                    "  --pgrad-per-apply  rebuild the nodal pressure gradient inside every apply\n"
                    "                     (--jac-action always rebuilds the DIRECTION's gradient,\n"
                    "                      which cannot be hoisted, and reports it separately)\n"
                    "                 instead of hoisting it out of the timed loop. The gradient is\n"
                    "                 a full element sweep, about 39%% of an apply, and hoisting it\n"
                    "                 across a Krylov solve is worth 1.26x off the whole linear\n"
                    "                 solve -- so which of the two is measured has to be said.\n"
                    "                 Requires --rhie-chow.\n"
                    "  --boundary     close the boundary control volumes with the boundary\n"
                    "                 sub-control-surface terms. Off by default; the benchmark\n"
                    "                 otherwise has no boundary handling. Reaches ~43%% of the\n"
                    "                 nodes on this channel at N=8.\n",
                    argv[0]);
            if (own_mpi) MPI_Finalize();
            return 0;
        } else {
            // An unrecognised token used to be ignored in silence, so a typo such as
            // --rhie_chow produced a plain-kernel number that looked like a measurement of
            // something else. A wrong number is worse than an error.
            std::fprintf(stderr, "unknown option '%s' (try --help)\n", arg.c_str());
            if (own_mpi) MPI_Finalize();
            return 1;
        }
    }

    // Rhie-Chow exists only in the sum-factorised kernels. cvfem_hex8_ns_upwind_residual
    // (the hand-written `current` kernel) and every generated sympy kernel take no
    // Hex8RhieChow argument at all, so asking for it there would silently measure a kernel
    // without it -- exactly the confusion this option exists to remove.
    if (rhie_chow && kernel != "sumfact") {
        std::fprintf(stderr,
                     "--rhie-chow requires --kernel sumfact (got '%s'): the hand-written and\n"
                     "generated kernels carry no Rhie-Chow term.\n",
                     kernel.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    // --boundary is a separate element sweep and so is layout-independent, but it only
    // reaches the residual and the Jacobian action that way. The assembled matrix needs the
    // boundary blocks written through the element BSR slots, which so far only the atomic
    // assembly does.
    if (boundary && (assemble || assemble_diag || bsr_apply) && layout != "atomic") {
        std::fprintf(stderr,
                     "--boundary with --assemble/--assemble-diag/--bsr-apply is implemented "
                     "for --layout atomic only (got '%s')\n",
                     layout.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    // Rhie-Chow lives inside the element kernel, so it has to be staged per layout. The
    // packed, colored and store sweeps carry their element data through Hex8RhieChowPack
    // and that staging is not wired up here yet; asking for it there has to fail rather
    // than quietly return a number measured without it.
    // The residual and the Jacobian action carry Rhie-Chow on atomic and on the
    // packed/store SIMD sumfact path. The colored sweep and the assembled Jacobian on the
    // packed layouts still need their staging extended, so those combinations are refused
    // rather than measured without it.
    const bool rc_layout_ok =
            layout == "atomic" ||
            ((layout == "packed" || layout == "store") && !assemble && !assemble_diag && !bsr_apply);
    if (rhie_chow && !rc_layout_ok) {
        std::fprintf(stderr,
                     "--rhie-chow is implemented for --layout atomic, and for the residual and "
                     "the Jacobian action on --layout packed|store (got '%s'%s)\n",
                     layout.c_str(),
                     (assemble || assemble_diag || bsr_apply) ? " with an assembly operation" : "");
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    // The isoparametric Jacobian-action kernels take no Rhie-Chow argument at all
    // (cvfem_hex8_ns_upwind_jacobian_action_isoparam_simd), so this combination would
    // report a throughput measured without the term. Refuse it rather than measure it.
    if (rhie_chow && jac_action && geom == "isoparam") {
        std::fprintf(stderr, "--rhie-chow with --jac-action is implemented for --geom affine only\n");
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    if (pgrad_per_apply && !rhie_chow) {
        std::fprintf(stderr, "--pgrad-per-apply is meaningless without --rhie-chow\n");
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    // Refused rather than ignored: the working set only means something for an operation
    // that a Krylov method actually repeats, and silently dropping it would report a
    // warm-cache number under a flag that asked for a cold one.
    if (live_vectors > 0 && !(jac_action || bsr_apply)) {
        std::fprintf(stderr, "--live-vectors applies to --jac-action or --bsr-apply\n");
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    if (live_vectors < 0) {
        std::fprintf(stderr, "--live-vectors must be >= 0 (got %d)\n", live_vectors);
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    if (rhie_chow && rc_scale == scalar_t(0)) {
        std::fprintf(stderr, "--rhie-chow 0 is the same as omitting it; say so explicitly\n");
        if (own_mpi) MPI_Finalize();
        return 1;
    }

    if (!kernel_is_valid(kernel)) {
        std::fprintf(stderr,
                     "invalid --kernel '%s' (expected sumfact, current, fd, sympy, sympy_block, sympy_row, sympy_face, or split)\n",
                     kernel.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    if ((assemble ? 1 : 0) + (jac_action ? 1 : 0) + (bsr_apply ? 1 : 0) + (assemble_diag ? 1 : 0) > 1) {
        std::fprintf(stderr,
                     "specify at most one of --assemble, --assemble-diag, --jac-action, --bsr-apply\n");
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    const KernelKind kernel_kind = parse_kernel(kernel);
    // sympy_row and sympy_face lost the saturated evaluation and were moved to subpar/.
    // Rejected by name here, which is what keeps the stubs in cvfem_hex8_layout_common.hpp
    // unreachable -- and, more to the point, means a run cannot report a throughput under
    // a kernel name that did not execute. This spike has produced that failure three
    // times; a rejection is cheap insurance against a fourth.
#ifndef CVFEM_ENABLE_SUBPAR
    if (kernel_kind == KernelKind::SympyRow || kernel_kind == KernelKind::SympyFace) {
        std::fprintf(stderr,
                     "--kernel %s was moved to subpar/: it is not the fastest kernel in any "
                     "measured configuration (see subpar/README.md).\n"
                     "Rebuild with -DCVFEM_ENABLE_SUBPAR=ON to measure it again.\n",
                     kernel.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }
#endif
    if (geom != "affine" && geom != "isoparam") {
        std::fprintf(stderr, "invalid --geom '%s' (expected affine or isoparam)\n", geom.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    const GeomKind geom_kind = parse_geom(geom);
    // The flat sympy kernel now has an isoparametric form; the blockwise, rowwise and
    // facewise CSE arrangements do not, so those combinations are still rejected rather
    // than silently falling back to a different kernel.
    // Isoparametric geometry has exactly two element kernels -- the hand-written one
    // (`current`) and the generated one (`sympy`) -- plus the finite-difference reference
    // and the split. Sum factorisation has no isoparametric form at all: it exists
    // because an affine element has one constant Jacobian to factor out, which is
    // precisely what isoparametric geometry does not have. The other CSE arrangements
    // were never generated isoparametrically.
    //
    // These are rejected rather than mapped onto the hand-written kernel. Mapping them is
    // what the atomic layout used to do, and it meant `--kernel sumfact --geom isoparam`
    // reported the hand-written kernel's throughput under the name `sumfact`.
    // Only for the operations that actually consult the kernel. The Jacobian action and
    // the SpMV ignore --kernel entirely -- no apply_jacobian_action_* takes a KernelKind
    // -- so rejecting them on the default kernel name would refuse a run that never uses
    // it. That is exactly what happened: `--jac-action --geom isoparam` inherits the
    // default `sumfact` and was refused for a kernel it does not call.
    // --assemble-diag dispatches on geometry alone too (see diag_fn), so it belongs on
    // this list for the same reason.
    const bool kernel_is_consulted = !(jac_action || bsr_apply || assemble_diag);
    if (kernel_is_consulted && geom_kind == GeomKind::Isoparam &&
        kernel_kind != KernelKind::Current && kernel_kind != KernelKind::Sympy &&
        kernel_kind != KernelKind::Fd && kernel_kind != KernelKind::Split) {
        std::fprintf(stderr,
                     "--geom isoparam supports --kernel current|sympy|fd|split; '%s' has no "
                     "isoparametric form\n",
                     kernel.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    if (layout != "packed" && layout != "atomic" && layout != "colored" && layout != "store") {
        std::fprintf(stderr, "invalid --layout '%s' (expected packed, atomic, colored or store)\n", layout.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    // The split assembly restores a saved constant half into the global BSR values
    // and adds the velocity-dependent half through precomputed element slots, so it
    // is defined only for the atomic layout. Reject the other combinations rather
    // than letting them fall through to the layout's default kernel: a silent
    // fallback here reports a throughput and a verification result for a kernel
    // that never ran.
    if (kernel_kind == KernelKind::Split && layout != "atomic") {
        std::fprintf(stderr, "--kernel split requires --layout atomic (got '%s')\n", layout.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }

    for (int i = 0; i < 64; ++i) g_identity_slots[i] = i;

    MeshData d;
    d.mesh = smesh::Mesh::create_hex8_cube(smesh::Communicator::self(), n, n, n, 0, 0, 0, 1, 1, 1);
    if (!d.mesh || d.mesh->element_type(0) != smesh::HEX8) {
        std::fprintf(stderr, "failed to create HEX8 mesh\n");
        if (own_mpi) MPI_Finalize();
        return 1;
    }

    if (use_sfc) {
        auto sfc = smesh::SFC::create_from_env();
        sfc->reorder(*d.mesh);
    }

    PackedData packed;
    if (layout == "packed" || layout == "colored" || layout == "store" || verify || verify_jac || jac_action ||
        bsr_apply)
        packed = make_packed(d.mesh, pack_size);
    PackColoring colors;
    if (layout == "colored" || verify || verify_jac)
        colors = cvfem_build_pack_coloring(packed.n_packs, packed.owned_nodes_ptr, packed.ghost_ptr, packed.ghost_idx);

    d.nnodes    = d.mesh->n_nodes();
    d.nelements = d.mesh->n_elements(0);
    d.elems     = d.mesh->elements(0)->data();
    d.points    = d.mesh->points()->data();

    if (warp != scalar_t(0)) {
        const scalar_t pi = std::acos(scalar_t(-1));
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            d.points[0][i] += smesh::geom_t(warp * std::sin(pi * scalar_t(d.points[1][i])));
        }
    }

    fill_fields(d);
    precompute_affine_geometry(d);

    // --- optional terms -------------------------------------------------------------
    //
    // The boundary mask is built from the bounding box rather than from a sideset. That
    // is enough here and it is what the solver's own fallback does when no sideset is
    // named (build_face_mask in cvfem_hex8_ns_op.cpp): the benchmark mesh is a box, so a
    // coordinate test and a topological skin agree on it by construction. On a mesh with
    // a re-entrant face they would not, which is why the solver prefers the sideset --
    // but a benchmark that measured a non-box mesh would be measuring something else.
    if (boundary) {
        d.Lx = d.Ly = d.Lz = 0;
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            d.Lx = std::max(d.Lx, (scalar_t)d.points[0][i]);
            d.Ly = std::max(d.Ly, (scalar_t)d.points[1][i]);
            d.Lz = std::max(d.Lz, (scalar_t)d.points[2][i]);
        }
        d.face_mask.assign((size_t)d.nelements, 0);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t e = 0; e < d.nelements; ++e) {
            scalar_t x[CVFEM_HEX8_N_NODES], y[CVFEM_HEX8_N_NODES], z[CVFEM_HEX8_N_NODES];
            gather_element_coords(d, e, x, y, z);
            uint8_t m = 0;
            for (int f = 0; f < 6; ++f)
                if (hex8_face_on_domain(f, x, y, z, d.Lx, d.Ly, d.Lz)) m |= (uint8_t)(1u << f);
            d.face_mask[(size_t)e] = m;
        }
        ptrdiff_t nfaces = 0, nel_touched = 0;
        for (ptrdiff_t e = 0; e < d.nelements; ++e) {
            const int m = d.face_mask[(size_t)e];
            if (m) ++nel_touched;
            for (int f = 0; f < 6; ++f) nfaces += (m >> f) & 1;
        }
        std::printf("boundary: %td faces on %td of %td elements\n", nfaces, nel_touched, d.nelements);
    }

    if (rhie_chow) {
        d.rhie_chow_scale = rc_scale;
        // Built once here, so the timed loop measures the apply with the gradient already
        // available -- the hoisted case, which is what SFEM_PGRAD_CACHE=1 gives the solver.
        // The other case, rebuilding it inside every apply, is the more expensive one and
        // is NOT measured yet: the gradient is a full element sweep costing about 39% of
        // an apply, and caching it is worth 1.26x off the whole linear solve
        // (docs/README_alps.md). Exposing both is the point of a cascade and is the next
        // step here; until it exists, a --rhie-chow number is the hoisted figure.
        cvfem_hex8_assemble_nodal_grad(d, geom_kind == GeomKind::Isoparam ? 1 : 0, d.p.data(), 1,
                                       d.pgx, d.pgy, d.pgz);
        std::printf("rhie_chow: scale %g, nodal gradient %s\n", (double)rc_scale,
                    pgrad_per_apply ? "rebuilt per apply" : "hoisted out of the timed loop");
    }

    BSR4                  bsr;
    std::vector<scalar_t> jac_linear;
    if (assemble || verify_jac || bsr_apply) bsr = make_bsr4(d.mesh);
    if (assemble || verify_jac || bsr_apply) {
        if (layout == "packed" || verify_jac || bsr_apply)
            build_pack_local_crs(packed, d.nelements, bsr.rowptr, bsr.colidx);
        if (layout == "atomic" || layout == "colored") precompute_element_bsr_slots(d, bsr);
        if (kernel_kind == KernelKind::Split) {
            // One-time cost in a Newton loop, so it is built before the timed region.
            precompute_element_bsr_slots(d, bsr);
            if (geom_kind == GeomKind::Isoparam)
                assemble_jacobian_atomic_linear_isoparam(d, bsr, mu, jac_linear);
            else
                assemble_jacobian_atomic_linear(d, bsr, mu, jac_linear);
        }
        if (layout == "store") build_pack_store_crs(packed, d.nelements, bsr.rowptr, bsr.colidx);
    }

    if (layout == "packed" || layout == "colored" || layout == "store" || verify || verify_jac || jac_action ||
        bsr_apply) {
        const size_t scratch_n = packed_scratch_n(packed);
        const size_t bsr_n =
                16 * (size_t)std::max<ptrdiff_t>(std::max(packed.max_local_nnz, packed.st_max_local_nnz), 1);
        const size_t slot2_n   = std::max(scratch_n, bsr_n);
#pragma omp parallel
        {
            (void)thread_scratch<scalar_t>(0, scratch_n);
            (void)thread_scratch<scalar_t>(1, scratch_n);
            if (assemble || verify_jac || jac_action || bsr_apply) (void)thread_scratch<scalar_t>(2, slot2_n);
            // Slot 3 grows from three arrays to six under --rhie-chow (coordinates plus the
            // nodal pressure gradient) and slot 4 appears only for the Jacobian action's
            // direction gradient. Touched here so the first timed call does not pay for the
            // allocation.
            if (geom_kind == GeomKind::Isoparam || verify || rhie_chow)
                (void)thread_scratch<scalar_t>(3, rhie_chow ? packed_rc_n(packed) : packed_xyz_n(packed));
            if (rhie_chow && jac_action) (void)thread_scratch<scalar_t>(4, packed_qg_n(packed));
        }
    }

    if (verify) {
        apply_residual_atomic(d, rho, mu);
        std::vector<scalar_t> current_r;
        pack_residual(d, current_r);

        apply_residual_atomic_sumfact(d, rho, mu);
        std::vector<scalar_t> sumfact_r;
        pack_residual(d, sumfact_r);
        const scalar_t sumfact_err = max_abs_diff(current_r.data(), sumfact_r.data(), (ptrdiff_t)current_r.size());
        std::printf("verify_sumfact_residual_vs_current_abs: %.6e\n", sumfact_err);
        if (sumfact_err > 1.0e-10) {
            std::fprintf(stderr, "HEX8 sumfact residual mismatch\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }

        apply_residual_atomic_isoparam(d, rho, mu);
        std::vector<scalar_t> isoparam_r;
        pack_residual(d, isoparam_r);
        const scalar_t iso_err = max_abs_diff(current_r.data(), isoparam_r.data(), (ptrdiff_t)current_r.size());
        std::printf("verify_isoparam_residual_vs_affine_abs: %.6e\n", iso_err);

        // The generated isoparametric kernel against the hand-written one. Both compute
        // the same discretisation, so this must be at rounding level -- unlike the
        // comparison above, which is isoparametric against affine and is a property of
        // the mesh rather than of the code.
        apply_residual_atomic_isoparam_sympy(d, rho, mu);
        std::vector<scalar_t> isoparam_sympy_r;
        pack_residual(d, isoparam_sympy_r);
        const scalar_t iso_sympy_err = max_abs_diff(isoparam_r.data(), isoparam_sympy_r.data(),
                                                    (ptrdiff_t)isoparam_r.size());
        std::printf("verify_sympy_isoparam_residual_vs_handwritten_abs: %.6e\n", iso_sympy_err);
        if (iso_sympy_err > 1.0e-12) {
            std::fprintf(stderr, "HEX8 sympy isoparametric residual mismatch\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }
        if (warp == scalar_t(0)) {
            if (iso_err > 1.0e-12) {
                std::fprintf(stderr, "HEX8 cube isoparam residual mismatch vs affine\n");
                if (own_mpi) MPI_Finalize();
                return 1;
            }
        } else if (iso_err <= 1.0e-12) {
            std::fprintf(stderr, "HEX8 warped isoparam residual unexpectedly matches affine\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }

        if (layout == "packed" || verify_jac) {
            apply_residual_packed(d, packed, rho, mu, KernelKind::Current, GeomKind::Affine);
            std::vector<scalar_t> packed_current_r;
            pack_residual(d, packed_current_r);
            const scalar_t packed_err =
                    max_abs_diff(current_r.data(), packed_current_r.data(), (ptrdiff_t)current_r.size());
            std::printf("verify_packed_residual_vs_atomic_abs: %.6e\n", packed_err);
            if (packed_err > 1.0e-10) {
                std::fprintf(stderr, "HEX8 packed residual mismatch\n");
                if (own_mpi) MPI_Finalize();
                return 1;
            }

            apply_residual_packed(d, packed, rho, mu, KernelKind::Sumfact, GeomKind::Affine);
            std::vector<scalar_t> packed_sumfact_r;
            pack_residual(d, packed_sumfact_r);
            const scalar_t packed_sf_err =
                    max_abs_diff(current_r.data(), packed_sumfact_r.data(), (ptrdiff_t)current_r.size());
            std::printf("verify_packed_sumfact_residual_vs_current_abs: %.6e\n", packed_sf_err);
            if (packed_sf_err > 1.0e-10) {
                std::fprintf(stderr, "HEX8 packed sumfact residual mismatch\n");
                if (own_mpi) MPI_Finalize();
                return 1;
            }

            apply_residual_packed(d, packed, rho, mu, KernelKind::Sumfact, GeomKind::Isoparam);
            std::vector<scalar_t> packed_iso_r;
            pack_residual(d, packed_iso_r);
            const scalar_t packed_iso_err =
                    max_abs_diff(isoparam_r.data(), packed_iso_r.data(), (ptrdiff_t)packed_iso_r.size());
            std::printf("verify_packed_isoparam_residual_vs_atomic_abs: %.6e\n", packed_iso_err);
            if (packed_iso_err > 1.0e-10) {
                std::fprintf(stderr, "HEX8 packed isoparam residual mismatch\n");
                if (own_mpi) MPI_Finalize();
                return 1;
            }
        }

        {
            apply_residual_colored(d, packed, colors, rho, mu, KernelKind::Sumfact, GeomKind::Affine);
            std::vector<scalar_t> colored_r;
            pack_residual(d, colored_r);
            const scalar_t colored_err = max_abs_diff(current_r.data(), colored_r.data(), (ptrdiff_t)current_r.size());
            std::printf("verify_colored_residual_vs_atomic_abs: %.6e\n", colored_err);
            if (colored_err > 1.0e-10) {
                std::fprintf(stderr, "HEX8 colored residual mismatch\n");
                if (own_mpi) MPI_Finalize();
                return 1;
            }

            apply_residual_colored(d, packed, colors, rho, mu, KernelKind::Sympy, GeomKind::Affine);
            std::vector<scalar_t> colored_sympy_r;
            pack_residual(d, colored_sympy_r);
            const scalar_t colored_sympy_err =
                    max_abs_diff(current_r.data(), colored_sympy_r.data(), (ptrdiff_t)current_r.size());
            std::printf("verify_colored_sympy_residual_vs_atomic_abs: %.6e\n", colored_sympy_err);
            if (colored_sympy_err > 1.0e-10) {
                std::fprintf(stderr, "HEX8 colored SymPy residual mismatch\n");
                if (own_mpi) MPI_Finalize();
                return 1;
            }

            apply_residual_colored(d, packed, colors, rho, mu, KernelKind::Sumfact, GeomKind::Isoparam);
            std::vector<scalar_t> colored_iso_r;
            pack_residual(d, colored_iso_r);
            const scalar_t colored_iso_err =
                    max_abs_diff(isoparam_r.data(), colored_iso_r.data(), (ptrdiff_t)colored_iso_r.size());
            std::printf("verify_colored_isoparam_residual_vs_atomic_abs: %.6e\n", colored_iso_err);
            if (colored_iso_err > 1.0e-10) {
                std::fprintf(stderr, "HEX8 colored isoparam residual mismatch\n");
                if (own_mpi) MPI_Finalize();
                return 1;
            }
        }

        if (layout == "packed")
            apply_residual_packed(d, packed, rho, mu, KernelKind::Sympy, GeomKind::Affine);
        else
            apply_residual_atomic_sympy(d, rho, mu);
        std::vector<scalar_t> sympy_r;
        pack_residual(d, sympy_r);
        const scalar_t max_err = max_abs_diff(current_r.data(), sympy_r.data(), (ptrdiff_t)current_r.size());
        std::printf("verify_sympy_residual_vs_current_abs: %.6e\n", max_err);
        if (max_err > 1.0e-10) {
            std::fprintf(stderr, "HEX8 SymPy residual mismatch\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }
    }

    // The boundary closure is one extra element sweep after the layout's own, which is
    // how the solver arranges it too -- see apply_boundary_scs_residual_pass. Doing it
    // here rather than inside each layout means --boundary works for all four.
    auto apply_fn = [&]() {
        if (pgrad_per_apply)
            cvfem_hex8_assemble_nodal_grad(d, geom_kind == GeomKind::Isoparam ? 1 : 0, d.p.data(), 1,
                                           d.pgx, d.pgy, d.pgz);
        if (geom_kind == GeomKind::Isoparam) {
            if (layout == "colored")
                apply_residual_colored(d, packed, colors, rho, mu, kernel_kind, GeomKind::Isoparam);
            else if (layout == "packed" || layout == "store")
                apply_residual_packed(d, packed, rho, mu, kernel_kind, GeomKind::Isoparam);
            else if (kernel_kind == KernelKind::Sympy)
                apply_residual_atomic_isoparam_sympy(d, rho, mu);
            else
                apply_residual_atomic_isoparam(d, rho, mu);
        } else if (layout == "colored")
            apply_residual_colored(d, packed, colors, rho, mu, kernel_kind, GeomKind::Affine);
        else if (layout == "packed" || layout == "store")
            apply_residual_packed(d, packed, rho, mu, kernel_kind, GeomKind::Affine);
        else if (kernel_uses_sympy_residual(kernel_kind))
            apply_residual_atomic_sympy(d, rho, mu);
        else if (kernel_kind == KernelKind::Sumfact)
            apply_residual_atomic_sumfact(d, rho, mu);
        else
            apply_residual_atomic(d, rho, mu);
            if (boundary)
            apply_boundary_scs_residual_pass(d, rho, mu, geom_kind == GeomKind::Isoparam ? 1 : 0);
    };
    auto jac_fn = [&]() {
        if (geom_kind == GeomKind::Isoparam) {
            if (layout == "store")
                assemble_jacobian_store(d, packed, bsr, rho, mu, kernel_kind, GeomKind::Isoparam);
            else if (layout == "colored")
                assemble_jacobian_colored(d, packed, colors, bsr, rho, mu, kernel_kind, GeomKind::Isoparam);
            else if (layout == "packed")
                assemble_jacobian_packed(d, packed, bsr, rho, mu, kernel_kind, GeomKind::Isoparam);
            else if (kernel_kind == KernelKind::Split)
                assemble_jacobian_atomic_nonlinear_isoparam(d, bsr, rho, mu, jac_linear);
            else if (kernel_kind == KernelKind::Sympy)
                assemble_jacobian_atomic_isoparam_sympy(d, bsr, rho, mu);
            else if (kernel_kind == KernelKind::Fd)
                assemble_jacobian_atomic_fd_isoparam(d, bsr, rho, mu);
            else
                // Current: the hand-written isoparametric kernel. Every other name is
                // rejected during validation, so this is not a fallback.
                assemble_jacobian_atomic_isoparam(d, bsr, rho, mu);
        } else if (layout == "store") {
            assemble_jacobian_store(d, packed, bsr, rho, mu, kernel_kind, GeomKind::Affine);
        } else if (layout == "colored") {
            assemble_jacobian_colored(d, packed, colors, bsr, rho, mu, kernel_kind, GeomKind::Affine);
        } else if (layout == "packed") {
            assemble_jacobian_packed(d, packed, bsr, rho, mu, kernel_kind, GeomKind::Affine);
        } else if (kernel_kind == KernelKind::Sumfact)
            assemble_jacobian_atomic_sumfact(d, bsr, rho, mu);
        else if (kernel_kind == KernelKind::Sympy)
            assemble_jacobian_atomic_sympy(d, bsr, rho, mu);
        else if (kernel_kind == KernelKind::SympyBlock)
            assemble_jacobian_atomic_sympy_block(d, bsr, rho, mu);
        else if (kernel_kind == KernelKind::SympyRow)
            assemble_jacobian_atomic_sympy_row(d, bsr, rho, mu);
        else if (kernel_kind == KernelKind::SympyFace)
            assemble_jacobian_atomic_sympy_face(d, bsr, rho, mu);
        else if (kernel_kind == KernelKind::Split)
            // Restore the geometry-only half built once at setup, then add only
            // the velocity-dependent half. The linear half is not rebuilt here:
            // that is the whole point of the split.
            assemble_jacobian_atomic_nonlinear(d, bsr, rho, mu, jac_linear);
        else
            // Current and Fd both land here. There is no dedicated `current`
            // assembly kernel -- the loop residual kernel has no assembled
            // counterpart -- so `--kernel current --assemble` measures the
            // finite-difference kernel. Kept as the fallback rather than
            // rejected, because fd is also the correctness reference, but the
            // two rows are the same kernel and should not be read as distinct.
            assemble_jacobian_atomic_fd(d, bsr, rho, mu);
    };

    std::vector<scalar_t> jac_dir, jac_out;
    if (jac_action || verify_jac || bsr_apply) {
        jac_dir.resize((size_t)d.nnodes * N_FIELDS);
        jac_out.assign((size_t)d.nnodes * N_FIELDS, 0.0);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < d.nnodes * N_FIELDS; ++i) jac_dir[(size_t)i] = 1.0 + 0.01 * scalar_t(i % 7);
    }

    // The Krylov working set, when --live-vectors asks for it.
    //
    // The default benchmark replays one apply on one direction, so after the first call the
    // direction, the output and the pack staging are all warm and the measurement is of the
    // kernel with the memory system on its side. A Krylov iteration is not that: it carries
    // its own basis -- BiCGStab holds about seven vectors of the solution size -- touches
    // every one of them between applies, and hands the operator a different direction each
    // time. On Grace those seven are ~35 MB each against 117 MB of L3, so which of the two
    // is being measured is not a detail.
    //
    // Only the apply is timed. The churn below runs between applies, outside the clock, so
    // the number reported stays the apply's throughput and only its cache environment
    // changes.
    std::vector<std::vector<scalar_t>> live;
    if (live_vectors > 0 && (jac_action || bsr_apply)) {
        live.resize((size_t)live_vectors);
        for (int k = 0; k < live_vectors; ++k) {
            live[(size_t)k].resize((size_t)d.nnodes * N_FIELDS);
            scalar_t *const SFEM_RESTRICT v = live[(size_t)k].data();
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < d.nnodes * N_FIELDS; ++i)
                v[(size_t)i] = 1.0 + 0.01 * scalar_t((i + k) % 7);
        }
        std::printf("live_vectors: %d x %td dof (%.1f MiB total)\n", live_vectors,
                    (ptrdiff_t)d.nnodes * N_FIELDS,
                    double(live_vectors) * double(d.nnodes) * N_FIELDS * sizeof(scalar_t) / (1024.0 * 1024.0));
    }
    // The direction the last timed apply actually used, so the cross-layout check below
    // compares like with like when the rotation is on.
    const scalar_t *last_dir = jac_dir.data();
    long            live_k   = 0;

    // One Krylov iteration's worth of vector traffic: two axpys and a dot over the basis,
    // which is the shape BiCGStab has and enough to evict what the apply would otherwise
    // have kept. Deliberately not a real BiCGStab -- the point is the memory traffic, not
    // the algorithm.
    auto churn_fn = [&]() {
        if (live.empty()) return;
        const ptrdiff_t n = d.nnodes * N_FIELDS;
        scalar_t        acc = 0;
        for (size_t k = 0; k < live.size(); ++k) {
            scalar_t *const SFEM_RESTRICT v = live[k].data();
            const scalar_t *const SFEM_RESTRICT y = jac_out.data();
            const scalar_t alpha = scalar_t(1e-8) * scalar_t(k + 1);
            scalar_t       part  = 0;
#pragma omp parallel for schedule(static) reduction(+ : part)
            for (ptrdiff_t i = 0; i < n; ++i) {
                v[(size_t)i] += alpha * y[(size_t)i];
                part += v[(size_t)i] * y[(size_t)i];
            }
            acc += part;
        }
        // Consumed so the loop above cannot be optimised away.
        g_churn_sink += acc;
    };
    // The Jacobian action's Rhie-Chow term differentiates through the nodal gradient
    // reconstruction, so it needs that reconstruction applied to the Krylov DIRECTION's
    // pressure as well as to the state's. The two have opposite lifetimes and that is the
    // whole point of measuring this: the state's gradient is a function of the Newton
    // iterate and is hoisted out of the entire Krylov solve (SFEM_PGRAD_CACHE), while the
    // direction's changes with every matvec and cannot be hoisted out of anything. It is
    // therefore rebuilt inside the timed lambda, unconditionally, and timed separately so
    // its share of the matvec is visible rather than folded into the element sweep.
    double qgrad_seconds = 0;
    const bool with_qgrad = rhie_chow && jac_action;
    auto jac_action_fn = [&]() {
        // A Krylov iteration never sees the same direction twice, so neither does this when
        // the live set exists: the rotation is what stops the direction being resident.
        const scalar_t *const dir_v = live.empty() ? jac_dir.data() : live[(size_t)(live_k++ % (long)live.size())].data();
        last_dir                    = dir_v;
        if (with_qgrad) {
            const double t0 = wall_time();
            cvfem_hex8_assemble_nodal_grad(d, geom_kind == GeomKind::Isoparam ? 1 : 0, dir_v + 3, N_FIELDS,
                                           d.qgx, d.qgy, d.qgz);
            qgrad_seconds += wall_time() - t0;
        }
        if (layout == "colored")
            apply_jacobian_action_colored(d, packed, colors, rho, mu, dir_v, jac_out.data(), geom_kind);
        else if (layout == "packed" || layout == "store")
            apply_jacobian_action_packed(d, packed, rho, mu, dir_v, jac_out.data(), geom_kind);
        else if (geom_kind == GeomKind::Isoparam)
            apply_jacobian_action_atomic_isoparam(d, rho, mu, dir_v, jac_out.data());
        else
            apply_jacobian_action_atomic(d, rho, mu, dir_v, jac_out.data());
            if (boundary)
            apply_boundary_scs_jacobian_action_pass(d, rho, mu, geom_kind == GeomKind::Isoparam ? 1 : 0,
                                                    dir_v, jac_out.data());
    };

    // Block diagonal, for the block-Jacobi preconditioner. Assembles only the 4x4
    // diagonal blocks -- 16 doubles per node instead of the whole matrix.
    std::vector<scalar_t> diag_blocks;
    auto diag_fn = [&]() {
        if (geom_kind == GeomKind::Isoparam)
            assemble_diag_atomic_isoparam(d, rho, mu, diag_blocks);
        else
            assemble_diag_atomic(d, rho, mu, diag_blocks);
    };

    if (bsr_apply) jac_fn();
    decltype(sfem::h_bsr_spmv<smesh::count_t, smesh::idx_t, scalar_t>(
            d.nnodes, d.nnodes, 4, bsr.graph->rowptr(), bsr.graph->colidx(), bsr.values, scalar_t(0))) bsr_apply_op;
    if (bsr_apply) {
        bsr_apply_op = sfem::h_bsr_spmv<smesh::count_t, smesh::idx_t, scalar_t>(
                d.nnodes, d.nnodes, 4, bsr.graph->rowptr(), bsr.graph->colidx(), bsr.values, scalar_t(0));
    }
    auto bsr_apply_fn = [&]() { bsr_apply_op->apply(jac_dir.data(), jac_out.data()); };

    if (verify_jac) {
        jac_fn();
        const scalar_t rel = verify_jacobian_fd(d, bsr, rho, mu, geom_kind);
        std::printf("verify_jac_spmv_vs_fd_rel: %.6e\n", rel);
        if (rel > 1.0e-6) {
            std::fprintf(stderr, "HEX8 BSR Jacobian mismatch\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }

        std::vector<scalar_t> jv_spmv((size_t)d.nnodes * N_FIELDS), jv_mf((size_t)d.nnodes * N_FIELDS),
                jv_mf_atomic((size_t)d.nnodes * N_FIELDS), jv_mf_colored((size_t)d.nnodes * N_FIELDS);
        bsr4_spmv(bsr, d.nnodes, jac_dir.data(), jv_spmv.data());
        apply_jacobian_action_packed(d, packed, rho, mu, jac_dir.data(), jv_mf.data(), geom_kind);
        if (geom_kind == GeomKind::Isoparam)
            apply_jacobian_action_atomic_isoparam(d, rho, mu, jac_dir.data(), jv_mf_atomic.data());
        else
            apply_jacobian_action_atomic(d, rho, mu, jac_dir.data(), jv_mf_atomic.data());
        if (colors.n_colors > 0)
            apply_jacobian_action_colored(
                    d, packed, colors, rho, mu, jac_dir.data(), jv_mf_colored.data(), geom_kind);
        const scalar_t mf_err      = max_abs_diff(jv_spmv.data(), jv_mf.data(), d.nnodes * N_FIELDS);
        const scalar_t atomic_err  = max_abs_diff(jv_mf.data(), jv_mf_atomic.data(), d.nnodes * N_FIELDS);
        const scalar_t colored_err = colors.n_colors > 0
                                             ? max_abs_diff(jv_mf.data(), jv_mf_colored.data(), d.nnodes * N_FIELDS)
                                             : scalar_t(0);
        std::printf("verify_jac_mf_action_vs_spmv_abs: %.6e\n", mf_err);
        std::printf("verify_jac_mf_atomic_action_vs_packed_abs: %.6e\n", atomic_err);
        if (colors.n_colors > 0) std::printf("verify_jac_mf_colored_action_vs_packed_abs: %.6e\n", colored_err);
        if (colored_err > 1.0e-12) {
            std::fprintf(stderr, "HEX8 colored Jacobian-action mismatch\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }
        if (mf_err > 1.0e-8 || atomic_err > 1.0e-12) {
            std::fprintf(stderr, "HEX8 Jacobian-action mismatch\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }
    }

    // Verify the two strategies that reuse the full element kernel through a modified
    // slot array: they must reproduce the full assembly exactly, not approximately.
    if (verify_jac && (assemble_diag || kernel_kind == KernelKind::Split)) {
        if (geom_kind == GeomKind::Isoparam)
            assemble_jacobian_atomic_isoparam(d, bsr, rho, mu);
        else
            assemble_jacobian_atomic_sumfact(d, bsr, rho, mu);
        const scalar_t *const ref = bsr.values->data();

        if (assemble_diag) {
            // Pull the diagonal blocks out of the full matrix and compare.
            std::vector<scalar_t> ref_diag((size_t)d.nnodes * 16, scalar_t(0));
            for (ptrdiff_t r = 0; r < d.nnodes; ++r)
                for (smesh::count_t j = bsr.rowptr[r]; j < bsr.rowptr[r + 1]; ++j)
                    if (bsr.colidx[j] == (smesh::idx_t)r)
                        std::memcpy(&ref_diag[(size_t)r * 16], &ref[(size_t)j * 16],
                                    16 * sizeof(scalar_t));
            diag_fn();
            scalar_t dmax = 0;
            for (scalar_t v : ref_diag) dmax = std::max(dmax, std::fabs(v));
            const scalar_t rel =
                    max_abs_diff(ref_diag.data(), diag_blocks.data(), (ptrdiff_t)ref_diag.size()) /
                    (dmax > 0 ? dmax : scalar_t(1));
            std::printf("verify_diag_vs_full_assembly_rel: %.6e\n", rel);
            if (rel > 1.0e-12) {
                std::fprintf(stderr, "HEX8 block-diagonal mismatch\n");
                if (own_mpi) MPI_Finalize();
                return 1;
            }
        } else if (geom_kind == GeomKind::Isoparam) {
            // Isoparametric split: linear + nonlinear must reproduce the full assembly.
            std::vector<scalar_t> full(ref, ref + (size_t)bsr.nnz * 16);
            scalar_t              fmax = 0;
            for (scalar_t v : full) fmax = std::max(fmax, std::fabs(v));
            assemble_jacobian_atomic_linear_isoparam(d, bsr, mu, jac_linear);
            assemble_jacobian_atomic_nonlinear_isoparam(d, bsr, rho, mu, jac_linear);
            const scalar_t rel =
                    max_abs_diff(full.data(), bsr.values->data(), (ptrdiff_t)full.size()) /
                    (fmax > 0 ? fmax : scalar_t(1));
            std::printf("verify_split_isoparam_vs_full_rel: %.6e\n", rel);
            if (rel > 1.0e-12) {
                std::fprintf(stderr, "HEX8 isoparametric split mismatch\n");
                if (own_mpi) MPI_Finalize();
                return 1;
            }
        }
    }

    for (int i = 0; i < warmup; ++i) {
        if (assemble)
            jac_fn();
        else if (assemble_diag)
            diag_fn();
        else if (jac_action)
            jac_action_fn();
        else if (bsr_apply)
            bsr_apply_fn();
        else
            apply_fn();
    }

    phase_reset();
    // With a live set the clock has to be stopped for the churn: what is being measured is
    // still the apply, only now with the caches in the state a Krylov iteration leaves them
    // in. Without one this is the same single interval it always was, so the default path's
    // timing is unchanged rather than merely equivalent.
    double       t0 = 0, t1 = 0;
    if (live.empty()) {
        t0 = wall_time();
        for (int i = 0; i < repeat; ++i) {
            if (assemble)
                jac_fn();
            else if (assemble_diag)
                diag_fn();
            else if (jac_action)
                jac_action_fn();
            else if (bsr_apply)
                bsr_apply_fn();
            else
                apply_fn();
        }
        t1 = wall_time();
    } else {
        double acc = 0;
        for (int i = 0; i < repeat; ++i) {
            // Before the apply, not after: the churn writes into the live vectors, and one
            // of them is the direction the apply is about to take. Running it afterwards
            // would leave the last apply's input modified, and the cross-layout check below
            // -- which re-applies the atomic kernel to that same vector -- would report a
            // mismatch that is nothing but the churn.
            churn_fn();
            const double a = wall_time();
            if (jac_action)
                jac_action_fn();
            else
                bsr_apply_fn();
            acc += wall_time() - a;
        }
        t0 = 0;
        t1 = acc;
    }

    // The new staging checked against the reference layout, outside the timed region.
    // The `checksum` printed below cannot do this job here: the Jacobian action telescopes,
    // so summing it over a near-uniform direction gives ~1e-15 whether or not the
    // Rhie-Chow term is present, and two layouts that disagree everywhere still agree in
    // that sum. A max-abs difference does not cancel.
    if (with_qgrad && layout != "atomic") {
        std::vector<scalar_t> jv_ref((size_t)d.nnodes * N_FIELDS, 0.0);
        // last_dir, not jac_dir: under --live-vectors the timed loop rotates the direction,
        // and comparing the packed result against the atomic action on a different vector
        // would fail for a reason that has nothing to do with the staging.
        apply_jacobian_action_atomic(d, rho, mu, last_dir, jv_ref.data());
        scalar_t ref_max = 0;
        for (ptrdiff_t i = 0; i < d.nnodes * N_FIELDS; ++i) ref_max = std::max(ref_max, std::fabs(jv_ref[(size_t)i]));
        const scalar_t err = max_abs_diff(jv_ref.data(), jac_out.data(), d.nnodes * N_FIELDS);
        const scalar_t rel = ref_max > scalar_t(0) ? err / ref_max : err;
        std::printf("jac_action_rc_vs_atomic_rel: %.6e\n", (double)rel);
        if (!(rel < 1.0e-10)) {
            std::fprintf(stderr,
                         "packed Jacobian action with Rhie-Chow disagrees with the atomic "
                         "reference (rel %.3e)\n",
                         (double)rel);
            if (own_mpi) MPI_Finalize();
            return 1;
        }
    }

    const double seconds          = t1 - t0;
    const double seconds_per_call = seconds / double(repeat);
    // Primary metric: unique mesh degrees of freedom per second, i.e. the number of
    // unknowns the solver actually carries (one velocity triple + pressure per node)
    // divided by the time to sweep them once. Element visits count each node once per
    // adjacent element, so they overstate throughput by the nodal valence (8 for HEX8);
    // they are reported too because they measure the element kernel rather than the
    // discretisation, but MDOF/s is the number to compare across element types.
    const ptrdiff_t n_dofs        = d.nnodes * N_FIELDS;
    const double    mdofs         = double(n_dofs) / seconds_per_call / 1.0e6;
    const double    melems        = double(d.nelements) / seconds_per_call / 1.0e6;
    const double    visit_mdofs   = double(d.nelements) * double(CVFEM_HEX8_N_DOF) / seconds_per_call / 1.0e6;

    const double residual_flops =
            geom_kind == GeomKind::Isoparam ? CVFEM_HEX8_ISOPARAM_RESIDUAL_FLOPS_PER_ELEMENT
                                            : CVFEM_HEX8_RESIDUAL_FLOPS_PER_ELEMENT;
    const double jac_action_flops =
            geom_kind == GeomKind::Isoparam ? CVFEM_HEX8_ISOPARAM_JAC_ACTION_FLOPS_PER_ELEMENT
                                            : CVFEM_HEX8_JAC_ACTION_FLOPS_PER_ELEMENT;
    const double assemble_flops =
            geom_kind == GeomKind::Isoparam ? CVFEM_HEX8_ISOPARAM_ASSEMBLE_FLOPS_PER_ELEMENT
                                            : CVFEM_HEX8_ASSEMBLE_FLOPS_PER_ELEMENT;
    const double elem_apps = double(repeat) * double(d.nelements);

    scalar_t checksum = 0;
    if (assemble) {
        for (ptrdiff_t i = 0; i < bsr.nnz * 16; ++i) checksum += bsr.values->data()[i];
    } else if (jac_action || bsr_apply) {
        for (ptrdiff_t i = 0; i < d.nnodes * N_FIELDS; ++i) checksum += jac_out[(size_t)i];
    } else {
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) checksum += d.rx[i] + d.ry[i] + d.rz[i] + d.rc[i];
    }

    phase_report(assemble ? "assemble" : (jac_action ? "jac_action" : "residual"), repeat, threads_active());
    std::printf("cvfem_hex8_ns_upwind_smesh\n");
    std::printf("  mesh_manager: smesh::Mesh::create_hex8_cube\n");
    std::printf("  operation: %s\n",
                bsr_apply ? "bsr_apply" : (jac_action ? "jacobian_action" : (assemble ? "jacobian_assemble" : "residual")));
    std::printf("  layout: %s\n", layout.c_str());
    std::printf("  kernel: %s\n", kernel.c_str());
    std::printf("  geom: %s\n", geom.c_str());
    std::printf("  warp: %.6e\n", warp);
    std::printf("  OpenMP_threads: %d\n", threads_active());
    if (layout == "store") {
        std::printf("  pack_size: %d\n", pack_size);
        std::printf("  n_packs: %td\n", packed.n_packs);
        std::printf("  st_max_local_nnz: %td\n", packed.st_max_local_nnz);
        std::printf("  st_local_matrix_KiB: %.1f\n", double(packed.st_max_local_nnz) * 128.0 / 1024.0);
    }
    if (layout == "colored") {
        std::printf("  pack_size: %d\n", pack_size);
        std::printf("  n_packs: %td\n", packed.n_packs);
        std::printf("  n_colors: %d\n", colors.n_colors);
        std::printf("  packs_per_color_min_max: %td %td\n", colors.min_packs_per_color, colors.max_packs_per_color);
        if (colors.min_packs_per_color < threads_active()) {
            std::printf("  WARNING: fewer packs per color (%td) than threads (%d); every color barrier\n"
                        "           leaves threads idle. Reduce --pack-size until packs_per_color >= threads.\n",
                        colors.min_packs_per_color,
                        threads_active());
        }
    }
    if (layout == "packed") {
        std::printf("  pack_size: %d\n", pack_size);
        std::printf("  n_packs: %td\n", packed.n_packs);
        std::printf("  n_elements_per_pack: %td\n", packed.n_elements_per_pack);
        std::printf("  mean_nodes_per_pack: %td\n", packed.mean_nodes_per_pack);
        std::printf("  max_actual_nodes_per_pack: %td\n", packed.max_actual_nodes_per_pack);
        if (assemble || bsr_apply) std::printf("  max_local_nnz: %td\n", packed.max_local_nnz);
    }
    std::printf("  cube_n: %d\n", n);
    std::printf("  nodes: %td\n", d.nnodes);
    std::printf("  elements: %td\n", d.nelements);
    std::printf("  dofs: %td\n", n_dofs);
    std::printf("  repeat: %d\n", repeat);
    std::printf("  MDOF/s: %.3f\n", mdofs);
    if (!bsr_apply) {
        std::printf("  MDOF/s_element_visits: %.3f\n", visit_mdofs);
        std::printf("  MELEM/s: %.3f\n", melems);
    }
    std::printf("  checksum: %.16e\n", checksum);
    if (!assemble && !jac_action && !bsr_apply) {
        std::printf("  seconds_per_apply: %.6e\n", seconds_per_call);
        std::printf("  MDOF/s_residual: %.3f\n", mdofs);
        std::printf("  GFLOP/s_model: %.3f\n", elem_apps * residual_flops / seconds / 1.0e9);
        std::printf("  flops_per_element_model: %.1f\n", residual_flops);
    }
    if (assemble || bsr_apply) {
        std::printf("  bsr_nnz: %td\n", bsr.nnz);
        std::printf("  bsr_nnz_per_node: %.3f\n", double(bsr.nnz) / double(d.nnodes));
        {
            smesh::count_t dmin = bsr.rowptr[1] - bsr.rowptr[0];
            smesh::count_t dmax = dmin;
            for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
                const smesh::count_t deg = bsr.rowptr[i + 1] - bsr.rowptr[i];
                dmin                     = std::min(dmin, deg);
                dmax                     = std::max(dmax, deg);
            }
            std::printf("  bsr_row_nnz_min: %d\n", (int)dmin);
            std::printf("  bsr_row_nnz_max: %d\n", (int)dmax);
            std::printf("  bsr_values_MiB: %.3f\n", double(bsr.nnz) * 16.0 * 8.0 / (1024.0 * 1024.0));
            std::printf("  bsr_x_KiB: %.3f\n", double(d.nnodes) * 4.0 * 8.0 / 1024.0);
        }
    }
    if (assemble) {
        std::printf("  seconds_per_assemble: %.6e\n", seconds_per_call);
        std::printf("  MDOF/s_assemble: %.3f\n", mdofs);
        std::printf("  MELEM/s_assemble: %.3f\n", melems);
        std::printf("  GFLOP/s_assemble_model: %.3f\n", elem_apps * assemble_flops / seconds / 1.0e9);
        std::printf("  flops_per_element_assemble_model: %.1f\n", assemble_flops);
    }
    if (jac_action) {
        std::printf("  seconds_per_jac_action: %.6e\n", seconds_per_call);
        std::printf("  MDOF/s_jac_action: %.3f\n", mdofs);
        std::printf("  MELEM/s_jac_action: %.3f\n", melems);
        std::printf("  GFLOP/s_jac_action_model: %.3f\n", elem_apps * jac_action_flops / seconds / 1.0e9);
        std::printf("  flops_per_element_jac_action_model: %.1f\n", jac_action_flops);
        if (with_qgrad) {
            // Reported next to the matvec, not subtracted from it: it IS part of every
            // matvec the solver performs. Splitting it out says how much of the gap
            // between this figure and a Rhie-Chow-free one is the element kernel and how
            // much is the reconstruction sweep in front of it.
            const double per_call = qgrad_seconds / double(repeat);
            std::printf("  seconds_per_jac_action_qgrad: %.6e\n", per_call);
            std::printf("  frac_jac_action_qgrad: %.4f\n", qgrad_seconds / seconds);
            std::printf("  MDOF/s_jac_action_kernel_only: %.3f\n",
                        double(n_dofs) / (seconds_per_call - per_call) / 1.0e6);
        }
    }
    if (bsr_apply) {
        const double bsr_apply_flops = double(bsr.nnz) * 2.0 * 16.0;
        const double bsr_apply_bytes = double(bsr.nnz) * 16.0 * double(sizeof(scalar_t)) +
                                       double(d.nnodes) * 8.0 * double(sizeof(scalar_t)) +
                                       double(bsr.nnz) * double(sizeof(smesh::idx_t));
        std::printf("  seconds_per_bsr_apply: %.6e\n", seconds_per_call);
        std::printf("  MDOF/s_bsr_apply: %.3f\n", mdofs);
        std::printf("  GFLOP/s_bsr_apply_model: %.3f\n", double(repeat) * bsr_apply_flops / seconds / 1.0e9);
        std::printf("  GB/s_bsr_apply_model: %.3f\n", double(repeat) * bsr_apply_bytes / seconds / 1.0e9);
        std::printf("  flops_per_bsr_apply_model: %.1f\n", bsr_apply_flops);
        std::printf("  bytes_per_bsr_apply_model: %.1f\n", bsr_apply_bytes);
    }

    if (!csv_path.empty()) {
        const double op_flops = bsr_apply       ? 0.0
                                : jac_action    ? jac_action_flops
                                : assemble      ? assemble_flops
                                                : residual_flops;
        CsvRow row{};
        row.tag                  = csv_tag.c_str();
        row.operation            = bsr_apply ? "bsr_apply"
                                             : (jac_action ? "jac_action" : (assemble ? "assemble" : "residual"));
        row.layout               = layout.c_str();
        row.kernel               = kernel.c_str();
        row.geom                 = geom.c_str();
        row.threads              = threads_active();
        row.pack_size            = (layout == "atomic") ? 0 : pack_size;
        row.cube_n               = n;
        row.nodes                = d.nnodes;
        row.elements             = d.nelements;
        row.dofs                 = n_dofs;
        row.bsr_nnz              = (assemble || bsr_apply) ? bsr.nnz : 0;
        row.bsr_values_mib       = (assemble || bsr_apply) ? double(bsr.nnz) * 16.0 * 8.0 / (1024.0 * 1024.0) : 0.0;
        row.repeat               = repeat;
        row.seconds_per_call     = seconds_per_call;
        row.mdofs                = mdofs;
        row.mdofs_element_visits = bsr_apply ? 0.0 : visit_mdofs;
        row.melems               = bsr_apply ? 0.0 : melems;
        row.gflops_model         = op_flops * elem_apps / seconds / 1.0e9;
        row.warp                 = warp;
        row.n_colors             = colors.n_colors;
        row.packs_per_color_min  = colors.min_packs_per_color;
        row.packs_per_color_max  = colors.max_packs_per_color;
        row.checksum             = checksum;
        row.rhie_chow            = rhie_chow;
        row.rhie_chow_scale      = rhie_chow ? (double)rc_scale : 0.0;
        row.boundary             = boundary;
        row.phase                = g_breakdown ? g_phase : nullptr;
        csv_write(csv_path, row);
    }

    d.mesh.reset();
    if (own_mpi) MPI_Finalize();
    return 0;
}
