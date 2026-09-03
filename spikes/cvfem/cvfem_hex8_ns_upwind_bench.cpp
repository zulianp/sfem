// HEX8 CVFEM Navier-Stokes benchmark driver.
//
// The operator implementations live in the per-layout headers below; this file
// holds the mesh/option setup, the verification harness and the timing loop.

#include "cvfem_hex8_layout_common.hpp"
#include "cvfem_hex8_layout_atomic.hpp"
#include "cvfem_hex8_layout_colored.hpp"
#include "cvfem_hex8_layout_packed.hpp"
#include "cvfem_hex8_layout_store.hpp"

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
    const double *phase;  // PH_N entries, thread-summed ms per call, or nullptr
};

static void csv_write(const std::string &path, const CsvRow &r) {
    if (path.empty()) return;

    bool need_header = true;
    if (FILE *probe = std::fopen(path.c_str(), "r")) {
        std::fseek(probe, 0, SEEK_END);
        need_header = std::ftell(probe) == 0;
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

    if (need_header) {
        std::fprintf(f,
                     "tag,host,element,operation,layout,kernel,geom,warp,threads,pack_size,cube_n,"
                     "nodes,elements,dofs,bsr_nnz,bsr_values_MiB,repeat,seconds_per_call,"
                     "MDOF_s,MDOF_s_element_visits,MELEM_s,GFLOP_s_model,"
                     "n_colors,packs_per_color_min,packs_per_color_max,checksum");
        for (int i = 0; i < PH_N; ++i) std::fprintf(f, ",ms_%s", g_phase_name[i]);
        std::fprintf(f, "\n");
    }

    std::fprintf(f,
                 "%s,%s,hex8,%s,%s,%s,%s,%.6e,%d,%d,%d,%td,%td,%td,%td,%.4f,%d,%.9e,%.4f,%.4f,%.4f,%.4f,%d,%td,%td,%.12e",
                 r.tag, host, r.operation, r.layout, r.kernel, r.geom, r.warp, r.threads, r.pack_size, r.cube_n,
                 r.nodes, r.elements, r.dofs, r.bsr_nnz, r.bsr_values_mib, r.repeat, r.seconds_per_call,
                 r.mdofs, r.mdofs_element_visits, r.melems, r.gflops_model,
                 r.n_colors, r.packs_per_color_min, r.packs_per_color_max, r.checksum);
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
                    "  --bsr-apply    assemble once, then time BSR SpMV y = J(u) v\n",
                    argv[0]);
            if (own_mpi) MPI_Finalize();
            return 0;
        }
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
    if (geom != "affine" && geom != "isoparam") {
        std::fprintf(stderr, "invalid --geom '%s' (expected affine or isoparam)\n", geom.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    const GeomKind geom_kind = parse_geom(geom);
    if (geom_kind == GeomKind::Isoparam && kernel_uses_sympy_residual(kernel_kind)) {
        std::fprintf(stderr, "--geom isoparam is incompatible with sympy kernels\n");
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
            if (geom_kind == GeomKind::Isoparam || verify) (void)thread_scratch<scalar_t>(3, packed_xyz_n(packed));
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

    auto apply_fn = [&]() {
        if (geom_kind == GeomKind::Isoparam) {
            if (layout == "colored")
                apply_residual_colored(d, packed, colors, rho, mu, kernel_kind, GeomKind::Isoparam);
            else if (layout == "packed" || layout == "store")
                apply_residual_packed(d, packed, rho, mu, kernel_kind, GeomKind::Isoparam);
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
            else
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
    auto jac_action_fn = [&]() {
        if (layout == "colored")
            apply_jacobian_action_colored(d, packed, colors, rho, mu, jac_dir.data(), jac_out.data(), geom_kind);
        else if (layout == "packed" || layout == "store")
            apply_jacobian_action_packed(d, packed, rho, mu, jac_dir.data(), jac_out.data(), geom_kind);
        else if (geom_kind == GeomKind::Isoparam)
            apply_jacobian_action_atomic_isoparam(d, rho, mu, jac_dir.data(), jac_out.data());
        else
            apply_jacobian_action_atomic(d, rho, mu, jac_dir.data(), jac_out.data());
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
    const double t0 = wall_time();
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
    const double t1 = wall_time();

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
        row.phase                = g_breakdown ? g_phase : nullptr;
        csv_write(csv_path, row);
    }

    d.mesh.reset();
    if (own_mpi) MPI_Finalize();
    return 0;
}
