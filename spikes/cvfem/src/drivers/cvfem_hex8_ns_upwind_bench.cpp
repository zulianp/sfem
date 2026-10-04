// HEX8 CVFEM Navier-Stokes benchmark driver.
//
// The operator implementations live in the per-layout headers below; this file
// holds the mesh/option setup, the verification harness and the timing loop.

#include <cstdint>
#include <cstring>

#include "best/cvfem_hex8_best_common.hpp"
#include "kernels/standard/cvfem_hex8_best_atomic.hpp"
#include "best/cvfem_hex8_best_colored.hpp"
#include "frontend/staging/cvfem_hex8_ecolored_launch.hpp"
#include "frontend/staging/cvfem_hex8_packed_launch.hpp"
#include "frontend/staging/cvfem_hex8_store_launch.hpp"

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

// One dispatch point for the reconstruction, so the four call sites cannot drift about
// which sweep they used. Over packs when there is a pack to sweep -- same operator, no
// atomics, deterministic -- and over the flat element table otherwise, or when
// --qgrad-atomic asks for it as a measurement escape hatch.
static int g_qgrad_atomic = 0;
// Force the PACKED nodal-gradient sweep on a layout that would otherwise use the atomic one.
// Exists only to reproduce measurements taken before the sweep followed the layout; see the
// note where these are resolved.
static int g_qgrad_packed = 0;

static void bench_nodal_grad(MeshData &d, PackedData &p, const GeomKind geom_kind,
                             const scalar_t *const SFEM_RESTRICT src, const int stride,
                             std::vector<scalar_t> &ox, std::vector<scalar_t> &oy, std::vector<scalar_t> &oz) {
    const int iso = geom_kind == GeomKind::Isoparam ? 1 : 0;
    // Held across calls so a repeated sweep does not allocate, as the solver's own buffer is.
    static std::vector<scalar_t> gbuf;
    if (p.n_packs > 0 && !g_qgrad_atomic)
        cvfem_hex8_assemble_nodal_grads_packed(d, p, iso, src, stride, ox, oy, oz, gbuf);
    else
        cvfem_hex8_assemble_nodal_grads_atomic(d, iso, src, stride, ox, oy, oz);
}

// THE EXACT HIGHER-ORDER JACOBIAN ACTION, AGAINST A FINITE DIFFERENCE OF THE RESIDUAL.
//
// The only check that can tell the exact action from the lagged one, and it turns on a detail
// that is easy to get wrong in the checker rather than in the kernel: the nodal velocity gradient
// must be REBUILT at each perturbed state. Freezing it would difference the operator the lagged
// action computes, both arms would agree, and the check would pass while testing nothing -- the
// same shape of vacuous gate this driver has already shipped once, when a packed sweep with no
// pack built compared two zeros.
//
// Central difference, so the error is O(eps^2) and the comparison is not dominated by the
// truncation of a one-sided one. Returns max|fd - jv| / max|fd|, and prints the lagged action
// against the same reference beside it, because the number that matters to a solver is not
// whether the exact action is right but how wrong the lagged one is.
// The geometry is a template parameter of the packed kernels now rather than a GeomKind they
// test per pack, so the runtime choice is resolved here. One dispatcher rather than a branch at
// each call site: --geom is a user-level option and this is the front end.
template <class... Args>
static void apply_jacobian_action_packed_geom(const GeomKind g, Args &&...args) {
    if (g == GeomKind::Isoparam)
        apply_jacobian_action_packed<true>(std::forward<Args>(args)...);
    else
        apply_jacobian_action_packed<false>(std::forward<Args>(args)...);
}

static scalar_t verify_ho_action_fd(MeshData &d, PackedData &p, const scalar_t rho, const scalar_t mu,
                                    const int limiter, const GeomKind geom_kind, scalar_t &lagged_rel,
                                    // The fraction of degrees of freedom where the two disagree by
                                    // more than the tolerance. A max norm cannot verify a
                                    // piecewise-differentiable operator: the limiters switch branch
                                    // somewhere inside the +/- eps step for a handful of surfaces,
                                    // and at such a surface the difference quotient is not a
                                    // derivative of anything. What must hold is that those surfaces
                                    // are a vanishing fraction and that everywhere else the
                                    // agreement is the unlimited arm's.
                                    scalar_t &frac_bad, scalar_t &lagged_frac_bad) {
    const ptrdiff_t n   = d.nnodes;
    const int       iso = geom_kind == GeomKind::Isoparam ? 1 : 0;
    // THE STATE IS PERTURBED FIRST, and this is not cosmetic. On a uniform hex mesh with a
    // smooth field the reconstruction lands EXACTLY on a limiter bound for a structural sixth of
    // the sub-control surfaces -- measured, 1024 of 6144 at n=8 -- and at such a surface the
    // limiter has a kink at the evaluation point. A central difference across a kink returns the
    // average of the two one-sided derivatives, which equals neither, so the reference is invalid
    // there however right the derivative is. Each node touches many surfaces, so a sixth of the
    // surfaces contaminates three quarters of the nodes, which is exactly the disagreement this
    // check reported before the ties were broken. A small deterministic perturbation moves the
    // state off the degenerate set; it changes nothing about the operator being verified.
    for (ptrdiff_t i = 0; i < n; ++i) {
        const uint32_t h = (uint32_t)i * 2246822519u + 374761393u;
        d.ux[i] += scalar_t(1e-3) * (scalar_t((h >> 5) & 0xffffu) / scalar_t(65535) - scalar_t(0.5));
        d.uy[i] += scalar_t(1e-3) * (scalar_t((h >> 11) & 0xffffu) / scalar_t(65535) - scalar_t(0.5));
        d.uz[i] += scalar_t(1e-3) * (scalar_t((h >> 19) & 0xffffu) / scalar_t(65535) - scalar_t(0.5));
    }
    std::vector<scalar_t> ux0(d.ux), uy0(d.uy), uz0(d.uz);
    std::vector<scalar_t> dir((size_t)n * N_FIELDS, scalar_t(0));
    // A node-varying direction in the velocity components, since the correction is a function of
    // the velocity and its gradient; a uniform one would be annihilated by the reconstruction.
    for (ptrdiff_t i = 0; i < n; ++i) {
        const uint32_t h = (uint32_t)i * 2654435761u;
        dir[(size_t)i * N_FIELDS + 0] = scalar_t(1) + scalar_t((h >> 8) & 0xffffu) / scalar_t(65535);
        dir[(size_t)i * N_FIELDS + 1] = scalar_t(0.5) + scalar_t((h >> 16) & 0xffffu) / scalar_t(65535);
        dir[(size_t)i * N_FIELDS + 2] = scalar_t(-0.25) + scalar_t((h >> 3) & 0xffffu) / scalar_t(65535);
    }

    std::vector<scalar_t> g, gbuf, rp((size_t)n * N_FIELDS), rm((size_t)n * N_FIELDS);
    auto residual_at = [&](const scalar_t sgn, std::vector<scalar_t> &out, const scalar_t eps) {
        for (ptrdiff_t i = 0; i < n; ++i) {
            d.ux[i] = ux0[(size_t)i] + sgn * eps * dir[(size_t)i * N_FIELDS + 0];
            d.uy[i] = uy0[(size_t)i] + sgn * eps * dir[(size_t)i * N_FIELDS + 1];
            d.uz[i] = uz0[(size_t)i] + sgn * eps * dir[(size_t)i * N_FIELDS + 2];
        }
        const scalar_t *srcs[3] = {d.ux.data(), d.uy.data(), d.uz.data()};
        cvfem_hex8_assemble_nodal_grads_packed(d, p, iso, srcs, 3, g, gbuf);
        apply_residual_packed_defcor(d, p, rho, mu, g.data(), limiter, scalar_t(0));
        // Interleaved into the same node-major order the action writes, so the two are compared
        // entry by entry rather than through two different layouts.
        for (ptrdiff_t i = 0; i < n; ++i) {
            out[(size_t)i * N_FIELDS + 0] = d.rx[(size_t)i];
            out[(size_t)i * N_FIELDS + 1] = d.ry[(size_t)i];
            out[(size_t)i * N_FIELDS + 2] = d.rz[(size_t)i];
            out[(size_t)i * N_FIELDS + 3] = d.rc[(size_t)i];
        }
    };

    // Overridable so the check can be swept: a derivative error is eps-independent while a
    // branch-crossing artefact shrinks with the step, and that is what tells them apart.
    const char *eps_env = std::getenv("CVFEM_FD_EPS");
    const scalar_t eps = eps_env ? (scalar_t)std::atof(eps_env) : scalar_t(1e-6);
    residual_at(scalar_t(+1), rp, eps);
    residual_at(scalar_t(-1), rm, eps);
    d.ux = ux0; d.uy = uy0; d.uz = uz0;

    std::vector<scalar_t> fd((size_t)n * N_FIELDS);
    for (size_t i = 0; i < fd.size(); ++i) fd[i] = (rp[i] - rm[i]) / (scalar_t(2) * eps);

    // The state's gradient at u0, and the direction's, which is the field the lagged action drops.
    const scalar_t *srcs[3] = {d.ux.data(), d.uy.data(), d.uz.data()};
    std::vector<scalar_t> gu, gv, vbuf;
    cvfem_hex8_assemble_nodal_grads_packed(d, p, iso, srcs, 3, gu, gbuf);
    const scalar_t *vsrcs[3] = {dir.data() + 0, dir.data() + 1, dir.data() + 2};
    const int       vst[3]   = {N_FIELDS, N_FIELDS, N_FIELDS};
    cvfem_hex8_assemble_nodal_grads_packed(d, p, iso, vsrcs, 3, gv, vbuf, vst);

    std::vector<scalar_t> jv((size_t)n * N_FIELDS), jl((size_t)n * N_FIELDS);
    apply_jacobian_action_packed_geom(geom_kind, d, p, rho, mu, dir.data(), jv.data(),
                                 gu.data(), gv.data(), limiter, scalar_t(0));
    apply_jacobian_action_packed_geom(geom_kind, d, p, rho, mu, dir.data(), jl.data());

    scalar_t den = 0, num = 0, numl = 0;
    for (size_t i = 0; i < fd.size(); ++i) {
        den  = std::max(den, std::fabs(fd[i]));
        num  = std::max(num, std::fabs(fd[i] - jv[i]));
        numl = std::max(numl, std::fabs(fd[i] - jl[i]));
    }
    // Per field, because a wrong component index and a wrong derivative look the same in a
    // total and quite different here.
    {
        scalar_t pf[N_FIELDS] = {0, 0, 0, 0};
        for (size_t i = 0; i < fd.size(); ++i)
            pf[i % N_FIELDS] = std::max(pf[i % N_FIELDS], std::fabs(fd[i] - jv[i]));
        std::printf("  per-field max |fd-jv|: x %.3e  y %.3e  z %.3e  c %.3e  (max|fd| %.3e)\n",
                    (double)pf[0], (double)pf[1], (double)pf[2], (double)pf[3], (double)den);
    }
    const scalar_t tol = scalar_t(1e-6) * den;
    ptrdiff_t bad = 0, badl = 0;
    for (size_t i = 0; i < fd.size(); ++i) {
        if (std::fabs(fd[i] - jv[i]) > tol) ++bad;
        if (std::fabs(fd[i] - jl[i]) > tol) ++badl;
    }
    frac_bad        = scalar_t(bad) / scalar_t(fd.size());
    lagged_frac_bad = scalar_t(badl) / scalar_t(fd.size());
    lagged_rel = den > 0 ? numl / den : numl;
    return den > 0 ? num / den : num;
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
    // Pressure only, because with Rhie-Chow off the residual is linear in the pressure and
    // the upwind switch cannot move -- so the central difference below is exact rather than
    // second-order, and no sign flip can masquerade as an error.
    //
    // Not a UNIFORM pressure, which is what this used to be. A closed control volume has
    // sum(A) = 0, so a constant pressure produces no net force and the difference collapses
    // to round-off: with --boundary on, max_fd fell to 7e-12 and the relative measure --
    // round-off over round-off -- read exactly 1.0 for every kernel. The check was not
    // failing, it had stopped testing anything. A node-varying direction exercises the same
    // columns and is annihilated by nothing.
    std::fill(dir.begin(), dir.end(), scalar_t(0));
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const uint32_t h            = (uint32_t)i * 2654435761u;
        dir[(size_t)i * 4 + 3]      = scalar_t(1) + scalar_t((h >> 8) & 0xffffu) / scalar_t(65535);
    }

    const scalar_t eps = scalar_t(1.0e-6);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        d.ux[i] = x0[(size_t)i * 4 + 0] - eps * dir[(size_t)i * 4 + 0];
        d.uy[i] = x0[(size_t)i * 4 + 1] - eps * dir[(size_t)i * 4 + 1];
        d.uz[i] = x0[(size_t)i * 4 + 2] - eps * dir[(size_t)i * 4 + 2];
        d.p[i]  = x0[(size_t)i * 4 + 3] - eps * dir[(size_t)i * 4 + 3];
    }
    if (geom_kind == GeomKind::Isoparam)
        apply_residual_atomic_isoparam(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.rc.data(), d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu);
    else
        apply_residual_atomic(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.rc.data(), d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu);
    // The assembled matrix carries the boundary closure whenever --boundary is on, so the
    // residual differenced here has to as well -- otherwise the check reports the closure
    // as the error. Adding it also makes this a finite-difference check of
    // boundary_scs_add_jacobian over a whole mesh, which until now existed only as a
    // single-element unit test.
    apply_boundary_scs_residual_pass(d, rho, mu, geom_kind == GeomKind::Isoparam ? 1 : 0);
    apply_transient_pass(d, rho);
    pack_residual(d, rm);

    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        d.ux[i] = x0[(size_t)i * 4 + 0] + eps * dir[(size_t)i * 4 + 0];
        d.uy[i] = x0[(size_t)i * 4 + 1] + eps * dir[(size_t)i * 4 + 1];
        d.uz[i] = x0[(size_t)i * 4 + 2] + eps * dir[(size_t)i * 4 + 2];
        d.p[i]  = x0[(size_t)i * 4 + 3] + eps * dir[(size_t)i * 4 + 3];
    }
    if (geom_kind == GeomKind::Isoparam)
        apply_residual_atomic_isoparam(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.rc.data(), d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu);
    else
        apply_residual_atomic(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.rc.data(), d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu);
    apply_boundary_scs_residual_pass(d, rho, mu, geom_kind == GeomKind::Isoparam ? 1 : 0);
    apply_transient_pass(d, rho);
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
// so a sweep can just append and the analysis script (python/cvfem_kernel_report.py)
// reads whatever accumulated.
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
    // "f64" | "f32", or "n/a" where no matrix was built. The storage type is not
    // recoverable from bsr_values_MiB alone, and the report keys its rows on the code path
    // that ran -- two SpMV runs differing only in storage must not collapse onto one key.
    const char *bsr_storage;
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
    // ---- what actually ran, as opposed to what was asked for ------------------------
    //
    // The three fields above record the REQUEST. That is how a row comes to claim a term
    // the code path dropped: the flag was passed, so the column says 1. These say what the
    // dispatch actually reached, and they are what a report must read.
    // The layout whose code actually ran. --layout store has no residual or Jacobian-action
    // sweep of its own: both branches read `layout == "packed" || layout == "store"` and call
    // apply_residual_packed / apply_jacobian_action_packed. Recording the request made the two
    // look like separate implementations that happened to agree to 0.4%, when they are one
    // call site measured twice -- the same misattribution the ran_* columns exist to stop.
    const char *ran_layout;
    const char *ran_kernel;    // the kernel that executed, or "n/a" where the operation ignores --kernel
    const char *ran_rc;        // "exact" | "frozen" | "off"
    const char *ran_boundary;  // "on" | "off"
    int         exact_rc;      // the direction's reconstructed gradient was staged
    int         pgrad_per_apply;
    int         live_vectors;
    double      upwind_eps;    // always 0 in this driver; recorded so the column is not silently absent
    int         transient;     // 1 when --transient made the operator unsteady
    // Wall seconds per call spent in the direction-gradient reconstruction, and its share of
    // the matvec. Measured whenever the exact Rhie-Chow Jacobian runs, --breakdown or not,
    // because it is the largest pass in that operator and a CSV that omits it invites the
    // reader to attribute the whole matvec to the element kernel. -1 where it does not apply.
    double      qgrad_seconds;
    double      qgrad_frac;
    // Bytes per degree of freedom of stored element tangent, 0 when there is none. A faster
    // matvec that caps the problem size per node is the trade full assembly already loses,
    // so the price belongs beside the rate.
    double      pa_bytes_per_dof;
    const double *phase;  // PH_N entries, thread-summed ms per call, or nullptr
};

static void csv_write(const std::string &path, const CsvRow &r) {
    if (path.empty()) return;

    // The column set changes when options are added, and this file is opened for APPEND.
    // Writing a new row shape under an old header produces a csv whose columns silently
    // shift partway down -- and python/cvfem_kernel_report.py reads these by
    // name. So the existing header is compared against the one we would write, and a
    // mismatch is refused rather than appended to.
    std::string header =
            "tag,host,element,operation,layout,ran_layout,kernel,geom,warp,threads,pack_size,cube_n,"
            "nodes,elements,dofs,bsr_nnz,bsr_values_MiB,bsr_storage,repeat,seconds_per_call,"
            "MDOF_s,MDOF_s_element_visits,MELEM_s,GFLOP_s_model,"
            "n_colors,packs_per_color_min,packs_per_color_max,checksum,"
            "rhie_chow,rhie_chow_scale,boundary,"
            "ran_kernel,ran_rc,ran_boundary,exact_rc,pgrad_per_apply,live_vectors,"
            "upwind_eps,transient,s_qgrad,frac_qgrad,pa_bytes_per_dof";
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
                 "%s,%s,hex8,%s,%s,%s,%s,%s,%.6e,%d,%d,%d,%td,%td,%td,%td,%.4f,%s,%d,%.9e,%.4f,%.4f,%.4f,%.4f,%d,%td,%td,%.12e",
                 r.tag, host, r.operation, r.layout, r.ran_layout, r.kernel, r.geom, r.warp, r.threads, r.pack_size, r.cube_n,
                 r.nodes, r.elements, r.dofs, r.bsr_nnz, r.bsr_values_mib, r.bsr_storage, r.repeat, r.seconds_per_call,
                 r.mdofs, r.mdofs_element_visits, r.melems, r.gflops_model,
                 r.n_colors, r.packs_per_color_min, r.packs_per_color_max, r.checksum);
    std::fprintf(f, ",%d,%.6f,%d", r.rhie_chow, r.rhie_chow_scale, r.boundary);
    std::fprintf(f, ",%s,%s,%s,%d,%d,%d,%.6g,%d", r.ran_kernel, r.ran_rc, r.ran_boundary,
                 r.exact_rc, r.pgrad_per_apply, r.live_vectors, r.upwind_eps, r.transient);
    if (r.qgrad_seconds >= 0)
        std::fprintf(f, ",%.9e,%.6f", r.qgrad_seconds, r.qgrad_frac);
    else
        std::fprintf(f, ",,");
    std::fprintf(f, ",%.3f", r.pa_bytes_per_dof);
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
    // --bsr-precision single: hold the assembled values as float while still accumulating in
    // double. h_bsr_spmv separates storage from compute, so this halves the matrix traffic the
    // apply is bound by without changing the arithmetic it performs.
    int         bsr_single = 0;
    int         verify     = 0;
    int         verify_jac = 0;
    // The higher-order oracles on their own. They live inside the --verify block, but that
    // block also exercises a colored sympy residual that is quarantined in subpar/ and aborts
    // by design in a default build, so --verify cannot be a ctest gate. This runs the two
    // higher-order equivalence checks and stops before anything else.
    int         verify_ho  = 0;
    int         use_sfc    = 1;
    scalar_t    rho        = 1.0;
    scalar_t    mu         = 0.01;
    std::string layout     = "atomic";
    std::string kernel     = "sumfact";
    std::string geom       = "affine";
    std::string csv_path;
    std::string csv_tag    = "run";
    scalar_t    warp       = 0;
    // 0 means derive it from the machine once the mesh exists; see cvfem_default_pack_size.
    int         pack_size  = 0;
    int         assemble_diag = 0;
    // Both off by default. The benchmark's job is to isolate the element kernel, and the
    // recorded throughput baselines were measured without either term; turning one on
    // changes what is being measured, so it has to be asked for.
    int         rhie_chow  = 0;
    scalar_t    rc_scale   = 1;
    int         boundary   = 0;
    // Deferred-correction higher-order convective flux, and which limiter arm. Residual only,
    // and only on the atomic sum-factored sweep; see the --conv-ho help text.
    int         conv_ho      = 0;
    int         conv_limiter = 0;
    // The higher-order correction in the JACOBIAN ACTION. Exact by default when --conv-ho is on,
    // matching --rhie-chow, whose exact form is also the default: both differentiate through a
    // reconstructed nodal gradient and both pay for it with a pass over the direction. --lagged-ho
    // drops the correction's derivative, which is the operator an assembled first-order matrix
    // holds and what this driver computed before the exact form existed.
    int         lagged_ho    = 0;
    int         verify_ho_jac = 0;
    // The nodal gradient reconstruction as an operation in its own right, at nf fields: 1 is the
    // pressure gradient every Rhie-Chow residual builds, 3 the velocity gradient the exact
    // higher-order Jacobian action rebuilds on every matvec. It has only ever been timed from
    // inside those operators -- as frac_jac_action_qgrad -- so neither perf nor the campaign
    // could see it, and it is now the whole of the exact action's cost.
    int         nodal_grad   = 0;
    // Lag the Rhie-Chow pressure-gradient sensitivity in the Jacobian action, matching what
    // the assembled matrix encodes. See the --lagged-rc help text.
    int         lagged_rc    = 0;
    // Select the SIMD-kernel packed higher-order sweep instead of the scalar one. Both compute
    // the same residual, agreeing to 3e-18.
    //
    // THE SCALAR ONE IS THE DEFAULT BECAUSE IT IS FASTER, which is the opposite of what
    // routing the correction through the 16-wide kernel was meant to achieve. Measured on
    // Grace at 8,586,756 dof, unlimited arm: 646.7 MDOF/s scalar against 494.3 SIMD, a 1.31x
    // loss; the laptop said 1.43x in the same direction. The reason is structural rather than
    // incidental -- the correction needs each lane's element coordinates and its eight nodal
    // gradients, 96 scalars, gathered INSIDE the lane loop, and that gather costs more than
    // vectorising the flux around it returns. Making it pay would need a lane-parallel
    // reconstruction rather than a per-lane gather, which means the reconstruction existing in
    // a second, vector form; that is a design decision about duplicating it, not a tweak.
    int         ho_simd      = 0;
    // Force the hand-written SCALAR higher-order kernel. It was the default until the generated
    // lane-blocked kernel measured 910 against its 657 MDOF/s on Grace (job 4814365); it remains
    // selectable because it is the reference the other two are checked against, and because it is
    // the only one of the three that carries every limiter arm.
    int         ho_scalar    = 0;
    // Report what each representation of the mesh costs in memory, then exit. Its own mode
    // rather than a column of the timing CSV: it needs the packed mesh built and nothing else
    // run, and the numbers it prints are sizes, not rates.
    int         mesh_footprint = 0;
    // Whether the nodal pressure gradient is rebuilt inside every apply or hoisted out of
    // the timed loop. This is the difference SFEM_PGRAD_CACHE makes in the solver, and it
    // is the other half of the cascade: the gradient is a full element sweep.
    int         pgrad_per_apply = 0;
    int         live_vectors    = 0;
    // Steady by default -- dt <= 0 means no transient term, which is what every recorded
    // baseline was measured with.
    scalar_t    dt         = 0;
    int         bdf_order  = 1;
    // Partial assembly: build the element tangent once and let the matvec read it.
    int         partial_assembly = 0;

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
        else if (arg == "--bsr-precision" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v == "single") bsr_single = 1;
            else if (v == "double") bsr_single = 0;
            else { std::fprintf(stderr, "--bsr-precision takes double or single\n"); return 1; }
        }
        else if (arg == "--verify")
            verify = 1;
        else if (arg == "--verify-ho")
            verify_ho = 1;
        else if (arg == "--verify-jac")
            verify_jac = 1;
        else if (arg == "--no-sfc")
            use_sfc = 0;
        else if (arg == "--breakdown")
            g_breakdown = 1;
        else if (arg == "--mesh-footprint")
            mesh_footprint = 1;
        else if (arg == "--kernel-only")
            g_kernel_only = 1;
        else if (arg == "--dense-flush")
            g_dense_flush = 1;
        else if (arg == "--rhie-chow") {
            rhie_chow = 1;
            // Optional scale, so --rhie-chow 0.5 works and a bare --rhie-chow means 1.
            if (i + 1 < argc && argv[i + 1][0] != '-') rc_scale = (scalar_t)std::atof(argv[++i]);
        } else if (arg == "--nodal-grad") {
            nodal_grad = 3;
            if (i + 1 < argc && argv[i + 1][0] != '-') nodal_grad = std::atoi(argv[++i]);
            if (nodal_grad != 1 && nodal_grad != 3) {
                std::fprintf(stderr, "--nodal-grad takes 1 (pressure) or 3 (velocity)\n");
                return 2;
            }
        } else if (arg == "--verify-jac-ho") {
            verify_ho_jac = 1;
        } else if (arg == "--lagged-ho") {
            lagged_ho = 1;
        } else if (arg == "--conv-ho") {
            conv_ho = 1;
            // Optional limiter id, so --conv-ho 2 selects Venkatakrishnan and a bare
            // --conv-ho means unlimited, which is the arm the solver defaults to.
            if (i + 1 < argc && argv[i + 1][0] != '-') conv_limiter = std::atoi(argv[++i]);
        } else if (arg == "--ho-simd")
            ho_simd = 1;
        else if (arg == "--ho-scalar")
            ho_scalar = 1;
        else if (arg == "--lagged-rc")
            lagged_rc = 1;
        else if (arg == "--boundary")
            boundary = 1;
        else if (arg == "--pgrad-per-apply")
            pgrad_per_apply = 1;
        else if (arg == "--qgrad-atomic")
            g_qgrad_atomic = 1;
        else if (arg == "--qgrad-packed")
            g_qgrad_packed = 1;
        else if (arg == "--partial-assembly")
            partial_assembly = 1;
        else if (arg == "--live-vectors" && i + 1 < argc)
            live_vectors = std::atoi(argv[++i]);
        else if (arg == "--transient" && i + 1 < argc)
            dt = (scalar_t)std::atof(argv[++i]);
        else if (arg == "--bdf" && i + 1 < argc)
            bdf_order = std::atoi(argv[++i]);
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
                    "          [--nodal-grad 1|3]\n"
                    "          [--breakdown] [--kernel-only] [--dense-flush]\n"
                    "          [--csv FILE] [--tag NAME]\n"
                    "  --layout NAME  layout used by residual / jac-action / assemble (default atomic)\n"
                    "                 atomic  : flat element sweep, #pragma omp atomic per entry\n"
                    "                           (cvfem_hex8_best_atomic.hpp)\n"
                    "                 packed  : pack-local buffer folded back into the global one,\n"
                    "                           ghost rows reduced after (cvfem_hex8_best_packed.hpp)\n"
                    "                 colored : colored pack sweep, no reduction and no atomics\n"
                    "                           (cvfem_hex8_best_colored.hpp). Best for\n"
                    "                           --assemble; for residual and --jac-action the\n"
                    "                           color barriers cost more than the ghost reduce\n"
                    "                           they replace, so prefer packed there\n"
                    "                 ecolor  : ELEMENT colouring -- the flat sweep with the atomics\n"
                    "                           removed. Elements are coloured so no two of a\n"
                    "                           colour share a node and renumbered into colour\n"
                    "                           order, so each colour is a contiguous range and\n"
                    "                           writes the global arrays with a plain +=\n"
                    "                           (cvfem_hex8_best_ecolored.hpp). This is the\n"
                    "                           colouring the literature means; `colored` above\n"
                    "                           colours PACKS. Residual and --jac-action only\n"
                    "                 store   : packed assembly whose owned rows carry the global\n"
                    "                           pattern and are flushed with one streaming memcpy;\n"
                    "                           every block written once, no zeroing pass\n"
                    "                           (cvfem_hex8_best_store.hpp; residual/jac-action\n"
                    "                           fall back to packed)\n"
                    "  --breakdown    per-phase timing of the assembly (thread-summed ms/call)\n"
                    "  --ho-scalar    packed --conv-ho through the hand-written SCALAR kernel. The\n"
                    "                 default for the unlimited no-Rhie-Chow arm is now the\n"
                    "                 GENERATED lane-blocked kernel, which measures 1.39x it; this\n"
                    "                 forces the reference, and is implied by any limiter or by\n"
                    "                 --rhie-chow, which the generated kernel does not carry.\n"
                    "  --verify-ho    the two higher-order equivalence oracles alone, then exit:\n"
                    "                 the layouts against each other and the SIMD kernel against\n"
                    "                 the scalar one. Needs --conv-ho; this is the ctest gate.\n"
                    "  --mesh-footprint  print the bytes each representation of the mesh occupies,\n"
                    "                 array by array, and exit. Every figure is the allocation's\n"
                    "                 own nbytes(), so nothing here is a hand computation.\n"
                    "  --kernel-only  element kernel writes to a dense stack buffer (no scatter);\n"
                    "                 measures the arithmetic floor of the element kernel\n"
                    "  --dense-flush  sumfact only: stage ke densely, then flush 64 contiguous\n"
                    "                 blocks (measured slower than direct scatter on Apple M1)\n"
                    "  --csv FILE     append one machine-readable row per run (header written if\n"
                    "                 the file is new); pairs with python/cvfem_kernel_report.py\n"
                    "  --tag NAME     free-form label carried into the csv (e.g. the machine)\n"
                    "  --kernel NAME  residual/Jacobian micro-kernel variant (default sumfact)\n"
                    "  --geom NAME    affine (constant J) or isoparam (12 SCS trilinear J)\n"
                    "  --warp EPS     x += EPS * sin(pi y) nodal perturbation\n"
                    "  --bsr-apply    assemble once, then time BSR SpMV y = J(u) v\n"
                    "  --bsr-precision double|single   SpMV value storage (default double);\n"
                    "                 single stores float and still accumulates in double\n"
                    "  --conv-ho [L]    deferred-correction higher-order convective flux, limiter L\n"
                    "                   (0 unlimited, 1 bounded-face clip, 2 Venkatakrishnan,\n"
                    "                   3 Darwish-Moukalled; default 0). Residual only. Runs on\n"
                    "                   --layout atomic and --layout packed, both through the\n"
                    "                   SCALAR sum-factored kernel, which is the only one that\n"
                    "                   accepts ugrad8: so packed-vs-atomic here isolates the\n"
                    "                   layout with the kernel held fixed, and a higher-order\n"
                    "                   packed number is NOT comparable with a first-order one,\n"
                    "                   which is 16-wide SIMD. The nodal velocity gradient is\n"
                    "                   hoisted, as the solver lags the correction one Newton\n"
                    "                   step.\n"
                    "  --ho-simd        packed --conv-ho through the SIMD kernel instead of the\n"
                    "                   scalar one. Same residual to 3e-18, and SLOWER by 1.31x\n"
                    "                   on Grace -- the per-lane gather the correction needs\n"
                    "                   costs more than vectorising the flux returns.\n"
                    "  --lagged-rc      lag the Rhie-Chow pressure-gradient sensitivity in the\n"
                    "                   Jacobian action, giving the FROZEN Jacobian that the\n"
                    "                   assembled matrix encodes. The exact action differentiates\n"
                    "                   through the nodal gradient reconstruction and pays a\n"
                    "                   second pass for the direction's gradient; the assembled\n"
                    "                   matrix cannot carry that term without widening its\n"
                    "                   one-ring sparsity, so it lags it. Use this for a\n"
                    "                   like-for-like comparison against --bsr-apply.\n"
                    "  --rhie-chow [S]  include the Rhie-Chow pressure-velocity coupling at scale\n"
                    "                 S (default 1), and the nodal pressure gradient it needs.\n"
                    "                 sumfact only -- the hand-written and generated kernels carry\n"
                    "                 no Rhie-Chow term. Off by default: without it there is no\n"
                    "                 pressure-pressure coupling at all, which is a smaller and\n"
                    "                 faster operator than the one the solver runs.\n"
                    "  --partial-assembly  build the element tangent once -- five scalars per\n"
                    "                 sub-control surface, sixty per element -- and read it in the\n"
                    "                 Jacobian action instead of gathering the state and rederiving\n"
                    "                 the mass flux and upwind switch on every matvec. MEASURED AND\n"
                    "                 LOST: 17-19%% slower than direct evaluation (subpar/README.md),\n"
                    "                 so it needs -DCVFEM_ENABLE_SUBPAR=ON to run at all.\n"
                    "  --qgrad-atomic   reconstruct the nodal gradient with the flat atomic sweep\n"
                    "                 instead of over packs. Same operator; the packed one has no\n"
                    "                 atomics and is deterministic. Here to measure the difference,\n"
                    "                 not as a mode to run in.\n"
                    "  --transient DT   make the operator unsteady with timestep DT: the BDF mass\n"
                    "                 term rho V a0 / dt on the velocity diagonal, applied as a\n"
                    "                 per-node post-pass because in a control-volume scheme the mass\n"
                    "                 matrix IS the control volume. Reaches the residual, the\n"
                    "                 Jacobian action, the assembled matrix and the block diagonal,\n"
                    "                 whatever the layout and kernel. Off by default.\n"
                    "  --bdf ORDER    1 or 2 (default 1). Order 2 needs two history levels; the\n"
                    "                 driver fills both, so it does not fall back the way a first\n"
                    "                 timestep in the solver does.\n"
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

    // The float storage belongs to the SpMV, not to the matrix. Assembly always writes
    // scalar_t whatever --bsr-precision says, so on an assembly run the flag would relabel
    // a double matrix as single and halve a footprint that did not change. Clamping it here
    // -- rather than qualifying it at each of the four sites downstream -- keeps one meaning
    // for the variable: "the SpMV reads float values".
    bsr_single &= bsr_apply;

    // --boundary used to be refused for an assembly on anything but --layout atomic,
    // because the boundary blocks had to be written through the element BSR slots and only
    // the atomic assembly did that. assemble_boundary_scs_jacobian_pass scatters through
    // b.element_slots, the global map every layout shares, so the closure is now
    // layout-independent for the assembly exactly as it already was for the residual and
    // the Jacobian action. (--assemble-diag is atomic-only for a different reason and is
    // refused for any other layout below.)
    // The layout no longer decides. Every one of them carries Rhie-Chow for every operation
    // it supports: the residual and the Jacobian action through the per-pack staging that
    // only `packed` used to have, the assembly through the same Hex8RhieChow the atomic
    // sweep builds -- those assembly sweeps are scalar per element, so the only difference
    // is whether the coordinates and the nodal gradient are read from the pack or from the
    // mesh. What decides is the KERNEL, and rc_kernel_ok below is where that is settled.
    // Measured and lost: 17-19% SLOWER than direct evaluation at 4.1M and 8.6M dof on
    // Grace, in two independent implementations. Rejected by name here for the reason
    // --kernel sympy_row and sympy_face are -- a run must not report a throughput under a
    // configuration that was shown not to be worth using -- and kept compiling so the
    // measurement stays repeatable if the hardware changes. See subpar/README.md.
#ifndef CVFEM_ENABLE_SUBPAR
    if (partial_assembly) {
        std::fprintf(stderr,
                     "--partial-assembly was measured and lost: 17-19%% slower than direct\n"
                     "evaluation at 4.1M and 8.6M dof on Grace (see subpar/README.md).\n"
                     "Rebuild with -DCVFEM_ENABLE_SUBPAR=ON to measure it again.\n");
        if (own_mpi) MPI_Finalize();
        return 1;
    }
#endif
    // The store holds a per-element tangent built from one adjugate at the element centre,
    // which is not what the isoparametric kernels evaluate, and it is read by a packed SIMD
    // sweep. Refused elsewhere rather than measured on an operator it does not describe.
    if (partial_assembly && !(jac_action && (layout == "packed" || layout == "store") && geom == "affine")) {
        std::fprintf(stderr,
                     "--partial-assembly needs --jac-action on --layout packed|store with "
                     "--geom affine (got %s/%s/%s)\n",
                     jac_action ? "--jac-action" : "another operation", layout.c_str(), geom.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    // The tangent is a function of the state AND of the state's reconstructed pressure
    // gradient. Rebuilding that gradient inside every apply is the one case where the
    // gradient is not a per-Newton-step quantity, so a tangent built once would be stale
    // from the second apply onward -- a wrong operator, not a slow one.
    if (partial_assembly && pgrad_per_apply) {
        std::fprintf(stderr,
                     "--partial-assembly and --pgrad-per-apply are incompatible: the tangent is "
                     "built from the reconstructed gradient, which --pgrad-per-apply changes on "
                     "every apply\n");
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    if (pgrad_per_apply && !rhie_chow) {
        std::fprintf(stderr, "--pgrad-per-apply is meaningless without --rhie-chow\n");
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    // THE ONE MIXED FORM THAT IS THE JACOBIAN OF NO RESIDUAL.
    //
    // The two sensitivities are switchable independently, which is right: three of the four
    // combinations are operators somebody applies. Both exact is the exact Jacobian of the
    // higher-order residual; both lagged is deferred correction, the standard treatment, whose
    // implicit operator is the first-order one; and the Rhie-Chow term exact with the correction
    // lagged IS deferred correction as published, since the correction is meant to stay an
    // explicit source.
    //
    // The fourth is not. Lagging the pressure-gradient sensitivity while differentiating the
    // correction is exact in one reconstruction and frozen in the other, which no residual has
    // as its Jacobian, so a rate measured for it prices an operator with no use. Refused rather
    // than ignored, because the two flags read as independent and nothing else would say so.
    //
    // Note what this does NOT refuse: --conv-ho with no --rhie-chow at all. There the coupling is
    // absent rather than lagged, so the exact correction is the exact Jacobian of the residual
    // being solved -- which is what jobs/ho_exact_jac.sbatch verifies against a finite
    // difference, and what a prescribed-velocity transport problem such as Smith-Hutton wants.
    if (jac_action && rhie_chow && lagged_rc && conv_ho && !lagged_ho) {
        std::fprintf(stderr,
                     "--lagged-rc with an exact higher-order correction is the Jacobian of no "
                     "residual: it lags the pressure-gradient sensitivity and differentiates the "
                     "velocity-gradient one. Add --lagged-ho for the deferred-correction action, "
                     "or drop --lagged-rc for the exact one\n");
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
    if ((assemble ? 1 : 0) + (jac_action ? 1 : 0) + (bsr_apply ? 1 : 0) + (assemble_diag ? 1 : 0) +
                (nodal_grad ? 1 : 0) >
        1) {
        std::fprintf(stderr,
                     "specify at most one of --assemble, --assemble-diag, --jac-action, --bsr-apply\n");
        if (own_mpi) MPI_Finalize();
        return 1;
    }

    const KernelKind kernel_kind = parse_kernel(kernel);
    // ------------------------------------------------- what this build cannot honour
    //
    // Each of these was ACCEPTED before, ran different code than it named, and wrote a row
    // claiming the configuration it was asked for. That is the failure this driver already
    // rejects by name elsewhere ("a run cannot report a throughput under a kernel name that
    // did not execute", above) -- these are the combinations that slipped through the same
    // net on a different axis.

    // The generated Jacobian-action arrangements run only for that operation, and only on
    // the atomic layout: they are scalar kernels, so the packed and colored sweeps -- which
    // are SIMD over a pack -- have nothing to call them from. Refused elsewhere rather than
    // mapped onto whatever would have run.
    if (kernel_is_action_only(kernel_kind) && !(jac_action && layout == "atomic")) {
        std::fprintf(stderr,
                     "--kernel %s is a Jacobian-action arrangement: it needs --jac-action "
                     "--layout atomic (got %s/%s)\n",
                     kernel.c_str(), jac_action ? "--jac-action" : "another operation", layout.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    // They carry no Rhie-Chow term, so a row would claim one it did not compute.
    if (kernel_is_action_only(kernel_kind) && rhie_chow) {
        std::fprintf(stderr, "--kernel %s carries no Rhie-Chow term\n", kernel.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }

    // `split` is assembly-only by construction and `fd` is a Jacobian reference with no
    // residual form; both fall through to the hand-written `current` residual and would be
    // recorded under their own name.
    const bool residual_op = !(assemble || assemble_diag || jac_action || bsr_apply || nodal_grad);
    if (residual_op && (kernel_kind == KernelKind::Split || kernel_kind == KernelKind::Fd)) {
        std::fprintf(stderr,
                     "--kernel %s has no residual form; it would run and report the "
                     "hand-written `current` kernel\n",
                     kernel.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }

    // The verification chains compare against `current` and against a finite difference of
    // it, and neither carries Rhie-Chow. With the term on they report a mismatch that is
    // the missing term in the reference, not a defect in what is being verified.
    //
    // With one exception, and it is the one that matters here. The block diagonal and the
    // split both work by handing the FULL element kernel a modified slot array, so their
    // check is against this driver's own assembly rather than against a Rhie-Chow-free
    // reference -- both sides carry whatever the run asked for. That check is the only
    // thing that can show the new staging is right, so it must be allowed to run with the
    // term on. The chains that cannot are skipped rather than refused; see below.
    // Each verification block now decides for itself which of its comparisons Rhie-Chow
    // invalidates and skips those, rather than the whole run being refused: the checks that
    // go against a Rhie-Chow-free reference are skipped, and the ones that go across this
    // driver's own implementations -- packed against atomic against colored, the block
    // diagonal against the full assembly, linear-plus-nonlinear against the whole -- run
    // with the term on, which is what makes them the oracle for the staging work.

    // Read only inside the sumfact branch of the colored and store assemblies.
    if (g_dense_flush && !(assemble && kernel_kind == KernelKind::Sumfact &&
                           (layout == "colored" || layout == "store"))) {
        std::fprintf(stderr,
                     "--dense-flush is read only by --assemble --kernel sumfact on "
                     "--layout colored|store\n");
        if (own_mpi) MPI_Finalize();
        return 1;
    }

    // The block diagonal dispatches on geometry alone -- see diag_fn -- so --layout would
    // be recorded in the row while atomic code ran, together with pack_size and the colour
    // counts of a layout that was never used.
    if (assemble_diag && layout != "atomic") {
        std::fprintf(stderr,
                     "--assemble-diag ignores --layout (it is always the atomic diagonal); "
                     "pass --layout atomic or drop it (got '%s')\n",
                     layout.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    // The assembled boundary closure used to exist in exactly one function --
    // assemble_jacobian_atomic_sumfact -- so every other assembly kernel, and every
    // isoparametric one, dropped it while the row recorded boundary=1. It is now
    // assemble_boundary_scs_jacobian_pass, one sweep over the compacted boundary shell
    // that runs after whichever assembly the layout and kernel chose, so there is nothing
    // left to refuse: the term is carried for every kernel, layout and geometry.
    //
    // ---- which configurations carry the Rhie-Chow term ------------------------------
    //
    // One gate, because the answer depends on the kernel, the geometry, the layout AND the
    // operation together. The three scattered rules this replaces got it wrong in both
    // directions: "--rhie-chow requires --kernel sumfact" refused the isoparametric
    // hand-written kernels, which do take a Hex8RhieChow and are what --geom isoparam runs
    // on the atomic layout; and nothing at all stopped `--rhie-chow --assemble --kernel
    // sympy`, which ran a generated arrangement with no such term and wrote rhie_chow=1
    // and ran_rc=frozen into the CSV.
    //
    // What carries it, read off the element kernels rather than off the flags:
    //
    //   affine    sumfact   residual / action / assembly    add_slots, *_sumfact_simd
    //   affine    split     assembly, nonlinear half        add_slots_nonlinear
    //   isoparam  current   residual / action / assembly    the SCALAR isoparam kernels
    //   either    n/a       block diagonal                  add_slots{,_isoparam}
    //
    // What does not: every generated kernel, because the term was never put into the SymPy
    // expressions; the finite-difference reference, because it differences a residual that
    // takes no pressure gradient; the hand-written affine `current` residual; and the
    // isoparametric SIMD kernels -- which is what confines the isoparametric case to
    // --layout atomic.
    const bool iso_scalar_kernel = kernel_kind == KernelKind::Current || kernel_kind == KernelKind::Split;
    const bool rc_kernel_ok =
            // These two do not consult --kernel: they run the hand-written scalar kernels,
            // which take the term on both geometries.
            assemble_diag || (jac_action && (geom == "affine" || layout == "atomic")) ||
            (geom == "affine" && (kernel_kind == KernelKind::Sumfact || kernel_kind == KernelKind::Split)) ||
            // Isoparametric geometry splits by OPERATION, not by layout. The residual and
            // the action on a pack-based layout run the isoparametric SIMD kernels, which
            // carry no term -- hence --layout atomic there. Assembly is scalar per element
            // on every layout and runs the isoparametric kernel that does carry it.
            (geom == "isoparam" && layout == "atomic" && iso_scalar_kernel) ||
            (geom == "isoparam" && (assemble || bsr_apply) && iso_scalar_kernel);
    if (rhie_chow && !rc_kernel_ok) {
        std::fprintf(stderr,
                     "--rhie-chow is carried by: --kernel sumfact|split on --geom affine, "
                     "--kernel current|split on --geom isoparam --layout atomic, and by "
                     "--jac-action and --assemble-diag, which run the hand-written kernels "
                     "whatever --kernel says (got '%s'/%s/%s)\n",
                     kernel.c_str(), geom.c_str(), layout.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    // sympy_row and sympy_face lost the saturated evaluation and were moved to subpar/.
    // Rejected by name here, which is what keeps the stubs in cvfem_hex8_best_common.hpp
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
    // The six generated Jacobian-action arrangements, all of them. Measured on Grace at
    // 8,586,756 dof they reach 0.37x to 0.57x of the hand-written atomic action and 0.16x to
    // 0.25x of the packed one (perf/campaign_generated_arms.csv).
    if (kernel_is_action_only(kernel_kind)) {
        std::fprintf(stderr,
                     "--kernel %s was moved to subpar/: every generated Jacobian-action "
                     "arrangement is between 0.37x and 0.57x of the hand-written one "
                     "(see subpar/README.md).\n"
                     "Rebuild with -DCVFEM_ENABLE_SUBPAR=ON to measure it again.\n",
                     kernel.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    // AFFINE ONLY. KernelKind::Sympy names two different generated functions and the geometry
    // picks between them: the affine `_sympy_residual`, which is quarantined, and
    // `_sympy_residual_isoparam`, which is the isoparametric scalar WINNER and stays. A
    // rejection that ignored --geom would retire a winner along with a loser, which is the
    // same conflation the split rule in the generator was just hardened against.
    // RESIDUAL ONLY, and affine only. KernelKind::Sympy selects a different generated
    // function for each operation: the retired affine `_sympy_residual` for the residual, and
    // `_jacobian_add_bsr_slots` for --assemble, which is the atomic-assembly WINNER and stays.
    // A rejection on the kernel name alone takes the winner down with the loser -- exactly the
    // conflation the generator's split rule was hardened against a few lines of work earlier,
    // reproduced here. cvfem_bench_staging is what caught it.
    const bool sympy_residual_run = kernel_kind == KernelKind::Sympy && geom == "affine" &&
                                    !assemble && !assemble_diag && !jac_action && !bsr_apply;
    if (sympy_residual_run) {
        std::fprintf(stderr,
                     "--kernel sympy --geom affine was moved to subpar/: it gives up 21.6%% "
                     "on the packed layout and ties sumfact on the atomic one, so it is "
                     "fastest nowhere (see subpar/README.md).\n"
                     "Its isoparametric form is NOT retired -- `--kernel sympy --geom "
                     "isoparam` still runs. Rebuild with -DCVFEM_ENABLE_SUBPAR=ON for the "
                     "affine one.\n");
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
    // ...with one exception since the action gained generated kernels: the three
    // `sympy_action*` arrangements ARE dispatched by --jac-action on the atomic layout.
    const bool kernel_is_consulted =
            !(jac_action || bsr_apply || assemble_diag) || kernel_is_action_only(kernel_kind);
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
    if (layout != "packed" && layout != "atomic" && layout != "colored" && layout != "ecolor" &&
        layout != "store") {
        std::fprintf(stderr, "invalid --layout '%s' (expected packed, atomic, colored, ecolor or store)\n",
                     layout.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    // The pack-based oracles cannot verify this layout, and the combination is refused rather
    // than silently producing a number. --verify-jac and --verify-ho compare against a PackedMesh
    // built from the pre-permutation element order, so what they would report is the renumbering,
    // not a defect: the atomic-vs-packed action came out at 2.4e-04 that way. --verify is the
    // gate for this layout; it compares against the atomic sweep on the same connectivity and
    // checks the cached geometry against the permuted connectivity directly.
    if (layout == "ecolor" && (verify_jac || verify_ho)) {
        std::fprintf(stderr,
                     "--layout ecolor renumbers the elements, so the pack-based oracles behind "
                     "--verify-jac / --verify-ho do not apply; use --verify\n");
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

    // THE NODAL GRADIENT SWEEP FOLLOWS THE LAYOUT UNDER TEST.
    //
    // It used to follow whether a pack EXISTED, and a pack is built whenever --jac-action is on
    // whatever --layout says -- so an atomic Jacobian-action row ran a PACKED gradient
    // reconstruction inside an otherwise atomic operator, and the layout comparison was measuring
    // the same pass on both sides. Measured on the exact higher-order action, the confusion is
    // worth 3%: an atomic arm reads 17.6 MDOF/s with the packed sweep and 18.2 with its own.
    // That matters most where the gradient pass is large, which is exactly the exact Rhie-Chow
    // and exact higher-order actions this benchmark exists to compare.
    // The element-coloured layout takes the atomic gradient sweep for the same reason, and for
    // one more: it renumbers the elements, so a pack built from the pre-permutation order would
    // be describing a different mesh. The atomic sweep walks whatever element order it is given.
    if ((layout == "atomic" || layout == "ecolor") && !g_qgrad_packed) g_qgrad_atomic = 1;

    // The pack size, if the caller did not name one. Derived here rather than at declaration
    // because the rule needs the element count and the thread count, and both are known only now.
    if (pack_size <= 0) pack_size = cvfem_default_pack_size(d.mesh->n_elements(0), threads_active());

    PackedData packed;
    // verify_ho belongs here for the same reason every other verify flag does: the higher-order
    // oracles compare PACKED sweeps, and a packed sweep with no pack built writes nothing at all
    // rather than failing, so the comparison would silently read whichever residual was left in
    // place and report perfect agreement. That is exactly what it did until this was added -- a
    // deliberately broken SIMD kernel still measured 0.0 -- and it is the same artefact
    // docs/kernel_prose/30_layout_margin.md records for a throughput row.
    // Never for the element-coloured layout. It uses no pack, and building one is not harmless:
    // make_packed renumbers the NODES, and the pack size it chooses follows the thread count, so
    // a --jac-action run came out with a different fingerprint at every thread count purely from
    // a decomposition it never reads. The layout's own decomposition -- the colouring -- depends
    // on neither.
    if (layout != "ecolor" &&
        (layout == "packed" || layout == "colored" || layout == "store" || verify || verify_jac ||
         jac_action || bsr_apply || mesh_footprint || verify_ho))
        packed = make_packed(d.mesh, pack_size);
    PackColoring colors;
    if (layout == "colored" || verify || verify_jac)
        colors = cvfem_build_pack_coloring(packed.n_packs, packed.owned_nodes_ptr, packed.ghost_ptr, packed.ghost_idx);

    d.nnodes    = d.mesh->n_nodes();
    d.nelements = d.mesh->n_elements(0);
    d.elems     = d.mesh->elements(0)->data();
    d.points    = d.mesh->points()->data();

    // ELEMENT colouring, and the renumbering that goes with it.
    //
    // This must happen HERE: after the connectivity is bound and before anything indexed by
    // element is derived from it -- the adjugate and determinant arrays, the Rhie-Chow surface
    // tables, the partially assembled tangent. Permuting the connectivity underneath those is
    // the producer/consumer mistake this spike has already made three times with the node
    // numbering, and the cube oracles cannot see it (tests/cvfem_warped_geometry.sh can).
    //
    // Renumbering rather than carrying an order array is what lets the sweep stay the atomic
    // sweep: a colour becomes a contiguous element range, so the geometry gather stays a memcpy
    // and the only differences left are the loop bounds and the plain += . The colouring and the
    // renumbering are smesh's (smesh::ElementColoring); the device path takes the same colouring
    // in its order-array form instead, since it cannot move its elements.
    ElementColoring ecolors;
    double          ecolor_build_s = 0.0;
    if (layout == "ecolor") {
        const double t_ec = wall_time();
        ecolors           = cvfem_build_element_coloring(d.mesh);
        ecolor_build_s = wall_time() - t_ec;
        // The nodal-gradient reconstruction follows the layout too. Leaving it atomic made the
        // Rhie-Chow arm the one nondeterministic part of an otherwise reproducible operator.
        cvfem_hex8_set_qgrad_ecolors(&ecolors);
        // NO HYBRID PIPELINE. Every pass of an operator application must belong to the layout
        // under test, and the one that can silently stop doing so is this one: the gradient
        // sweep picks its scatter from a pointer rather than from the dispatch, so a new call
        // site or a caller outside this file would fall back to atomics and produce an
        // element-coloured element sweep feeding on an atomically reduced gradient. Checked
        // here, where the pointer is set, rather than trusted.
        if (cvfem_hex8_qgrad_ecolors() != &ecolors || ecolors.n_colors <= 0) {
            std::fprintf(stderr, "element-coloured layout could not claim the nodal-gradient "
                                 "sweep; refusing to run a hybrid pipeline\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }
    }

    if (mesh_footprint) {
        // Every figure below is an allocation's own nbytes(). Nothing is
        // nelements * nodes_per_element * sizeof(index) worked out by hand, because the
        // arrays that are easy to get wrong that way are exactly the ones that decide the
        // answer: the ghost list and the reduction graph have no closed form -- their
        // lengths depend on how the packs happen to cut the mesh -- and the packed element
        // array is padded past n_packed_elements.
        //
        // Two totals are printed, with and without the coordinates, because the coordinates
        // are byte-for-byte the same in both representations. Including them is the honest
        // answer to "what does the mesh cost"; excluding them is the honest answer to "what
        // does the format change", and the two ratios are very different. Reporting only one
        // invites the reader to apply it to the other question.
        const auto      pm    = packed.packed;
        const long long dofs  = (long long)d.nnodes * N_FIELDS;
        const size_t    pts   = d.mesh->points()->nbytes();
        const size_t    st_el = d.mesh->elements(0)->nbytes();

        const size_t pk_el   = pm->elements(0)->nbytes();
        const size_t pk_own  = pm->owned_nodes_ptr(0)->nbytes();
        const size_t pk_shr  = pm->n_shared(0)->nbytes();
        const size_t pk_gptr = pm->ghost_ptr(0)->nbytes();
        const size_t pk_gidx = pm->ghost_idx(0)->nbytes();
        const size_t pk_rptr = pm->ghost_reduce_ptr(0) ? pm->ghost_reduce_ptr(0)->nbytes() : 0;
        const size_t pk_ridx = pm->ghost_reduce_idx(0) ? pm->ghost_reduce_idx(0)->nbytes() : 0;
        const size_t pk_rdst = pm->ghost_reduce_dest(0) ? pm->ghost_reduce_dest(0)->nbytes() : 0;
        const size_t pk_nmap = pm->node_map() ? pm->node_map()->nbytes() : 0;
        const size_t pk_top  = pk_el + pk_own + pk_shr + pk_gptr + pk_gidx + pk_rptr + pk_ridx +
                              pk_rdst + pk_nmap;

        std::printf("### mesh footprint  n_nodes=%td n_elements=%td dofs=%lld\n",
                    d.nnodes, d.nelements, dofs);
        std::printf("### n_packs=%td elements_per_pack=%td packed_elements=%td "
                    "ghost_entries=%td ghost_reduce_rows=%td\n",
                    packed.n_packs, packed.n_elements_per_pack, packed.n_packed_elements,
                    packed.n_ghost_entries, packed.n_ghost_reduce_rows);
        std::printf("\n%-28s %14s %12s\n", "array", "bytes", "B/dof");
#define CV_FP_ROW(name, b) \
    std::printf("%-28s %14zu %12.3f\n", name, (size_t)(b), (double)(b) / (double)dofs)
        CV_FP_ROW("standard.elements", st_el);
        CV_FP_ROW("standard.topology_total", st_el);
        CV_FP_ROW("packed.elements", pk_el);
        CV_FP_ROW("packed.owned_nodes_ptr", pk_own);
        CV_FP_ROW("packed.n_shared", pk_shr);
        CV_FP_ROW("packed.ghost_ptr", pk_gptr);
        CV_FP_ROW("packed.ghost_idx", pk_gidx);
        CV_FP_ROW("packed.ghost_reduce_ptr", pk_rptr);
        CV_FP_ROW("packed.ghost_reduce_idx", pk_ridx);
        CV_FP_ROW("packed.ghost_reduce_dest", pk_rdst);
        CV_FP_ROW("packed.node_map", pk_nmap);
        CV_FP_ROW("packed.topology_total", pk_top);
        CV_FP_ROW("shared.points", pts);
        CV_FP_ROW("standard.with_points", st_el + pts);
        CV_FP_ROW("packed.with_points", pk_top + pts);
#undef CV_FP_ROW
        std::printf("\npacked/standard topology_only   %.4f\n", (double)pk_top / (double)st_el);
        std::printf("packed/standard with_points     %.4f\n",
                    (double)(pk_top + pts) / (double)(st_el + pts));
        return 0;
    }

    if (warp != scalar_t(0)) {
        const scalar_t pi = std::acos(scalar_t(-1));
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            d.points[0][i] += smesh::geom_t(warp * std::sin(pi * scalar_t(d.points[1][i])));
        }
    }

    fill_fields(d);

    // The transient term. A history that is a small, node-varying displacement of the
    // state: the term is linear in it, so its VALUE cannot change any cost, but a history
    // equal to the state would make the BDF1 residual contribution identically zero and a
    // run could then not tell a term that was applied from one that was not.
    if (dt > scalar_t(0)) {
        d.dt        = dt;
        d.bdf_order = bdf_order;
        d.u_prev.resize((size_t)d.nnodes * 3);
        if (bdf_order >= 2) d.u_prev2.resize((size_t)d.nnodes * 3);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            const scalar_t s1 = scalar_t(0.97) + scalar_t(0.02) * (scalar_t)((i * 7) % 11) / scalar_t(11);
            const scalar_t s2 = scalar_t(0.94) + scalar_t(0.03) * (scalar_t)((i * 5) % 13) / scalar_t(13);
            d.u_prev[(size_t)i * 3 + 0] = s1 * d.ux[i];
            d.u_prev[(size_t)i * 3 + 1] = s1 * d.uy[i];
            d.u_prev[(size_t)i * 3 + 2] = s1 * d.uz[i];
            if (!d.u_prev2.empty()) {
                d.u_prev2[(size_t)i * 3 + 0] = s2 * d.ux[i];
                d.u_prev2[(size_t)i * 3 + 1] = s2 * d.uy[i];
                d.u_prev2[(size_t)i * 3 + 2] = s2 * d.uz[i];
            }
        }
        build_node_volume(d, d.node_vol);
        scalar_t vol = 0;
        for (const scalar_t v : d.node_vol) vol += v;
        std::printf("transient: dt %g, BDF%d, control volumes sum to %.12g\n", (double)dt,
                    bdf_coeffs(d).order, (double)vol);
    }
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
            gather_element_coords(d.elems, d.points, e, x, y, z);
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
        bench_nodal_grad(d, packed, geom_kind, d.p.data(), 1, d.pgx, d.pgy, d.pgz);
        std::printf("rhie_chow: scale %g, nodal gradient %s\n", (double)rc_scale,
                    pgrad_per_apply ? "rebuilt per apply" : "hoisted out of the timed loop");
    }

    // Built here, once, for the same reason the nodal gradient is: it is a function of the
    // Newton iterate, and a Krylov solve applies the operator hundreds of times at a state
    // that does not move. Outside the timed loop because that is where the solver would
    // pay it -- in update(), not in apply().
    if (partial_assembly) {
        cvfem_hex8_build_pa_tangent(d, rho, mu, scalar_t(0));
        std::printf("partial_assembly: %d scalars/element, %.1f bytes/dof (assembled matrix is ~847)\n",
                    CVFEM_HEX8_PA_PER_ELEM, cvfem_hex8_pa_bytes_per_dof(d));
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
            jac_linear.assign((size_t)(bsr.nnz * 16), scalar_t(0));
            if (geom_kind == GeomKind::Isoparam)
                assemble_jacobian_atomic_linear_isoparam(d.elems, d.nelements, d.p.data(), d.points, d.ux.data(), d.uy.data(), d.uz.data(), bsr.element_slots.empty() ? nullptr : bsr.element_slots.data(), mu, jac_linear.data());
            else
                assemble_jacobian_atomic_linear(d.adj_ptr, d.det_ptr, d.nelements, bsr.element_slots.empty() ? nullptr : bsr.element_slots.data(), mu, jac_linear.data());
        }
        if (layout == "store") build_pack_store_crs(packed, d.nelements, bsr.rowptr, bsr.colidx);
    }

    if (layout == "packed" || layout == "colored" || layout == "store" || verify || verify_jac || jac_action ||
        bsr_apply) {
        const size_t scratch_n = packed_scratch_n(packed.max_actual_nodes_per_pack);
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
            if (geom_kind == GeomKind::Isoparam || verify || verify_ho || rhie_chow)
                (void)thread_scratch<scalar_t>(3, rhie_chow ? packed_rc_n(packed.max_actual_nodes_per_pack) : packed_xyz_n(packed.max_actual_nodes_per_pack));
            if (rhie_chow && jac_action) (void)thread_scratch<scalar_t>(4, packed_qg_n(packed.max_actual_nodes_per_pack));
        }
    }

    // With Rhie-Chow on, the chain below cannot run: most of its references are kernels
    // that carry no such term, so every comparison would report the term as the error.
    // What can be checked -- and, since the colored sweep learned to stage Rhie-Chow, is
    // the only thing that says it stages it the way the packed sweep does -- is that the
    // three matrix-free residual sweeps agree with each other. All three run the same
    // sum-factorised kernel with the same per-pack staging; a difference here is a staging
    // bug and nothing else.
    if (rhie_chow && (verify || verify_jac) && layout != "ecolor") {
        apply_residual_atomic_sumfact(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.rc.data(), d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu);
        std::vector<scalar_t> atomic_r;
        pack_residual(d, atomic_r);
        apply_residual_packed<false>(d, packed, rho, mu, KernelKind::Sumfact);
        std::vector<scalar_t> packed_r;
        pack_residual(d, packed_r);
        apply_residual_colored(d, packed, colors, rho, mu, KernelKind::Sumfact, GeomKind::Affine);
        std::vector<scalar_t> colored_r;
        pack_residual(d, colored_r);
        const scalar_t packed_err  = max_abs_diff(atomic_r.data(), packed_r.data(), (ptrdiff_t)atomic_r.size());
        const scalar_t colored_err = max_abs_diff(atomic_r.data(), colored_r.data(), (ptrdiff_t)atomic_r.size());
        std::printf("verify_rc_packed_residual_vs_atomic_abs: %.6e\n", packed_err);
        std::printf("verify_rc_colored_residual_vs_atomic_abs: %.6e\n", colored_err);
        if (packed_err > 1.0e-10 || colored_err > 1.0e-10) {
            std::fprintf(stderr, "HEX8 Rhie-Chow residual mismatch across layouts\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }
    }

    // The element-coloured sweep against the atomic one, on the same connectivity.
    //
    // This cannot join the bitwise checksum gate the other layouts share: each node takes its
    // contributions one per colour, which is a different summation order from the atomic sweep's,
    // so the last bits differ by construction. A per-node comparison is the right instrument, and
    // it is a strong one -- both sweeps run the same element kernel over the same element
    // numbering, so anything beyond round-off is the scatter.
    if (layout == "ecolor" && (verify || verify_jac)) {
        // FIRST, the invariant the element renumbering can break: the cached geometry must
        // describe the element the connectivity now names. Recompute the adjugate straight from
        // the permuted connectivity and the coordinates, and compare against the cached array.
        //
        // This check exists because the obvious one is blind. Comparing the element-coloured
        // sweep against the atomic sweep cannot see a permutation applied after the geometry was
        // cached: both sweeps read the same connectivity and the same cache, so they agree with
        // each other while both describe a mesh that does not exist. Verified by reintroducing
        // the mistake -- the sweeps still agreed to 1.7e-18, and only this check moved.
        scalar_t geom_err = 0;
        for (ptrdiff_t e = 0; e < d.nelements; ++e) {
            scalar_t x[8], y[8], z[8], adj[9], det;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const smesh::idx_t g = d.elems[a][e];
                x[a]                 = scalar_t(d.points[0][g]);
                y[a]                 = scalar_t(d.points[1][g]);
                z[a]                 = scalar_t(d.points[2][g]);
            }
            cvfem_hex8_affine_adj(x, y, z, adj, &det);
            for (int c = 0; c < 9; ++c)
                geom_err = std::max(geom_err, (scalar_t)std::fabs(adj[c] - d.jacobian_adjugate[c][(size_t)e]));
            geom_err = std::max(geom_err, (scalar_t)std::fabs(det - d.jacobian_determinant[(size_t)e]));
        }
        std::printf("verify_ecolor_geometry_consistent_abs: %.6e\n", geom_err);
        if (geom_err > 1.0e-12) {
            std::fprintf(stderr,
                         "HEX8 element-coloured geometry is stale: the element permutation ran "
                         "after the adjugate cache was built\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }

        apply_residual_atomic_sumfact_simd(d.adj_ptr, d.det_ptr, d.elems, d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.rc.data(), d.rhie_chow_scale, d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu);
        std::vector<scalar_t> atomic_r;
        pack_residual(d, atomic_r);
        apply_residual_ecolored(d, ecolors, rho, mu);
        std::vector<scalar_t> ecolor_r;
        pack_residual(d, ecolor_r);
        const scalar_t err = max_abs_diff(atomic_r.data(), ecolor_r.data(), (ptrdiff_t)atomic_r.size());
        std::printf("verify_ecolor_residual_vs_atomic_abs: %.6e\n", err);
        if (err > 1.0e-10) {
            std::fprintf(stderr, "HEX8 element-coloured residual mismatch against atomic\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }
    }

    // The higher-order kernels WITH Rhie-Chow. The block below is gated off when Rhie-Chow is on,
    // because most of its comparisons are between kernels that do not all carry the term -- but the
    // generated higher-order kernel does carry it, in its own variant, and shipping that unverified
    // is exactly what these oracles exist to prevent. So this one runs here instead.
    if (verify_ho && rhie_chow && conv_ho) {
        std::vector<scalar_t> ug;
        const scalar_t       *srcs[3] = {d.ux.data(), d.uy.data(), d.uz.data()};
        cvfem_hex8_assemble_nodal_grads_atomic(d, 0, srcs, 3, ug);

        apply_residual_packed_defcor_scalar(d, packed, rho, mu, ug.data(), conv_limiter, scalar_t(0));
        std::vector<scalar_t> rc_scalar_r;
        pack_residual(d, rc_scalar_r);

        apply_residual_packed_defcor(d, packed, rho, mu, ug.data(), conv_limiter, scalar_t(0),
                                     /*sympy=*/true);
        std::vector<scalar_t> rc_sympy_r;
        pack_residual(d, rc_sympy_r);

        const scalar_t rc_err = max_abs_diff(rc_scalar_r.data(), rc_sympy_r.data(),
                                             (ptrdiff_t)rc_scalar_r.size());
        std::printf("verify_packed_ho_rc_sympy_vs_scalar_abs: %.6e\n", rc_err);
        if (rc_err > 1.0e-10) {
            std::fprintf(stderr, "HEX8 generated higher-order Rhie-Chow kernel mismatch\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }
        if (own_mpi) MPI_Finalize();
        return 0;
    }

    if ((verify || verify_ho) && !rhie_chow) {
        apply_residual_atomic(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.rc.data(), d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu);
        std::vector<scalar_t> current_r;
        pack_residual(d, current_r);

        apply_residual_atomic_sumfact(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.rc.data(), d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu);
        std::vector<scalar_t> sumfact_r;
        pack_residual(d, sumfact_r);
        const scalar_t sumfact_err = max_abs_diff(current_r.data(), sumfact_r.data(), (ptrdiff_t)current_r.size());
        std::printf("verify_sumfact_residual_vs_current_abs: %.6e\n", sumfact_err);
        if (sumfact_err > 1.0e-10) {
            std::fprintf(stderr, "HEX8 sumfact residual mismatch\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }

        apply_residual_atomic_isoparam(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.rc.data(), d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu);
        std::vector<scalar_t> isoparam_r;
        pack_residual(d, isoparam_r);
        const scalar_t iso_err = max_abs_diff(current_r.data(), isoparam_r.data(), (ptrdiff_t)current_r.size());
        std::printf("verify_isoparam_residual_vs_affine_abs: %.6e\n", iso_err);

        // The generated isoparametric kernel against the hand-written one. Both compute
        // the same discretisation, so this must be at rounding level -- unlike the
        // comparison above, which is isoparametric against affine and is a property of
        // the mesh rather than of the code.
        apply_residual_atomic_isoparam_sympy(d.elems, d.nelements, d.nnodes, d.p.data(), d.points, d.rc.data(), d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(), rho, mu);
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

        if ((layout == "packed" || verify_jac) && layout != "ecolor") {
            apply_residual_packed<false>(d, packed, rho, mu, KernelKind::Current);
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

            apply_residual_packed<false>(d, packed, rho, mu, KernelKind::Sumfact);
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

            apply_residual_packed<true>(d, packed, rho, mu, KernelKind::Sumfact);
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

        // The two higher-order layouts must agree. Run in ONE process, so the pack has been
        // built once and both see the same node numbering -- a comparison across two processes
        // would be invalid, because packing renumbers the mesh in place and the arrays would
        // not be index-comparable. Checksums cannot stand in for this: the residual of this
        // state sums to roughly 1e-14, so the sum is all cancellation and two layouts differ
        // in it by round-off whether or not they agree per node.
        if (conv_ho) {
            std::vector<scalar_t> ug;
            const scalar_t       *srcs[3] = {d.ux.data(), d.uy.data(), d.uz.data()};
            cvfem_hex8_assemble_nodal_grads_atomic(d, 0, srcs, 3, ug);

            apply_residual_atomic_sumfact_defcor(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.rc.data(), d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu, ug.data(), conv_limiter, scalar_t(0));
            std::vector<scalar_t> ho_atomic_r;
            pack_residual(d, ho_atomic_r);

            apply_residual_packed_defcor_scalar(d, packed, rho, mu, ug.data(), conv_limiter, scalar_t(0));
            std::vector<scalar_t> ho_packed_r;
            pack_residual(d, ho_packed_r);

            // SAME LAYOUT, TWO KERNELS: isolates the kernel difference from everything the
            // packed sweep does around it.
            apply_residual_packed_defcor(d, packed, rho, mu, ug.data(), conv_limiter, scalar_t(0));
            std::vector<scalar_t> ho_packed_simd_r;
            pack_residual(d, ho_packed_simd_r);
            // A GATE, not a print. The two kernels agree to 1.3e-18 -- eight orders inside
            // this bound -- and the bound is what stops a vectorisation change shipping a
            // wrong kernel. It is deliberately 1e-10 rather than 1e-18: the two instantiations
            // may legitimately reassociate differently, so demanding the measured value would
            // fail on a reordering that is not a defect.
            const scalar_t ho_kernel_err = max_abs_diff(ho_packed_r.data(), ho_packed_simd_r.data(),
                                                        (ptrdiff_t)ho_packed_r.size());
            std::printf("verify_packed_ho_simd_vs_packed_ho_scalar_abs: %.6e\n", ho_kernel_err);
            if (ho_kernel_err > 1.0e-10) {
                std::fprintf(stderr, "HEX8 packed higher-order SIMD/scalar kernel mismatch\n");
                if (own_mpi) MPI_Finalize();
                return 1;
            }

            // THE GENERATED KERNEL, against the same scalar reference. Every limiter arm is
            // generated now, so this runs on all four; the checksum cannot stand in for it, as the
            // residual of this state sums to ~1e-14 and is almost all cancellation.
            {
                apply_residual_packed_defcor(d, packed, rho, mu, ug.data(), conv_limiter, scalar_t(0),
                                             /*sympy=*/true);
                std::vector<scalar_t> ho_sympy_r;
                pack_residual(d, ho_sympy_r);
                const scalar_t ho_sympy_err = max_abs_diff(ho_packed_r.data(), ho_sympy_r.data(),
                                                           (ptrdiff_t)ho_packed_r.size());
                std::printf("verify_packed_ho_sympy_vs_packed_ho_scalar_abs: %.6e\n", ho_sympy_err);
                if (ho_sympy_err > 1.0e-10) {
                    std::fprintf(stderr, "HEX8 generated higher-order kernel mismatch\n");
                    if (own_mpi) MPI_Finalize();
                    return 1;
                }
            }

            const scalar_t ho_err =
                    max_abs_diff(ho_atomic_r.data(), ho_packed_r.data(), (ptrdiff_t)ho_atomic_r.size());
            std::printf("verify_packed_ho_residual_vs_atomic_abs: %.6e\n", ho_err);
            if (ho_err > 1.0e-10) {
                std::fprintf(stderr, "HEX8 packed higher-order residual mismatch\n");
                if (own_mpi) MPI_Finalize();
                return 1;
            }
        }

        // --verify-ho stops here. What follows includes a colored sympy residual that lives in
        // subpar/ and aborts unless built with -DCVFEM_ENABLE_SUBPAR, so continuing would make
        // the higher-order gate fail for a reason that has nothing to do with it.
        if (verify_ho && !verify) {
            if (own_mpi) MPI_Finalize();
            return 0;
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
            apply_residual_packed<false>(d, packed, rho, mu, KernelKind::Sympy);
        else
            apply_residual_atomic_sympy(d.adj_ptr, d.det_ptr, d.elems, d.nelements, d.nnodes, d.p.data(), d.rc.data(), d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(), rho, mu);
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
    // Nodal velocity gradient for the deferred correction: nine interleaved components per
    // node. Hoisted, because the solver lags the correction one Newton step -- so what the
    // timed apply measures is the higher-order FLUX, not the reconstruction that feeds it,
    // exactly as the Rhie-Chow state gradient is hoisted by default. The same fused
    // multi-field sweep the solver uses, so this is not a second code path.
    std::vector<scalar_t> ugrad;
    if (conv_ho) {
        const scalar_t *srcs[3] = {d.ux.data(), d.uy.data(), d.uz.data()};
        const int       iso     = geom_kind == GeomKind::Isoparam ? 1 : 0;
        if (packed.n_packs > 0 && !g_qgrad_atomic) {
            std::vector<scalar_t> gbuf;
            cvfem_hex8_assemble_nodal_grads_packed(d, packed, iso, srcs, 3, ugrad, gbuf);
        } else {
            cvfem_hex8_assemble_nodal_grads_atomic(d, iso, srcs, 3, ugrad);
        }
    }

    if (verify_ho_jac) {
        if (packed.n_packs <= 0) {
            std::fprintf(stderr, "--verify-jac-ho needs a pack; use --layout packed\n");
            if (own_mpi) MPI_Finalize();
            return 2;
        }
        scalar_t lag = 0, fbad = 0, fbadl = 0;
        const scalar_t rel = verify_ho_action_fd(d, packed, rho, mu, conv_limiter, geom_kind, lag,
                                                 fbad, fbadl);
        std::printf("verify_jac_ho: limiter %d  exact_rel %.3e (%.4f%% of dofs disagree)  "
                    "lagged_rel %.3e (%.4f%%)\n",
                    conv_limiter, (double)rel, 100.0 * (double)fbad, (double)lag,
                    100.0 * (double)fbadl);
        // The lagged action differing from the finite difference is the POINT, not a failure, so
        // only the exact arm is gated. A lagged arm that agreed would mean the correction never
        // reached the residual and the whole comparison is vacuous, so it is gated from below.
        // Two standards, because the operators differ. The unlimited arm is differentiable
        // everywhere, so its max norm must hold. A limited arm is differentiable almost
        // everywhere, so what must hold is that the exceptions are a vanishing fraction of the
        // mesh -- the surfaces whose limiter branch the perturbation crosses -- and that the
        // lagged action is wrong on a far larger share.
        const scalar_t tol = scalar_t(1e-6);
        const bool ok = conv_limiter == 0 ? (rel < tol) : (fbad < scalar_t(0.01));
        if (!ok) {
            std::fprintf(stderr, "FAIL: exact higher-order action differs from the finite "
                                 "difference: max %.3e, %.4f%% of dofs (limiter %d)\n",
                         (double)rel, 100.0 * (double)fbad, conv_limiter);
            if (own_mpi) MPI_Finalize();
            return 1;
        }
        if (conv_limiter != 0 && !(fbadl > scalar_t(20) * fbad + scalar_t(0.05))) {
            std::fprintf(stderr, "FAIL: the lagged action disagrees on %.4f%% of dofs against the "
                                 "exact one's %.4f%% -- too close to call this a check\n",
                         100.0 * (double)fbadl, 100.0 * (double)fbad);
            if (own_mpi) MPI_Finalize();
            return 1;
        }
        if (!(lag > scalar_t(10) * rel)) {
            std::fprintf(stderr, "FAIL: the lagged action is as close to the finite difference as "
                                 "the exact one (%.3e vs %.3e) -- the correction did not reach "
                                 "the residual, so this comparison tested nothing\n",
                         (double)lag, (double)rel);
            if (own_mpi) MPI_Finalize();
            return 1;
        }
    }

    // here rather than inside each layout means --boundary works for all four.
    auto apply_fn = [&]() {
        if (pgrad_per_apply)
            bench_nodal_grad(d, packed, geom_kind, d.p.data(), 1, d.pgx, d.pgy, d.pgz);
        if (geom_kind == GeomKind::Isoparam) {
            if (layout == "colored")
                apply_residual_colored(d, packed, colors, rho, mu, kernel_kind, GeomKind::Isoparam);
            else if (layout == "packed" || layout == "store")
                apply_residual_packed<true>(d, packed, rho, mu, kernel_kind);
            else if (kernel_kind == KernelKind::Sympy)
                apply_residual_atomic_isoparam_sympy(d.elems, d.nelements, d.nnodes, d.p.data(), d.points, d.rc.data(), d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(), rho, mu);
            else
                apply_residual_atomic_isoparam(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.rc.data(), d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu);
        } else if (layout == "ecolor")
            apply_residual_ecolored(d, ecolors, rho, mu, conv_ho ? ugrad.data() : nullptr, conv_limiter,
                                    scalar_t(0));
        else if (layout == "colored")
            apply_residual_colored(d, packed, colors, rho, mu, kernel_kind, GeomKind::Affine);
        else if ((layout == "packed" || layout == "store") && conv_ho && ho_simd)
            apply_residual_packed_defcor(d, packed, rho, mu, ugrad.data(), conv_limiter, scalar_t(0));
        else if ((layout == "packed" || layout == "store") && conv_ho && !ho_scalar)
            // THE DEFAULT where it applies: the generated lane-blocked kernel, 1.39x the scalar one
            // it replaced. It covers the unlimited arm with and without Rhie-Chow -- two generated
            // variants per limiter arm, eight in all, dispatched inside the sweep. --ho-scalar
            // forces the hand-written reference, and is required for a non-zero Venkatakrishnan
            // eps^2, which the generated kernels do not carry and refuse rather than drop.
            apply_residual_packed_defcor(d, packed, rho, mu, ugrad.data(), conv_limiter, scalar_t(0),
                                         /*sympy=*/true);
        else if ((layout == "packed" || layout == "store") && conv_ho)
            apply_residual_packed_defcor_scalar(d, packed, rho, mu, ugrad.data(), conv_limiter, scalar_t(0));
        else if (layout == "packed" || layout == "store")
            apply_residual_packed<false>(d, packed, rho, mu, kernel_kind);
        else if (kernel_uses_sympy_residual(kernel_kind))
            apply_residual_atomic_sympy(d.adj_ptr, d.det_ptr, d.elems, d.nelements, d.nnodes, d.p.data(), d.rc.data(), d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(), rho, mu);
        else if (kernel_kind == KernelKind::Sumfact && conv_ho && !ho_scalar)
            apply_residual_atomic_sumfact_simd(d.adj_ptr, d.det_ptr, d.elems, d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.rc.data(), d.rhie_chow_scale, d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu, ugrad.data(), conv_limiter, scalar_t(0));
        else if (kernel_kind == KernelKind::Sumfact && conv_ho)
            apply_residual_atomic_sumfact_defcor(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.rc.data(), d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu, ugrad.data(), conv_limiter, scalar_t(0));
        else if (kernel_kind == KernelKind::Sumfact)
            // The lane-blocked sweep, unconditionally. The scalar one is not an option the
            // standard layout should be measured on: it issues 0.1% vector instructions where
            // this issues 23%, and reporting it would credit the packed format with a
            // vectorisation difference that has nothing to do with the format.
            apply_residual_atomic_sumfact_simd(d.adj_ptr, d.det_ptr, d.elems, d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.rc.data(), d.rhie_chow_scale, d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu);
        else
            apply_residual_atomic(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.rc.data(), d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu);

        if (boundary) apply_boundary_scs_residual_pass(d, rho, mu, geom_kind == GeomKind::Isoparam ? 1 : 0);
        apply_transient_pass(d, rho);
    };
    auto jac_fn = [&]() {
        if (geom_kind == GeomKind::Isoparam) {
            if (layout == "store")
                assemble_jacobian_store<true>(d, packed, bsr, rho, mu, kernel_kind);
            else if (layout == "colored")
                assemble_jacobian_colored(d, packed, colors, bsr, rho, mu, kernel_kind, GeomKind::Isoparam);
            else if (layout == "packed")
                assemble_jacobian_packed<true>(d, packed, bsr, rho, mu, kernel_kind);
            else if (kernel_kind == KernelKind::Split)
                assemble_jacobian_atomic_nonlinear_isoparam(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), bsr.element_slots.empty() ? nullptr : bsr.element_slots.data(), bsr.nnz, bsr.values->data(), rho, mu, jac_linear.data());
            else if (kernel_kind == KernelKind::Sympy)
                assemble_jacobian_atomic_isoparam_sympy(d.elems, d.nelements, d.p.data(), d.points, d.ux.data(), d.uy.data(), d.uz.data(), bsr.element_slots.empty() ? nullptr : bsr.element_slots.data(), bsr.nnz, bsr.values->data(), rho, mu);
            else if (kernel_kind == KernelKind::Fd)
                assemble_jacobian_atomic_fd_isoparam(d.elems, d.nelements, d.p.data(), d.points, d.ux.data(), d.uy.data(), d.uz.data(), bsr.colidx, bsr.element_slots.empty() ? nullptr : bsr.element_slots.data(), bsr.nnz, bsr.rowptr, bsr.values->data(), rho, mu);
            else
                // Current: the hand-written isoparametric kernel. Every other name is
                // rejected during validation, so this is not a fallback.
                assemble_jacobian_atomic_isoparam(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), bsr.element_slots.empty() ? nullptr : bsr.element_slots.data(), bsr.nnz, bsr.values->data(), rho, mu);
        } else if (layout == "store") {
            assemble_jacobian_store<false>(d, packed, bsr, rho, mu, kernel_kind);
        } else if (layout == "colored") {
            assemble_jacobian_colored(d, packed, colors, bsr, rho, mu, kernel_kind, GeomKind::Affine);
        } else if (layout == "packed") {
            assemble_jacobian_packed<false>(d, packed, bsr, rho, mu, kernel_kind);
        } else if (kernel_kind == KernelKind::Sumfact)
            assemble_jacobian_atomic_sumfact(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), bsr.element_slots.empty() ? nullptr : bsr.element_slots.data(), bsr.nnz, bsr.values->data(), rho, mu);
        else if (kernel_kind == KernelKind::Sympy)
            assemble_jacobian_atomic_sympy(d.adj_ptr, d.det_ptr, d.elems, d.nelements, d.p.data(), d.ux.data(), d.uy.data(), d.uz.data(), bsr.element_slots.empty() ? nullptr : bsr.element_slots.data(), bsr.nnz, bsr.values->data(), rho, mu);
        else if (kernel_kind == KernelKind::SympyBlock)
            assemble_jacobian_atomic_sympy_block(d.adj_ptr, d.det_ptr, d.elems, d.nelements, d.p.data(), d.ux.data(), d.uy.data(), d.uz.data(), bsr.element_slots.empty() ? nullptr : bsr.element_slots.data(), bsr.nnz, bsr.values->data(), rho, mu);
        else if (kernel_kind == KernelKind::SympyRow)
            assemble_jacobian_atomic_sympy_row(d.adj_ptr, d.det_ptr, d.elems, d.nelements, d.p.data(), d.ux.data(), d.uy.data(), d.uz.data(), bsr.element_slots.empty() ? nullptr : bsr.element_slots.data(), bsr.nnz, bsr.values->data(), rho, mu);
        else if (kernel_kind == KernelKind::SympyFace)
            assemble_jacobian_atomic_sympy_face(d.adj_ptr, d.det_ptr, d.elems, d.nelements, d.p.data(), d.ux.data(), d.uy.data(), d.uz.data(), bsr.element_slots.empty() ? nullptr : bsr.element_slots.data(), bsr.nnz, bsr.values->data(), rho, mu);
        else if (kernel_kind == KernelKind::Split)
            // Restore the geometry-only half built once at setup, then add only
            // the velocity-dependent half. The linear half is not rebuilt here:
            // that is the whole point of the split.
            assemble_jacobian_atomic_nonlinear(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), bsr.element_slots.empty() ? nullptr : bsr.element_slots.data(), bsr.nnz, bsr.values->data(), rho, mu, jac_linear.data());
        else
            // Current and Fd both land here. There is no dedicated `current`
            // assembly kernel -- the loop residual kernel has no assembled
            // counterpart -- so `--kernel current --assemble` measures the
            // finite-difference kernel. Kept as the fallback rather than
            // rejected, because fd is also the correctness reference, but the
            // two rows are the same kernel and should not be read as distinct.
            assemble_jacobian_atomic_fd(d.adj_ptr, d.det_ptr, d.elems, d.nelements, d.p.data(), d.ux.data(), d.uy.data(), d.uz.data(), bsr.colidx, bsr.element_slots.empty() ? nullptr : bsr.element_slots.data(), bsr.nnz, bsr.rowptr, bsr.values->data(), rho, mu);

        if (boundary)
            assemble_boundary_scs_jacobian_pass(d, bsr, rho, mu, geom_kind == GeomKind::Isoparam ? 1 : 0);
        assemble_transient_diag_pass(d, rho, bsr);
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
    // --lagged-rc: the FROZEN Rhie-Chow Jacobian, which is the operator the assembled matrix
    // encodes. The exact action differentiates through the nodal pressure-gradient
    // reconstruction, so it needs the direction's gradient and pays a second pass for it. The
    // assembled Jacobian cannot carry that term at all -- including it would widen the
    // pressure coupling past nearest neighbours and break the one-ring sparsity the matrix is
    // built on -- so it lags it.
    //
    // Comparing the exact matrix-free action against an SpMV of that matrix therefore compares
    // two different operators, and in the direction that flatters the matrix: the matrix-free
    // side is doing strictly more. This flag makes the like-for-like comparison available by
    // clearing the direction gradient, which is exactly what Hex8Extras keys `with_qg` on.
    if (lagged_rc) { d.qgx.clear(); d.qgy.clear(); d.qgz.clear(); }
    const bool with_qgrad = rhie_chow && jac_action && !lagged_rc;
    // The exact higher-order action. Packed only for now: the atomic sweep runs a different,
    // scalar element kernel and carrying the term there is its own change, so the driver refuses
    // the combination rather than reporting an atomic row that silently ran the lagged operator.
    const bool with_hograd = conv_ho && jac_action && !lagged_ho;
    std::vector<scalar_t> vgrad, vgbuf;
    double                hograd_seconds = 0;
    auto jac_action_fn = [&]() {
        // A Krylov iteration never sees the same direction twice, so neither does this when
        // the live set exists: the rotation is what stops the direction being resident.
        const scalar_t *const dir_v = live.empty() ? jac_dir.data() : live[(size_t)(live_k++ % (long)live.size())].data();
        last_dir                    = dir_v;
        // BOTH DIRECTION GRADIENTS IN ONE SWEEP when both are wanted. They read different fields
        // of the SAME array at the same stride -- the direction's three velocity components and
        // its pressure -- so together they are one four-field sweep instead of a one-field and a
        // three-field one, and the half of a sweep that does not scale with the field count is
        // paid once. Each keeps the layout its consumer reads, which is what the per-component
        // output stride is for. The whole cost lands in the qgrad phase, since it is one pass.
        // THE EXACT ACTION RECONSTRUCTS THE STATE'S GRADIENTS TOO, ON EVERY APPLICATION.
        //
        // It used to rebuild only the direction's and read the state's from a cache filled once
        // at setup, which makes the operator exact in its sensitivities and frozen in the fields
        // those sensitivities are evaluated at -- and makes the measured rate exclude two full
        // element sweeps the operator needs. An exact application is self-contained: it
        // evaluates every reconstruction it reads. The frozen-state form is the LAGGED one, and
        // it keeps the cache.
        //
        // All of it is ONE sweep, which is the only reason this is affordable. The sweep takes a
        // base pointer and a stride per field, so the direction's four interleaved fields and
        // the state's four contiguous ones go in together: eight fields on the higher-order arm,
        // two on the first-order one. Outputs run three per field in field order, which is what
        // fixes the index arithmetic below.
        const int iso_ng = geom_kind == GeomKind::Isoparam ? 1 : 0;
        if (with_qgrad && with_hograd) {
            const double    t0      = wall_time();
            const scalar_t *srcs[8] = {dir_v + 0,     dir_v + 1,     dir_v + 2,     dir_v + 3,
                                       d.ux.data(),   d.uy.data(),   d.uz.data(),   d.p.data()};
            const int       st[8]   = {N_FIELDS, N_FIELDS, N_FIELDS, N_FIELDS, 1, 1, 1, 1};
            vgrad.resize((size_t)d.nnodes * 9);
            ugrad.resize((size_t)d.nnodes * 9);
            d.qgx.resize((size_t)d.nnodes);
            d.qgy.resize((size_t)d.nnodes);
            d.qgz.resize((size_t)d.nnodes);
            d.pgx.resize((size_t)d.nnodes);
            d.pgy.resize((size_t)d.nnodes);
            d.pgz.resize((size_t)d.nnodes);
            scalar_t *outp[24];
            ptrdiff_t ostr[24];
            for (int c = 0; c < 9; ++c) { outp[c] = vgrad.data() + c; ostr[c] = 9; }
            outp[9]  = d.qgx.data(); ostr[9]  = 1;
            outp[10] = d.qgy.data(); ostr[10] = 1;
            outp[11] = d.qgz.data(); ostr[11] = 1;
            for (int c = 0; c < 9; ++c) { outp[12 + c] = ugrad.data() + c; ostr[12 + c] = 9; }
            outp[21] = d.pgx.data(); ostr[21] = 1;
            outp[22] = d.pgy.data(); ostr[22] = 1;
            outp[23] = d.pgz.data(); ostr[23] = 1;
            if (packed.n_packs > 0 && !g_qgrad_atomic)
                cvfem_hex8_assemble_nodal_grads_packed(d, packed, iso_ng, srcs, st, 8, outp, ostr, vgbuf);
            else
                cvfem_hex8_assemble_nodal_grads_atomic(d, iso_ng, srcs, st, 8, outp, ostr);
            // Charged ONCE, to the higher-order account, because it is one pass. Adding it to
            // both would make the two fractions sum past the application they are fractions of.
            // What frac_jac_action_hograd then reports on a fused arm is the whole
            // reconstruction -- all four gradients -- which is the quantity section 7.8 wants.
            const double dt = wall_time() - t0;
            hograd_seconds += dt;
            if (g_breakdown) g_phase[PH_QGRAD] += dt;
        } else if (with_qgrad) {
            // First-order exact: the coupling's sensitivity needs the direction's pressure
            // gradient, and the coupling itself the state's. Two fields, one sweep.
            const double    t0      = wall_time();
            const scalar_t *srcs[2] = {dir_v + 3, d.p.data()};
            const int       st[2]   = {N_FIELDS, 1};
            d.qgx.resize((size_t)d.nnodes);
            d.qgy.resize((size_t)d.nnodes);
            d.qgz.resize((size_t)d.nnodes);
            d.pgx.resize((size_t)d.nnodes);
            d.pgy.resize((size_t)d.nnodes);
            d.pgz.resize((size_t)d.nnodes);
            scalar_t *outp[6] = {d.qgx.data(), d.qgy.data(), d.qgz.data(),
                                 d.pgx.data(), d.pgy.data(), d.pgz.data()};
            ptrdiff_t ostr[6] = {1, 1, 1, 1, 1, 1};
            if (packed.n_packs > 0 && !g_qgrad_atomic)
                cvfem_hex8_assemble_nodal_grads_packed(d, packed, iso_ng, srcs, st, 2, outp, ostr, vgbuf);
            else
                cvfem_hex8_assemble_nodal_grads_atomic(d, iso_ng, srcs, st, 2, outp, ostr);
            const double dt_qg = wall_time() - t0;
            qgrad_seconds += dt_qg;
            // Also into the phase table, so the pass that dominates this application appears
            // beside the phases of the sweep it precedes rather than only in a stdout line.
            if (g_breakdown) g_phase[PH_QGRAD] += dt_qg;
        }
        if (with_hograd && !with_qgrad) {
            // The direction's nodal velocity gradient, reconstructed from scratch on every
            // matvec because the direction changes every Krylov iteration. This is the pass the
            // lagged action does not run, and it is what the exact form costs.
            //
            // Read in place: the direction arrives interleaved by node, and the sweep takes a
            // base pointer and a stride per field, so its three velocity components are three
            // views of the same array. This used to de-interleave them into scratch first --
            // three full-length copies inside the timed region on every matvec.
            // The state's velocity gradient goes in with it, for the reason above: six fields,
            // one sweep. The Rhie--Chow term is lagged on this arm, so its state pressure
            // gradient stays cached -- that is what lagging it means.
            const double    t0       = wall_time();
            const scalar_t *vsrcs[6] = {dir_v + 0,   dir_v + 1,   dir_v + 2,
                                        d.ux.data(), d.uy.data(), d.uz.data()};
            const int       vst[6]   = {N_FIELDS, N_FIELDS, N_FIELDS, 1, 1, 1};
            vgrad.resize((size_t)d.nnodes * 9);
            ugrad.resize((size_t)d.nnodes * 9);
            scalar_t *outp[18];
            ptrdiff_t ostr[18];
            for (int c = 0; c < 9; ++c) { outp[c] = vgrad.data() + c; ostr[c] = 9; }
            for (int c = 0; c < 9; ++c) { outp[9 + c] = ugrad.data() + c; ostr[9 + c] = 9; }
            if (packed.n_packs > 0 && !g_qgrad_atomic)
                cvfem_hex8_assemble_nodal_grads_packed(d, packed, iso_ng, vsrcs, vst, 6, outp, ostr, vgbuf);
            else
                cvfem_hex8_assemble_nodal_grads_atomic(d, iso_ng, vsrcs, vst, 6, outp, ostr);
            hograd_seconds += wall_time() - t0;
        }
        if (partial_assembly)
            apply_jacobian_action_packed_pa(d, packed, rho, mu, dir_v, jac_out.data());
        else if (layout == "ecolor")
            apply_jacobian_action_ecolored(d, ecolors, rho, mu, dir_v, jac_out.data(),
                                           with_hograd ? ugrad.data() : nullptr,
                                           with_hograd ? vgrad.data() : nullptr,
                                           conv_limiter, scalar_t(0));
        else if (layout == "colored")
            apply_jacobian_action_colored(d, packed, colors, rho, mu, dir_v, jac_out.data(), geom_kind);
        else if (layout == "packed" || layout == "store")
            apply_jacobian_action_packed_geom(geom_kind, d, packed, rho, mu, dir_v, jac_out.data(),
                                         with_hograd ? ugrad.data() : nullptr,
                                         with_hograd ? vgrad.data() : nullptr,
                                         conv_limiter, scalar_t(0));
        else if (geom_kind == GeomKind::Isoparam)
            apply_jacobian_action_atomic_isoparam(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu, dir_v, jac_out.data());
        // The lane-blocked sweep wherever it applies, for the reason the residual gives: the
        // standard layout is measured at its best or the comparison credits the format with a
        // vectorisation difference. It carries the first-order flux, Rhie-Chow exact or frozen,
        // and the EXACT higher-order action -- the SIMD kernel takes ho and hov, which is what
        // the packed Jacobian passes it. Only the generated kernel arrangements are outside it,
        // and the recorded row says which one ran.
        else if (kernel_kind == KernelKind::Sumfact) {
            // Braced on purpose: the build below is a second statement in this branch, and an
            // unbraced one would attach the else that follows to the wrong call.
            if (cvfem_hex8_extras_of(d).with_rc) cvfem_hex8_build_rc_coeff(d, rho, mu);
            apply_jacobian_action_atomic_simd(d.adj_ptr, d.det_ptr, d.elems, d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.rc_coeff.data(), d.rc_w.data(), d.rhie_chow_scale, d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu, dir_v, jac_out.data(),
                                              with_hograd ? ugrad.data() : nullptr,
                                              with_hograd ? vgrad.data() : nullptr,
                                              conv_limiter, scalar_t(0));
        }
        else
            apply_jacobian_action_atomic(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu, dir_v, jac_out.data(), kernel_kind,
                                         with_hograd ? ugrad.data() : nullptr,
                                         with_hograd ? vgrad.data() : nullptr,
                                         conv_limiter, scalar_t(0));

        if (boundary)
            apply_boundary_scs_jacobian_action_pass(d, rho, mu, geom_kind == GeomKind::Isoparam ? 1 : 0, dir_v,
                                                    jac_out.data());
        apply_transient_action_pass(d, rho, dir_v, jac_out.data());
    };

    // The reconstruction on its own. The same call the operators make -- bench_nodal_grad for
    // one field, the fused sweep for three -- so this times the kernel a solver runs rather than
    // a copy of it, and the layout follows --layout like every other operation here.
    std::vector<scalar_t> ng_out, ng_buf;
    auto grad_fn = [&]() {
        const int iso = geom_kind == GeomKind::Isoparam ? 1 : 0;
        if (nodal_grad == 1) {
            bench_nodal_grad(d, packed, geom_kind, d.p.data(), 1, d.pgx, d.pgy, d.pgz);
        } else {
            const scalar_t *srcs[3] = {d.ux.data(), d.uy.data(), d.uz.data()};
            if (packed.n_packs > 0 && !g_qgrad_atomic)
                cvfem_hex8_assemble_nodal_grads_packed(d, packed, iso, srcs, 3, ng_out, ng_buf);
            else
                cvfem_hex8_assemble_nodal_grads_atomic(d, iso, srcs, 3, ng_out);
        }
    };

    // Block diagonal, for the block-Jacobi preconditioner. Assembles only the 4x4
    // diagonal blocks -- 16 doubles per node instead of the whole matrix.
    std::vector<scalar_t> diag_blocks;
    auto diag_fn = [&]() {
        diag_blocks.assign((size_t)(d.nnodes * 16), scalar_t(0));
        // Hoisted out of assemble_diag_boundary_scs_pass, which both branches below reach: it is
        // a once-per-solve cache on the mesh, not part of an element pass. Above the `if` rather
        // than inside each branch, so no branch gains a second unbraced statement.
        cvfem_hex8_build_face_mask_eff(d);
        if (geom_kind == GeomKind::Isoparam)
            assemble_diag_atomic_isoparam(d.Lx, d.Ly, d.Lz, d.adj_ptr, d.bnd_elems.data(), d.det_ptr, d.elems, d.face_mask.data(), d.face_mask_eff.data(), d.nelements, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), (ptrdiff_t)d.bnd_elems.size(), cvfem_hex8_extras_of(d), rho, mu, diag_blocks.data());
        else
            assemble_diag_atomic(d.Lx, d.Ly, d.Lz, d.adj_ptr, d.bnd_elems.data(), d.det_ptr, d.elems, d.face_mask.data(), d.face_mask_eff.data(), d.nelements, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), (ptrdiff_t)d.bnd_elems.size(), cvfem_hex8_extras_of(d), rho, mu, diag_blocks.data());
        assemble_diag_transient_pass(d, rho, diag_blocks);
    };

    if (bsr_apply) jac_fn();
    // Operator<scalar_t> rather than the concrete BSR type: BSR<R, C, float, double> and
    // BSR<R, C, double, double> are different types and both derive from it, so this is what can
    // hold either. A decltype of one call pins the storage type and makes the other unreachable.
    std::shared_ptr<sfem::Operator<scalar_t>> bsr_apply_op;
    smesh::SharedBuffer<float>                bsr_values_f32;
    if (bsr_apply) {
        if (bsr_single) {
            bsr_values_f32 = bsr4_values_f32(bsr);
            bsr_apply_op   = sfem::h_bsr_spmv<smesh::count_t, smesh::idx_t, float, scalar_t>(
                    d.nnodes, d.nnodes, 4, bsr.graph->rowptr(), bsr.graph->colidx(), bsr_values_f32, scalar_t(0));
        } else {
            bsr_apply_op = sfem::h_bsr_spmv<smesh::count_t, smesh::idx_t, scalar_t>(
                    d.nnodes, d.nnodes, 4, bsr.graph->rowptr(), bsr.graph->colidx(), bsr.values, scalar_t(0));
        }
    }
    auto bsr_apply_fn = [&]() { bsr_apply_op->apply(jac_dir.data(), jac_out.data()); };

    // Two comparisons here go against the ASSEMBLED matrix and one goes across the
    // matrix-free implementations, and Rhie-Chow separates them. The assembled Jacobian
    // carries the frozen form by design while the action carries the exact one, so with the
    // term on the first two would fail by construction and are skipped -- that gap is the
    // decision recorded in ran_rc, not a defect. The third is unaffected: packed, atomic
    // and colored are three spellings of the same matrix-free operator whatever terms are
    // on, and with Rhie-Chow on it is the ONLY check that says the colored sweep stages the
    // term the same way the packed one does.
    if (verify_jac) {
        const bool vs_matrix = !rhie_chow;
        if (vs_matrix) {
            jac_fn();
            const scalar_t rel = verify_jacobian_fd(d, bsr, rho, mu, geom_kind);
            std::printf("verify_jac_spmv_vs_fd_rel: %.6e\n", rel);
            if (rel > 1.0e-6) {
                std::fprintf(stderr, "HEX8 BSR Jacobian mismatch\n");
                if (own_mpi) MPI_Finalize();
                return 1;
            }
        }

        std::vector<scalar_t> jv_spmv((size_t)d.nnodes * N_FIELDS), jv_mf((size_t)d.nnodes * N_FIELDS),
                jv_mf_atomic((size_t)d.nnodes * N_FIELDS), jv_mf_colored((size_t)d.nnodes * N_FIELDS);
        // The exact form differentiates through the nodal gradient reconstruction, so the
        // direction's pressure needs the same reconstruction. Without this the three sweeps
        // would agree in the frozen form and the exact term would go unchecked.
        if (rhie_chow)
            bench_nodal_grad(d, packed, geom_kind, jac_dir.data() + 3, N_FIELDS, d.qgx, d.qgy, d.qgz);
        if (vs_matrix) bsr4_spmv(bsr, d.nnodes, jac_dir.data(), jv_spmv.data());
        apply_jacobian_action_packed_geom(geom_kind, d, packed, rho, mu, jac_dir.data(), jv_mf.data());
        if (geom_kind == GeomKind::Isoparam)
            apply_jacobian_action_atomic_isoparam(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu, jac_dir.data(), jv_mf_atomic.data());
        else
            apply_jacobian_action_atomic(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu, jac_dir.data(), jv_mf_atomic.data());
        if (colors.n_colors > 0)
            apply_jacobian_action_colored(
                    d, packed, colors, rho, mu, jac_dir.data(), jv_mf_colored.data(), geom_kind);
        // The assembled matrix carries the boundary closure, so the three matrix-free
        // actions compared against it have to be closed too -- this is the same pass
        // jac_action_fn runs, applied to each of them.
        if (boundary) {
            const int iso = geom_kind == GeomKind::Isoparam ? 1 : 0;
            apply_boundary_scs_jacobian_action_pass(d, rho, mu, iso, jac_dir.data(), jv_mf.data());
            apply_boundary_scs_jacobian_action_pass(d, rho, mu, iso, jac_dir.data(), jv_mf_atomic.data());
            if (colors.n_colors > 0)
                apply_boundary_scs_jacobian_action_pass(d, rho, mu, iso, jac_dir.data(), jv_mf_colored.data());
        }
        apply_transient_action_pass(d, rho, jac_dir.data(), jv_mf.data());
        apply_transient_action_pass(d, rho, jac_dir.data(), jv_mf_atomic.data());
        if (colors.n_colors > 0)
            apply_transient_action_pass(d, rho, jac_dir.data(), jv_mf_colored.data());
        // The atomic sweep runs the SCALAR isoparametric kernel, which takes a
        // Hex8RhieChow; the packed and colored sweeps run the SIMD one, which does not. So
        // with Rhie-Chow on and isoparametric geometry the two sides are honestly different
        // operators and this comparison would report 3.2e-2 -- the term itself. That is the
        // remaining staging gap in this family, and it is refused for a RUN
        // (--rhie-chow --geom isoparam needs --layout atomic); here it only means there is
        // nothing to compare the atomic action against. Packed against colored still holds,
        // since both lack the term equally.
        const bool     atomic_comparable = !(rhie_chow && geom_kind == GeomKind::Isoparam);
        const scalar_t mf_err      = vs_matrix ? max_abs_diff(jv_spmv.data(), jv_mf.data(), d.nnodes * N_FIELDS)
                                                : scalar_t(0);
        const scalar_t atomic_err  = atomic_comparable
                                             ? max_abs_diff(jv_mf.data(), jv_mf_atomic.data(), d.nnodes * N_FIELDS)
                                             : scalar_t(0);
        const scalar_t colored_err = colors.n_colors > 0
                                             ? max_abs_diff(jv_mf.data(), jv_mf_colored.data(), d.nnodes * N_FIELDS)
                                             : scalar_t(0);
        if (vs_matrix) std::printf("verify_jac_mf_action_vs_spmv_abs: %.6e\n", mf_err);
        if (atomic_comparable)
            std::printf("verify_jac_mf_atomic_action_vs_packed_abs: %.6e\n", atomic_err);
        else
            std::printf("verify_jac_mf_atomic_action_vs_packed_abs: skipped (the isoparametric SIMD "
                        "kernel carries no Rhie-Chow term)\n");
        if (colors.n_colors > 0) std::printf("verify_jac_mf_colored_action_vs_packed_abs: %.6e\n", colored_err);
        if (colored_err > 1.0e-12) {
            std::fprintf(stderr, "HEX8 colored Jacobian-action mismatch\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }
        // `fd` assembles the matrix by central differences with eps=1e-6, so it agrees with
        // the analytic action only to the truncation error of that -- about 5e-4 here, and
        // the same figure with the boundary closure on or off. Comparing it at 1e-8 was
        // measuring the reference against itself and failing every time; the looser bound
        // is the right yardstick for a finite-difference matrix, not a concession.
        const scalar_t mf_tol = kernel_kind == KernelKind::Fd ? scalar_t(1.0e-3) : scalar_t(1.0e-8);
        if (mf_err > mf_tol || atomic_err > 1.0e-12) {
            std::fprintf(stderr, "HEX8 Jacobian-action mismatch\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }
    }

    // Verify the two strategies that reuse the full element kernel through a modified
    // slot array: they must reproduce the full assembly exactly, not approximately.
    if (verify_jac && (assemble_diag || kernel_kind == KernelKind::Split)) {
        if (geom_kind == GeomKind::Isoparam)
            assemble_jacobian_atomic_isoparam(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), bsr.element_slots.empty() ? nullptr : bsr.element_slots.data(), bsr.nnz, bsr.values->data(), rho, mu);
        else
            assemble_jacobian_atomic_sumfact(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), bsr.element_slots.empty() ? nullptr : bsr.element_slots.data(), bsr.nnz, bsr.values->data(), rho, mu);
        // The reference has to be closed the same way the thing under test is, or the
        // comparison reports the boundary term as a mismatch. Same for the transient
        // diagonal.
        if (boundary)
            assemble_boundary_scs_jacobian_pass(d, bsr, rho, mu, geom_kind == GeomKind::Isoparam ? 1 : 0);
        assemble_transient_diag_pass(d, rho, bsr);
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
            jac_linear.assign((size_t)(bsr.nnz * 16), scalar_t(0));
            assemble_jacobian_atomic_linear_isoparam(d.elems, d.nelements, d.p.data(), d.points, d.ux.data(), d.uy.data(), d.uz.data(), bsr.element_slots.empty() ? nullptr : bsr.element_slots.data(), mu, jac_linear.data());
            assemble_jacobian_atomic_nonlinear_isoparam(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), bsr.element_slots.empty() ? nullptr : bsr.element_slots.data(), bsr.nnz, bsr.values->data(), rho, mu, jac_linear.data());
            if (boundary)
                assemble_boundary_scs_jacobian_pass(d, bsr, rho, mu, 1);
            assemble_transient_diag_pass(d, rho, bsr);
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
        else if (nodal_grad)
            grad_fn();
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
            else if (nodal_grad)
                grad_fn();
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
        // The reference carries whatever the timed sweep carried, including the exact
        // higher-order correction. Comparing a packed sweep that has the term against an atomic
        // reference that does not is not a layout check -- it reports the term as a defect.
        apply_jacobian_action_atomic(d.adj_ptr, d.det_ptr, d.elems, d.face_mask.data(), d.nelements, d.nnodes, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.ux.data(), d.uy.data(), d.uz.data(), cvfem_hex8_extras_of(d), rho, mu, last_dir, jv_ref.data(), KernelKind::Sumfact,
                                     with_hograd ? ugrad.data() : nullptr,
                                     with_hograd ? vgrad.data() : nullptr,
                                     conv_limiter, scalar_t(0));
        // The reference has to carry everything the timed apply carried, or the check
        // reports the boundary closure and the transient term as staging errors. It did:
        // `--jac-action --rhie-chow --boundary --layout packed` failed here at 8.3e-1, and
        // that combination has been accepted since the closure became a separate pass.
        if (boundary)
            apply_boundary_scs_jacobian_action_pass(d, rho, mu, geom_kind == GeomKind::Isoparam ? 1 : 0, last_dir,
                                                    jv_ref.data());
        apply_transient_action_pass(d, rho, last_dir, jv_ref.data());
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

    // The deferred correction is residual-only and lagged, so it enters the residual's work
    // model and not the Jacobian's -- which is the same asymmetry the kernels implement.
    const double residual_flops =
            (geom_kind == GeomKind::Isoparam ? CVFEM_HEX8_ISOPARAM_RESIDUAL_FLOPS_PER_ELEMENT
                                             : CVFEM_HEX8_RESIDUAL_FLOPS_PER_ELEMENT)
            + (conv_ho ? cvfem_hex8_defcor_flops_per_element(conv_limiter) : 0.0);
    // The exact higher-order action differentiates the correction, so its work model has to carry
    // that derivative; the lagged one does not see the correction at all and keeps the
    // first-order model. The comment above this block predates the exact action and said the
    // correction "enters the residual's work model and not the Jacobian's" without qualification,
    // which left every --conv-ho Jacobian arm reporting a first-order flop count -- and kept those
    // arms off the roofline, since a point there is placed by this number.
    const double jac_action_flops =
            (geom_kind == GeomKind::Isoparam ? CVFEM_HEX8_ISOPARAM_JAC_ACTION_FLOPS_PER_ELEMENT
                                             : CVFEM_HEX8_JAC_ACTION_FLOPS_PER_ELEMENT)
            + ((conv_ho && !lagged_ho) ? cvfem_hex8_defcor_jac_flops_per_element(conv_limiter) : 0.0);
    const double assemble_flops =
            geom_kind == GeomKind::Isoparam ? CVFEM_HEX8_ISOPARAM_ASSEMBLE_FLOPS_PER_ELEMENT
                                            : CVFEM_HEX8_ASSEMBLE_FLOPS_PER_ELEMENT;
    const double elem_apps = double(repeat) * double(d.nelements);

    // A bitwise fingerprint of the result, beside the checksum.
    //
    // The checksum cannot answer whether a run is reproducible, and that is not a matter of
    // degree: it is a running sum of millions of values, so a last-bit difference on a
    // handful of nodes is far below the ULP of the total and rounds away completely. Five
    // repeats of the boundary closure scattered atomically and gathered per node produced
    // byte-identical checksums for both, while a per-node comparison of the same operator
    // showed hundreds of doubles differing.
    //
    // This folds every value's bit pattern in, so any single bit that changes anywhere
    // changes it. The multiply-and-rotate is there because a plain XOR would cancel a value
    // that appears twice, which node arrays are full of.
    uint64_t fingerprint = 0xcbf29ce484222325ull;
    const auto fold = [&fingerprint](const scalar_t v) {
        uint64_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        fingerprint ^= bits;
        fingerprint *= 0x100000001b3ull;
        fingerprint = (fingerprint << 7) | (fingerprint >> 57);
    };

    scalar_t checksum = 0;
    if (assemble) {
        for (ptrdiff_t i = 0; i < bsr.nnz * 16; ++i) {
            checksum += bsr.values->data()[i];
            fold(bsr.values->data()[i]);
        }
    } else if (assemble_diag) {
        // The diagonal had no branch here, so it fell through to the residual arrays it
        // never writes and every --assemble-diag row carried checksum 0 -- a number that
        // cannot distinguish any two runs, which is the whole purpose of the column.
        for (scalar_t v : diag_blocks) {
            checksum += v;
            fold(v);
        }
    } else if (jac_action || bsr_apply) {
        for (ptrdiff_t i = 0; i < d.nnodes * N_FIELDS; ++i) {
            checksum += jac_out[(size_t)i];
            fold(jac_out[(size_t)i]);
        }
    } else {
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            checksum += d.rx[i] + d.ry[i] + d.rz[i] + d.rc[i];
            fold(d.rx[i]);
            fold(d.ry[i]);
            fold(d.rz[i]);
            fold(d.rc[i]);
        }
    }

    // --assemble-diag was missing from both of these, so a diagonal run announced itself
    // on stdout as a residual. The CSV column was fixed earlier; these were not, and a log
    // is what a person reads.
    phase_report(assemble ? "assemble"
                          : assemble_diag ? "assemble_diag"
                          : jac_action    ? "jac_action"
                                          : "residual",
                 repeat, threads_active());
    std::printf("cvfem_hex8_ns_upwind_smesh\n");
    std::printf("  mesh_manager: smesh::Mesh::create_hex8_cube\n");
    std::printf("  operation: %s\n",
                bsr_apply      ? "bsr_apply"
                : jac_action   ? "jacobian_action"
                : assemble     ? "jacobian_assemble"
                : assemble_diag ? "jacobian_block_diagonal"
                : nodal_grad   ? "nodal_gradient"
                               : "residual");
    std::printf("  layout: %s\n", layout.c_str());
    std::printf("  kernel: %s\n", kernel.c_str());
    std::printf("  geom: %s\n", geom.c_str());
    std::printf("  warp: %.6e\n", warp);
    std::printf("  OpenMP_threads: %d\n", threads_active());
    // THE PIPELINE, pass by pass, because "which layout ran" is not one answer but four and a
    // hybrid is a correctness question rather than a performance one. An element sweep from one
    // layout feeding on a gradient reduced by another is not that layout's operator, and it
    // would show up only as a reproducibility result nobody could explain.
    //
    // Two of the four passes belong to no layout and that is a property, not an omission. The
    // boundary closure gathers per owned node rather than scattering (the atomic scatter it
    // replaced made the result depend on thread arrival, and survives only as SFEM_BND_ATOMIC);
    // the transient term is a pure node loop with no scatter at all. Neither can be hybrid.
    {
        // Every row names a LAYOUT, in the spelling --layout takes, so the comparison a reader
        // makes is row against row. Naming the same implementation two ways -- "ecolor" on one
        // line and "element-coloured" on the next -- reads as two implementations and hides
        // exactly the disagreement this block exists to show.
        const char *qg = (layout == "ecolor")   ? "ecolor"
                         : g_qgrad_atomic       ? "atomic"
                         : (packed.n_packs > 0) ? "packed"
                                                : "atomic";
        std::printf("  pipeline_element_sweep: %s\n", layout.c_str());
        std::printf("  pipeline_nodal_gradient: %s\n", qg);
        std::printf("  pipeline_boundary: %s\n", cvfem_env_flag("SFEM_BND_ATOMIC") ? "atomic scatter" : "node gather");
        std::printf("  pipeline_transient: node loop\n");
    }
    if (layout == "store") {
        std::printf("  pack_size: %d\n", pack_size);
        std::printf("  n_packs: %td\n", packed.n_packs);
        std::printf("  st_max_local_nnz: %td\n", packed.st_max_local_nnz);
        std::printf("  st_local_matrix_KiB: %.1f\n", double(packed.st_max_local_nnz) * 128.0 / 1024.0);
    }
    if (layout == "ecolor") {
        std::printf("  n_colors: %d\n", ecolors.n_colors);
        std::printf("  elements_per_color_min_max: %td %td\n", ecolors.min_per_color, ecolors.max_per_color);
        // Setup, not throughput, but reported because it is where this layout is asymmetric with
        // the pack colouring beside it: that one colours a couple of thousand packs, this one
        // walks every element's incident elements through a node-to-element graph.
        std::printf("  ecolor_build_seconds: %.3f\n", ecolor_build_s);
        if (ecolors.min_per_color < (ptrdiff_t)threads_active() * CVFEM_HEX8_VEC_SIZE) {
            std::printf("  WARNING: a colour holds fewer elements (%td) than threads x lanes (%d);\n"
                        "           its barrier is paid for a partly idle sweep\n",
                        ecolors.min_per_color, threads_active() * CVFEM_HEX8_VEC_SIZE);
        }
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
    std::printf("  fingerprint: %016llx\n", (unsigned long long)fingerprint);
    if (nodal_grad) {
        // Reported per NODE-COMPONENT as well as per dof, because the two say different things:
        // MDOF/s keeps the row comparable with every other operator in this benchmark, and the
        // component rate is what scales with nf and is the quantity the kernel actually produces.
        std::printf("  nodal_grad_fields: %d\n", nodal_grad);
        std::printf("  seconds_per_nodal_grad: %.6e\n", seconds_per_call);
        std::printf("  MDOF/s_nodal_grad: %.3f\n", mdofs);
        std::printf("  MCOMP/s_nodal_grad: %.3f\n",
                    double(d.nnodes) * double(3 * nodal_grad) / seconds_per_call / 1.0e6);
    }
    if (!assemble && !jac_action && !bsr_apply && !nodal_grad) {
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
            std::printf("  bsr_values_MiB: %.3f\n",
                    double(bsr.nnz) * 16.0 * (bsr_single ? 4.0 : double(sizeof(scalar_t))) / (1024.0 * 1024.0));
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
        // The same split for the higher-order reconstruction, and it is worth having for the
        // same reason the Rhie-Chow one is: the exact action costs several times the lagged one,
        // and without this there is nothing to say how much of that is the reconstruction in
        // front of the sweep and how much is the correction's own arithmetic inside it. Guessing
        // from two separate runs' rates is not the same measurement -- it was off by an order of
        // magnitude when tried.
        if (with_hograd) {
            const double per_call = hograd_seconds / double(repeat);
            std::printf("  seconds_per_jac_action_hograd: %.6e\n", per_call);
            std::printf("  frac_jac_action_hograd: %.4f\n", hograd_seconds / seconds);
        }
    }
    if (bsr_apply) {
        const double bsr_apply_flops = double(bsr.nnz) * 2.0 * 16.0;
        // The values carry the storage type; x and y stay scalar_t whatever the storage is.
        const double bsr_apply_bytes = double(bsr.nnz) * 16.0 * (bsr_single ? 4.0 : double(sizeof(scalar_t))) +
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
        row.operation            = bsr_apply       ? "bsr_apply"
                                   : jac_action    ? "jac_action"
                                   : assemble      ? "assemble"
                                   : assemble_diag ? "assemble_diag"
                                                   : "residual";
        row.layout               = layout.c_str();
        row.ran_layout           = bsr_apply     ? "n/a"      // a matvec over CRS rows has no layout
                                 : assemble_diag ? "atomic"   // always the atomic diagonal, whatever was asked
                                 : assemble      ? layout.c_str()
                                 : (layout == "store") ? "packed"
                                                       : layout.c_str();
        row.kernel               = kernel.c_str();
        row.geom                 = geom.c_str();
        row.threads              = threads_active();
        row.pack_size            = (layout == "atomic") ? 0 : pack_size;
        row.cube_n               = n;
        row.nodes                = d.nnodes;
        row.elements             = d.nelements;
        row.dofs                 = n_dofs;
        row.bsr_nnz              = (assemble || bsr_apply) ? bsr.nnz : 0;
        row.bsr_storage          = (assemble || bsr_apply) ? (bsr_single ? "f32" : "f64") : "n/a";
        row.bsr_values_mib       = (assemble || bsr_apply)
                                           ? double(bsr.nnz) * 16.0 * (bsr_single ? 4.0 : double(sizeof(scalar_t))) /
                                                     (1024.0 * 1024.0)
                                           : 0.0;
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

        // ---- what actually ran ------------------------------------------------------
        //
        // This MUST mirror the dispatch in apply_fn / jac_fn / jac_action_fn / diag_fn
        // above; it is the one place where a row can be made to describe the code that
        // executed rather than the flags that were passed. Everything it can get wrong is
        // now either refused at the top of main or covered by a case here, and
        // tests/cvfem_bench_coverage_test checks the mapping.
        //
        // Three operations ignore --kernel entirely -- no apply_jacobian_action_* or
        // assemble_diag_* takes a KernelKind, and the SpMV takes nothing at all -- so the
        // requested name says nothing about what ran and "n/a" is the honest entry.
        const bool kernel_ran = !(jac_action || bsr_apply || assemble_diag) ||
                                kernel_is_action_only(kernel_kind);
        row.ran_kernel =
                partial_assembly ? "pa_sumfact"
                : !kernel_ran ? "n/a"
                // The isoparametric residual on a pack-based layout always runs the
                // isoparametric SIMD kernel; the requested name is not consulted.
                : (geom_kind == GeomKind::Isoparam && layout != "atomic" && !assemble) ? "isoparam_simd"
                // Isoparametric assembly on a pack-based layout runs the hand-written
                // scalar kernel for every name but `fd`: the branch there tests only for
                // fd, so `sympy` reaches the same code `current` does.
                : (geom_kind == GeomKind::Isoparam && layout != "atomic" && assemble &&
                   kernel_kind != KernelKind::Fd)                  ? "current"
                // There is no dedicated `current` assembly kernel, so it lands on fd.
                : (assemble && kernel_kind == KernelKind::Current) ? "fd"
                                                                   : kernel.c_str();
        // The exact form -- differentiating through the nodal gradient reconstruction --
        // is staged only by the Jacobian action. The assembled Jacobian keeps the frozen
        // form deliberately: the exact term couples pressures beyond nearest neighbours
        // and would widen the BSR pattern, and that operator exists only to build the
        // preconditioner.
        row.ran_rc       = !rhie_chow                              ? "off"
                           : jac_action                            ? "exact"
                           : (assemble || bsr_apply || assemble_diag) ? "frozen"
                                                                     : "on";
        row.ran_boundary = boundary ? "on" : "off";
        row.exact_rc     = (rhie_chow && jac_action && !lagged_rc) ? 1 : 0;
        row.pgrad_per_apply = pgrad_per_apply;
        row.live_vectors    = live_vectors;
        // Recorded as columns rather than left absent: this driver runs every kernel with
        // the hard upwind switch and with no transient term, while the solver plumbs both
        // (SFEM_UPWIND_EPS, and BDF1/BDF2). A reader comparing a benchmark rate against a
        // solver scope needs to see that, and a zero that is present says it.
        row.upwind_eps      = 0.0;
        row.transient       = dt > scalar_t(0) ? 1 : 0;
        row.pa_bytes_per_dof = partial_assembly ? cvfem_hex8_pa_bytes_per_dof(d) : 0.0;
        row.qgrad_seconds   = with_qgrad ? qgrad_seconds / double(repeat) : -1.0;
        row.qgrad_frac      = with_qgrad ? qgrad_seconds / seconds : -1.0;
        row.phase                = g_breakdown ? g_phase : nullptr;
        csv_write(csv_path, row);
    }

    d.mesh.reset();
    if (own_mpi) MPI_Finalize();
    return 0;
}
