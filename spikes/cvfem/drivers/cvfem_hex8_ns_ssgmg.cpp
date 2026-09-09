// HEX8 CVFEM Navier-Stokes through the SFEM frontend.
//
// Same channel problem as cvfem_hex8_ns_steady, driven through sfem::Function rather
// than through the spike's own mesh state: FunctionSpace of block size 4, the CVFEM
// operator, and DirichletConditions instead of a hand-maintained constraint mask. The
// destination is the semi-structured multigrid hierarchy, which needs a Function to
// derefine; this is the step that gets there and can still be checked against the
// standalone driver, which solves the same problem to the same tolerances.
//
// Note what it includes: the operator header and the channel case, and nothing else from
// this directory. No MeshData, no BSR4, no kernels. That is the point of the split -- a
// driver states the problem and the operator stays opaque.

#include "cvfem_hex8_ns_op.hpp"
#include "cvfem_fgmres.hpp"
#include "cvfem_ss_transfer.hpp"
#include "cvfem_ss_galerkin_api.hpp"

#include "sfem_CRS_X_BSR.hpp"
#include "cvfem_ns_channel_case.hpp"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_GeometricMultigrid.hpp"
#include "sfem_Multigrid.hpp"
#include "sfem_context.hpp"
#include "sfem_mask.hpp"

#include "smesh_env.hpp"
#include "smesh_sideset.hpp"
#include "smesh_glob.hpp"
#include "smesh_buffer.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <omp.h>

#include <map>
#include <limits>
#include <random>
#include <vector>

using cvfem_case::FlowCase;

// Relative floor for a scalar about to be inverted.
//
// Guards of the form `fabs(d) > 1e-30 ? 1/d : fallback` put an absolute threshold on a
// quantity that only means anything relative to its matrix. A 3x3 determinant of 1e-25, or a
// pressure diagonal of 1e-20, passes such a test and yields an inverse of 1e25 or 1e20, which
// a smoother or preconditioner then applies to the residual on every application. That is
// exactly how the Vanka smoother came to amplify one degree of freedom by 2.8e16 per sweep.
//
// The floor is deliberately far below anything legitimate: it guards against catastrophe and
// is not a conditioning heuristic. Placed anywhere near a plausible value it zeroes healthy
// entries and cripples the smoother -- measured, at 1e-11 it took Poiseuille from 178 linear
// iterations to 1119.
static inline bool cvfem_invertible(const real_t d, const real_t scale) {
    return std::fabs(d) > scale * real_t(1e-14) && std::fabs(d) > real_t(1e-300);
}

// The damping the smoother is actually run with.
//
// This was two parameters describing one thing. SFEM_GMG_OMEGA (0.35) drove the point-block
// smoother and the standalone smoother check; SFEM_VANKA_OMEGA (1) drove the Vanka smoother
// that replaces it inside the cycle. Vanka is the default, so the cycle ran undamped while
// every diagnostic reported 0.35 -- the check that exists to certify the smoother was
// measuring a different operator than the one being certified, and duly certified it.
//
// On the closed box that was survivable: additive Vanka at omega = 1 is still just
// convergent there, so the V-cycle worked and nothing pointed at the damping. With the
// do-nothing outflow the same smoother crosses one, and then more smoothing makes the cycle
// diverge faster rather than slower -- measured at 59x per cycle for two sweeps, 1000x for
// three, 1.5e4 for four and 9e8 for eight. That monotonicity is the signature, and it is
// what distinguishes a divergent smoother from a bad coarse space, which does not care how
// often the smoother runs.
//
// One function now, so the cycle and the check cannot disagree again.
//
// The default stays at 1, which is the measured-best value and not the thing that was wrong.
// On the closed-box Poiseuille regression at Re = 3200, omega = 1 reaches the target in 178
// linear iterations against 509 at 0.35 and 1711 at 0.5 (which does not even reach Re = 3200),
// so damping the smoother globally would cost a factor of three on every case that already
// works, to help one that needs more than damping anyway. What was wrong was that the check
// said 0.35 while the cycle ran 1; the value itself was chosen on evidence.
//
// A case with an open outflow needs damping to make the cycle converge at all -- set
// SFEM_VANKA_OMEGA explicitly there. It is a per-problem property, not a default.
static real_t smoother_omega() {
    return smesh::Env::read<real_t>("SFEM_VANKA_OMEGA", real_t(1));
}


namespace {

    constexpr int N_FIELDS = 4;

// Pressure gauge by zero mean, for the cases where nothing constrains the pressure.
//
// Pinning one pressure node nominally removes the constant mode, but it leaves a nearly
// constant one that costs the operator almost nothing, and it acts like a point source in
// the pressure. Measured on the 3D cavity at Re=100 with Vanka preconditioning, the pin puts
// a single isolated eigenvalue 51x (h=1/2) to 71x (h=1/4) below the rest of the spectrum,
// whose eigenvector is 96 percent pressure energy with no sign changes -- a smooth pressure
// mode, not a checkerboard. It takes cond(M^-1 A) from 9.1 to 447 at h=1/2 and from 27.4 to
// 1946 at h=1/4, so its damage grows like h^-2 and gets worse the finer the mesh.
//
// The remedy is to leave the system singular and work in the complement of its null space,
// which is what a zero-mean constraint or a Lagrange multiplier does. Two projections are
// needed and both are the same index set here, the pressure component:
//
//   null(A)   is the constant pressure: pressure enters the momentum equations only through
//             differences, so a uniform shift produces no force. Projecting the *solution*
//             fixes the gauge.
//   null(A^T) is the constant on the continuity rows: summing continuity over every node
//             cancels the interior fluxes and leaves the net boundary flux, which is zero
//             for a closed domain. Projecting the *residual* makes the right-hand side
//             compatible, which is what lets a Krylov method solve a consistent singular
//             system at all.
//
// The projection is unweighted because both null vectors are unweighted constants -- the
// same quantity the "sum of continuity residual" conservation check already reports.
class PressureGauge {
public:
    PressureGauge(const ptrdiff_t ndof, const mask_t *const cmask) {
        for (ptrdiff_t k = 3; k < ndof; k += N_FIELDS)
            if (!mask_get(k, cmask)) idx_.push_back(k);
    }

    // Off when something already constrains the pressure -- a pin, or a Dirichlet value.
    // Adding a gauge on top of one would over-determine the system.
    bool active() const { return active_ && !idx_.empty(); }
    void set_active(const bool a) { active_ = a; }
    size_t size() const { return idx_.size(); }

    void project(real_t *const v) const {
        if (!active()) return;
        long double s = 0;
        for (const ptrdiff_t k : idx_) s += (long double)v[(size_t)k];
        const real_t m = (real_t)(s / (long double)idx_.size());
        for (const ptrdiff_t k : idx_) v[(size_t)k] -= m;
    }

private:
    std::vector<ptrdiff_t> idx_;
    bool                   active_{true};
};

// A preconditioner whose output carries no constant-pressure component.
//
// Without this the preconditioner reintroduces the null direction the residual projection
// just removed, and the Krylov space drifts out of the complement it is supposed to stay in.
// Projection is linear and idempotent, so applying it to an accumulated result is correct
// whether or not the incoming vector was already projected.
class GaugedPreconditioner final : public sfem::Operator<real_t> {
public:
    GaugedPreconditioner(std::shared_ptr<sfem::Operator<real_t>> p, const PressureGauge *g)
        : p_(std::move(p)), g_(g) {}

    int apply(const real_t *const b, real_t *const x) override {
        const int ret = p_->apply(b, x);
        g_->project(x);
        return ret;
    }

    ptrdiff_t rows() const override { return p_->rows(); }
    ptrdiff_t cols() const override { return p_->cols(); }
    sfem::ExecutionSpace execution_space() const override { return p_->execution_space(); }

private:
    std::shared_ptr<sfem::Operator<real_t>> p_;
    const PressureGauge                    *g_;
};

    void usage(const char *argv0) {
        std::fprintf(stderr,
                     "usage: %s <output_folder>\n"
                     "\n"
                     "HEX8 CVFEM Navier-Stokes channel, driven through the SFEM frontend.\n"
                     "Same problem and defaults as cvfem_hex8_ns_steady, so the two are\n"
                     "directly comparable.\n"
                     "\n"
                     "Environment:\n"
                     "  SFEM_CASE            poiseuille | couette | cavity | cavity_reg | mms (required)\n"
                     "  SFEM_N               cells in y (default 8)\n"
                     "  SFEM_NX SFEM_NY SFEM_NZ   override cells per direction\n"
                     "  SFEM_LX SFEM_LY SFEM_LZ   channel size (default 4, 1, 1)\n"
                     "  SFEM_RHO SFEM_MU SFEM_U   density, viscosity, velocity scale\n"
                     "  SFEM_GEOM            affine | isoparam (default affine)\n"
                     "  SFEM_NL_MAX_IT SFEM_NL_RTOL SFEM_NL_ATOL\n"
                     "  SFEM_DT              timestep; <= 0 (default) is the steady solve\n"
                     "  SFEM_NSTEPS          time steps to take (default 1)\n"
                     "  SFEM_BDF_ORDER       1 or 2 (default 2; step 1 falls back to BDF1)\n"
                     "  SFEM_LSOLVE_RTOL SFEM_LSOLVE_ATOL SFEM_LSOLVE_MAX_IT\n"
                     "  SFEM_PACK_SIZE       affine packed SIMD (default 2048; 0 = atomic)\n"
                     "  SFEM_MATRIX_FREE     1: Krylov uses J(u)v (default); 0: assembled BSR\n"
                     "  SFEM_CHECK_JV        1: compare |J_mf v - J_asm v| on the first Jacobian\n"
                     "  SFEM_GMG             1: V-cycle preconditioner (needs a refine level > 1)\n"
                     "                       2: cost-matched control -- fine-level smoother, no hierarchy\n"
                     "  SFEM_GMG_SMOOTH      block-Jacobi smoothing steps (default 3)\n"
                     "  SFEM_ELEMENT_REFINE_LEVEL  >1: semi-structured macro-elements at that\n"
                     "                       internal level (default 1 = flat)\n"
                     "  SFEM_PGRAD_CACHE     1: reuse the nodal pressure gradient across a\n"
                     "                       Krylov solve rather than rebuilding it per apply\n"
                     "  SFEM_VERIFY_TOL      fail if velocity Linf exceeds this (default 1e-2)\n",
                     argv0);
    }

    // Inverse of the 4x4 node blocks, used as the Krylov preconditioner. Velocity and
    // pressure are inverted separately, exactly as the standalone driver does: the 3x3
    // velocity block by cofactors, and the pressure entry as a reciprocal. The pressure
    // diagonal is nonzero here only because Rhie-Chow stabilisation puts it there -- see
    // the SFEM_PC_PSCALE discussion in the standalone driver for what governs its size.
    // Phase timing.
    //
    // Cost has so far been inferred from operator-application counts, which is a model, not
    // a measurement -- it assumes every application costs the same and ignores the sparse
    // coarse work, the transfers and the assembly entirely. This wraps each thing the cycle
    // does in a timer so the breakdown is measured. SFEM's own tracing needs
    // SMESH_ENABLE_TRACE compiled into smesh, which the installed one lacks.
    struct Phase {
        double t{0};
        long   n{0};
    };
    std::map<std::string, Phase> g_phases;

    void phase_add(const std::string &k, const double dt) {
        auto &p = g_phases[k];
        p.t += dt;
        p.n += 1;
    }

    // Thread clamp for small levels.
    //
    // A coarse level has too little work to fill a thread team, and paying to start one per
    // vector operation costs far more than the arithmetic. Measured on an 81-node level
    // (324 unknowns): 156 us per smoother application on one thread, 4.6 ms on four,
    // 14.2 ms on eight -- ninety times slower for having more cores. It also inverts the
    // whole solve, which ran 14.9 s on one thread, 4.9 s on four and 13.0 s on eight, and it
    // is why the same configuration that was merely slow on a laptop was an order of
    // magnitude off on 72 Grace cores.
    //
    // So each level runs with a thread count matched to its size rather than the machine's.
    // These calls happen between parallel regions, never inside one.
    //
    // The threshold was 20000 dofs per thread, and that was far too high: at 242,500 dofs on
    // 72 Grace cores it gave the *fine* level 12 threads and level 1 exactly one, throttling
    // the two phases that are 65% of the cycle. perf saw it three ways at once -- 4.21
    // instructions per cycle and a 0.57% cache miss rate, so the code runs well when it runs;
    // cores idle 83% of the wall time; and 12/72 = 16.7% matching the 17% of peak cycles
    // actually retired. Lowering it to 1000 leaves the coarse protection intact, since a
    // 324-dof level still gets one thread either way -- the threshold only governs when a
    // level is allowed *more* threads, and 20000 was high enough that nothing on this machine
    // ever got the full count.
    //
    // Measured at 242,500 dofs, 72 cores, fixed work, three repeats each: preconditioner
    // phases 1.657 s against 0.681 s, a factor of 2.43; smooth[L0] 4327 us against 1723,
    // smooth[L1] 2292 us against 161. Run-to-run spread was under 2%.
    //
    // This is also why the pathology never showed on a laptop: at 8 threads the clamp is
    // nearly inert, and it only bites once the machine is wide enough for the ratio to matter.
    std::shared_ptr<sfem::Operator<real_t>> thread_clamped(const ptrdiff_t                                ndofs,
                                                           const std::shared_ptr<sfem::Operator<real_t>> &op) {
        if (!op) return op;
        // SFEM_GMG_CLAMP_BELOW: engage the clamp only on levels smaller than this, and leave
        // everything above it on the full team.
        //
        // The clamp changes the team size, and with schedule(static) over the element loops a
        // different team size means a different partition of the same array. The grid
        // transfers are *not* clamped, so restriction writes a coarse vector partitioned
        // across 72 threads and the coarse smoother then reads it partitioned across 30 -- the
        // mapping does not line up, and a V-cycle crosses that boundary at every level. Add
        // the per-call omp_set_num_threads and there is team-resize churn on top. Clamping
        // only where a level genuinely cannot fill the machine keeps one partition for
        // everything large enough to care about locality.
        //
        // 0 keeps the pure dofs-per-thread rule.
        //
        // OFF BY DEFAULT, because measurement says the mechanism costs more than it saves.
        // At 242,500 dofs on 72 Grace cores, three repeats, fixed work, t_solve:
        //
        //     dofs/thread 1000, no floor      1.394 s
        //     dofs/thread 1000, floor 10000   1.163 s   (clamp only the coarse space)
        //     no clamp at all                 0.642 s
        //
        // and the *phase totals are the same in all three*, ~0.69 s. The whole difference is
        // outside the timed phases: omp_set_num_threads is called around every clamped apply,
        // and resizing the team degrades the unclamped parallel regions that follow -- the
        // outer Krylov vector operations. Clamping only the coarse levels halves the damage
        // but does not remove it, because the churn is per clamped operator, not per dof.
        //
        // The coarse levels do not need it any more either: coarse_solve is 209 us unclamped
        // against 318 us clamped. The pathology it was written for -- an 81-node level at 156
        // us on one thread and 14.2 ms on eight -- was measured before the deterministic
        // scatter and the assembled coarse operators existed, and no longer reproduces.
        //
        // Set SFEM_GMG_DOFS_PER_THREAD to a positive value to bring it back.
        const ptrdiff_t per = (ptrdiff_t)smesh::Env::read<int>("SFEM_GMG_DOFS_PER_THREAD", 0);
        if (per <= 0) return op;

        const ptrdiff_t clamp_below = (ptrdiff_t)smesh::Env::read<int>("SFEM_GMG_CLAMP_BELOW", 0);
        if (clamp_below > 0 && ndofs >= clamp_below) return op;
        const int       mx  = omp_get_max_threads();
        int             n   = (int)std::min<ptrdiff_t>(mx, std::max<ptrdiff_t>(1, ndofs / std::max<ptrdiff_t>(1, per)));
        if (n >= mx) return op;  // big enough to use the machine as configured
        return sfem::make_op<real_t>(
                op->rows(), op->cols(),
                [op, n, mx](const real_t *const x, real_t *const y) {
                    omp_set_num_threads(n);
                    op->apply(x, y);
                    omp_set_num_threads(mx);
                },
                sfem::EXECUTION_SPACE_HOST);
    }

    std::shared_ptr<sfem::Operator<real_t>> timed(const std::string                             &name,
                                                  const std::shared_ptr<sfem::Operator<real_t>> &op) {
        if (!op) return op;
        return sfem::make_op<real_t>(
                op->rows(), op->cols(),
                [op, name](const real_t *const x, real_t *const y) {
                    const double t0 = smesh::time_seconds();
                    op->apply(x, y);
                    phase_add(name, smesh::time_seconds() - t0);
                },
                sfem::EXECUTION_SPACE_HOST);
    }

    // Some phases contain others: precond_total wraps the whole V-cycle, so the smoothers,
    // transfers and coarse solve are inside it. Summing every row therefore double counts,
    // and shares taken against that sum understate everything -- which is exactly how an
    // earlier reading of this table came to report fine-level smoothing at 42% when it is
    // 79% of wall time. Containers are excluded from the denominator and printed apart.
    bool is_container(const std::string &k) { return k == "precond_total"; }

    void phase_report() {
        if (g_phases.empty()) return;
        double total = 0;
        for (auto &kv : g_phases)
            if (!is_container(kv.first)) total += kv.second.t;
        std::vector<std::pair<std::string, Phase>> v(g_phases.begin(), g_phases.end());
        std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) { return a.second.t > b.second.t; });
        std::printf("\nphase breakdown (%.3f s in top-level phases; shares are of that)\n", total);
        std::printf("  %-22s %10s %10s %12s %7s\n", "phase", "seconds", "calls", "us/call", "share");
        for (auto &kv : v) {
            if (is_container(kv.first)) continue;
            std::printf("  %-22s %10.3f %10ld %12.1f %6.1f%%\n", kv.first.c_str(), kv.second.t, kv.second.n,
                        1e6 * kv.second.t / (double)std::max(1L, kv.second.n),
                        100.0 * kv.second.t / std::max(1e-30, total));
        }
        for (auto &kv : v)
            if (is_container(kv.first))
                std::printf("  %-22s %10.3f %10ld %12.1f %6.1f%%  (container: the rows above it\n"
                            "  %-22s %10s %10s %12s %7s   marked op/smooth/transfer/coarse are inside)\n",
                            kv.first.c_str(), kv.second.t, kv.second.n,
                            1e6 * kv.second.t / (double)std::max(1L, kv.second.n),
                            100.0 * kv.second.t / std::max(1e-30, total), "", "", "", "", "");
    }

    class BlockJacobi final : public sfem::Operator<real_t> {
    public:
        BlockJacobi(const ptrdiff_t nnodes, std::vector<real_t> inv) : nnodes_(nnodes), inv_(std::move(inv)) {}

        int apply(const real_t *const x, real_t *const y) override {
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nnodes_; ++i) {
                const real_t *const m  = inv_.data() + (size_t)i * 16;
                const real_t *const xx = x + (size_t)i * 4;
                real_t *const       yy = y + (size_t)i * 4;
                for (int r = 0; r < 4; ++r) {
                    real_t s = 0;
                    for (int c = 0; c < 4; ++c) s += m[r * 4 + c] * xx[c];
                    yy[r] += s;
                }
            }
            return SFEM_SUCCESS;
        }

        ptrdiff_t rows() const override { return nnodes_ * 4; }
        ptrdiff_t cols() const override { return nnodes_ * 4; }
        sfem::ExecutionSpace execution_space() const override { return sfem::EXECUTION_SPACE_HOST; }

    private:
        ptrdiff_t           nnodes_;
        std::vector<real_t> inv_;
    };

    // SIMPLE smoother.
    //
    // Block-Jacobi is not a smoother for this system: measured as the stationary iteration
    // it becomes inside a cycle, its rate crosses 1 at the damping that looked best by
    // iteration count, and even at its best damping it only reaches about 0.97 per sweep.
    // That is the failure a saddle-point smoother exists to fix, and it is what the 2x2
    // block split was built for -- SIMPLE needs the off-diagonal blocks on their own, and
    // evaluating the whole operator to get one of them would discard most of the work.
    //
    // One application, given a residual r = (r_u, r_p):
    //   du   = Du^-1 r_u                        velocity predictor, 3x3 solves per node
    //   rp   = r_p - (C du)_p                   continuity defect of that predictor
    //   dp   = solve(S dp = rp),  S = Dpp - C Du^-1 B, approximated by its diagonal
    //   du  -= Du^-1 (B dp)_u                   make the predictor respect the new pressure
    // With one inner sweep from dp = 0 the Schur apply drops out and the cost is two block
    // applications, PU and UP, rather than two full ones.
    //
    // Accumulates into `out`, which is the Operator convention here and is what the
    // stationary iteration relies on: it computes r = b - A x and then calls the
    // preconditioner as x += M^-1 r.
    class SimpleSmoother final : public sfem::Operator<real_t> {
    public:
        SimpleSmoother(sfem::CVFEMNavierStokes &op, const real_t *const state, const ptrdiff_t nnodes,
                       std::vector<real_t> dinv_u, std::vector<real_t> dinv_p, std::vector<uint8_t> free_dof,
                       const real_t omega, const int inner)
            : op_(op), state_(state), nnodes_(nnodes), dinv_u_(std::move(dinv_u)), dinv_p_(std::move(dinv_p)),
              free_(std::move(free_dof)), omega_(omega), inner_(inner) {}

        int apply(const real_t *const r, real_t *const out) override {
            const ptrdiff_t    nd = nnodes_ * 4;
            std::vector<real_t> du((size_t)nd, 0), tmp((size_t)nd, 0), dp((size_t)nd, 0);

            // velocity predictor
            for (ptrdiff_t i = 0; i < nnodes_; ++i) {
                const real_t *const m = dinv_u_.data() + (size_t)i * 9;
                const real_t *const rr = r + (size_t)i * 4;
                real_t *const       dd = du.data() + (size_t)i * 4;
                for (int a = 0; a < 3; ++a)
                    dd[a] = m[a * 3 + 0] * rr[0] + m[a * 3 + 1] * rr[1] + m[a * 3 + 2] * rr[2];
            }

            // continuity defect of the predictor
            op_.apply_blocks(state_, du.data(), tmp.data(), sfem::CVFEM_BLOCK_PU);
            std::vector<real_t> rp((size_t)nnodes_, 0);
            for (ptrdiff_t i = 0; i < nnodes_; ++i)
                rp[(size_t)i] = r[(size_t)i * 4 + 3] - tmp[(size_t)i * 4 + 3];

            // pressure correction. The first sweep starts from dp = 0, so the Schur
            // application is zero and is skipped rather than computed.
            for (int it = 0; it < inner_; ++it) {
                std::vector<real_t> s((size_t)nnodes_, 0);
                if (it > 0) {
                    std::fill(tmp.begin(), tmp.end(), real_t(0));
                    std::vector<real_t> t2((size_t)nd, 0), w((size_t)nd, 0), t3((size_t)nd, 0);
                    op_.apply_blocks(state_, dp.data(), tmp.data(), sfem::CVFEM_BLOCK_PP);
                    op_.apply_blocks(state_, dp.data(), t2.data(), sfem::CVFEM_BLOCK_UP);
                    for (ptrdiff_t i = 0; i < nnodes_; ++i) {
                        const real_t *const m  = dinv_u_.data() + (size_t)i * 9;
                        const real_t *const tt = t2.data() + (size_t)i * 4;
                        real_t *const       ww = w.data() + (size_t)i * 4;
                        for (int a = 0; a < 3; ++a)
                            ww[a] = m[a * 3 + 0] * tt[0] + m[a * 3 + 1] * tt[1] + m[a * 3 + 2] * tt[2];
                    }
                    op_.apply_blocks(state_, w.data(), t3.data(), sfem::CVFEM_BLOCK_PU);
                    for (ptrdiff_t i = 0; i < nnodes_; ++i)
                        s[(size_t)i] = tmp[(size_t)i * 4 + 3] - t3[(size_t)i * 4 + 3];
                }
                for (ptrdiff_t i = 0; i < nnodes_; ++i)
                    dp[(size_t)i * 4 + 3] += dinv_p_[(size_t)i] * (rp[(size_t)i] - s[(size_t)i]);
            }

            // velocity correction for the new pressure
            std::fill(tmp.begin(), tmp.end(), real_t(0));
            op_.apply_blocks(state_, dp.data(), tmp.data(), sfem::CVFEM_BLOCK_UP);

            for (ptrdiff_t i = 0; i < nnodes_; ++i) {
                const real_t *const m  = dinv_u_.data() + (size_t)i * 9;
                const real_t *const tt = tmp.data() + (size_t)i * 4;
                real_t              c[3];
                for (int a = 0; a < 3; ++a)
                    c[a] = m[a * 3 + 0] * tt[0] + m[a * 3 + 1] * tt[1] + m[a * 3 + 2] * tt[2];
                for (int a = 0; a < 3; ++a) {
                    const ptrdiff_t k = i * 4 + a;
                    if (free_[(size_t)k]) out[k] += omega_ * (du[(size_t)k] - c[a]);
                }
                const ptrdiff_t kp = i * 4 + 3;
                if (free_[(size_t)kp]) out[kp] += omega_ * dp[(size_t)kp];
            }
            return SFEM_SUCCESS;
        }

        ptrdiff_t rows() const override { return nnodes_ * 4; }
        ptrdiff_t cols() const override { return nnodes_ * 4; }
        sfem::ExecutionSpace execution_space() const override { return sfem::EXECUTION_SPACE_HOST; }

    private:
        sfem::CVFEMNavierStokes &op_;
        const real_t *const      state_;
        ptrdiff_t                nnodes_;
        std::vector<real_t>      dinv_u_, dinv_p_;
        std::vector<uint8_t>     free_;
        real_t                   omega_;
        int                      inner_;
    };

    // `mask` marks the constrained dofs. It is needed because hessian_block_diag reports
    // the operator's own diagonal and knows nothing about boundary conditions, while the
    // matrix the Krylov method actually sees has identity rows there -- Function applies
    // the constraints to it after the operators. A preconditioner built from the
    // unconstrained diagonal scales those rows by something unrelated to 1.
    // Builds the pieces SIMPLE needs from the operator's own 4x4 block diagonal: the 3x3
    // velocity inverse per node, and a diagonal approximation of the Schur complement.
    // `ds_scale` tunes the latter, which is the one genuinely approximate ingredient --
    // S = Dpp - C Du^-1 B is not diagonal and its true diagonal is not available
    // matrix-free, so the pressure block's own diagonal stands in for it.
    std::shared_ptr<SimpleSmoother> make_simple(sfem::CVFEMNavierStokes &op,
                                                const real_t *const      x,
                                                const mask_t *const      mask,
                                                const ptrdiff_t          nnodes,
                                                const real_t             omega,
                                                const int                inner,
                                                const real_t             ds_scale) {
        // SIMPLE takes the frozen-pressure-gradient block apply, not the exact restriction
        // of the operator.
        //
        // sscvfem_apply_blocks can differentiate through the nodal pressure-gradient
        // reconstruction, and by default it does, because the four field blocks are checked
        // against the requirement that they sum back to the operator. A preconditioner is
        // under no such obligation, and here the exact term is measurably not worth its
        // price: on 72 Grace cores it adds one nodal gradient pass per pressure-column apply
        // -- +140% on B^T and +217% on C at 34,147,332 dofs -- and moves this smoother's
        // standalone convergence rate not at all. Measured to 40 sweeps at 75,140 dofs, the
        // exact and frozen forms agree to six decimals at every single sweep, ending at
        // 0.990157 against 0.990157.
        //
        // The block diagonal SIMPLE builds below is frozen anyway and cannot be otherwise:
        // the exact term is nonlocal and does not fit the BSR sparsity pattern. So this also
        // makes the two halves of the preconditioner consistent with each other.
        //
        // SFEM_SIMPLE_EXACT_RC=1 restores the exact form, which is how the numbers above
        // were obtained and how they can be re-obtained.
        op.set_option("blocks_exact_rc", smesh::Env::read<int>("SFEM_SIMPLE_EXACT_RC", 0) != 0);

        std::vector<real_t> bd((size_t)nnodes * 16, real_t(0));
        op.hessian_block_diag(x, bd.data());

        std::vector<real_t>  du((size_t)nnodes * 9, real_t(0));
        std::vector<real_t>  dp((size_t)nnodes, real_t(0));
        std::vector<uint8_t> freed((size_t)nnodes * 4, 1);

        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            const real_t *const b = bd.data() + (size_t)i * 16;
            real_t              a[9];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) a[r * 3 + c] = b[r * 4 + c];

            const real_t det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
                               a[2] * (a[3] * a[7] - a[4] * a[6]);
            real_t *const m = du.data() + (size_t)i * 9;
            // A determinant scales as the cube of the entries, so its floor must too.
            real_t ascale = 0;
            for (int k = 0; k < 9; ++k) ascale = std::max(ascale, std::fabs(a[k]));
            if (cvfem_invertible(det, ascale * ascale * ascale)) {
                const real_t id = real_t(1) / det;
                m[0] = (a[4] * a[8] - a[5] * a[7]) * id;
                m[1] = (a[2] * a[7] - a[1] * a[8]) * id;
                m[2] = (a[1] * a[5] - a[2] * a[4]) * id;
                m[3] = (a[5] * a[6] - a[3] * a[8]) * id;
                m[4] = (a[0] * a[8] - a[2] * a[6]) * id;
                m[5] = (a[2] * a[3] - a[0] * a[5]) * id;
                m[6] = (a[3] * a[7] - a[4] * a[6]) * id;
                m[7] = (a[1] * a[6] - a[0] * a[7]) * id;
                m[8] = (a[0] * a[4] - a[1] * a[3]) * id;
            } else {
                m[0] = m[4] = m[8] = real_t(1);
            }

            // Scale the pressure diagonal against the pressure row, not against the whole
            // block. In a colocated scheme the pressure diagonal is the Rhie-Chow term,
            // h^2/(2 mu) times an area, and it is legitimately orders of magnitude below the
            // momentum entries beside it. Flooring it relative to the block maximum therefore
            // rejects perfectly healthy entries -- measured, it stopped the backward-facing
            // step converging at all under the block-Jacobi preconditioner.
            const real_t pp = b[15];
            real_t       prow = 0;
            for (int k = 12; k < 16; ++k) prow = std::max(prow, std::fabs(b[k]));
            dp[(size_t)i] = cvfem_invertible(pp, prow) ? ds_scale / pp : real_t(0);

            for (int c = 0; c < 4; ++c) {
                const ptrdiff_t k = i * 4 + c;
                if (mask_get(k, mask)) freed[(size_t)k] = 0;
            }
        }

        return std::make_shared<SimpleSmoother>(op, x, nnodes, std::move(du), std::move(dp),
                                                std::move(freed), omega, inner);
    }

    // Inverts a given 4x4 block diagonal. Split out from make_block_jacobi so a smoother
    // can be built from an assembled matrix's own diagonal rather than from the operator's,
    // which is what the Galerkin levels need: they smooth an assembled matrix, and
    // preconditioning it with the rediscretised diagonal is what made those levels diverge.
    std::shared_ptr<BlockJacobi> make_block_jacobi_from_diag(std::vector<real_t> blocks,
                                                             const mask_t *const mask,
                                                             const ptrdiff_t     nnodes,
                                                             const real_t        omega = real_t(1),
                                                             const real_t        prow  = real_t(1)) {

        // `prow` scales the continuity row of every block, matching SFEM_GMG_PSCALE's
        // scaling of the level operator. The two must agree: the smoother preconditions
        // the operator it smooths, so preconditioning a scaled operator with an unscaled
        // diagonal leaves the pressure update wrong by 1/prow and turns the coarse
        // smoother divergent as prow shrinks -- which is what an earlier reading of
        // SFEM_GMG_PSCALE was actually measuring.
        if (prow != real_t(1))
            for (ptrdiff_t i = 0; i < nnodes; ++i)
                for (int c = 0; c < 4; ++c) blocks[(size_t)i * 16 + 12 + c] *= prow;

        std::vector<real_t> inv((size_t)nnodes * 16, 0);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            const real_t *const b = blocks.data() + (size_t)i * 16;
            real_t *const       m = inv.data() + (size_t)i * 16;

            const real_t a00 = b[0], a01 = b[1], a02 = b[2];
            const real_t a10 = b[4], a11 = b[5], a12 = b[6];
            const real_t a20 = b[8], a21 = b[9], a22 = b[10];
            const real_t c00 = a11 * a22 - a12 * a21;
            const real_t c01 = a02 * a21 - a01 * a22;
            const real_t c02 = a01 * a12 - a02 * a11;
            const real_t det = a00 * c00 + a10 * c01 + a20 * c02;

            real_t vscale = 0;
            for (const real_t q : {a00, a01, a02, a10, a11, a12, a20, a21, a22})
                vscale = std::max(vscale, std::fabs(q));
            if (cvfem_invertible(det, vscale * vscale * vscale)) {
                const real_t d = real_t(1) / det;
                m[0] = c00 * d;
                m[1] = c01 * d;
                m[2] = c02 * d;
                m[4] = (a12 * a20 - a10 * a22) * d;
                m[5] = (a00 * a22 - a02 * a20) * d;
                m[6] = (a02 * a10 - a00 * a12) * d;
                m[8]  = (a10 * a21 - a11 * a20) * d;
                m[9]  = (a01 * a20 - a00 * a21) * d;
                m[10] = (a00 * a11 - a01 * a10) * d;
            } else {
                m[0] = m[5] = m[10] = real_t(1);
            }

            // Same reasoning as above: the pressure row sets the scale for its own diagonal.
            const real_t pp = b[15];
            real_t       prow2 = 0;
            for (int k = 12; k < 16; ++k) prow2 = std::max(prow2, std::fabs(b[k]));
            m[15] = cvfem_invertible(pp, prow2) ? real_t(1) / pp : real_t(1);

            // Damping. Undamped block-Jacobi is fine as a Krylov preconditioner, where it
            // is applied once, and is not a smoother: as a stationary iteration on this
            // saddle-point system it does not converge. SFEM's own multigrid damps its
            // block-Jacobi by 1/block_size for the same reason.
            if (omega != real_t(1))
                for (int k = 0; k < 16; ++k) m[k] *= omega;

            // Constrained rows are identity in the assembled matrix, so they must be
            // identity here too.
            for (int c = 0; c < 4; ++c) {
                if (!mask_get(i * 4 + c, mask)) continue;
                for (int k = 0; k < 4; ++k) m[c * 4 + k] = real_t(0);
                m[c * 4 + c] = real_t(1);
            }
        }
        return std::make_shared<BlockJacobi>(nnodes, std::move(inv));
    }

    std::shared_ptr<BlockJacobi> make_block_jacobi(sfem::CVFEMNavierStokes &op,
                                                   const real_t *const      x,
                                                   const mask_t *const      mask,
                                                   const ptrdiff_t          nnodes,
                                                   const real_t             omega = real_t(1),
                                                   const real_t             prow  = real_t(1)) {
        std::vector<real_t> blocks((size_t)nnodes * 16, 0);
        const double        t0 = smesh::time_seconds();
        op.hessian_block_diag(x, blocks.data());
        phase_add("hessian_block_diag", smesh::time_seconds() - t0);
        return make_block_jacobi_from_diag(std::move(blocks), mask, nnodes, omega, prow);
    }


    // ---------------------------------------------------------------------------
    // Semi-structured geometric multigrid, as a preconditioner for the Jacobian solve.
    //
    // sfem::create_gmg_data builds the hierarchy: it derefines the Function level by level,
    // which calls derefine_op on the CVFEM operator, which reassembles itself on the coarse
    // space. That is rediscretisation rather than Galerkin coarsening, and it has to be --
    // the Rhie-Chow coefficient carries h^2/(2 mu), so the coarse pressure operator differs
    // from the fine one by about 8x per level in 3D and P^T A P would inherit the wrong one.
    //
    // Two things here are not create_gmg_operators and create_gmg_default_smoothers_and_solver,
    // and neither could be:
    //
    //   * create_gmg_operators builds each level with a null state. That is fine for a
    //     linear operator and fatal for this one, whose Jacobian depends on where it is
    //     linearised. Each level gets the fine state restricted onto it instead. The
    //     restriction divides by the node incidence count before accumulating, so it
    //     averages rather than sums and is the right transfer for a state.
    //
    //   * the default smoothers compute sym_block_size as (block_size == 3 ? 6 : 3), which
    //     silently yields 3 for a block size of 4 where a symmetric 4x4 needs 10; they then
    //     call hessian_block_diag_sym, which assumes a symmetry Navier-Stokes does not have;
    //     and the coarse solver is CG, which needs an SPD operator. All three are wrong here,
    //     so the smoother is the operator's own 4x4 block diagonal and the coarse solver is
    //     BiCGStab.
    struct GmgLevels {
        std::shared_ptr<sfem::MultigridData>                   data;
        std::vector<sfem::SharedBuffer<real_t>>                states;
        // R = P^T sums; dividing by R applied to the constant 1 turns it into the
        // partition-of-unity average that a state transfer needs. One per coarse level.
        std::vector<std::vector<real_t>>                       state_weights;   // kept alive for the operators
        std::vector<std::shared_ptr<sfem::Operator<real_t>>>   ops;
        std::vector<std::shared_ptr<sfem::CVFEMNavierStokes>>  level_ops;
        std::shared_ptr<sfem::Multigrid<real_t>>               mg;
        int                                                    smoothing_steps{3};

        // Transfer matrices, built once: they depend on the lattice, not the linearisation.
        // Pmat[i] maps level i (coarse) to level i-1 (fine); Rmat[i] is its transpose.
        using CRS_t = sfem::CRS<sfem::count_t, sfem::idx_t, real_t, real_t>;
        using BSR_t = sfem::BSR<sfem::count_t, sfem::idx_t, real_t, real_t>;
        std::vector<std::shared_ptr<CRS_t>> Pmat, Rmat;
        // The assembled coarse operators, kept as matrices so the level below can be formed
        // from the level above by a triple product instead of by probing.
        std::vector<std::shared_ptr<BSR_t>> Amat;
    };

    // Zero the source component of every block feeding a constrained dof.
    //
    // The grid transfers zero constrained dofs on their output, and they do it per
    // *component* -- ux,uy,uz at a wall, not p. A scalar nodal transfer cannot express that,
    // so R*A*P alone does not reproduce the composite the driver probes. Folding the mask
    // into A's columns first does, exactly and in O(nnz): blocks are row-major
    // values[a*16 + r*4 + c] with c the source component, so a constrained dof (j,c) means
    // clearing column c of every block in block-column j.
    std::shared_ptr<GmgLevels::BSR_t> mask_block_columns(const std::shared_ptr<GmgLevels::BSR_t> &a,
                                                         const mask_t *const                      mask) {
        const ptrdiff_t nbr = a->row_ptr->size() - 1;
        auto            rp  = a->row_ptr;
        auto            ci  = a->col_idx;
        auto            va  = smesh::create_host_buffer<real_t>(a->values->size());
        std::copy(a->values->data(), a->values->data() + a->values->size(), va->data());

        const sfem::count_t *const rpd = rp->data();
        const sfem::idx_t *const   cid = ci->data();
        real_t *const              vd  = va->data();

#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < nbr; ++i)
            for (sfem::count_t k = rpd[i]; k < rpd[i + 1]; ++k) {
                const ptrdiff_t j = cid[k];
                for (int c = 0; c < N_FIELDS; ++c) {
                    if (!mask_get(j * N_FIELDS + c, mask)) continue;
                    for (int r = 0; r < N_FIELDS; ++r) vd[(size_t)k * 16 + (size_t)r * N_FIELDS + c] = 0;
                }
            }

        return sfem::h_bsr_spmv<sfem::count_t, sfem::idx_t, real_t, real_t>(
                nbr, a->cols() / N_FIELDS, N_FIELDS, rp, ci, va, real_t(0));
    }

    // Apply a scalar nodal matrix to a block vector. The transfers are scalar because the
    // interpolation is identical for all four fields; the CRS x BSR rap knows that (a scalar
    // entry scales a whole block), but a hand composition has to spell it out.
    void apply_scalar_to_blocks(const std::shared_ptr<GmgLevels::CRS_t> &m,
                                const real_t *const                     in,
                                real_t *const                           out) {
        const ptrdiff_t nr = m->rows(), nc = m->cols();
        std::vector<real_t> a((size_t)nc), b((size_t)nr);
        for (int c = 0; c < N_FIELDS; ++c) {
            for (ptrdiff_t k = 0; k < nc; ++k) a[(size_t)k] = in[k * N_FIELDS + c];
            std::fill(b.begin(), b.end(), real_t(0));
            m->apply(a.data(), b.data());
            for (ptrdiff_t k = 0; k < nr; ++k) out[k * N_FIELDS + c] += b[(size_t)k];
        }
    }

    // Identity rows for constrained dofs, searching for the diagonal rather than assuming
    // its position -- mm emits unsorted column indices.
    void patch_identity_rows(const std::shared_ptr<GmgLevels::BSR_t> &a, const mask_t *const mask) {
        const ptrdiff_t            nbr = a->row_ptr->size() - 1;
        const sfem::count_t *const rp  = a->row_ptr->data();
        const sfem::idx_t *const   ci  = a->col_idx->data();
        real_t *const              vd  = a->values->data();

#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < nbr; ++i)
            for (int r = 0; r < N_FIELDS; ++r) {
                if (!mask_get(i * N_FIELDS + r, mask)) continue;
                for (sfem::count_t k = rp[i]; k < rp[i + 1]; ++k) {
                    const bool diag = (ci[k] == i);
                    for (int c = 0; c < N_FIELDS; ++c)
                        vd[(size_t)k * 16 + (size_t)r * N_FIELDS + c] =
                                (diag && c == r) ? real_t(1) : real_t(0);
                }
            }
    }

    std::shared_ptr<GmgLevels> build_gmg(const std::shared_ptr<sfem::Function>       &f,
                                         const std::shared_ptr<sfem::CVFEMNavierStokes> &fine_op,
                                         const sfem::SharedBuffer<real_t>            &x_fine,
                                         const int smoothing_steps) {
        auto data = sfem::create_gmg_data(f);
        if (!data) return nullptr;
        int nlevels = (int)data->functions.size();
        if (nlevels < 2) return nullptr;

        // SFEM_GMG_MAX_LEVELS caps the depth, keeping the finest levels and solving on the
        // deepest one kept.
        //
        // The hierarchy bottoms out at the macro mesh, which for a small N is a handful of
        // cells. A convection-dominated Navier-Stokes discretisation there does not
        // approximate the fine operator in any useful sense, and because that level is
        // solved to tolerance the cycle takes its answer at face value and prolongs a
        // confidently wrong correction. Stopping the hierarchy while the coarse mesh still
        // resolves the flow is the standard remedy.
        {
            const int cap = smesh::Env::read<int>("SFEM_GMG_MAX_LEVELS", 0);
            if (cap > 1 && cap < nlevels) nlevels = cap;
        }

        auto out             = std::make_shared<GmgLevels>();
        out->data            = data;
        out->smoothing_steps = smoothing_steps;

        // Level states. The matrix-free operators read these buffers live, so the buffers
        // are allocated once and refilled per Newton step rather than reallocated -- which
        // is also what lets the operators below be built once.
        out->states.resize(nlevels);
        out->states[0] = x_fine;
        for (int i = 1; i < nlevels; ++i)
            out->states[i] = smesh::create_host_buffer<real_t>((size_t)data->functions[i]->space()->n_dofs());

        // The operator chain, so each level's block diagonal is reachable.
        out->level_ops.resize(nlevels);
        out->level_ops[0] = fine_op;
        const real_t rc_decay = smesh::Env::read<real_t>("SFEM_GMG_RC_DECAY", real_t(1));
        for (int i = 1; i < nlevels; ++i) {
            out->level_ops[i] = out->level_ops[i - 1] ? out->level_ops[i - 1]->coarser() : nullptr;
            if (!out->level_ops[i]) return nullptr;

            // Rhie-Chow does not survive rediscretisation unscaled. Its coefficient is
            // Df = rc_scale * h^2 / (2 mu), so halving the lattice resolution per level
            // quadruples Df, and the coarse pressure block ends up far stiffer than the
            // fine block whose error it is supposed to correct. The operator inherits
            // rc_scale from its parent, so left alone every level stabilises for its own
            // h. SFEM_GMG_RC_DECAY rescales it per level; 0.25 keeps Df fixed at the fine
            // level's value, which is what makes the coarse correction commensurate.
            out->level_ops[i]->rhie_chow_scale = out->level_ops[i - 1]->rhie_chow_scale * rc_decay;
        }

        for (int i = 0; i < nlevels; ++i)
            out->ops.push_back(sfem::create_linear_operator(sfem::op_type::MATRIX_FREE, data->functions[i],
                                                            out->states[i], sfem::EXECUTION_SPACE_HOST));
        return out;
    }

    // Per Newton step: push the new state down the levels and rebuild the smoothers around
    // it. The hierarchy, its transfer operators and the level operators are all reused --
    // only the linearisation moved. Rebuilding the whole hierarchy here instead was the
    // first attempt and it dominated the solve, since create_gmg_data derefines every
    // Function again.
    // Transfer sanity check (SFEM_GMG_CHECK=1).
    //
    // A V-cycle that measures identically to its own smoother is suspicious in a specific
    // way: it suggests the coarse-grid correction is not weak but absent. This applies the
    // transfers to a smooth test field and reports what survives each hop, which separates
    // "the correction is small" from "the correction is zero".
    void check_transfers(GmgLevels &g) {
        const int nlevels = (int)g.ops.size();
        auto nrm = [](const std::vector<real_t> &v) {
            real_t s = 0;
            for (auto e : v) s += e * e;
            return std::sqrt(s);
        };

        // Is the Jacobian action element-local?
        //
        // Element-wise Galerkin (sum_e P_e^T A_e P_e) is only exact if A is a sum of
        // macro-element contributions. Rhie-Chow couples through a *nodal* pressure
        // gradient, which is not element-local -- unless it is frozen from the state rather
        // than recomputed from the direction. Test it directly: put the direction on the
        // interior of a single macro element and see whether the response stays inside that
        // element's nodes. If it spreads, the operator is not a sum of element operators and
        // the element-wise construction is wrong.
        {
            auto           &fop = *g.level_ops[0];
            const ptrdiff_t nd  = g.data->functions[0]->space()->n_dofs();
            const ptrdiff_t nn0 = nd / N_FIELDS;

            auto     &m0   = g.data->functions[0]->space()->mesh();
            auto      b0   = m0.block(0);
            const int lvl  = smesh::semistructured_level(m0);
            const int nxe0 = (lvl + 1) * (lvl + 1) * (lvl + 1);

            std::vector<uint8_t> in_elem((size_t)nn0, 0);
            for (int a = 0; a < nxe0; ++a) in_elem[(size_t)b0->elements()->data()[a][0]] = 1;

            std::vector<real_t> dir((size_t)nd, 0), out((size_t)nd, 0);
            // Interior lattice nodes of element 0 only, so nothing is shared with a neighbour.
            for (int z = 1; z < lvl; ++z)
                for (int y = 1; y < lvl; ++y)
                    for (int x = 1; x < lvl; ++x) {
                        const ptrdiff_t gnode = b0->elements()->data()[smesh::sshex8_lidx(lvl, x, y, z)][0];
                        for (int c = 0; c < N_FIELDS; ++c) dir[(size_t)gnode * N_FIELDS + c] = 1;
                    }

            fop.apply(g.states[0]->data(), dir.data(), out.data());

            real_t inside = 0, outside = 0;
            for (ptrdiff_t k = 0; k < nn0; ++k)
                for (int c = 0; c < N_FIELDS; ++c) {
                    const real_t v = std::fabs(out[(size_t)k * N_FIELDS + c]);
                    if (in_elem[(size_t)k]) inside = std::max(inside, v);
                    else                    outside = std::max(outside, v);
                }
            std::printf("element-locality: max |out| inside elem 0 = %.4e, outside = %.4e  %s\n",
                        inside, outside,
                        (outside <= inside * 1e-12) ? "LOCAL (element-wise Galerkin is exact)"
                                                    : "NON-LOCAL (element-wise Galerkin would be wrong)");
        }

        // Element-wise Galerkin, identity gate.
        //
        // At q = 1 the prolongation is the identity, so P^T A P is A itself and the assembled
        // matrix must reproduce the matrix-free apply to round-off. That one comparison
        // covers everything the construction rests on at once: that identity slots really do
        // yield the micro-cell matrix, that the hoisted geometry and Rhie-Chow struct fed to
        // the assembly kernel are the ones the apply uses, that the derived pattern holds
        // every entry, and that the inverted-index accumulation lands each block where it
        // belongs. A failure here means the coarse operators are wrong for a reason that has
        // nothing to do with coarsening, so it runs before any q > 1 comparison.
        {
            auto           &fop = *g.level_ops[0];
            const auto      fs  = g.data->functions[0]->space();
            const ptrdiff_t nd  = fs->n_dofs();

            std::vector<real_t> diag;
            auto                A1 = cvfem_ss::assemble_coarse_operator(fop, fs, fs, &diag);

            std::vector<real_t> v((size_t)nd), a((size_t)nd, 0), b((size_t)nd, 0);
            for (ptrdiff_t k = 0; k < nd; ++k) v[(size_t)k] = std::sin(real_t(0.7) * (real_t)k + real_t(0.3));

            fop.apply(g.states[0]->data(), v.data(), a.data());
            A1->apply(v.data(), b.data());

            real_t num = 0, den = 0;
            for (ptrdiff_t k = 0; k < nd; ++k) {
                const real_t d = a[(size_t)k] - b[(size_t)k];
                num += d * d;
                den += a[(size_t)k] * a[(size_t)k];
            }
            const real_t rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
            std::printf("egal identity (q=1): rel |A_egal v - A v| = %.3e over %td blocks  %s\n", rel,
                        (ptrdiff_t)A1->col_idx->size(), rel < 1e-12 ? "OK" : "FAILED");
        }

        // Element-wise Galerkin against the composite it claims to equal, at q > 1.
        //
        // The identity gate above proves the assembly reproduces A; this proves the
        // coarsening reproduces P^T A P. Both sides are built without constraints -- the
        // matrix-free composite from the structured transfer and the raw operator, the
        // assembled one straight from the element matrices -- so nothing here depends on
        // the constraint treatment, which is applied identically to both afterwards and
        // would otherwise hide a discrepancy inside the rows it overwrites.
        std::shared_ptr<cvfem_ss::CoarseBSR> egal_prev;
        for (int i = 1; i < (int)g.data->functions.size(); ++i) {
            auto           &fop = *g.level_ops[0];
            const auto      fs  = g.data->functions[0]->space();
            const auto      cs  = g.data->functions[i]->space();
            const ptrdiff_t ndc = cs->n_dofs();
            const ptrdiff_t nnc = ndc / N_FIELDS;

            // Level 1 is checked against the matrix-free composite. Below that the check is
            // the stronger one: the direct construction at ratio q must equal the same
            // operator coarsened one 2:1 hop from the level above -- which is the claim that
            // levels do not chain, that piecewise-linear interpolation on nested uniform
            // lattices composes to the direct map. If it were false, building each level
            // straight from the fine element matrices would silently differ from the
            // hierarchy the transfers actually implement.
            const auto      up  = i == 1 ? fs : g.data->functions[i - 1]->space();
            const ptrdiff_t ndf = up->n_dofs();

            cvfem_ss::ProlongationPattern pat;
            cvfem_ss::build_from_spaces(cs, up, pat);
            if (!pat.uniform) continue;  // apply_structured covers the 2:1 hops only

            auto A = cvfem_ss::assemble_coarse_operator(fop, cs, fs, nullptr);
            if (i > 1 && !egal_prev) continue;

            std::vector<real_t> vc((size_t)ndc), tf((size_t)ndf, 0), wf((size_t)ndf, 0);
            std::vector<real_t> a((size_t)ndc, 0), b((size_t)ndc, 0);
            for (ptrdiff_t k = 0; k < ndc; ++k) vc[(size_t)k] = std::cos(real_t(0.41) * (real_t)k + real_t(0.9));

            // P applied per component, the transfer being scalar and nodal.
            for (int c = 0; c < N_FIELDS; ++c) {
                std::vector<real_t> cin((size_t)nnc), fout((size_t)(ndf / N_FIELDS), 0);
                for (ptrdiff_t k = 0; k < nnc; ++k) cin[(size_t)k] = vc[(size_t)k * N_FIELDS + c];
                cvfem_ss::apply_structured(pat, cin.data(), fout.data());
                for (ptrdiff_t k = 0; k < ndf / N_FIELDS; ++k) tf[(size_t)k * N_FIELDS + c] = fout[(size_t)k];
            }

            if (i == 1) fop.apply(g.states[0]->data(), tf.data(), wf.data());
            else        egal_prev->apply(tf.data(), wf.data());

            // P^T, the adjoint of apply_structured with the same implied weights.
            for (ptrdiff_t r = 0; r < pat.n_fine; ++r) {
                const sfem::count_t bk = pat.rowptr[(size_t)r], ek = pat.rowptr[(size_t)r + 1];
                const real_t        w  = real_t(1) / (real_t)(ek - bk);
                for (sfem::count_t k = bk; k < ek; ++k)
                    for (int c = 0; c < N_FIELDS; ++c)
                        a[(size_t)pat.colidx[(size_t)k] * N_FIELDS + c] += w * wf[(size_t)r * N_FIELDS + c];
            }

            A->apply(vc.data(), b.data());

            real_t num = 0, den = 0;
            for (ptrdiff_t k = 0; k < ndc; ++k) {
                const real_t d = a[(size_t)k] - b[(size_t)k];
                num += d * d;
                den += a[(size_t)k] * a[(size_t)k];
            }
            const real_t rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
            std::printf("egal galerkin (0->%d): rel |A_egal - P^T %s P| = %.3e  %s\n", i,
                        i == 1 ? "A" : "A_egal(prev)", rel, rel < 1e-11 ? "OK" : "FAILED");
            egal_prev = A;
        }

        // Block-split gate: the four blocks must sum to the full Jacobian action, and
        // each must be non-trivial. A smoother built on the block split is only as good as
        // this, and a block application that silently produced nothing would make SIMPLE
        // degenerate into block-Jacobi without saying so.
        {
            auto           &fop = *g.level_ops[0];
            const ptrdiff_t nd  = g.data->functions[0]->space()->n_dofs();
            std::vector<real_t> dir((size_t)nd), full((size_t)nd, 0), sum((size_t)nd, 0);
            for (ptrdiff_t k = 0; k < nd; ++k) dir[(size_t)k] = std::sin(0.7 * (real_t)k) + 0.3;

            fop.apply(g.states[0]->data(), dir.data(), full.data());

            const int   sel[4] = {sfem::CVFEM_BLOCK_UU, sfem::CVFEM_BLOCK_UP,
                                  sfem::CVFEM_BLOCK_PU, sfem::CVFEM_BLOCK_PP};
            const char *bn[4]  = {"uu", "up", "pu", "pp"};
            std::printf("block split:");
            for (int b = 0; b < 4; ++b) {
                std::vector<real_t> one((size_t)nd, 0);
                fop.apply_blocks(g.states[0]->data(), dir.data(), one.data(), sel[b]);
                real_t n = 0;
                for (ptrdiff_t k = 0; k < nd; ++k) {
                    n += one[(size_t)k] * one[(size_t)k];
                    sum[(size_t)k] += one[(size_t)k];
                }
                std::printf("  |%s| %.4e", bn[b], std::sqrt(n));
            }
            real_t dn = 0, fn = 0;
            for (ptrdiff_t k = 0; k < nd; ++k) {
                const real_t d = sum[(size_t)k] - full[(size_t)k];
                dn += d * d;
                fn += full[(size_t)k] * full[(size_t)k];
            }
            // Full-precision checksum of one operator application, plus a repeat, so
            // non-determinism is visible rather than hidden by the print width above.
            // Compare this line across thread counts and across runs.
            {
                std::vector<real_t> y1((size_t)nd, 0), y2((size_t)nd, 0);
                fop.apply(g.states[0]->data(), dir.data(), y1.data());
                fop.apply(g.states[0]->data(), dir.data(), y2.data());
                real_t s1 = 0, dmax = 0;
                for (ptrdiff_t k = 0; k < nd; ++k) {
                    s1 += y1[(size_t)k];
                    dmax = std::max(dmax, std::fabs(y1[(size_t)k] - y2[(size_t)k]));
                }
                real_t sx = 0;
                for (ptrdiff_t k = 0; k < nd; ++k) sx += g.states[0]->data()[(size_t)k];
                // Sorted checksum too: summing in sorted order is invariant to node
                // numbering, so if the plain sum varies and this one does not, the mesh is
                // being numbered differently between runs and the values are the same.
                std::vector<real_t> sv(g.states[0]->data(), g.states[0]->data() + nd);
                std::sort(sv.begin(), sv.end());
                real_t sxs = 0;
                for (auto v : sv) sxs += v;
                std::printf("  state checksum: %.17g  sorted: %.17g\n", (double)sx, (double)sxs);
                std::printf("  apply checksum: %.17g   repeat-diff %.3e\n", (double)s1, (double)dmax);

                // The other three kernels that scatter: residual, block diagonal, and one
                // block of the 2x2 split. Same question, same evidence.
                {
                    std::vector<real_t> r1((size_t)nd, 0), r2((size_t)nd, 0);
                    fop.gradient(g.states[0]->data(), r1.data());
                    fop.gradient(g.states[0]->data(), r2.data());
                    real_t cs = 0, dm = 0;
                    for (ptrdiff_t k = 0; k < nd; ++k) {
                        cs += r1[(size_t)k];
                        dm = std::max(dm, std::fabs(r1[(size_t)k] - r2[(size_t)k]));
                    }
                    std::printf("  residual checksum: %.17g   repeat-diff %.3e\n", (double)cs, (double)dm);

                    const ptrdiff_t nnod = nd / N_FIELDS;
                    std::vector<real_t> b1((size_t)nnod * 16, 0), b2((size_t)nnod * 16, 0);
                    fop.hessian_block_diag(g.states[0]->data(), b1.data());
                    fop.hessian_block_diag(g.states[0]->data(), b2.data());
                    real_t bs = 0, bm = 0;
                    for (size_t k = 0; k < b1.size(); ++k) {
                        bs += b1[k];
                        bm = std::max(bm, std::fabs(b1[k] - b2[k]));
                    }
                    std::printf("  blockdiag checksum: %.17g  repeat-diff %.3e\n", (double)bs, (double)bm);

                    std::vector<real_t> k1((size_t)nd, 0), k2((size_t)nd, 0);
                    fop.apply_blocks(g.states[0]->data(), dir.data(), k1.data(), sfem::CVFEM_BLOCK_PU);
                    fop.apply_blocks(g.states[0]->data(), dir.data(), k2.data(), sfem::CVFEM_BLOCK_PU);
                    real_t ks = 0, km = 0;
                    for (ptrdiff_t k = 0; k < nd; ++k) {
                        ks += k1[(size_t)k];
                        km = std::max(km, std::fabs(k1[(size_t)k] - k2[(size_t)k]));
                    }
                    std::printf("  blocksplit checksum: %.17g repeat-diff %.3e\n", (double)ks, (double)km);
                }
            }
            std::printf("\n  sum vs full: rel %.4e  %s\n", (fn > 0) ? std::sqrt(dn / fn) : 0.0,
                        (fn > 0 && std::sqrt(dn / fn) < 1e-10) ? "OK" : "MISMATCH");
        }

        // Constraint census per level. A colocated Navier-Stokes system with velocity
        // Dirichlet data all round fixes pressure only up to a constant, so the pressure
        // needs a pin. If the fine level has one and a coarse level does not, that coarse
        // operator is singular, its solve wanders along the constant-pressure null vector,
        // and the prolonged correction carries that spurious mode back up -- which looks
        // exactly like a V-cycle that diverges after the first couple of cycles.
        for (int i = 0; i < nlevels; ++i) {
            const ptrdiff_t     nd = g.data->functions[i]->space()->n_dofs();
            std::vector<mask_t> m(mask_count(nd), 0);
            g.data->functions[i]->constraints_mask(m.data());
            int per[N_FIELDS] = {0};
            for (ptrdiff_t k = 0; k < nd; ++k)
                if (mask_get(k, m.data())) per[k % N_FIELDS]++;
            std::printf("constraints level %d: n %td  ux %d  uy %d  uz %d  p %d\n",
                        i, nd, per[0], per[1], per[2], per[3]);
        }

        for (int i = 0; i + 1 < nlevels; ++i) {
            const ptrdiff_t nf = g.data->functions[i]->space()->n_dofs();
            const ptrdiff_t nc = g.data->functions[i + 1]->space()->n_dofs();

            std::vector<real_t> fine((size_t)nf), coarse((size_t)nc, 0), back((size_t)nf, 0);
            // Smooth and non-zero on every component, so a component-selective bug shows.
            for (ptrdiff_t k = 0; k < nf; ++k) fine[(size_t)k] = 1 + 0.1 * (real_t)(k % 7);

            g.data->restrictions[i]->apply(fine.data(), coarse.data());
            if (g.data->prolongations[i + 1]) g.data->prolongations[i + 1]->apply(coarse.data(), back.data());

            {
                // Gate for the structured prolongation: it must reproduce the matrix-free
                // transfer it is meant to replace. The structured form carries no values at
                // all -- the weight is 1/nnz for a 2:1 hop -- so this also checks that the
                // "weights are implied by row length" claim actually holds on the real
                // numbering, not just on paper.
                cvfem_ss::ProlongationPattern pp;
                cvfem_ss::build_from_spaces(g.data->functions[i + 1]->space(),
                                            g.data->functions[i]->space(), pp);

                std::vector<real_t> cs((size_t)nc), ref((size_t)nf, 0), got((size_t)nf, 0);
                unsigned            st = 4242u;
                for (auto &v : cs) {
                    st = st * 1103515245u + 12345u;
                    v  = (real_t)((st >> 16) & 0x7fff) / (real_t)0x7fff - real_t(0.5);
                }

                // The driver's prolongation is block-valued; compare one component by
                // scattering the scalar field into that component and reading it back.
                std::vector<real_t> cb((size_t)nc * 0 + (size_t)g.data->functions[i + 1]->space()->n_dofs(), 0);
                std::vector<real_t> fb((size_t)g.data->functions[i]->space()->n_dofs(), 0);
                for (ptrdiff_t k = 0; k < pp.n_coarse; ++k) cb[(size_t)k * N_FIELDS] = cs[(size_t)k];
                if (g.data->prolongations[i + 1]) {
                    auto praw = sfem::create_hierarchical_prolongation(g.data->functions[i + 1]->space(),
                                                                       g.data->functions[i]->space(),
                                                                       sfem::EXECUTION_SPACE_HOST);
                    praw->apply(cb.data(), fb.data());
                    for (ptrdiff_t k = 0; k < pp.n_fine; ++k) ref[(size_t)k] = fb[(size_t)k * N_FIELDS];

                    cvfem_ss::apply_structured(pp, cs.data(), got.data());

                    real_t dn = 0, rn = 0;
                    for (ptrdiff_t k = 0; k < pp.n_fine; ++k) {
                        const real_t d = got[(size_t)k] - ref[(size_t)k];
                        dn += d * d;
                        rn += ref[(size_t)k] * ref[(size_t)k];
                    }
                    const real_t rel = (rn > 0) ? std::sqrt(dn / rn) : 0.0;
                    std::printf("  structured P %d->%d: nnz %td  uniform %d  vs matrix-free rel %.4e  %s\n",
                                i + 1, i, (ptrdiff_t)pp.rowptr[(size_t)pp.n_fine], (int)pp.uniform, rel,
                                (rel < 1e-12) ? "OK" : "MISMATCH");

                    // Does P reproduce a constant?
                    //
                    // Neither test above asks this. The structured-vs-matrix-free comparison
                    // checks two implementations of the same P against each other, and the
                    // adjoint check verifies R = P^T; both pass whatever P is. A prolongation
                    // whose rows do not sum to one interpolates a constant field into
                    // something else, and the multigrid correction is then wrong by an amount
                    // proportional to the solution itself rather than to the error.
                    //
                    // This goes unnoticed on every case in this file but one. Where the whole
                    // skin is Dirichlet, the correction at boundary nodes is zeroed after the
                    // prolongation, so a defect confined to boundary rows is masked exactly.
                    // The do-nothing outflow leaves those nodes free, and it is the only
                    // configuration here that exposes them -- which is why it is also the only
                    // one whose V-cycle diverges.
                    std::vector<real_t> one((size_t)g.data->functions[i + 1]->space()->n_dofs(), 0),
                            pone((size_t)g.data->functions[i]->space()->n_dofs(), 0);
                    for (ptrdiff_t k = 0; k < pp.n_coarse; ++k) one[(size_t)k * N_FIELDS] = 1;
                    praw->apply(one.data(), pone.data());
                    real_t    worst = 0;
                    ptrdiff_t worst_k = -1, n_bad = 0;
                    for (ptrdiff_t k = 0; k < pp.n_fine; ++k) {
                        const real_t d = std::fabs(pone[(size_t)k * N_FIELDS] - real_t(1));
                        if (d > 1e-10) ++n_bad;
                        if (d > worst) { worst = d; worst_k = k; }
                    }
                    std::printf("  P %d->%d reproduces constants: worst |P1-1| = %.4e at node %td, %td of %td nodes off  %s\n",
                                i + 1, i, (double)worst, worst_k, n_bad, (ptrdiff_t)pp.n_fine,
                                (worst < 1e-10) ? "OK" : "BROKEN");
                }
            }

            {
                // Full-precision checksums of the transfers themselves. The restriction
                // accumulates fine contributions into coarse nodes, which is the same kind
                // of operation the element scatter was, so it is a candidate for the same
                // problem.
                long double cr = 0, cp = 0;
                for (ptrdiff_t k = 0; k < nc; ++k) cr += (long double)coarse[(size_t)k];
                for (ptrdiff_t k = 0; k < nf; ++k) cp += (long double)back[(size_t)k];
                std::printf("  transfer checksums: R %.17g  P %.17g\n", (double)cr, (double)cp);
            }
            std::printf("transfer %d->%d: n %td->%td  |x| %.4e  |Rx| %.4e  |PRx| %.4e\n",
                        i, i + 1, nf, nc, nrm(fine), nrm(coarse), nrm(back));

            // Adjoint test. A coarse-grid correction is only consistent when the residual
            // restriction is the transpose of the correction prolongation, so that the
            // coarse problem minimises the same error the fine level sees. If it is not,
            // the correction is scaled wrongly and the cycle over- or under-corrects; a
            // constant ratio here is exactly the factor it is out by.
            if (g.data->prolongations[i + 1]) {
                std::vector<real_t> xr((size_t)nf), yc((size_t)nc), Rx((size_t)nc, 0), Py((size_t)nf, 0);
                unsigned            seed = 12345;
                auto                rnd  = [&seed]() {
                    seed = seed * 1103515245u + 12345u;
                    return (real_t)((seed >> 16) & 0x7fff) / (real_t)0x7fff - real_t(0.5);
                };
                for (auto &e : xr) e = rnd();
                for (auto &e : yc) e = rnd();

                // Both transfers zero constrained dofs on their output, so the adjoint
                // identity only holds on vectors that already satisfy the constraints.
                // Probing with unconstrained noise measures the constraint handling, not
                // the transfers, and reports a spurious mismatch.
                g.data->functions[i]->apply_zero_constraints(xr.data());
                g.data->functions[i + 1]->apply_zero_constraints(yc.data());

                g.data->restrictions[i]->apply(xr.data(), Rx.data());
                g.data->prolongations[i + 1]->apply(yc.data(), Py.data());

                real_t lhs = 0, rhs2 = 0;
                for (ptrdiff_t k = 0; k < nc; ++k) lhs += Rx[(size_t)k] * yc[(size_t)k];
                for (ptrdiff_t k = 0; k < nf; ++k) rhs2 += xr[(size_t)k] * Py[(size_t)k];
                std::printf("  adjoint %d: <Rx,y>=%.6e  <x,Py>=%.6e  ratio=%.6f\n",
                            i, lhs, rhs2, (rhs2 != 0) ? lhs / rhs2 : 0.0);
            }

            // Coarse-operator consistency: A_c v against R A_f P v on a smooth coarse
            // vector.
            //
            // The coarse level is rediscretised rather than assembled as R A_f P, which is
            // the whole reason the hierarchy is affordable, but it is only a legitimate
            // substitute if it acts like the Galerkin operator on the smooth vectors a
            // coarse grid is supposed to carry. If the two disagree by O(1) here, the
            // coarse solve is answering a different question from the one the fine level
            // asked, and no smoother can repair the correction that comes back.
            if (g.data->prolongations[i + 1]) {
                std::vector<real_t> vc((size_t)nc), vf((size_t)nf, 0), wf((size_t)nf, 0),
                        g1((size_t)nc, 0), g2((size_t)nc, 0);
                for (ptrdiff_t k = 0; k < nc; ++k) vc[(size_t)k] = 1 + 0.1 * (real_t)(k % 7);
                g.data->functions[i + 1]->apply_zero_constraints(vc.data());

                g.data->prolongations[i + 1]->apply(vc.data(), vf.data());
                g.ops[i]->apply(vf.data(), wf.data());
                g.data->restrictions[i]->apply(wf.data(), g1.data());
                g.ops[i + 1]->apply(vc.data(), g2.data());

                // Split by component. The momentum and continuity rows coarsen very
                // differently: Rhie-Chow's Df = rc_scale * h^2 / (2 mu) is the only term
                // that depends on the lattice spacing outright, so an inconsistency
                // concentrated in the pressure rows implicates the stabilisation, and one
                // spread evenly implicates the discretisation as a whole.
                real_t dn[N_FIELDS] = {0}, rn[N_FIELDS] = {0};
                real_t ab[N_FIELDS] = {0}, aa[N_FIELDS] = {0};
                for (ptrdiff_t k = 0; k < nc; ++k) {
                    const int    c = (int)(k % N_FIELDS);
                    const real_t d = g1[(size_t)k] - g2[(size_t)k];
                    dn[c] += d * d;
                    rn[c] += g1[(size_t)k] * g1[(size_t)k];
                    ab[c] += g2[(size_t)k] * g1[(size_t)k];   // <A_c v, R A P v>
                    aa[c] += g2[(size_t)k] * g2[(size_t)k];
                }
                const char *nm[N_FIELDS] = {"ux", "uy", "uz", "p"};
                std::printf("  coarse-op %d rel:  ", i);
                for (int c = 0; c < N_FIELDS; ++c)
                    std::printf("%s %.4f  ", nm[c], (rn[c] > 0) ? std::sqrt(dn[c] / rn[c]) : 0.0);
                // Best-fit scale per component. A rediscretised coarse operator is not
                // meant to equal R A P -- the two differ by a fixed factor from the
                // h-scaling convention -- so the raw mismatch above conflates that known
                // factor with real disagreement. What matters is whether the velocity and
                // pressure rows carry the SAME factor. If they do not, the coarse operator
                // is not a scalar multiple of the Galerkin one, and no single scaling of
                // the correction can reconcile it, which is why SFEM_GMG_CGC failed.
                std::printf("\n  coarse-op %d scale:", i);
                real_t su = 0;
                for (int c = 0; c < N_FIELDS; ++c) {
                    const real_t sc = (aa[c] > 0) ? ab[c] / aa[c] : 0.0;
                    std::printf("  %s %.4f", nm[c], sc);
                    if (c < 3) su += sc / 3;
                }
                const real_t sp = (aa[3] > 0) ? ab[3] / aa[3] : 0.0;
                std::printf("   -> pressure/velocity %.4f", (su != 0) ? sp / su : 0.0);
                // Residual left after removing each component's own best-fit scale. This
                // is the part of the disagreement that no rescaling of any kind can reach,
                // and it decides whether a cheap fix exists at all.
                std::printf("\n  coarse-op %d after-scale:", i);
                for (int c = 0; c < N_FIELDS; ++c) {
                    const real_t sc = (aa[c] > 0) ? ab[c] / aa[c] : 0.0;
                    real_t       rr2 = 0;
                    for (ptrdiff_t k = c; k < nc; k += N_FIELDS) {
                        const real_t d = sc * g2[(size_t)k] - g1[(size_t)k];
                        rr2 += d * d;
                    }
                    std::printf("  %s %.4f", nm[c], (rn[c] > 0) ? std::sqrt(rr2 / rn[c]) : 0.0);
                }
                std::printf("\n");
            }
        }
    }

    // R is the adjoint of the prolongation, which is what the residual transfer in a
    // V-cycle must be, and is exactly wrong for moving a state down. Applied to a field it
    // sums rather than averages and inflates it by the number of fine nodes feeding each
    // coarse node -- measured here as a factor of about 3.8 per level. A coarse operator
    // linearised about a state that large is not an approximation of the fine operator at
    // all, so its correction is not a correction. Normalising by R applied to the constant
    // 1 recovers the average, which is exact for constants and leaves a smooth field alone.
    // Transfer matrices, built once. They depend only on the lattice, so rebuilding them
    // per Newton step -- as the probing path effectively did with its pattern and colouring
    // -- is pure waste.
    void build_transfer_matrices(GmgLevels &g) {
        const int nlevels = (int)g.ops.size();
        g.Pmat.assign((size_t)nlevels, nullptr);
        g.Rmat.assign((size_t)nlevels, nullptr);
        g.Amat.assign((size_t)nlevels, nullptr);
        for (int i = 1; i < nlevels; ++i) {
            cvfem_ss::ProlongationPattern pp;
            cvfem_ss::build_from_spaces(g.data->functions[i]->space(), g.data->functions[i - 1]->space(), pp);
            g.Pmat[(size_t)i] = cvfem_ss::to_crs(pp);
            g.Rmat[(size_t)i] = g.Pmat[(size_t)i]->transpose();
        }
    }

    // Exact probe pattern for the first coarse level, derived rather than guessed.
    //
    // Probing needs a sparsity pattern up front, and an entry falling outside it is folded
    // into the wrong slot rather than dropped -- so a guess that is too narrow yields a
    // wrong matrix, which is why the old code guessed, checked, and widened to dense. There
    // is no need to guess: the fine operator's stencil is the fine node graph, so the
    // pattern of R*G*P with G the graph carrying unit values is a guaranteed superset of the
    // true Galerkin pattern (it can only over-estimate, under cancellation). Probing on a
    // superset is correct by construction.
    std::shared_ptr<GmgLevels::CRS_t> symbolic_rap_pattern(const std::shared_ptr<sfem::Function> &f_fine,
                                                           const std::shared_ptr<GmgLevels::CRS_t> &R,
                                                           const std::shared_ptr<GmgLevels::CRS_t> &P) {
        auto            graph = f_fine->space()->node_to_node_graph();
        const ptrdiff_t nn    = graph->rowptr()->size() - 1;
        const ptrdiff_t nnz   = graph->rowptr()->data()[nn];

        auto rp = smesh::create_host_buffer<sfem::count_t>((size_t)nn + 1);
        auto ci = smesh::create_host_buffer<sfem::idx_t>((size_t)nnz);
        auto va = smesh::create_host_buffer<real_t>((size_t)nnz);
        std::copy(graph->rowptr()->data(), graph->rowptr()->data() + nn + 1, rp->data());
        std::copy(graph->colidx()->data(), graph->colidx()->data() + nnz, ci->data());
        std::fill(va->data(), va->data() + nnz, real_t(1));

        auto G = sfem::h_crs_spmv<sfem::count_t, sfem::idx_t, real_t, real_t>(nn, nn, rp, ci, va, real_t(0));
        return sfem::rap(R, G, P);
    }

    void build_state_weights(GmgLevels &g) {
        const int nlevels = (int)g.ops.size();
        g.state_weights.assign((size_t)nlevels, {});
        for (int i = 1; i < nlevels; ++i) {
            const ptrdiff_t nf = g.data->functions[i - 1]->space()->n_dofs();
            const ptrdiff_t nc = g.data->functions[i]->space()->n_dofs();
            std::vector<real_t> ones((size_t)nf, real_t(1)), w((size_t)nc, real_t(0));
            g.data->restrictions[i - 1]->apply(ones.data(), w.data());
            g.state_weights[(size_t)i] = std::move(w);
        }
    }

    // Is the derefined coarse operator the same operator as one built directly on the
    // coarse space?
    //
    // The coarse operators come from derefine_op, walking down from the fine one. A driver
    // run at that refine level would instead construct the operator on that space from
    // scratch. Those two ought to be the same object, and if they are not, the hierarchy is
    // not solving a coarse version of the problem at all -- which would be a defect rather
    // than the known fact that rediscretisation differs from Galerkin coarsening.
    void check_derefined_op(GmgLevels &g) {
        auto            fs = g.data->functions[1]->space();
        const ptrdiff_t nd = fs->n_dofs();

        auto fresh = std::make_shared<sfem::CVFEMNavierStokes>(fs);
        fresh->rho             = g.level_ops[0]->rho;
        fresh->mu              = g.level_ops[0]->mu;
        fresh->rhie_chow_scale = g.level_ops[1]->rhie_chow_scale;
        fresh->geom            = g.level_ops[0]->geom;
        fresh->pack_size       = 0;
        if (fresh->initialize() != SFEM_SUCCESS) {
            std::printf("derefined-op check: could not build a fresh coarse operator\n");
            return;
        }

        std::vector<real_t> dir((size_t)nd), a((size_t)nd, 0), b((size_t)nd, 0);
        for (ptrdiff_t k = 0; k < nd; ++k) dir[(size_t)k] = std::sin(0.9 * (real_t)k) + 0.2;

        g.level_ops[1]->apply(g.states[1]->data(), dir.data(), a.data());
        fresh->apply(g.states[1]->data(), dir.data(), b.data());

        real_t dn[N_FIELDS] = {0}, rn[N_FIELDS] = {0};
        for (ptrdiff_t k = 0; k < nd; ++k) {
            const int    c = (int)(k % N_FIELDS);
            const real_t d = a[(size_t)k] - b[(size_t)k];
            dn[c] += d * d;
            rn[c] += a[(size_t)k] * a[(size_t)k];
        }
        const char *nm[N_FIELDS] = {"ux", "uy", "uz", "p"};
        std::printf("derefined vs freshly built coarse operator, rel:");
        for (int c = 0; c < N_FIELDS; ++c)
            std::printf("  %s %.3e", nm[c], (rn[c] > 0) ? std::sqrt(dn[c] / rn[c]) : 0.0);
        std::printf("\n");
    }

    // Exact coarse solve by dense LU.
    //
    // The coarsest level is solved, not smoothed, so the cycle takes its answer at face
    // value and an iterative method that fails there poisons everything above it. Measured
    // at N=2, L=16: BiCGStab on the 81-node coarsest operator diverges outright, its
    // residual going from 1.577 to 36066 over a hundred iterations, and it then returns
    // that amplified vector as the coarse-grid correction. The V-cycle amplified by 1e6 to
    // 1e9 per cycle, FGMRES could not converge, and Newton stepped from a badly solved
    // system to an answer three orders of magnitude wrong.
    //
    // At this size the question should not be asked of an iterative solver at all. The
    // coarsest level here is a few hundred unknowns and already stored dense, so a
    // factorisation is both exact and cheap, and it cannot diverge. The matrix is recovered
    // by applying the operator to unit vectors, which costs n small matrix-vector products
    // once per Newton step.
    // The coarse solve is rank-revealing, because the coarse operator need not have full rank.
    //
    // A pivot is small or large only relative to the matrix it came from, so the old absolute
    // `1e-300` test let a pivot of 1e-18 through in a matrix of scale one and back-substituted
    // an inverse of 1e18. On the backward-facing step that produced a coarse correction of
    // norm 1.6e12 from a fine residual of 5e-3, which the prolongation then added to the
    // solution -- the cycle's divergence at 1e19 per iteration was this and nothing else.
    //
    // Treating such a pivot as a null direction (y = 0 there) makes the solve a truncated
    // least-squares one: the coarse space's null components are simply not corrected, which is
    // the right thing for a multigrid coarse solve, since a null direction of A_H carries no
    // information about the fine residual. The dropped count is printed rather than swallowed;
    // a coarse operator that suddenly loses rank is a defect worth seeing.
    class DenseLU final : public sfem::Operator<real_t> {
    public:
        DenseLU(const ptrdiff_t n, std::vector<real_t> a) : n_(n), a_(std::move(a)), piv_((size_t)n) {
            for (ptrdiff_t i = 0; i < n_; ++i) piv_[(size_t)i] = i;
            real_t amax = 0;
            for (const real_t v : a_) amax = std::max(amax, std::fabs(v));
            const real_t ptol =
                    amax * (real_t)smesh::Env::read<double>("SFEM_COARSE_LU_TOL", 1e-14);
            for (ptrdiff_t k = 0; k < n_; ++k) {
                ptrdiff_t p = k;
                real_t    m = std::fabs(a_[(size_t)k * n_ + k]);
                for (ptrdiff_t i = k + 1; i < n_; ++i) {
                    const real_t v = std::fabs(a_[(size_t)i * n_ + k]);
                    if (v > m) { m = v; p = i; }
                }
                if (p != k) {
                    for (ptrdiff_t j = 0; j < n_; ++j)
                        std::swap(a_[(size_t)k * n_ + j], a_[(size_t)p * n_ + j]);
                    std::swap(piv_[(size_t)k], piv_[(size_t)p]);
                }
                real_t d = a_[(size_t)k * n_ + k];
                if (std::fabs(d) <= ptol) {
                    // Rank-deficient column: record it, zero the pivot so back-substitution
                    // takes y = 0 here, and leave the rest of the column alone.
                    a_[(size_t)k * n_ + k] = 0;
                    if (dropped_.size() < 32) dropped_.push_back(piv_[(size_t)k]);
                    ++n_dropped_;
                    continue;
                }
                for (ptrdiff_t i = k + 1; i < n_; ++i) {
                    const real_t f = a_[(size_t)i * n_ + k] / d;
                    a_[(size_t)i * n_ + k] = f;
                    if (f == real_t(0)) continue;
                    for (ptrdiff_t j = k + 1; j < n_; ++j)
                        a_[(size_t)i * n_ + j] -= f * a_[(size_t)k * n_ + j];
                }
            }
        }

        int apply(const real_t *const b, real_t *const x) override {
            std::vector<real_t> y((size_t)n_);
            for (ptrdiff_t i = 0; i < n_; ++i) {
                real_t s = b[piv_[(size_t)i]];
                for (ptrdiff_t j = 0; j < i; ++j) s -= a_[(size_t)i * n_ + j] * y[(size_t)j];
                y[(size_t)i] = s;
            }
            for (ptrdiff_t i = n_ - 1; i >= 0; --i) {
                real_t s = y[(size_t)i];
                for (ptrdiff_t j = i + 1; j < n_; ++j) s -= a_[(size_t)i * n_ + j] * y[(size_t)j];
                const real_t d = a_[(size_t)i * n_ + i];
                y[(size_t)i] = (d != real_t(0)) ? s / d : real_t(0);
            }
            for (ptrdiff_t i = 0; i < n_; ++i) x[i] += y[(size_t)i];
            return SFEM_SUCCESS;
        }

        ptrdiff_t rows() const override { return n_; }
        ptrdiff_t cols() const override { return n_; }
        sfem::ExecutionSpace execution_space() const override { return sfem::EXECUTION_SPACE_HOST; }

        // Number of coarse directions the factorisation could not resolve.
        ptrdiff_t                     n_dropped() const { return n_dropped_; }
        const std::vector<ptrdiff_t> &dropped() const { return dropped_; }

    private:
        ptrdiff_t              n_;
        std::vector<real_t>    a_;
        std::vector<ptrdiff_t> piv_;
        ptrdiff_t              n_dropped_{0};
        std::vector<ptrdiff_t> dropped_;
    };

    // Densify from the assembled matrix instead of applying it once per column.
    //
    // make_dense_lu below recovers the coarse operator by probing it with n unit vectors,
    // which is the same anti-pattern the coarse operators themselves no longer use: n
    // applications of an operator whose entries are already sitting in memory. It costs 832
    // applies on a 208-node coarsest level and showed up at 2.85% of a profiled V-cycle run,
    // and it grows as n * cost(apply) -- so it gets worse in exactly the regime where a
    // larger terminal problem is wanted. Reading the blocks is O(nnz) and does not touch the
    // operator at all.
    std::shared_ptr<DenseLU> make_dense_lu_from_bsr(const std::shared_ptr<GmgLevels::BSR_t> &a,
                                                    const ptrdiff_t                          n) {
        std::vector<real_t>        dense((size_t)n * (size_t)n, real_t(0));
        const sfem::count_t *const rp = a->row_ptr->data();
        const sfem::idx_t *const   ci = a->col_idx->data();
        const real_t *const        vd = a->values->data();
        const ptrdiff_t            nb = n / N_FIELDS;

        for (ptrdiff_t r = 0; r < nb; ++r)
            for (sfem::count_t k = rp[r]; k < rp[r + 1]; ++k) {
                const ptrdiff_t c = (ptrdiff_t)ci[k];
                for (int i = 0; i < N_FIELDS; ++i)
                    for (int j = 0; j < N_FIELDS; ++j)
                        dense[(size_t)(r * N_FIELDS + i) * (size_t)n + (size_t)(c * N_FIELDS + j)] =
                                vd[(size_t)k * 16 + (size_t)(i * N_FIELDS + j)];
            }
        return std::make_shared<DenseLU>(n, std::move(dense));
    }

    // Write a densified operator out so its spectrum can be examined offline.
    //
    // Component structure is what matters here -- which field a near-null mode lives in, and
    // whether it is smooth or oscillatory -- and that is far easier to read from an SVD than
    // to infer from residual norms. Gated on SFEM_GMG_DUMP_COARSE and off by default.
    static void dump_dense(const char *path, const ptrdiff_t n, const std::vector<real_t> &a) {
        FILE *f = std::fopen(path, "w");
        if (!f) {
            std::fprintf(stderr, "dump_dense: cannot open %s\n", path);
            return;
        }
        std::fprintf(f, "%td\n", n);
        for (ptrdiff_t i = 0; i < n; ++i) {
            for (ptrdiff_t j = 0; j < n; ++j)
                std::fprintf(f, "%.17g%c", (double)a[(size_t)i * n + j], (j + 1 == n) ? '\n' : ' ');
        }
        std::fclose(f);
        std::printf("dump_dense: wrote %td x %td to %s\n", n, n, path);
    }

    std::shared_ptr<DenseLU> make_dense_lu(const std::shared_ptr<sfem::Operator<real_t>> &op,
                                           const ptrdiff_t                                n) {
        std::vector<real_t> a((size_t)n * (size_t)n, real_t(0));
        std::vector<real_t> e((size_t)n, real_t(0)), col((size_t)n, real_t(0));
        for (ptrdiff_t j = 0; j < n; ++j) {
            std::fill(e.begin(), e.end(), real_t(0));
            std::fill(col.begin(), col.end(), real_t(0));
            e[(size_t)j] = real_t(1);
            op->apply(e.data(), col.data());
            for (ptrdiff_t i = 0; i < n; ++i) a[(size_t)i * n + j] = col[(size_t)i];
        }
        return std::make_shared<DenseLU>(n, std::move(a));
    }

    // Two-level coarse-grid correction, measured on one prescribed error mode.
    //
    // The cycle's rate decays to the smoother's own and the stalled residual is pressure,
    // so the question is narrow: given a smooth error the smoother cannot touch, does the
    // coarse grid reproduce it? This applies the textbook correction operator
    // P A_c^-1 R A to a chosen mode and reports what fraction of it survives. A working
    // coarse grid leaves little; a value near 1 means the correction is doing nothing for
    // that mode, and comparing a pressure mode against a velocity mode says whether the
    // failure is specific to the pressure equation.
    void check_cgc(GmgLevels &g, const int mode) {
        const ptrdiff_t nf = g.data->functions[0]->space()->n_dofs();
        const ptrdiff_t nc = g.data->functions[1]->space()->n_dofs();

        std::vector<real_t> e((size_t)nf, 0), r((size_t)nf, 0), rc((size_t)nc, 0),
                ec((size_t)nc, 0), ef((size_t)nf, 0);

        // The mode is built as P applied to a coarse field, not as a formula in the node
        // index. Node ids are not positions, so an index-based "smooth" mode need not be
        // smooth at all, and a rough mode is supposed to survive a coarse correction. A
        // mode in the range of the prolongation is exactly representable on the coarse
        // grid by construction, so a correct two-level correction must reproduce it almost
        // perfectly: with the Galerkin operator A_c = R A P the surviving fraction would be
        // zero. Whatever survives is the rediscretisation error, measured on the modes the
        // coarse grid is supposed to own.
        {
            std::vector<real_t> seed((size_t)nc, 0);
            unsigned            st = 7u;
            auto                rnd = [&st]() {
                st = st * 1103515245u + 12345u;
                return (real_t)((st >> 16) & 0x7fff) / (real_t)0x7fff - real_t(0.5);
            };
            // A random coarse field is oscillatory at the coarse scale, which is the
            // harshest case for a rediscretised operator. SFEM_GMG_CGC_SMOOTH seeds two
            // levels down and prolongs, giving a field that is smooth relative to the
            // coarse grid -- the case rediscretisation is actually supposed to handle. If
            // the correction fails on that too, the verdict does not rest on an unfair test.
            if (smesh::Env::read<int>("SFEM_GMG_CGC_SMOOTH", 0) && (int)g.ops.size() > 2) {
                const ptrdiff_t n2 = g.data->functions[2]->space()->n_dofs();
                std::vector<real_t> s2((size_t)n2, 0);
                for (ptrdiff_t i = 0; i < n2 / N_FIELDS; ++i)
                    s2[(size_t)i * N_FIELDS + (mode == 3 ? 3 : 0)] = rnd();
                g.data->functions[2]->apply_zero_constraints(s2.data());
                g.data->prolongations[2]->apply(s2.data(), seed.data());
            } else {
                for (ptrdiff_t i = 0; i < nc / N_FIELDS; ++i)
                    seed[(size_t)i * N_FIELDS + (mode == 3 ? 3 : 0)] = rnd();
            }
            g.data->functions[1]->apply_zero_constraints(seed.data());
            g.data->prolongations[1]->apply(seed.data(), e.data());
        }
        g.data->functions[0]->apply_zero_constraints(e.data());

        g.ops[0]->apply(e.data(), r.data());
        g.data->restrictions[0]->apply(r.data(), rc.data());

        // Two coarse operators, same everything else. `galerkin` builds R A P explicitly by
        // composing the transfers with the fine operator -- far too expensive for
        // production, and exactly the right thing for a diagnostic, because with it the
        // surviving fraction is zero by construction if the transfers are sound. Comparing
        // the two separates a wrong rediscretisation from wrong transfers, which nothing
        // measured so far has been able to do.
        const bool use_galerkin = smesh::Env::read<int>("SFEM_GMG_GALERKIN", 0) != 0;
        auto       coarse_op    = g.ops[1];
        if (use_galerkin) {
            auto Pop = g.data->prolongations[1];
            auto Rop = g.data->restrictions[0];
            auto Af  = g.ops[0];
            coarse_op = sfem::make_op<real_t>(
                    nc, nc,
                    [Pop, Rop, Af, nf, nc](const real_t *const xc, real_t *const yc) {
                        std::vector<real_t> t1((size_t)nf, 0), t2((size_t)nf, 0);
                        Pop->apply(xc, t1.data());
                        Af->apply(t1.data(), t2.data());
                        Rop->apply(t2.data(), yc);
                    },
                    sfem::EXECUTION_SPACE_HOST);
        }

        auto cs = sfem::create_bcgs<real_t>(coarse_op, sfem::EXECUTION_SPACE_HOST);
        cs->set_max_it(500);
        cs->set_rtol(1e-10);
        cs->verbose = false;
        cs->apply(rc.data(), ec.data());

        g.data->prolongations[1]->apply(ec.data(), ef.data());

        real_t ne[N_FIELDS] = {0}, nd[N_FIELDS] = {0};
        for (ptrdiff_t k = 0; k < nf; ++k) {
            const int    c = (int)(k % N_FIELDS);
            const real_t d = e[(size_t)k] - ef[(size_t)k];
            ne[c] += e[(size_t)k] * e[(size_t)k];
            nd[c] += d * d;
        }
        const char *nm[N_FIELDS] = {"ux", "uy", "uz", "p"};
        std::printf("cgc [%s] on %s mode: surviving fraction",
                    use_galerkin ? "galerkin" : "rediscretised", mode == 3 ? "pressure" : "velocity");
        for (int c = 0; c < N_FIELDS; ++c)
            if (ne[c] > 0) std::printf("  %s %.4f", nm[c], std::sqrt(nd[c] / ne[c]));
        std::printf("\n");
    }

    // Assembles the Galerkin coarse operator A_c = R A P into BSR, once per Newton step.
    //
    // Composing R A P at solve time works -- it is what SFEM_GMG_GALERKIN=1 measures -- but
    // puts fine-level work under every coarse application, which is precisely what a
    // hierarchy exists to avoid. Assembling it instead pays that cost once per Newton step
    // and leaves the cycle applying a sparse matrix, so a coarse level never reaches back
    // up to a finer one during the solve.
    //
    // Assembly also fixes the other half of the problem. A coarse smoother needs the
    // diagonal of the operator it smooths, and the matrix-free composite cannot supply one;
    // using the rediscretised diagonal instead mismatches the Galerkin operator by the
    // per-block scale factors (about 1.6 in velocity and 8 in pressure) and makes the
    // coarse smoother diverge. An assembled matrix hands over its own diagonal.
    //
    // The entries are recovered by probing. With a distance-2 colouring of the coarse node
    // graph, no node has two neighbours of the same colour, so one application per colour
    // and component reveals a whole set of blocks at once: colours x 4 applications rather
    // than one per coarse degree of freedom.
    std::shared_ptr<GmgLevels::BSR_t> g_last_assembled;  // set by assemble_galerkin

    std::shared_ptr<sfem::Operator<real_t>> assemble_galerkin(const std::shared_ptr<sfem::Function>         &f_coarse,
                                                              const std::shared_ptr<sfem::Operator<real_t>> &A_above,
                                                              const std::shared_ptr<sfem::Operator<real_t>> &P,
                                                              const std::shared_ptr<sfem::Operator<real_t>> &R,
                                                              const ptrdiff_t                                n_fine,
                                                              std::vector<real_t>                           *diag_out,
                                                              const GmgLevels::CRS_t *const                  pattern = nullptr) {
        auto            graph = f_coarse->space()->node_to_node_graph();
        const ptrdiff_t nn    = f_coarse->space()->n_dofs() / N_FIELDS;
        const count_t *const g_rp = graph->rowptr()->data();
        const idx_t *const   g_ci = graph->colidx()->data();

        // The pattern is the coarse mesh graph, widened if that turns out to be too narrow.
        //
        // Probing recovers A_c(i,j) only for j inside the pattern being probed; a non-zero
        // of R A P outside it is not dropped but folded into the wrong entry, so a pattern
        // that is too narrow yields a wrong matrix rather than an approximate one. The mesh
        // graph is right while the coarse mesh is fine enough that R A P does not reach past
        // it, and stops being right on the coarsest levels, where a few nodes are all within
        // reach of each other. `widen` squares the adjacency; at the second retry the level
        // is small enough that a dense pattern costs nothing.
        std::vector<std::vector<idx_t>> adj((size_t)nn);
        auto build_pattern = [&](const int widen) {
            for (ptrdiff_t i = 0; i < nn; ++i) {
                std::vector<idx_t> row(g_ci + g_rp[i], g_ci + g_rp[i + 1]);
                if (widen == 2) {
                    row.clear();
                    for (ptrdiff_t j = 0; j < nn; ++j) row.push_back((idx_t)j);
                } else if (widen == 1) {
                    for (count_t a = g_rp[i]; a < g_rp[i + 1]; ++a) {
                        const idx_t j = g_ci[a];
                        row.insert(row.end(), g_ci + g_rp[j], g_ci + g_rp[j + 1]);
                    }
                    std::sort(row.begin(), row.end());
                    row.erase(std::unique(row.begin(), row.end()), row.end());
                }
                adj[(size_t)i] = std::move(row);
            }
        };

        std::shared_ptr<sfem::Operator<real_t>> assembled;
        for (int attempt = 0; attempt < 3; ++attempt) {
        if (pattern) {
            // Derived pattern: exact by construction, so there is nothing to retry.
            const sfem::count_t *const prp = pattern->row_ptr->data();
            const sfem::idx_t *const   pci = pattern->col_idx->data();
            for (ptrdiff_t i = 0; i < nn; ++i)
                adj[(size_t)i].assign(pci + prp[i], pci + prp[i + 1]);
        } else {
            build_pattern(attempt);
        }
        std::vector<count_t> rpv((size_t)nn + 1, 0);
        for (ptrdiff_t i = 0; i < nn; ++i) rpv[(size_t)i + 1] = rpv[(size_t)i] + (count_t)adj[(size_t)i].size();
        std::vector<idx_t> civ;
        civ.reserve((size_t)rpv[(size_t)nn]);
        for (ptrdiff_t i = 0; i < nn; ++i) civ.insert(civ.end(), adj[(size_t)i].begin(), adj[(size_t)i].end());
        const count_t *const rp  = rpv.data();
        const idx_t *const   ci  = civ.data();
        const ptrdiff_t      nnz = rp[nn];

        // Greedy distance-2 colouring: two nodes sharing a neighbour must differ, so that a
        // probe on one colour never mixes two contributions into the same row.
        std::vector<int> color((size_t)nn, -1);
        {
            std::vector<int> used;
            for (ptrdiff_t i = 0; i < nn; ++i) {
                used.assign(64, 0);
                for (count_t a = rp[i]; a < rp[i + 1]; ++a) {
                    const idx_t j = ci[a];
                    if (color[(size_t)j] >= 0) {
                        if ((size_t)color[(size_t)j] >= used.size()) used.resize((size_t)color[(size_t)j] + 1, 0);
                        used[(size_t)color[(size_t)j]] = 1;
                    }
                    for (count_t b = rp[j]; b < rp[j + 1]; ++b) {
                        const idx_t k = ci[b];
                        if (color[(size_t)k] >= 0) {
                            if ((size_t)color[(size_t)k] >= used.size()) used.resize((size_t)color[(size_t)k] + 1, 0);
                            used[(size_t)color[(size_t)k]] = 1;
                        }
                    }
                }
                int c = 0;
                while (c < (int)used.size() && used[(size_t)c]) ++c;
                color[(size_t)i] = c;
            }
        }
        const int ncolors = 1 + *std::max_element(color.begin(), color.end());

        auto rowptr = smesh::create_host_buffer<count_t>((size_t)nn + 1);
        auto colidx = smesh::create_host_buffer<idx_t>((size_t)nnz);
        auto values = smesh::create_host_buffer<real_t>((size_t)nnz * 16);
        std::copy(rp, rp + nn + 1, rowptr->data());
        std::copy(ci, ci + nnz, colidx->data());
        std::fill(values->data(), values->data() + (size_t)nnz * 16, real_t(0));

        const ptrdiff_t     ndc = nn * N_FIELDS;
        std::vector<real_t> v((size_t)ndc), y((size_t)ndc), t1((size_t)n_fine), t2((size_t)n_fine);

        // With null transfers this assembles A itself rather than R A P, which is how the
        // fine level gets a matrix. Worth having for its own sake: a matrix-free apply
        // accumulates with atomics, so its summation order changes with the thread count and
        // the solve is not reproducible; a BSR apply accumulates each row in one thread and
        // is deterministic however many threads are used.
        auto composite = [&](const real_t *const xc, real_t *const yc) {
            if (P && R) {
                std::fill(t1.begin(), t1.end(), real_t(0));
                std::fill(t2.begin(), t2.end(), real_t(0));
                P->apply(xc, t1.data());
                A_above->apply(t1.data(), t2.data());
                R->apply(t2.data(), yc);
            } else {
                A_above->apply(xc, yc);
            }
        };

        for (int c = 0; c < ncolors; ++c) {
            for (int b = 0; b < N_FIELDS; ++b) {
                std::fill(v.begin(), v.end(), real_t(0));
                for (ptrdiff_t j = 0; j < nn; ++j)
                    if (color[(size_t)j] == c) v[(size_t)j * N_FIELDS + b] = real_t(1);

                std::fill(y.begin(), y.end(), real_t(0));
                composite(v.data(), y.data());

                for (ptrdiff_t i = 0; i < nn; ++i)
                    for (count_t a = rp[i]; a < rp[i + 1]; ++a)
                        if (color[(size_t)ci[a]] == c)
                            for (int r = 0; r < N_FIELDS; ++r)
                                values->data()[(size_t)a * 16 + (size_t)r * N_FIELDS + b] =
                                        y[(size_t)i * N_FIELDS + r];
            }
        }

        // Both transfers zero constrained degrees of freedom on output, so the assembled
        // rows for those are empty and the matrix would be singular. Restore the identity
        // rows the constrained system actually has.
        {
            std::vector<mask_t> m(mask_count(f_coarse->space()->n_dofs()), 0);
            f_coarse->constraints_mask(m.data());
            for (ptrdiff_t i = 0; i < nn; ++i)
                for (int r = 0; r < N_FIELDS; ++r) {
                    if (!mask_get(i * N_FIELDS + r, m.data())) continue;
                    for (count_t a = rp[i]; a < rp[i + 1]; ++a)
                        for (int cc = 0; cc < N_FIELDS; ++cc)
                            values->data()[(size_t)a * 16 + (size_t)r * N_FIELDS + cc] =
                                    (ci[a] == i && cc == r) ? real_t(1) : real_t(0);
                }
        }

        if (diag_out) {
            diag_out->assign((size_t)nn * 16, real_t(0));
            for (ptrdiff_t i = 0; i < nn; ++i)
                for (count_t a = rp[i]; a < rp[i + 1]; ++a)
                    if (ci[a] == i)
                        std::copy(values->data() + (size_t)a * 16, values->data() + (size_t)a * 16 + 16,
                                  diag_out->data() + (size_t)i * 16);
        }

        auto assembled_bsr = sfem::h_bsr_spmv<count_t, idx_t, real_t, real_t>(nn, nn, N_FIELDS, rowptr,
                                                                              colidx, values, real_t(0));
        assembled          = assembled_bsr;
        g_last_assembled   = assembled_bsr;
        bool gate_ok       = false;

        // Gate: the assembled matrix must reproduce the composite R A P it was probed from.
        // Probing is only valid if every non-zero of R A P falls inside the pattern being
        // probed; anything outside it lands in the wrong row and is silently absorbed.
        {
            std::vector<real_t> v((size_t)ndc), ya((size_t)ndc, 0), yb((size_t)ndc, 0);
            unsigned            st = 991u;
            for (ptrdiff_t k = 0; k < ndc; ++k) {
                st = st * 1103515245u + 12345u;
                v[(size_t)k] = (real_t)((st >> 16) & 0x7fff) / (real_t)0x7fff - real_t(0.5);
            }
            f_coarse->apply_zero_constraints(v.data());

            assembled->apply(v.data(), ya.data());
            composite(v.data(), yb.data());

            real_t dn = 0, rnv = 0;
            for (ptrdiff_t k = 0; k < ndc; ++k) {
                const real_t d = ya[(size_t)k] - yb[(size_t)k];
                dn += d * d;
                rnv += yb[(size_t)k] * yb[(size_t)k];
            }
            const real_t rel = (rnv > 0) ? std::sqrt(dn / rnv) : 0.0;
            gate_ok          = rel < 1e-10;
            if (gate_ok || attempt == 2)
                std::printf("assembly gate [%s]: rel = %.4e  %s\n", (P && R) ? "RAP" : "A", rel,
                            gate_ok ? "OK" : "MISMATCH");
        }

        if (gate_ok || attempt == 2 || pattern) {
            std::printf("galerkin assembly: %td nodes, %td blocks, %d colours, %d applications%s\n",
                        nn, nnz, ncolors, ncolors * N_FIELDS,
                        pattern ? "  (derived pattern)"
                                : (attempt == 0 ? "" : (attempt == 1 ? "  (widened pattern)" : "  (dense pattern)")));
            return assembled;
        }
        }  // attempt

        return assembled;
    }

    void refresh_gmg(GmgLevels &g) {
        const int nlevels = (int)g.ops.size();

        // SFEM_GMG_CONST_STATE=1: a diagnostic that removes the state transfer as a
        // variable. Every level is given the *same* constant field, which averaging and an
        // L2 projection reproduce identically, so a directly-assembled coarse operator and
        // the Galerkin one are then evaluated at genuinely the same state. Whatever gap
        // survives is the discretisation, not the state -- which is the thing an L2
        // projection could and could not fix, respectively.
        if (smesh::Env::read<int>("SFEM_GMG_CONST_STATE", 0)) {
            const real_t cu = smesh::Env::read<real_t>("SFEM_GMG_CONST_U", real_t(1));
            for (int i = 0; i < nlevels; ++i) {
                const ptrdiff_t nd = g.data->functions[i]->space()->n_dofs();
                real_t *const   xs = g.states[i]->data();
                for (ptrdiff_t k = 0; k < nd; k += N_FIELDS) {
                    xs[k + 0] = cu;
                    xs[k + 1] = 0;
                    xs[k + 2] = 0;
                    xs[k + 3] = 0;
                }
            }
        } else
        for (int i = 1; i < nlevels; ++i) {
            g.data->restrictions[i - 1]->apply(g.states[i - 1]->data(), g.states[i]->data());

            const auto     &w  = g.state_weights[(size_t)i];
            real_t *const   sc = g.states[i]->data();
            const ptrdiff_t nc = g.data->functions[i]->space()->n_dofs();
            // Constrained dofs come back zeroed by the transfer and so does their weight;
            // apply_constraints below writes the boundary values over them regardless.
            for (ptrdiff_t k = 0; k < nc; ++k)
                if (w[(size_t)k] > real_t(1e-12)) sc[(size_t)k] /= w[(size_t)k];

            g.data->functions[i]->apply_constraints(g.states[i]->data());
        }

        g.mg          = sfem::h_mg<real_t>();
        g.mg->verbose = false;

        // SFEM_GMG_PFILTER=1: strip the constant pressure mode from each prolonged
        // correction.
        //
        // Pressure here is fixed only by a single pin, so it is determined up to a
        // constant and each level's pin is its own gauge. Nothing makes the coarse pin the
        // same physical node as the fine one, so a coarse pressure correction can arrive
        // carrying an arbitrary constant offset. That constant is a near-null mode of the
        // fine operator, which is precisely what the smoother is worst at removing, so it
        // accumulates from cycle to cycle instead of being damped.
        const bool pfilter = smesh::Env::read<int>("SFEM_GMG_PFILTER", 0) != 0;

        // SFEM_GMG_CGC scales the prolonged coarse-grid correction. The rediscretised
        // coarse operator measures about six times the Galerkin operator R A P that the
        // transfers imply (see the coarse-op line under SFEM_GMG_CHECK=1), so its inverse
        // returns a correction scaled by the reciprocal of that. This is the knob that
        // says whether the mismatch is a single scalar per level -- in which case one
        // factor repairs the cycle -- or a genuine difference in what the two operators
        // do, which no scalar can fix.
        const real_t cgc    = smesh::Env::read<real_t>("SFEM_GMG_CGC", real_t(1));
        const real_t pscale   = smesh::Env::read<real_t>("SFEM_GMG_PSCALE", real_t(1));
        // 0 rediscretised, 1 Galerkin composed matrix-free (diagnostic), 2 Galerkin
        // assembled once per Newton step (the usable form).
        const int  galerkin_mode = smesh::Env::read<int>("SFEM_GMG_GALERKIN", 0);
        const bool galerkin      = galerkin_mode == 1;
        std::shared_ptr<sfem::Operator<real_t>> level_op_below;
        std::vector<real_t>                     galerkin_diag;

        // The whole element-wise hierarchy, built in one pass before the level loop.
        //
        // Level 1 comes from the micro-cell matrices; every level below is one element-local
        // coarsening hop from the level above, masking that level's columns with its own
        // constraints. Those constraints already exist -- create_gmg_data derefines the
        // Function at each level -- which is what makes the hops reproduce the composite the
        // transfers define (Z_i Rhat A_{i-1} Z_{i-1} Phat) instead of skipping the
        // intermediate Z's, the omission that gave wrong block diagonals when every level was
        // built straight from level 0.
        cvfem_ss::CoarseHierarchy egal_hier;
        const bool                egal_on = smesh::Env::read<int>("SFEM_GMG_EGAL", 1) && galerkin_mode == 2 &&
                                            g.level_ops[0] && g.level_ops[0]->is_semi_structured();
        if (egal_on) {
            const double t_h = smesh::time_seconds();
            std::vector<std::shared_ptr<sfem::FunctionSpace>> spaces;
            std::vector<std::vector<uint8_t>>                 masks;
            for (int l = 0; l < nlevels; ++l) {
                spaces.push_back(g.data->functions[l]->space());
                const ptrdiff_t     nd = spaces.back()->n_dofs();
                std::vector<mask_t> m(mask_count(nd), 0);
                g.data->functions[l]->constraints_mask(m.data());
                std::vector<uint8_t> b((size_t)nd, 0);
                for (ptrdiff_t k = 0; k < nd; ++k) b[(size_t)k] = mask_get(k, m.data()) ? 1 : 0;
                masks.push_back(std::move(b));
            }
                        // SFEM_GMG_EGAL_EM keeps every level but the coarsest as element matrices, so
            // nothing above the level that is factorised is ever assembled. The hops coarsen
            // element matrices to element matrices, so this costs no extra construction -- it
            // stops at the element form instead of going on to a BSR.
            egal_hier = cvfem_ss::assemble_hierarchy(*g.level_ops[0], g.states[0]->data(), spaces, masks,
                                                     smesh::Env::read<int>("SFEM_GMG_EGAL_EM", 0) != 0);
            phase_add("galerkin_assembly", smesh::time_seconds() - t_h);
        }
        auto       wrap_p  = [&](const int i) -> std::shared_ptr<sfem::Operator<real_t>> {
            auto P = g.data->prolongations[i];
            if (!P || (!pfilter && cgc == real_t(1))) return P;
            const ptrdiff_t nf = g.data->functions[i - 1]->space()->n_dofs();
            return sfem::make_op<real_t>(
                    P->rows(), P->cols(),
                    [P, nf, pfilter, cgc](const real_t *const from, real_t *const to) {
                        P->apply(from, to);
                        if (cgc != real_t(1))
                            for (ptrdiff_t k = 0; k < nf; ++k) to[k] *= cgc;
                        if (!pfilter) return;
                        real_t    sum = 0;
                        ptrdiff_t cnt = 0;
                        for (ptrdiff_t k = 3; k < nf; k += N_FIELDS) {
                            sum += to[k];
                            ++cnt;
                        }
                        if (!cnt) return;
                        const real_t mean = sum / (real_t)cnt;
                        for (ptrdiff_t k = 3; k < nf; k += N_FIELDS) to[k] -= mean;
                    },
                    sfem::EXECUTION_SPACE_HOST);
        };
        for (int i = 0; i < nlevels; ++i) {
            auto            fi = g.data->functions[i];
            const ptrdiff_t nn = fi->space()->n_dofs() / N_FIELDS;
            std::vector<mask_t> mask(mask_count(fi->space()->n_dofs()), 0);
            fi->constraints_mask(mask.data());
            const real_t omega = (i + 1 < nlevels)
                                         ? smesh::Env::read<real_t>("SFEM_GMG_OMEGA", real_t(0.35))
                                         : real_t(1);
            auto lop = g.ops[i];

            // SFEM_GMG_GALERKIN=1: use R A P as the coarse operator instead of the
            // rediscretised one, composed matrix-free and recursively, so level i applies
            // the level i-1 operator between its transfers.
            //
            // This is deliberately the expensive form. Every coarse application reaches all
            // the way up to the fine level, which is exactly what a hierarchy exists to
            // avoid, so it is not a solution -- it is the experiment that says whether
            // Galerkin coarsening fixes the cycle before any effort is spent making it
            // affordable. If it does, the affordable version is to assemble these operators
            // once per Newton step and apply them as sparse matrices.
            if (i > 0 && galerkin) {
                auto Pop   = g.data->prolongations[i];
                auto Rop   = g.data->restrictions[i - 1];
                auto below = level_op_below;           // already Galerkin for i-1
                const ptrdiff_t nfine = g.data->functions[i - 1]->space()->n_dofs();
                const ptrdiff_t ncrs  = fi->space()->n_dofs();
                lop = sfem::make_op<real_t>(
                        ncrs, ncrs,
                        [Pop, Rop, below, nfine](const real_t *const xc, real_t *const yc) {
                            std::vector<real_t> t1((size_t)nfine, 0), t2((size_t)nfine, 0);
                            Pop->apply(xc, t1.data());
                            below->apply(t1.data(), t2.data());
                            Rop->apply(t2.data(), yc);
                        },
                        sfem::EXECUTION_SPACE_HOST);
            }

            if (i > 0 && galerkin_mode == 2) {
                const double t_asm = smesh::time_seconds();

                // Element-wise Galerkin, when the fine operator is semi-structured: every
                // level is built straight from the fine macro-elements as sum_e P_e^T A_e P_e.
                //
                // This replaces both branches below. Nothing is probed, so no level pays 108
                // to 192 operator applications and no pattern has to be guessed -- the one
                // here is derived from the lattice and is exact, which removes the failure
                // mode where an entry outside a too-narrow guess is folded into the wrong
                // slot instead of being dropped. Levels are built directly rather than
                // chained, so no error accumulates through repeated triple products; the
                // SFEM_GMG_CHECK=1 gates measure both claims at 2e-16.
                // Element-wise Galerkin replaces the *probe*, and the probe was only ever
                // needed at level 1 -- below that the level above is already a matrix and the
                // triple product is exact and cheap. Restricting it to level 1 is not a
                // limitation of the construction but of the constraint treatment: the
                // hierarchy's transfers zero constrained dofs at every hop, so level 2 built
                // by chaining coarsens a level-1 matrix that already carries identity rows,
                // and those rows contribute to the product. Building level 2 straight from
                // level 0 cannot see them. The operators still agree to 2e-16 on unconstrained
                // columns, which is why this was invisible until the block diagonals were
                // compared -- they differed by 1.2e-2 at level 2 and 5.7e-2 at level 3, and a
                // smoother given wrong diagonals cost the solve its convergence (40 Newton
                // steps against 27) while every operator gate still passed.
                const bool egal = egal_on && egal_hier.op[(size_t)i];

                if (egal) {

                    // Identity rows and the block diagonal are already applied to whichever
                    // form backs this level -- assembled BSR, or element matrices kept as they
                    // are -- so from here the two are interchangeable.
                    auto a_c      = egal_hier.op[(size_t)i];
                    galerkin_diag = egal_hier.diag[(size_t)i];

                    // Cross-check against the construction it replaces (SFEM_GMG_CHECK).
                    //
                    // The gates in build_gmg compare the two constructions unconstrained.
                    // This is the constrained comparison, and the probe is the right
                    // reference for it: it recovers the composite including the transfers'
                    // zero-constraint wrapping, which is the thing the column mask and the
                    // identity-row patch here are meant to reproduce. Level 1 probes the
                    // matrix-free fine operator, so it is the true composite; below that the
                    // probe sees the element-wise matrix above, making this the direct
                    // against the chained construction with constraints in place.
                    if (smesh::Env::read<int>("SFEM_GMG_CHECK", 0)) {
                        auto saved = g_last_assembled;
                        std::shared_ptr<GmgLevels::CRS_t> pat;
                        if (i == 1 && g.Rmat[1] && g.Pmat[1] &&
                            smesh::Env::read<int>("SFEM_GMG_SYMBOLIC_PATTERN", 1))
                            pat = symbolic_rap_pattern(g.data->functions[0], g.Rmat[1], g.Pmat[1]);
                        std::vector<real_t> pdiag;
                        assemble_galerkin(fi, level_op_below, g.data->prolongations[i],
                                          g.data->restrictions[i - 1],
                                          g.data->functions[i - 1]->space()->n_dofs(), &pdiag, pat.get());
                        auto probed = g_last_assembled;
                        g_last_assembled = saved;

                        // The constrained rows, and the block diagonal, deliberately left out
                        // of the comparison below. Both feed the smoother, so a difference
                        // there does not show up as a wrong coarse operator but as a worse
                        // cycle -- which is exactly the symptom that sent this back here.
                        {
                            real_t dd = 0, dr = 0;
                            for (size_t k = 0; k < pdiag.size() && k < galerkin_diag.size(); ++k) {
                                const real_t d = galerkin_diag[k] - pdiag[k];
                                dd += d * d;
                                dr += pdiag[k] * pdiag[k];
                            }
                            const real_t rd = dr > 0 ? std::sqrt(dd / dr) : std::sqrt(dd);
                            std::printf("egal diag  %d: rel |diag_egal - diag_probe| = %.4e  %s (n %zu vs %zu)\n",
                                        i, rd, rd < 1e-10 ? "OK" : "MISMATCH", galerkin_diag.size(),
                                        pdiag.size());
                        }

                        const ptrdiff_t ndc = fi->space()->n_dofs();
                        std::vector<real_t> v((size_t)ndc), ya((size_t)ndc, 0), yb((size_t)ndc, 0);
                        for (ptrdiff_t k = 0; k < ndc; ++k)
                            v[(size_t)k] = std::sin(real_t(0.37) * (real_t)k + real_t(1.1));
                        fi->apply_zero_constraints(v.data());
                        a_c->apply(v.data(), ya.data());
                        probed->apply(v.data(), yb.data());
                        // Whole-vector comparison, constrained rows included. The earlier
                        // version zeroed them on both sides, copying the rap gate, and that
                        // hid whatever lives there.
                        real_t dn = 0, rn = 0;
                        for (ptrdiff_t k = 0; k < ndc; ++k) {
                            const real_t d = ya[(size_t)k] - yb[(size_t)k];
                            dn += d * d;
                            rn += yb[(size_t)k] * yb[(size_t)k];
                        }
                        const real_t rel = rn > 0 ? std::sqrt(dn / rn) : std::sqrt(dn);
                        std::printf("egal level %d: %s (probe %td blocks)  vs probed composite rel = %.4e  %s\n", i,
                                    egal_hier.A[(size_t)i]
                                            ? (std::to_string((long long)egal_hier.A[(size_t)i]->col_idx->size()) +
                                               " blocks")
                                                      .c_str()
                                            : "element matrices",
                                    (ptrdiff_t)probed->col_idx->size(), rel, rel < 1e-10 ? "OK" : "MISMATCH");
                    }

                    g.Amat[(size_t)i] = egal_hier.A[(size_t)i];  // null when kept as element matrices
                    lop               = a_c;
                } else
                // Level 1 is the only level that must be probed: the operator above it is
                // matrix-free and has no matrix form. Below that the level above IS a
                // matrix, so the Galerkin operator is a sparse triple product -- exact
                // sparsity, no colouring, no pattern guess, no dense fallback.
                if (i == 1 || !g.Amat[(size_t)i - 1] || galerkin_mode != 2 ||
                    smesh::Env::read<int>("SFEM_GMG_RAP", 1) == 0) {
                    std::shared_ptr<GmgLevels::CRS_t> pat;
                    if (i == 1 && g.Rmat[1] && g.Pmat[1] &&
                        smesh::Env::read<int>("SFEM_GMG_SYMBOLIC_PATTERN", 1))
                        pat = symbolic_rap_pattern(g.data->functions[0], g.Rmat[1], g.Pmat[1]);

                    lop = assemble_galerkin(fi, level_op_below, g.data->prolongations[i],
                                            g.data->restrictions[i - 1],
                                            g.data->functions[i - 1]->space()->n_dofs(), &galerkin_diag,
                                            pat.get());
                    g.Amat[(size_t)i] = g_last_assembled;
                } else {
                    const ptrdiff_t nd_up = g.data->functions[i - 1]->space()->n_dofs();
                    std::vector<mask_t> mup(mask_count(nd_up), 0);
                    g.data->functions[i - 1]->constraints_mask(mup.data());

                    auto masked = mask_block_columns(g.Amat[(size_t)i - 1], mup.data());
                    auto a_c    = sfem::rap(g.Rmat[(size_t)i], masked, g.Pmat[(size_t)i]);
                    patch_identity_rows(a_c, mask.data());

                    g.Amat[(size_t)i] = a_c;
                    lop               = a_c;

                    // Block diagonal for the smoother: search for the diagonal, since mm
                    // emits unsorted column indices.
                    galerkin_diag.assign((size_t)nn * 16, real_t(0));
                    {
                        const sfem::count_t *const rp = a_c->row_ptr->data();
                        const sfem::idx_t *const   ci = a_c->col_idx->data();
                        const real_t *const        vd = a_c->values->data();
                        for (ptrdiff_t r = 0; r < nn; ++r)
                            for (sfem::count_t k = rp[r]; k < rp[r + 1]; ++k)
                                if (ci[k] == r)
                                    std::copy(vd + (size_t)k * 16, vd + (size_t)k * 16 + 16,
                                              galerkin_diag.data() + (size_t)r * 16);
                    }

                    if (smesh::Env::read<int>("SFEM_GMG_CHECK", 0) && !fi->space()->has_semi_structured_mesh()) {
                        // How far is a directly-assembled (rediscretised) operator from the
                        // Galerkin one at this level?
                        //
                        // hessian_bsr refuses on the semi-structured path but works on an
                        // unstructured level, so the coarsest level could be assembled
                        // outright and the probe dropped. That trades Galerkin for
                        // rediscretisation, which measured badly higher up (5.52 against
                        // 0.000 on a coarse-representable mode) -- but that was with a
                        // partition-of-unity averaged state. This reports the gap on the
                        // real coarse operator, per component, before any effort goes into a
                        // better state transfer.
                        auto direct = sfem::create_linear_operator(sfem::op_type::BSR, fi, g.states[i],
                                                                   sfem::EXECUTION_SPACE_HOST);
                        const ptrdiff_t     ndc = nn * N_FIELDS;
                        std::vector<real_t> v((size_t)ndc), yg((size_t)ndc, 0), yd((size_t)ndc, 0);
                        unsigned            st = 17u;
                        for (auto &e : v) {
                            st = st * 1103515245u + 12345u;
                            e  = (real_t)((st >> 16) & 0x7fff) / (real_t)0x7fff - real_t(0.5);
                        }
                        fi->apply_zero_constraints(v.data());
                        a_c->apply(v.data(), yg.data());
                        direct->apply(v.data(), yd.data());
                        fi->apply_zero_constraints(yg.data());
                        fi->apply_zero_constraints(yd.data());

                        real_t dn[N_FIELDS] = {0}, rn[N_FIELDS] = {0};
                        for (ptrdiff_t k = 0; k < ndc; ++k) {
                            const int    c = (int)(k % N_FIELDS);
                            const real_t d = yd[(size_t)k] - yg[(size_t)k];
                            dn[c] += d * d;
                            rn[c] += yg[(size_t)k] * yg[(size_t)k];
                        }
                        // Separate a scale mismatch from a structural one. If the direct
                        // operator is close to a constant multiple of the Galerkin one, the
                        // difference is the h-dependent stabilisation and no state transfer
                        // can touch it. If a residual survives removing the best-fit scale,
                        // the two operators genuinely differ.
                        real_t ab[N_FIELDS] = {0}, aa[N_FIELDS] = {0};
                        for (ptrdiff_t k = 0; k < ndc; ++k) {
                            const int c = (int)(k % N_FIELDS);
                            ab[c] += yd[(size_t)k] * yg[(size_t)k];
                            aa[c] += yd[(size_t)k] * yd[(size_t)k];
                        }
                        const char *nm[N_FIELDS] = {"ux", "uy", "uz", "p"};
                        std::printf("  direct vs galerkin, level %d   raw:", i);
                        for (int c = 0; c < N_FIELDS; ++c)
                            std::printf("  %s %8.3f", nm[c], (rn[c] > 0) ? std::sqrt(dn[c] / rn[c]) : 0.0);
                        std::printf("\n                                scale:");
                        for (int c = 0; c < N_FIELDS; ++c)
                            std::printf("  %s %8.4f", nm[c], (aa[c] > 0) ? ab[c] / aa[c] : 0.0);
                        std::printf("\n                          after-scale:");
                        for (int c = 0; c < N_FIELDS; ++c) {
                            const real_t sc = (aa[c] > 0) ? ab[c] / aa[c] : 0.0;
                            real_t       rr = 0;
                            for (ptrdiff_t k = c; k < ndc; k += N_FIELDS) {
                                const real_t d = sc * yd[(size_t)k] - yg[(size_t)k];
                                rr += d * d;
                            }
                            std::printf("  %s %8.4f", nm[c], (rn[c] > 0) ? std::sqrt(rr / rn[c]) : 0.0);
                        }
                        std::printf("\n");
                    }

                    if (smesh::Env::read<int>("SFEM_GMG_CHECK", 0)) {
                        // Same gate the probing path uses: the assembled level must
                        // reproduce the matrix-free composite it stands for.
                        const ptrdiff_t     ndc = nn * N_FIELDS;
                        std::vector<real_t> v((size_t)ndc), ya((size_t)ndc, 0), t1((size_t)nd_up, 0),
                                t2((size_t)nd_up, 0), yb((size_t)ndc, 0);
                        unsigned st = 991u;
                        for (auto &e : v) {
                            st = st * 1103515245u + 12345u;
                            e  = (real_t)((st >> 16) & 0x7fff) / (real_t)0x7fff - real_t(0.5);
                        }
                        fi->apply_zero_constraints(v.data());
                        a_c->apply(v.data(), ya.data());
                        // Compose the SAME three matrices by hand. Comparing against the
                        // matrix-free composite instead would be comparing Galerkin against
                        // rediscretisation, which differ by construction -- that difference
                        // is the reason this work exists. This isolates the triple product,
                        // which is the untested part: rap has only ever been exercised at
                        // block size 1 in this repo.
                        apply_scalar_to_blocks(g.Pmat[(size_t)i], v.data(), t1.data());
                        masked->apply(t1.data(), t2.data());
                        apply_scalar_to_blocks(g.Rmat[(size_t)i], t2.data(), yb.data());
                        // a_c carries identity rows where the coarse level is constrained,
                        // while the raw product carries whatever R*A*P gives there. Those
                        // rows are not part of what is being tested, so both sides are
                        // zeroed on them. (Zeroing only the probe is not enough: the
                        // identity reproduces the zero, the raw product does not.)
                        fi->apply_zero_constraints(ya.data());
                        fi->apply_zero_constraints(yb.data());
                        real_t dn = 0, rn = 0;
                        for (ptrdiff_t k = 0; k < ndc; ++k) {
                            const real_t d = ya[(size_t)k] - yb[(size_t)k];
                            dn += d * d;
                            rn += yb[(size_t)k] * yb[(size_t)k];
                        }
                        const real_t rel = (rn > 0) ? std::sqrt(dn / rn) : 0.0;
                        std::printf("rap level %d: %td blocks  vs R*(A*(P*v)) rel = %.4e  %s\n", i,
                                    (ptrdiff_t)a_c->col_idx->size(), rel, (rel < 1e-10) ? "OK" : "MISMATCH");
                    }
                }
                phase_add("galerkin_assembly", smesh::time_seconds() - t_asm);
            }

            std::shared_ptr<BlockJacobi> prec;
            if (i > 0 && galerkin_mode == 2)
                prec = make_block_jacobi_from_diag(galerkin_diag, mask.data(), nn, omega);
            else
                prec = make_block_jacobi(*g.level_ops[i], g.states[i]->data(), mask.data(), nn, omega,
                                         (i > 0) ? pscale : real_t(1));

            // SFEM_GMG_PSCALE scales the continuity rows of every coarse level.
            //
            // Left-scaling a row does not change what the coarse system solves; it changes
            // the correction the cycle takes from it. Unlike SFEM_GMG_RC_DECAY this leaves
            // Df alone, so the balance between divergence and stabilisation inside the
            // continuity row is untouched and the coarse operator keeps the stabilisation
            // its own mesh needs. The value to use is not tuned: it is the pressure/velocity
            // ratio of the best-fit scales printed by SFEM_GMG_CHECK=1.
            if (i > 0 && pscale != real_t(1)) {
                const ptrdiff_t nd = fi->space()->n_dofs();
                auto            inner = lop;
                lop = sfem::make_op<real_t>(
                        inner->rows(), inner->cols(),
                        [inner, nd, pscale](const real_t *const x, real_t *const y) {
                            std::vector<real_t> t((size_t)nd, real_t(0));
                            inner->apply(x, t.data());
                            for (ptrdiff_t k = 0; k < nd; ++k)
                                y[k] += (k % N_FIELDS == 3) ? pscale * t[(size_t)k] : t[(size_t)k];
                        },
                        sfem::EXECUTION_SPACE_HOST);
            }

            if (i + 1 < nlevels) {
                // SFEM_GMG_KSMOOTH > 0 replaces the stationary smoother with that many
                // BiCGStab iterations, preconditioned by the same block-Jacobi.
                //
                // The Galerkin coarse operators approximate the fine operator far better
                // than the rediscretised ones and are far worse to smooth -- denser, and
                // without the diagonal dominance a stationary iteration needs, so they
                // diverge under block-Jacobi at every damping. A Krylov method does not
                // require that: it adapts to the operator it is given. The price is that
                // the resulting cycle is no longer a fixed linear operator, which is why
                // this must be paired with a flexible outer solver.
                // Coarse levels only. The Krylov smoother exists because the assembled
                // Galerkin operators lack the diagonal dominance block-Jacobi needs; the
                // fine level is the matrix-free rediscretised operator and has no such
                // problem, and it is the level where work is expensive. Smoothing it with
                // sixteen preconditioned BiCGStab iterations costs sixty-four fine
                // operator applications per cycle, which is what made the large cases run
                // an order of magnitude slower than plain block-Jacobi despite needing far
                // fewer iterations.
                // SFEM_GMG_KSMOOTH_FINE controls the fine level separately, because the
                // right answer depends on size. Krylov smoothing the fine level buys
                // iterations everywhere, but a BiCGStab iteration there costs two fine
                // operator applications, so at 16 sweeps a cycle spends 64 of them on the
                // finest level alone. When the coarse hierarchy dominates the work that is
                // cheap; when the fine level dominates it is not, and the same setting that
                // wins at one macro-element loses by an order of magnitude at twenty-seven.
                // Default is to follow SFEM_GMG_KSMOOTH, i.e. smooth every level the same.
                // The fine level gets far less smoothing than the coarse ones, and the
                // reason is pure arithmetic. The fine smoother is itself BiCGStab
                // preconditioned by block-Jacobi -- the same solver this whole cycle is
                // competing against -- so at k iterations a cycle spends 4k fine operator
                // applications on smoothing alone, against the two that solver spends per
                // iteration. At k = 16 that is 32 times the work per outer iteration, which
                // the eleven-fold drop in iteration count cannot pay for. At k = 2 it is
                // four times the work for a comparable drop, and that is the configuration
                // that finally beats the baseline. Coarse levels keep the strong smoother:
                // they need it, and they are cheap.
                const int kdefault = smesh::Env::read<int>("SFEM_GMG_KSMOOTH", 0);
                const int ksmooth  = (i > 0) ? kdefault
                                             : smesh::Env::read<int>("SFEM_GMG_KSMOOTH_FINE",
                                                                     kdefault > 0 ? 2 : 0);
                // SFEM_SMOOTHER=vanka replaces the point-block smoother on the fine level.
                //
                // Measured standalone at Re=1, 150 sweeps to the asymptote: block-Jacobi
                // 0.981 (and divergent for omega >= 0.5), additive Vanka 0.883, multiplicative
                // 8-colour Vanka 0.63 at omega = 1 with no damping needed. The point-block
                // smoother discards the velocity-pressure coupling that determines the
                // pressure; the patch solve keeps it.
                //
                // Fine level only: the patch operator is read from the assembled fine matrix
                // via the element-wise Galerkin path at q = 1, and the coarse levels have a
                // different lattice. They keep block-Jacobi for now.
                // Vanka is the default smoother. Block-Jacobi is retained only as a
                // reference and for meshes with no lattice (SFEM_SMOOTHER=bjacobi).
                //
                // Measured: block-Jacobi has an asymptotic factor of 0.981 at Re=1 and diverges
                // at omega >= 0.5, so it is not a smoother in any useful sense on this system;
                // it damps momentum while discarding the continuity constraint. At 33,124 dofs
                // on 72 cores it diverges outright at omega = 1 where Vanka converges at both
                // Re=1 and Re=100. And Re=200, which block-Jacobi could never converge -- 40
                // Newton steps, ramp stalling at Re~75 -- solves in 3 Newton steps with Vanka,
                // to u_linf 2.28e-10 against the 2.9e-7 plateau of the run that never
                // converged.
                std::shared_ptr<sfem::Operator<real_t>> prec_op = prec;
                // Coarse levels: same smoother, built from the matrix the element-wise Galerkin
                // assembly already produced. A cycle is limited by its worst level, and leaving
                // these on the point-block smoother would waste the fine-level gain.
                if (i > 0 && smesh::Env::read<std::string>("SFEM_SMOOTHER", "vanka") == "vanka" &&
                    g.Amat[(size_t)i] && g.level_ops[(size_t)i] &&
                    g.level_ops[(size_t)i]->is_semi_structured()) {
                    std::vector<uint8_t> cb((size_t)fi->space()->n_dofs(), 0);
                    for (ptrdiff_t k = 0; k < fi->space()->n_dofs(); ++k)
                        cb[(size_t)k] = mask_get(k, mask.data()) ? 1 : 0;
                    auto vk = cvfem_ss::make_diagonal_vanka_from_bsr(
                            *g.level_ops[(size_t)i], g.Amat[(size_t)i], cb.data(),
                            smoother_omega());
                    if (vk) prec_op = vk;
                }
                if (i == 0 && smesh::Env::read<std::string>("SFEM_SMOOTHER", "vanka") == "vanka" &&
                    g.level_ops[0] && g.level_ops[0]->is_semi_structured()) {
                    const ptrdiff_t      nd0 = g.data->functions[0]->space()->n_dofs();
                    std::vector<mask_t>  m0(mask_count(nd0), 0);
                    g.data->functions[0]->constraints_mask(m0.data());
                    std::vector<uint8_t> cb((size_t)nd0, 0);
                    for (ptrdiff_t k = 0; k < nd0; ++k) cb[(size_t)k] = mask_get(k, m0.data()) ? 1 : 0;
                    const double t_v = smesh::time_seconds();
                    prec_op = cvfem_ss::make_diagonal_vanka(*g.level_ops[0], g.data->functions[0]->space(),
                                                            g.states[0]->data(), cb.data(),
                                                            smoother_omega());
                    phase_add("vanka_setup", smesh::time_seconds() - t_v);
                }

                std::shared_ptr<sfem::MatrixFreeLinearSolver<real_t>> sm;
                if (ksmooth > 0) {
                    auto ks = sfem::create_bcgs<real_t>(lop, sfem::EXECUTION_SPACE_HOST);
                    ks->set_max_it(ksmooth);
                    ks->set_rtol(1e-12);
                    ks->set_atol(1e-30);
                    ks->verbose = false;
                    ks->set_preconditioner_op(prec_op);
                    sm = ks;
                } else {
                    auto st = sfem::create_stationary<real_t>(lop, prec_op, sfem::EXECUTION_SPACE_HOST);
                    st->set_max_it(g.smoothing_steps);
                    sm = st;
                }
                auto sm_unused = sm;
                level_op_below = lop;
                const ptrdiff_t nd_lvl = fi->space()->n_dofs();
                g.mg->add_level(timed("op[L" + std::to_string(i) + "]", thread_clamped(nd_lvl, lop)),
                                timed("smooth[L" + std::to_string(i) + "]", thread_clamped(nd_lvl, sm)),
                                i == 0 ? nullptr : timed("prolong[L" + std::to_string(i) + "->" + std::to_string(i - 1) + "]", wrap_p(i)),
                                timed("restrict[L" + std::to_string(i) + "->" + std::to_string(i + 1) + "]", g.data->restrictions[i]));
            } else {
                level_op_below = lop;
                // Coarse solve. Dense LU when the level is small enough to factorise,
                // which is exact and cannot diverge; BiCGStab otherwise, and not CG,
                // because the operator is not symmetric.
                const ptrdiff_t nd_coarse = fi->space()->n_dofs();
                // The coarsest level is solved directly.
                //
                // A cycle takes its coarse answer at face value, so an iterative coarse solve
                // that stagnates hands up a correction that is noise while reporting success.
                // The direct solve cannot do that. It is also the second place a varying
                // preconditioner came from, since an iterative coarse solve inherits every
                // nondeterminism below it.
                //
                // The cost is real and cubic: this is a dense LU, so a coarse level of n dofs
                // costs O(n^3) to factor and O(n^2) to store, once per Jacobian. Where that
                // bites, the answer is a deeper hierarchy so the coarsest level is genuinely
                // small, not a return to an iterative coarse solve. SFEM_GMG_DENSE_LU_BELOW
                // caps it for the cases where that is not yet possible.
                const ptrdiff_t lu_max =
                        (ptrdiff_t)smesh::Env::read<int>("SFEM_GMG_DENSE_LU_BELOW", 1 << 30);
                if (nd_coarse <= lu_max) {
                    // Prefer the assembled matrix when there is one; fall back to probing
                    // only for a level that has no matrix form.
                    auto lu = g.Amat[(size_t)i] ? make_dense_lu_from_bsr(g.Amat[(size_t)i], nd_coarse)
                                                : make_dense_lu(lop, nd_coarse);
                    {
                        const std::string dpath =
                                smesh::Env::read_string("SFEM_GMG_DUMP_COARSE", std::string());
                        if (!dpath.empty()) {
                            std::vector<real_t> dn((size_t)nd_coarse * (size_t)nd_coarse, 0);
                            std::vector<real_t> e((size_t)nd_coarse), c((size_t)nd_coarse);
                            for (ptrdiff_t j = 0; j < nd_coarse; ++j) {
                                std::fill(e.begin(), e.end(), real_t(0));
                                std::fill(c.begin(), c.end(), real_t(0));
                                e[(size_t)j] = 1;
                                lop->apply(e.data(), c.data());
                                for (ptrdiff_t r = 0; r < nd_coarse; ++r)
                                    dn[(size_t)r * nd_coarse + j] = c[(size_t)r];
                            }
                            dump_dense(dpath.c_str(), nd_coarse, dn);
                        }
                    }

                    // A rank-deficient coarse operator is reported by component, because which
                    // component loses rank says what is missing: pressure alone is a gauge
                    // (no Dirichlet pressure and an outflow that does not fix the level),
                    // velocity means the coarse boundary treatment itself is wrong.
                    if (lu->n_dropped()) {
                        int by_comp[4] = {0, 0, 0, 0};
                        for (const ptrdiff_t d : lu->dropped()) by_comp[(int)(d % 4)]++;
                        std::printf(
                                "coarse LU: %td of %td directions dropped as null  "
                                "(first %zu by component: ux %d  uy %d  uz %d  p %d)\n",
                                lu->n_dropped(), (ptrdiff_t)nd_coarse, lu->dropped().size(),
                                by_comp[0], by_comp[1], by_comp[2], by_comp[3]);
                    }

                    if (smesh::Env::read<int>("SFEM_GMG_CHECK", 0) && g.Amat[(size_t)i]) {
                        // The two densifications must agree: same operator, read two ways.
                        auto                ref = make_dense_lu(lop, nd_coarse);
                        std::vector<real_t> b((size_t)nd_coarse), x1((size_t)nd_coarse, 0),
                                x2((size_t)nd_coarse, 0);
                        for (ptrdiff_t k = 0; k < nd_coarse; ++k)
                            b[(size_t)k] = std::sin(real_t(0.61) * (real_t)k + real_t(0.2));
                        lu->apply(b.data(), x1.data());
                        ref->apply(b.data(), x2.data());
                        real_t dn = 0, rn = 0;
                        for (ptrdiff_t k = 0; k < nd_coarse; ++k) {
                            const real_t d = x1[(size_t)k] - x2[(size_t)k];
                            dn += d * d;
                            rn += x2[(size_t)k] * x2[(size_t)k];
                        }
                        const real_t rel = rn > 0 ? std::sqrt(dn / rn) : std::sqrt(dn);
                        std::printf("coarse LU: assembled vs probed, rel = %.4e over %td dofs  %s\n", rel,
                                    nd_coarse, rel < 1e-10 ? "OK" : "MISMATCH");
                    }
                    g.mg->add_level(timed("op[coarsest]", lop), timed("coarse_solve", lu),
                                    timed("prolong[L" + std::to_string(i) + "->" + std::to_string(i - 1) + "]", wrap_p(i)), nullptr);
                    continue;
                }

                auto cs = sfem::create_bcgs<real_t>(lop, sfem::EXECUTION_SPACE_HOST);
                cs->set_max_it(smesh::Env::read<int>("SFEM_GMG_COARSE_MAX_IT", 200));
                cs->set_rtol(1e-8);
                cs->set_atol(1e-14);
                // The coarse level is solved, not smoothed, so the cycle takes its answer
                // at face value. If that solve stagnates the correction is noise, and a
                // stagnating BiCGStab reports success by exhausting its iterations.
                cs->verbose = smesh::Env::read<int>("SFEM_GMG_COARSE_VERBOSE", 0) != 0;
                cs->set_preconditioner_op(prec);
                const ptrdiff_t nd_c = fi->space()->n_dofs();
                g.mg->add_level(timed("op[coarsest]", thread_clamped(nd_c, lop)),
                                timed("coarse_solve", thread_clamped(nd_c, cs)),
                                timed("prolong[L" + std::to_string(i) + "->" + std::to_string(i - 1) + "]", wrap_p(i)), nullptr);
            }
        }
        // Cycle index, following Brandt's TME report (NASA/CR-1998-207647).
        //
        // For non-aligned grids with open characteristics -- entering flow, which is what an
        // inlet/outlet channel is -- he identifies the difficulty as "the shorter distance
        // (along the characteristics) for which a coarser grid still approximates some smooth
        // solution components", and lists three cures: downstream-ordered relaxation marching,
        // semi-coarsening, and a cycle index of 2^(p/m), the last "not requiring ordered
        // relaxation". For closed characteristics -- a recirculation bubble, which the
        // backward-facing step also has -- the recommended cycles are likewise W-based
        // (defect correction within W cycles, or downstream ordering with doubled transferred
        // residuals).
        //
        // That matches what was measured here: the weakly damped modes are smooth streamwise
        // velocity, not pressure, and neither the repaired smoother nor the coarse space
        // removes them. A higher cycle index visits the coarse levels more often per fine
        // sweep, which is the cheapest of the three cures to try and the only one needing no
        // new machinery.
        g.mg->set_cycle_type(smesh::Env::read<int>("SFEM_GMG_CYCLE", 1));
        g.mg->set_max_it(1);  // one cycle per preconditioner application
    }

}  // namespace

// Exact Jacobian action by forward differencing of the residual:
//
//     J v  ~  ( R(x + eps v) - R(x) ) / eps
//
// The assembled Jacobian here is not the exact derivative: the Rhie-Chow term freezes the
// reconstructed nodal pressure gradient pg, dropping its dependence on p. SFEM_FD_CHECK
// measures 3.9e-2 relative error in the continuity rows against 1.3e-4 in the momentum rows,
// flat in eps and identical at L=8 and L=16. Differencing the residual has no such omission
// by construction, so it isolates that defect from everything else.
//
// eps follows Knoll & Keyes: sqrt(machine eps) * (1 + ||x||) / ||v||, balancing truncation
// against cancellation.
//
// Constrained rows must stay identity. The difference is exactly zero there -- v is zeroed at
// constraints, so x + eps v equals x and the two residuals cancel -- which would leave a zero
// row and a singular operator. The mask restores v on those rows.
class JFNKOperator final : public sfem::Operator<real_t> {
public:
    JFNKOperator(std::shared_ptr<sfem::Function> f, const ptrdiff_t n, const mask_t *const cmask)
        : f_(std::move(f)), n_(n), cmask_(cmask), xt_((size_t)n), rt_((size_t)n), r0_((size_t)n) {}

    void set_base(const real_t *const x) {
        base_.assign(x, x + n_);
        std::fill(r0_.begin(), r0_.end(), real_t(0));
        f_->gradient(base_.data(), r0_.data());
        f_->apply_zero_constraints(r0_.data());
        xnorm_ = 0;
        for (ptrdiff_t i = 0; i < n_; ++i) xnorm_ += base_[(size_t)i] * base_[(size_t)i];
        xnorm_ = std::sqrt(xnorm_);
    }

    int apply(const real_t *const v, real_t *const y) override {
        real_t vn = 0;
        for (ptrdiff_t i = 0; i < n_; ++i) vn += v[i] * v[i];
        vn = std::sqrt(vn);
        if (vn == real_t(0)) return SFEM_SUCCESS;
        const real_t eps =
                std::sqrt(std::numeric_limits<real_t>::epsilon()) * (real_t(1) + xnorm_) / vn;
        for (ptrdiff_t i = 0; i < n_; ++i) xt_[(size_t)i] = base_[(size_t)i] + eps * v[i];
        std::fill(rt_.begin(), rt_.end(), real_t(0));
        f_->gradient(xt_.data(), rt_.data());
        f_->apply_zero_constraints(rt_.data());
        for (ptrdiff_t i = 0; i < n_; ++i)  // Operator::apply accumulates into y
            y[i] += mask_get(i, cmask_) ? v[i] : (rt_[(size_t)i] - r0_[(size_t)i]) / eps;
        return SFEM_SUCCESS;
    }

    std::ptrdiff_t       rows() const override { return n_; }
    std::ptrdiff_t       cols() const override { return n_; }
    sfem::ExecutionSpace execution_space() const override { return sfem::EXECUTION_SPACE_HOST; }

private:
    std::shared_ptr<sfem::Function> f_;
    ptrdiff_t                       n_;
    const mask_t                   *cmask_;
    std::vector<real_t>             base_, xt_, rt_, r0_;
    real_t                          xnorm_{0};
};

int main(int argc, char **argv) {
    auto ctx = sfem::initialize(argc, argv);

    if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        usage(argv[0]);
        return EXIT_SUCCESS;
    }
    if (argc != 2) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    const std::string out_folder = argv[1];

    const std::string case_name  = smesh::Env::read_string("SFEM_CASE", "");
    const int         n          = smesh::Env::read<int>("SFEM_N", 8);
    int               ny         = smesh::Env::read<int>("SFEM_NY", n);
    int               nx         = smesh::Env::read<int>("SFEM_NX", 0);
    int               nz         = smesh::Env::read<int>("SFEM_NZ", 0);
    // The channel default is 4x1x1; a cavity wants a square box, so its default differs.
    // Explicit SFEM_LX/LY/LZ still win.
    // The manufactured pressure's gauge constant makes it zero-mean on [0,2]^2 and nowhere
    // else, so the MMS domain is not a free parameter -- default it, and reject an override.
    const std::string case_req    = smesh::Env::read_string("SFEM_CASE", "");
    const bool        want_mms    = case_req == "mms" || case_req == "manufactured";
    // Their cavity is the cube [0,2]^3 and the lid profile is written for it; on any other
    // box the polynomial no longer vanishes at the edges and it is a different problem.
    const bool        want_cavreg = case_req == "cavity_reg" || case_req == "regularized_cavity" ||
                             case_req == "cavity_regularized";
    // The backward-facing step is not a box: it needs its own generator and the topological
    // boundary mask, since a coordinate test cannot see the two step faces.
    const bool        want_step   = case_req == "step" || case_req == "backward_facing_step" ||
                           case_req == "bfs";
    const bool        want_cavity = smesh::Env::read_string("SFEM_CASE", "") == "cavity" ||
                             smesh::Env::read_string("SFEM_CASE", "") == "lid" ||
                             smesh::Env::read_string("SFEM_CASE", "") == "lid_driven_cavity";
    // The diaphragm pump: a closed chamber with a moving wall and one port. See the block
    // comment in src/cases/cvfem_ns_channel_case.hpp for the geometry and for the identity
    // it is checked against. A cube by default, so the diaphragm area is Lx*Lz and the
    // arithmetic in that identity is visible rather than buried.
    const bool        want_pump   = case_req == "pump" || case_req == "diaphragm" ||
                           case_req == "diaphragm_pump";
    const real_t      Lx         = smesh::Env::read<real_t>("SFEM_LX", want_step ? 10 : ((want_mms || want_cavreg) ? 2 : ((want_cavity || want_pump) ? 1 : 4)));
    const real_t      Ly         = smesh::Env::read<real_t>("SFEM_LY", (want_mms || want_cavreg || want_step) ? 2 : 1);
    const real_t      Lz         = smesh::Env::read<real_t>("SFEM_LZ", (want_mms || want_cavreg) ? 2 : 1);
    const real_t      rho        = smesh::Env::read<real_t>("SFEM_RHO", 1);
    const real_t      mu         = smesh::Env::read<real_t>("SFEM_MU", 0.01);
    const real_t      U          = smesh::Env::read<real_t>("SFEM_U", 1);
    const std::string geom_name  = smesh::Env::read_string("SFEM_GEOM", "affine");
    const int         max_newton = smesh::Env::read<int>("SFEM_NL_MAX_IT", 40);
    // Transient stepping. SFEM_DT <= 0 -- the default -- is the steady solve every existing
    // case runs, and nothing below changes for it. With a timestep the whole continuation
    // and Newton solve becomes one time step, repeated SFEM_NSTEPS times, with the velocity
    // history shifted after each. BDF2 falls back to BDF1 on the first step, which has no
    // second history level; that is the standard start-up.
    const real_t      dt_step   = smesh::Env::read<real_t>("SFEM_DT", real_t(0));
    const int         nsteps    = std::max(1, smesh::Env::read<int>("SFEM_NSTEPS", 1));
    const int         bdf_order = smesh::Env::read<int>("SFEM_BDF_ORDER", 2);
    const real_t      nl_rtol    = smesh::Env::read<real_t>("SFEM_NL_RTOL", 1e-8);
    const real_t      nl_atol    = smesh::Env::read<real_t>("SFEM_NL_ATOL", 1e-12);
    // Step-size convergence. The residual tests above cannot fire once ||R|| has reached its
    // round-off floor, because no further relative reduction is achievable; but a Newton step
    // of size 1e-10 has converged whatever the residual is doing. Without this the loop asks
    // for another linear solve, that solve chases a relative tolerance against a residual it
    // cannot reduce and burns its whole iteration cap, and the line search then correctly
    // finds no decrease -- abandoning a stage that had in fact succeeded.
    // Off by default: measured harmful. Accepting convergence on the step alone declares a
    // stage successful without ever confirming ||R|| came down, and the continuation then
    // grows its step from a state that has not converged. At Re=3200 it produced 8 such false
    // successes and took the reachable Reynolds number from 3200 (at target) down to 1682.
    // Retained only as an experiment knob.
    const real_t      nl_stol    = smesh::Env::read<real_t>("SFEM_NL_STOL", 0);
    const real_t      lin_rtol   = smesh::Env::read<real_t>("SFEM_LSOLVE_RTOL", 1e-8);
    const real_t      lin_atol   = smesh::Env::read<real_t>("SFEM_LSOLVE_ATOL", 1e-14);
    const int         lin_max_it = smesh::Env::read<int>("SFEM_LSOLVE_MAX_IT", 1000);
    const int         pack_size  = smesh::Env::read<int>("SFEM_PACK_SIZE", 2048);
    // Matrix-free by default. 0 assembles a BSR once per Newton step and hands the
    // Krylov method an SpMV instead; that path stays fully supported and gated, and at
    // p=1 it is still the faster of the two -- see the timing breakdown printed at the
    // end. The default reflects where the work is going, not where it is today: the
    // semi-structured hierarchy is what makes the matrix-free apply worth having, and
    // an assembled BSR per level is exactly the memory the hierarchy exists to avoid.
    const int         matrix_free = smesh::Env::read<int>("SFEM_MATRIX_FREE", 1);
    // 1: precondition the Jacobian solve with a semi-structured multigrid V-cycle instead
    // of point block-Jacobi. Needs a semi-structured mesh with more than one level, so it
    // is ignored on a flat one.
    const int         use_gmg     = smesh::Env::read<int>("SFEM_GMG", 0);
    const int         gmg_smooth  = smesh::Env::read<int>("SFEM_GMG_SMOOTH", 3);
    // Compares J_mf v against J_asm v once, on the first Jacobian. The two paths must
    // agree before any timing comparison between them means anything.
    const int         check_jv    = smesh::Env::read<int>("SFEM_CHECK_JV", 0);
    const real_t      verify_tol = smesh::Env::read<real_t>("SFEM_VERIFY_TOL", 1e-2);

    FlowCase flow;
    if (case_name.empty() || !cvfem_case::parse_case(case_name, flow)) {
        std::fprintf(stderr, "SFEM_CASE is required "
                                 "(poiseuille, couette, cavity, cavity_reg or mms)\n");
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (geom_name != "affine" && geom_name != "isoparam") {
        std::fprintf(stderr, "invalid SFEM_GEOM '%s' (expected affine or isoparam)\n", geom_name.c_str());
        return EXIT_FAILURE;
    }
    if (ny < 1) ny = 1;
    if (nx < 1) nx = std::max(1, (int)std::lround((double)ny * (double)Lx / (double)Ly));
    if (nz < 1) nz = std::max(1, (int)std::lround((double)ny * (double)Lz / (double)Ly));

    const double tick = smesh::time_seconds();

    auto mesh = want_step ? smesh::Mesh::create_hex8_lshape(ctx->communicator(), nx, ny, nz, Lx, Ly, Lz,
                                                            smesh::Env::read<real_t>("SFEM_STEP_X", 1),
                                                            smesh::Env::read<real_t>("SFEM_STEP_Y", 1))
                          : smesh::Mesh::create_hex8_cube(ctx->communicator(), nx, ny, nz, 0, 0, 0, Lx, Ly, Lz);
    if (!mesh) {
        std::fprintf(stderr, "mesh generation failed\n");
        return EXIT_FAILURE;
    }
    // Named sidesets, built once on the MACRO mesh and carried from there.
    //
    // These are the specification of the boundary; the per-element bitmask the kernels read
    // is the compiled form of them. Building them here has three consequences worth stating.
    //
    // The outlet is named rather than found by a coordinate test. Selecting it by comparing
    // corners against x = Lx was the same box-thinking that makes hex8_face_on_domain wrong
    // on this geometry -- it happens to work because this outlet is a plane, and would not on
    // one that is not.
    //
    // The Dirichlet set and the control-volume closure are then derived from the *same*
    // objects rather than from two independent tests. cvfem_ns_channel_case.hpp documents the
    // invariant that the two must decide the same thing, "and if they disagree a node gets a
    // closed control volume without a boundary condition, or the reverse"; one source removes
    // the possibility instead of testing for it.
    //
    // And they are level-invariant. Sideset stores (parent, lfi) on the macro element, which
    // a semi-structured level change leaves untouched, so every multigrid level compiles its
    // mask from these same sidesets instead of re-deriving the skin -- which cost an
    // element-adjacency pass per level.
    //
    // SFEM_OUTLET=natural puts the same do-nothing outflow on a plain box. The step is an
    // L-shape with an open outlet; when its multigrid cycle misbehaves, those two properties
    // are confounded, and every clean diagnostic in this file was obtained on a box. This
    // knob supplies the missing control -- a box that differs from the working Poiseuille
    // case in the outlet treatment and nothing else -- and it is off by default, so no
    // existing run changes.
    const bool want_natural_outlet =
            want_step ? smesh::Env::read_string("SFEM_STEP_OUTFLOW", "natural") != "dirichlet"
                      : smesh::Env::read_string("SFEM_OUTLET", "dirichlet") == "natural";
    // A traction or pressure condition names one of these sidesets, so they have to exist
    // whether or not the outlet is natural. Read here rather than where the operator is
    // configured, several hundred lines below, because the sidesets are built now and a
    // condition naming one that was never registered would fail for the wrong reason.
    const std::string want_traction_sideset = smesh::Env::read_string("SFEM_TRACTION_SIDESET", "");
    // The pump's port defaults to being the pressure boundary, because that is what a port
    // is; naming SFEM_PRESSURE_SIDESET explicitly still wins.
    const std::string want_pressure_sideset =
            smesh::Env::read_string("SFEM_PRESSURE_SIDESET", want_pump ? "port" : "");
    const bool        want_named_bc = !want_traction_sideset.empty() || !want_pressure_sideset.empty();
    // A face carrying a traction or pressure condition must not also carry Dirichlet
    // velocity data. It is the same invariant cvfem_ns_channel_case.hpp states for the
    // boundary mask -- two channels deciding the same face -- and getting it wrong is
    // silent in the worst way: the constraint wins, the condition is overridden, and the
    // run prints that the port was applied while reporting the Dirichlet answer. Measured
    // before this line existed, on 33,124 dofs: a port at p_bar = 1.5 on a Dirichlet-pinned
    // outlet moved u_linf from 3.754076e-05 to 3.754095e-05, which is nothing at all.
    //
    // "outlet" is this driver's own name for the x = Lx plane, registered above from the
    // same plane the `outlet` predicate tests, so matching on the name is the correct test
    // here rather than a coincidence of naming.
    const bool outlet_governed = want_natural_outlet || want_traction_sideset == "outlet" ||
                                 want_pressure_sideset == "outlet";
    std::shared_ptr<smesh::Sideset> step_skin, step_outlet;
    // Kept so they survive to_semistructured, which builds a new Mesh and copies none.
    std::shared_ptr<smesh::Sideset> pump_skin, pump_port, pump_diaphragm;
    if (!want_pump && (want_step || want_natural_outlet || want_named_bc)) {
        step_skin = smesh::skin_sideset(mesh);
        auto outs = smesh::Sideset::create_from_plane(mesh, 1, 0, 0, (smesh::geom_t)Lx, 1e-6);
        if (!step_skin || outs.empty()) {
            std::fprintf(stderr, "could not build the boundary sidesets\n");
            return EXIT_FAILURE;
        }
        step_outlet = outs.front();
        // Register them on the flat mesh as well, so a run at refine_level 1 -- which never
        // reaches the re-attachment below, because it does not rebuild the mesh -- still has
        // the named sidesets the operator asks for.
        mesh->add_sideset("skin", step_skin);
        mesh->add_sideset("outlet", step_outlet);
        std::printf("sidesets: skin %td faces, outlet %td faces\n",
                    (ptrdiff_t)step_skin->parent()->size(), (ptrdiff_t)step_outlet->parent()->size());
        setenv("SFEM_BOUNDARY_MASK", "1", 0);
    }

    // The pump's two openings, both derived from the SAME coordinate predicates the
    // Dirichlet set below uses. That is the point of building them here rather than from a
    // plane: cvfem_ns_channel_case.hpp states the invariant that the boundary marking and
    // the constraint set must decide the same faces, and deriving both from one predicate
    // removes the possibility of disagreement instead of testing for it.
    //
    // The port is a patch and not a whole face, so create_from_plane cannot express it;
    // create_from_selector takes the predicate directly.
    const real_t pump_port_frac = smesh::Env::read<real_t>("SFEM_PUMP_PORT", real_t(0.5));
    if (want_pump) {
        auto skin = smesh::skin_sideset(mesh);
        auto port = smesh::Sideset::create_from_selector(
                mesh, [&](const smesh::geom_t x, const smesh::geom_t y, const smesh::geom_t z) {
                    return cvfem_case::pump_on_port<real_t>((real_t)x, (real_t)y, (real_t)z, Lx, Ly, Lz,
                                                            pump_port_frac);
                });
        if (!skin || port.empty()) {
            std::fprintf(stderr, "pump: could not build the chamber sidesets\n");
            return EXIT_FAILURE;
        }
        auto diaphragm = smesh::Sideset::create_from_selector(
                mesh, [&](const smesh::geom_t /*x*/, const smesh::geom_t y, const smesh::geom_t /*z*/) {
                    return cvfem_case::pump_on_diaphragm<real_t>((real_t)y, Ly);
                });
        if (diaphragm.empty()) {
            std::fprintf(stderr, "pump: could not build the diaphragm sideset\n");
            return EXIT_FAILURE;
        }
        pump_skin      = skin;
        pump_port      = port.front();
        pump_diaphragm = diaphragm.front();
        mesh->add_sideset("skin", skin);
        mesh->add_sideset("port", port.front());
        // Named only so the flux through it can be measured; no boundary condition reads it,
        // because the diaphragm is Dirichlet and lives in the constraint set.
        mesh->add_sideset("diaphragm", diaphragm.front());
        const ptrdiff_t n_port = (ptrdiff_t)port.front()->parent()->size();
        // A port of no faces is a closed chamber with a moving wall pushing into an
        // incompressible fluid, which has no solution. Say so here rather than let the
        // linear solver discover it.
        if (n_port == 0) {
            std::fprintf(stderr,
                         "pump: SFEM_PUMP_PORT=%g selected no faces at this resolution -- the "
                         "chamber has no opening and the problem is unsolvable. Raise it, or "
                         "raise SFEM_N.\n",
                         (double)pump_port_frac);
            return EXIT_FAILURE;
        }
        std::printf("pump: chamber %gx%gx%g  diaphragm area %g  port %td faces (frac %g)\n",
                    (double)Lx, (double)Ly, (double)Lz, (double)(Lx * Lz), n_port,
                    (double)pump_port_frac);
        // The port is where the fluid leaves, so it carries the pressure and must not also
        // carry velocity data; the diaphragm carries velocity and must not carry pressure.
        // Both are set by name below.
        setenv("SFEM_BOUNDARY_MASK", "1", 0);
    }
    // SFEM_ELEMENT_REFINE_LEVEL > 1 turns the mesh semi-structured: the cells above become
    // macro-elements, each holding a level^3 lattice, and the operator switches to the
    // sshex8 kernels on its own from what the space carries. The requested cell counts then
    // describe macro-elements, so the problem is level^3 times larger than the flat run of
    // the same SFEM_N -- which is the point, but worth knowing when comparing.
    const int refine_level = smesh::Env::read<int>("SFEM_ELEMENT_REFINE_LEVEL", 1);
    if (refine_level > 1) {
        mesh = smesh::to_semistructured(refine_level, mesh, true, false);
        // to_semistructured builds a new Mesh and does not copy sidesets. The macro elements
        // are the same and (parent, lfi) refers to them, so re-attaching is exact rather than
        // a re-derivation.
        if (mesh && step_skin) {
            mesh->add_sideset("skin", step_skin);
            mesh->add_sideset("outlet", step_outlet);
        }
        // The pump's, for the same reason. (parent, lfi) addresses the MACRO element, which
        // the conversion leaves alone, so re-attaching is exact and not a re-derivation --
        // and it is what lets one coordinate predicate keep governing a mesh whose nodes it
        // was never evaluated on.
        if (mesh && pump_port) {
            mesh->add_sideset("skin", pump_skin);
            mesh->add_sideset("port", pump_port);
            mesh->add_sideset("diaphragm", pump_diaphragm);
        }
        if (!mesh) {
            std::fprintf(stderr, "to_semistructured failed for level %d\n", refine_level);
            return EXIT_FAILURE;
        }
    }
    auto fs   = sfem::FunctionSpace::create(mesh, N_FIELDS);
    auto f    = sfem::Function::create(fs);

    auto op  = std::make_shared<sfem::CVFEMNavierStokes>(fs);
    real_t upwind_eps_ref = 0;
    // Harten band for the upwind switch, expressed relatively so it does not depend on the
    // mesh or the units.
    //
    // The band has to be compared against a mass flux, so it is scaled by one: rho times a
    // velocity scale times a fine-cell face area. SFEM_UPWIND_EPS_REL is the fraction of that
    // flux inside which the upwind switch is rounded off; 0, the default, is the hard switch
    // and reproduces every existing result bit for bit. SFEM_UPWIND_EPS overrides with an
    // absolute value for when a specific band is wanted.
    {
        const int    Lref_u = std::max(1, refine_level);
        const real_t h_u    = std::min({Lx / (real_t)(nx * Lref_u), Ly / (real_t)(ny * Lref_u),
                                        Lz / (real_t)(nz * Lref_u)});
        const real_t rel_u  = smesh::Env::read<real_t>("SFEM_UPWIND_EPS_REL", real_t(0));
        const real_t abs_u  = smesh::Env::read<real_t>("SFEM_UPWIND_EPS", real_t(-1));
        // One power of h below the physical flux scale, which is what makes the band a
        // discriminator rather than a perturbation.
        //
        // A sub-control-surface flux in a real flow is of order rho * U * h^2. Sizing the
        // band at that order -- which is what rho*U*h^2 did -- swamps genuine fluxes instead
        // of separating them from noise, and it does not vanish under refinement relative to
        // the thing it is being compared against. Venkatakrishnan's eps^2 = (K dx)^3 is the
        // same idea for a limiter: in smooth regions the differences are O(dx) so eps^2 is
        // an order smaller and the limiter stays active, while in the near-constant regions
        // -- where the quantity has collapsed to noise -- eps^2 dominates and the switch
        // turns off. Here that means eps / (rho U h^2) = K h / L, going to zero with the
        // mesh, so the band separates a flux that is physically small from one that is
        // merely round-off.
        const real_t L_ref = std::max({Lx, Ly, Lz});
        upwind_eps_ref = (abs_u >= 0) ? abs_u : rel_u * rho * U * h_u * h_u * h_u / L_ref;
        if (upwind_eps_ref > 0)
            std::printf("upwind switch: band eps = %.6e (K %g, rho %g, U %g, h %g)%s\n",
                        (double)upwind_eps_ref, (double)rel_u, (double)rho, (double)U, (double)h_u,
                        smesh::Env::read<int>("SFEM_UPWIND_ADAPT", 0) ? "  [adaptive]" : "");
        if (!smesh::Env::read<int>("SFEM_UPWIND_ADAPT", 0)) op->upwind_eps = upwind_eps_ref;
    }
    op->rho  = rho;
    op->mu   = mu;
    op->geom = (geom_name == "isoparam") ? sfem::CVFEMGeometry::Isoparam : sfem::CVFEMGeometry::Affine;
    op->pack_size = pack_size;
    // Two ways to leave an outlet open, and they are different boundary conditions.
    //
    //   donothing   (default) drop (p I - tau).n on the outlet face. This imposes a
    //               traction-free condition, which is the natural condition the weak form
    //               produces, and it is what the Farrell/Mitchell/Wechsung step is specified
    //               with. Dropping p_i*a is also what fixes the pressure level, so no gauge
    //               is applied.
    //   extrapolate keep the face as an ordinary closed boundary face evaluated from the
    //               interior state, while leaving the velocity unconstrained -- a
    //               zero-gradient outflow. It imposes nothing on the traction, so the
    //               constant-pressure mode returns and the zero-mean gauge takes over.
    //
    // The two are not related by a pressure shift: the retained term carries the local p_i,
    // not a constant, so the velocity fields genuinely differ near the outlet. The reason to
    // have both is that the extrapolation form gives the outflow momentum diagonal a viscous
    // contribution that the do-nothing form has no source for, and it is the absence of that
    // contribution that the Vanka patch solve degenerates on.
    const std::string outflow_mode =
            smesh::Env::read_string("SFEM_OUTFLOW_MODE", std::string("donothing"));
    // An open outlet switches the Vanka sweep to the additive form.
    //
    // The multiplicative 8-colour sweep is the better smoother on a closed box -- the
    // measurements recorded in this file give it 0.63 at omega = 1 against the additive
    // form's 0.883 -- and that ranking reverses once the outlet opens. Brandt's regime
    // diagnostic (SFEM_GMG_CHECK=7) localises it: on the backward-facing step the
    // multiplicative sweep decays error healthily through the whole upstream half, including
    // the recirculation bubble, and then amplifies it through the last third of the channel,
    // reaching a local rate of 1.99 at the outlet plane. The additive sweep is convergent
    // everywhere on the same problem, worst local rate 1.0013.
    //
    // The mechanism is ordering. A multiplicative sweep propagates information in colour
    // order, which roughly follows the characteristics where the flow is unidirectional --
    // which is why it wins on a box. At an outlet with more than half its faces reversed (24
    // of 45 on this case) colour order and characteristic direction disagree and the sweep
    // carries error against the flow. The additive form has no ordering to get wrong.
    //
    // Recorded honestly: this is a smoother measurement. Applied as a solver on the box with
    // an open outlet the additive V-cycle is *worse* -- diverging about 20x per cycle against
    // the multiplicative one's 7.3x -- so a locally convergent smoother is not by itself
    // buying a convergent cycle here, and that tension is unresolved.
    //
    // The gate matters in both directions, and cost is not the reason for it. Measured on
    // closed domains -- the side that keeps the multiplicative sweep -- everything else equal:
    //
    //     Poiseuille  5,508 dof   mult  Re 3200 reached,    25 its,   0.233 s, sweep 0.263 ms
    //                             add   Re 0,            12000 its,  67.6 s,   sweep 0.090 ms
    //     cavity     19,652 dof   mult  Re 50 reached,     100 its,   1.53 s,  sweep 0.554 ms
    //                             add   Re 0,            12000 its, 101.8 s,   sweep 0.157 ms
    //
    // The additive sweep is about three times cheaper per call and still loses by two orders
    // of magnitude in time to solution, needing 480 times the iterations and converging on
    // neither case. Per-sweep cost is the wrong figure of merit; only its product with the
    // iteration count decides anything.
    //
    // So this is not a cheap-versus-expensive trade with a happy side benefit. Each variant
    // is unusable where the other belongs -- multiplicative diverges at an open outlet,
    // additive cannot solve a closed domain -- which is why the choice is gated on the
    // geometry rather than offered as a tuning knob.
    //
    // An explicit SFEM_VANKA_MULT still wins: setenv's overwrite flag is 0.
    if (want_natural_outlet) setenv("SFEM_VANKA_MULT", "0", 0);
    if (want_natural_outlet && outflow_mode == "donothing") {
        // Do-nothing outflow at x = Lx. This drops (p I - tau).n there, which is what fixes
        // the pressure gauge -- so the pin must come off with it, or the system is
        // over-determined.
        op->natural_outflow_sideset = "outlet";
    }

    // Value-carrying boundary conditions, named by sideset. Both are off unless named, so
    // every existing case and every recorded number is untouched.
    //
    //   SFEM_TRACTION_SIDESET=<name> SFEM_TRACTION="tx ty tz"
    //       (pI - tau).n = t there. Those faces become natural, so a zero t is a second
    //       do-nothing outflow and a non-zero one is a surface being pushed -- which is
    //       what a diaphragm looks like when it is driven by force rather than motion.
    //   SFEM_PRESSURE_SIDESET=<name> SFEM_PRESSURE=<p>
    //       p = p there, with the viscous traction still from the interior state: a port
    //       held at a pressure.
    //
    // The sideset must already exist on the mesh under that name. Both need
    // SFEM_BOUNDARY_MASK=1, and the operator refuses rather than proceeding without it.
    op->traction_sideset = want_traction_sideset;
    if (!op->traction_sideset.empty()) {
        const std::string t = smesh::Env::read_string("SFEM_TRACTION", "0 0 0");
        if (std::sscanf(t.c_str(), "%lf %lf %lf", &op->traction[0], &op->traction[1], &op->traction[2]) != 3) {
            std::fprintf(stderr, "SFEM_TRACTION must be three numbers, got '%s'\n", t.c_str());
            return EXIT_FAILURE;
        }
    }
    op->pressure_sideset = want_pressure_sideset;
    if (!op->pressure_sideset.empty())
        op->pressure_value = smesh::Env::read<real_t>("SFEM_PRESSURE", real_t(0));
    if (op->initialize() != SFEM_SUCCESS) return EXIT_FAILURE;
    // The Newton loop below evaluates the residual immediately after every step and
    // before the linear solve, which is the condition this option asks for: the nodal
    // pressure gradient is then current for the whole Krylov sweep and need not be
    // rebuilt on each of its hundreds of applies. SFEM_PGRAD_CACHE=0 turns it off.
    const int pgrad_cache = smesh::Env::read<int>("SFEM_PGRAD_CACHE", 1);
    op->set_option("cache_nodal_pgrad", pgrad_cache != 0);
    f->add_operator(op);

    const ptrdiff_t     nnodes = mesh->n_nodes();
    const ptrdiff_t     ndof   = nnodes * N_FIELDS;
    std::vector<real_t> p_exact;
    ptrdiff_t           pin_node = 0;  // pressure pin, needed again by the MMS diagnostics
    std::shared_ptr<sfem::DirichletConditions> dirichlet;

    // Built after op->initialize(), and that order is required rather than incidental:
    // initialize() renumbers the mesh nodes for the packed layout, so node indices taken
    // before it would refer to the old numbering.
    //
    // Boundary conditions, node by node, matching mark_constraints in the standalone
    // driver: no-slip and inlet/outlet fix all three velocity components to the exact
    // profile, the spanwise planes fix uz alone, and the pressure gets a single pin
    // because the continuity equations only determine it up to a constant.
    {
        const auto *const px = mesh->points()->data()[0];
        const auto *const py = mesh->points()->data()[1];
        const auto *const pz = mesh->points()->data()[2];

        {
            // Checksum the mesh coordinates. p_exact is a serial function of these, so if
            // it varies between runs, they do.
            long double cx = 0, cy = 0, cz = 0;
            for (ptrdiff_t i = 0; i < nnodes; ++i) {
                cx += (long double)px[i];
                cy += (long double)py[i];
                cz += (long double)pz[i];
            }
            std::printf("mesh coords checksum: %.17g %.17g %.17g\n", (double)cx, (double)cy, (double)cz);
        }

        const bool step_outflow_natural =
                smesh::Env::read_string("SFEM_STEP_OUTFLOW", "natural") != "dirichlet";

        // Which nodes lie on the domain skin. Topological, so the step faces are included.
        std::vector<char> skin_node((size_t)nnodes, 0);
        if (flow == cvfem_case::FlowCase::Step) {
            auto skin = smesh::skin_sideset(mesh);
            if (!skin) {
                std::fprintf(stderr, "step: skin_sideset failed\n");
                return EXIT_FAILURE;
            }
            auto ns = smesh::create_nodeset_from_sideset(mesh, skin);
            if (!ns) {
                std::fprintf(stderr, "step: create_nodeset_from_sideset failed\n");
                return EXIT_FAILURE;
            }
            for (ptrdiff_t k = 0; k < ns->size(); ++k) skin_node[(size_t)ns->data()[k]] = 1;
            ptrdiff_t n_skin = 0;
            for (auto c : skin_node) n_skin += c;
            std::printf("step: skin nodes %td of %td\n", n_skin, nnodes);
        }

        std::vector<idx_t>  uvw_nodes, uz_nodes;
        std::vector<real_t> uvw_ux, uvw_uy, uvw_uz, uz_vals;
        p_exact.assign((size_t)nnodes, real_t(0));
        ptrdiff_t          &pin  = pin_node;
        real_t              best = 1e300;

        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            const real_t x = (real_t)px[i], y = (real_t)py[i], z = (real_t)pz[i];
            if (x + y + z < best) {
                best = x + y + z;
                pin  = i;
            }

            real_t ux, uy, uz, p;
            cvfem_case::exact_state(flow, mu, U, Lx, Ly, x, y, z, ux, uy, uz, p);
            p_exact[(size_t)i] = p;

            const bool wall_y = cvfem_case::on_plane(y, real_t(0), Ly) || cvfem_case::on_plane(y, Ly, Ly);
            const bool inlet  = cvfem_case::on_plane(x, real_t(0), Lx);
            const bool outlet = cvfem_case::on_plane(x, Lx, Lx);
            const bool span   = cvfem_case::on_plane(z, real_t(0), Lz) || cvfem_case::on_plane(z, Lz, Lz);

            if (flow == cvfem_case::FlowCase::Step) {
                // Constrain the skin, minus the outflow plane. The skin comes from the same
                // smesh::skin_sideset that builds the boundary mask, so the Dirichlet set and
                // the control-volume closure agree by construction.
                //
                // That matters more than it sounds. cvfem_ns_channel_case.hpp documents the
                // invariant that boundary marking and the sub-control-surface test must decide
                // the same thing, "and if they disagree a node gets a closed control volume
                // without a boundary condition, or the reverse". On a box two independent
                // coordinate tests happen to agree; on the L-shape they would not, because the
                // step faces lie on no bounding-box plane. Deriving both from one skin removes
                // the possibility rather than testing for it.
                //
                // Outflow nodes are simply left out: no Dirichlet condition, and the boundary
                // sub-control-surface term then evaluates the flux from the interior state.
                // SFEM_STEP_OUTFLOW selects the outlet treatment.
                //   natural   (default) leave x=Lx unconstrained; the boundary
                //             sub-control-surface term then evaluates the flux from the
                //             interior state -- a zero-gradient finite-volume outflow.
                //   dirichlet impose the inflow profile's fully-developed counterpart. Not
                //             their boundary condition, so any number produced under it must
                //             be labelled as such; it exists to separate an outflow problem
                //             from a geometry or marking problem.
                const bool outflow = cvfem_case::on_plane(x, Lx, Lx);
                const bool free_outlet = step_outflow_natural && outflow;
                if (skin_node[(size_t)i] && !free_outlet) {
                    uvw_nodes.push_back((idx_t)i);
                    uvw_ux.push_back(ux);
                    uvw_uy.push_back(uy);
                    uvw_uz.push_back(uz);
                }
            } else if (flow == cvfem_case::FlowCase::Pump) {
                // Every wall of the chamber, all three components, EXCEPT the port.
                //
                // The diaphragm is in here too and is not a special case: it is a wall whose
                // prescribed velocity happens to be non-zero and normal, which is what makes
                // transpiration cost no new constraint machinery. exact_state supplies
                // (0, -U, 0) there and zero elsewhere.
                //
                // The port is left out entirely -- no velocity data at all -- because it
                // carries the prescribed pressure, and a face cannot carry both. Constrain
                // it here and the port is overridden while the log still reports it applied,
                // which is the failure the outlet_governed test above exists to prevent.
                const bool on_port = cvfem_case::pump_on_port<real_t>(x, y, z, Lx, Ly, Lz, pump_port_frac);
                const bool on_wall = wall_y || inlet || outlet || span;  // the chamber is a box
                if (on_wall && !on_port) {
                    uvw_nodes.push_back((idx_t)i);
                    uvw_ux.push_back(ux);
                    uvw_uy.push_back(uy);
                    uvw_uz.push_back(uz);
                }
            } else if (flow == cvfem_case::FlowCase::CavityRegularized) {
                // No-slip on every wall including the spanwise pair, with the lid profile on
                // y = Ly. This is a genuinely three-dimensional cavity, which is what their
                // section 5.5 solves and what Table 5.6 is measured on.
                //
                // Note this differs deliberately from FlowCase::Cavity, which leaves the
                // z-planes with uz = 0 only -- a slip/symmetry condition that makes the flow
                // quasi-two-dimensional. That is the right choice there, because our
                // constant-lid cavity is compared against Ghia et al.'s 2D reference data;
                // it is the wrong choice here. Same geometry, different problem.
                if (wall_y || inlet || outlet || span) {
                    uvw_nodes.push_back((idx_t)i);
                    uvw_ux.push_back(ux);
                    uvw_uy.push_back(uy);
                    uvw_uz.push_back(uz);
                }
            } else if (flow == cvfem_case::FlowCase::MMS) {
                // Every boundary node, all three components. The manufactured field has a
                // nonzero tangential velocity on the z-planes too, so the channel pattern
                // below (which constrains only uz there) would impose the wrong data.
                if (wall_y || inlet || outlet || span) {
                    uvw_nodes.push_back((idx_t)i);
                    uvw_ux.push_back(ux);
                    uvw_uy.push_back(uy);
                    uvw_uz.push_back(uz);
                }
            } else if (wall_y || inlet || (outlet && !outlet_governed)) {
                uvw_nodes.push_back((idx_t)i);
                uvw_ux.push_back(ux);
                uvw_uy.push_back(uy);
                uvw_uz.push_back(uz);
            } else if (span) {
                // Only where the velocity is not already fully constrained, so no node
                // appears twice for the same component.
                uz_nodes.push_back((idx_t)i);
                uz_vals.push_back(uz);
            }
        }

        real_t pux, puy, puz, pp;
        cvfem_case::exact_state(
                flow, mu, U, Lx, Ly, (real_t)px[pin], (real_t)py[pin], (real_t)pz[pin], pux, puy, puz, pp);

        // Conditions are built with owned buffers rather than through the raw-pointer
        // add_condition overloads: those call manage_host_buffer, which takes ownership
        // of the pointer, so handing them a std::vector's storage both dangles and frees
        // memory the vector still owns.
        auto make_cond = [](const std::vector<idx_t>  &nodes,
                            const std::vector<real_t> &vals,
                            const int                  component) {
            sfem::DirichletConditions::Condition c;
            c.component = component;
            c.nodeset   = smesh::create_host_buffer<idx_t>(nodes.size());
            c.values    = smesh::create_host_buffer<real_t>(vals.size());
            std::copy(nodes.begin(), nodes.end(), c.nodeset->data());
            std::copy(vals.begin(), vals.end(), c.values->data());
            return c;
        };

        std::vector<sfem::DirichletConditions::Condition> conds;
        conds.push_back(make_cond(uvw_nodes, uvw_ux, 0));
        conds.push_back(make_cond(uvw_nodes, uvw_uy, 1));
        conds.push_back(make_cond(uvw_nodes, uvw_uz, 2));
        if (!uz_nodes.empty()) conds.push_back(make_cond(uz_nodes, uz_vals, 2));
        // The pressure is left unconstrained and its gauge is fixed by a zero-mean
        // projection instead (see PressureGauge). Pinning a node is the cheaper-looking
        // option and the more expensive one: it leaves a near-constant mode that puts an
        // isolated eigenvalue far below the rest of the spectrum and degrades like h^-2.
        //
        // SFEM_PIN_PRESSURE=1 restores the pin, which is worth having to reproduce older
        // numbers and to compare the two treatments directly.
        if (smesh::Env::read<int>("SFEM_PIN_PRESSURE", 0))
            conds.push_back(make_cond({(idx_t)pin}, {pp}, 3));
        else
            std::printf("pressure pin: DISABLED\n");

        // Kept rather than discarded: the pump drives its diaphragm by rescaling these
        // values every time step through set_time, which is the mechanism
        // DirichletConditions already has for a time-varying load and which spares this
        // driver from rebuilding the whole constraint set once per step.
        dirichlet = sfem::DirichletConditions::create(fs, conds);
        f->add_constraint(dirichlet);

        // The manufactured solution is driven by a body force. This must happen after
        // op->initialize() -- which ran above -- because the packed path renumbers mesh
        // nodes, and a force built against the old numbering would be silently scrambled
        // rather than rejected.
        if (flow == cvfem_case::FlowCase::MMS)
            std::printf("mms: forcing recomputed per continuation stage (rho varies, mu=%g, Re=%g)\n",
                        (double)mu, (double)(1.0 / mu));

        std::printf("constraints: uvw_nodes=%td  uz_nodes=%td  p_pin=%td\n",
                    (ptrdiff_t)uvw_nodes.size(),
                    (ptrdiff_t)uz_nodes.size(),
                    pin);
    }

    std::printf("case: %s  geom: %s  refine_level: %d  semi_structured: %d\n",
                case_name.c_str(), geom_name.c_str(), refine_level, op->is_semi_structured() ? 1 : 0);
    std::printf("channel: L=(%g,%g,%g)  cells=(%d,%d,%d)\n", Lx, Ly, Lz, nx, ny, nz);
    std::printf("nnodes: %td  nelements: %td  ndof: %td\n", nnodes, mesh->n_elements(0), ndof);
    if (flow == cvfem_case::FlowCase::MMS) {
        // Two different quantities would otherwise both be printed as "Re": the driver's
        // flow Reynolds number rho*U*Ly/mu, and the manufactured solution's own parameter
        // 1/mu which appears in its pressure formula. On the MMS domain Ly=2, so they differ
        // by a factor of two and a run asked for Re=200 would report 400.
        std::printf("rho: %g  mu: %g  mms_Re (=1/mu, the parameter in p): %g   "
                    "[continuation ramps rho to 1]\n", rho, mu, 1.0 / mu);
    } else {
        std::printf("rho: %g  mu: %g  U: %g  Re: %g\n", rho, mu, U, rho * U * Ly / mu);
    }

    // The state lives in a SharedBuffer because the Jacobian operator is built from it:
    // create_linear_operator assembles once, at construction, so a nonlinear problem has
    // to rebuild it per Newton step against the current state.
    auto                xbuf = smesh::create_host_buffer<real_t>((size_t)ndof);
    real_t *const       x    = xbuf->data();
    std::vector<real_t> r((size_t)ndof, 0), dx((size_t)ndof, 0), rhs((size_t)ndof, 0);
    std::fill(x, x + ndof, real_t(0));
    f->apply_constraints(x);
    // Seed the whole pressure field with the analytic pressure, not just the pinned node.
    // The standalone driver's init_fields does the same -- velocity respects the
    // constraint mask, pressure is set everywhere -- and it matters: starting from p = 0
    // leaves Newton converging linearly at a far worse rate, needing several times the
    // iterations for the same answer. Verification drivers against an analytic solution
    // are entitled to the better initial guess; the two must simply agree about it.
    for (ptrdiff_t i = 0; i < nnodes; ++i) x[(size_t)i * 4 + 3] = p_exact[(size_t)i];
    {
        // Checksum the state at construction, before any solver has touched it, to say
        // whether the variation seen later is built in or acquired.
        real_t s0 = 0, sp = 0;
        for (ptrdiff_t i = 0; i < ndof; ++i) s0 += x[(size_t)i];
        for (ptrdiff_t i = 0; i < nnodes; ++i) sp += p_exact[(size_t)i];
        std::printf("initial state checksum: %.17g   p_exact: %.17g\n", (double)s0, (double)sp);
    }

    std::vector<mask_t> cmask(mask_count(ndof), 0);
    f->constraints_mask(cmask.data());

    // Outlet nodes, for the active-set trace in the Newton loop. Empty for every case that
    // has no outlet, which is what switches the trace off for them.
    std::vector<ptrdiff_t> outlet_nodes;
    std::vector<uint8_t>   outlet_active;
    if (flow == cvfem_case::FlowCase::Step) {
        const auto *const pxo = mesh->points()->data()[0];
        for (ptrdiff_t i = 0; i < nnodes; ++i)
            if (cvfem_case::on_plane((real_t)pxo[i], Lx, Lx)) outlet_nodes.push_back(i);
        outlet_active.assign(outlet_nodes.size(), 0);
    }

    // The gauge applies exactly when nothing else determines the pressure level, and there
    // are two ways it can be determined -- one obvious, one not.
    //
    // The obvious one is a constrained pressure dof: a pin, or a Dirichlet value.
    //
    // The other is a do-nothing outflow. That boundary drops the p_i*a term, and dropping it
    // is exactly what removes the constant-pressure null mode: with p_i*a retained a uniform
    // pressure shift integrates to zero over every closed control volume and the level stays
    // free, whereas without it the shift leaves a net force on the outflow control volumes.
    // So an open outlet is a pressure condition even though it constrains no dof, and adding
    // a zero-mean condition on top over-determines the system. Measured, doing so took the
    // backward-facing step's continuity residual from 6.8e-09 to 5.4e-03.
    PressureGauge gauge(ndof, cmask.data());
    {
        ptrdiff_t n_p_free = 0, n_p = 0;
        for (ptrdiff_t k = 3; k < ndof; k += N_FIELDS) {
            ++n_p;
            if (!mask_get(k, cmask.data())) ++n_p_free;
        }
        // Ask the operator rather than testing one of the three conditions that can do it.
        // A traction surface and a pressure port fix the level exactly as the do-nothing
        // outflow does, and this test used to see only the outflow -- so naming a port
        // would have left the zero-mean gauge on top of it and over-determined the system.
        const bool outflow_fixes_it = op->fixes_pressure_level();
        gauge.set_active(n_p_free == n_p && !outflow_fixes_it);
        std::printf("pressure gauge: %s  (%td of %td pressure dofs free)\n",
                    gauge.active()      ? "zero mean"
                    : !op->pressure_sideset.empty() ? "determined by the prescribed pressure"
                    : !op->traction_sideset.empty() ? "determined by the traction surface"
                    : outflow_fixes_it  ? "determined by the do-nothing outflow"
                                        : "constrained (pin or Dirichlet)",
                    n_p_free, n_p);
    }

    // Reynolds continuation, matching the standalone driver. Newton from a zero state
    // does not converge at Re=100: the first stage solves the same geometry at Re=1 by
    // taking rho = mu / (U Ly), and the second continues from that solution at the
    // physical density. Without it this diverges to inf, which is how its absence was
    // found rather than reasoned about.
    const real_t Re_phys = rho * U * Ly / std::max(mu, real_t(1e-30));
    const real_t rho_re1 = mu / std::max(U * Ly, real_t(1e-30));

    // The two-stage scheme above stops working between Re=200 and Re=400.
    //
    // Measured at 33,124 dofs: Re=100 converges in 32 Newton steps; Re=200 reaches the
    // right answer (u_linf 2.90e-07 against the converged Re=100 run's 2.55e-07, both at
    // discretisation accuracy) but does not satisfy the Newton test within 40 steps; Re=400,
    // 800 and 3200 diverge outright, to 1e+13, 1e+112 and 1e+87. Every failing run reaches
    // the navier-stokes stage, so the Re=1 solve always succeeds and the blow-up is always
    // on the single jump to physical density. The linear solves stay healthy throughout --
    // 663 to 1289 iterations per Newton step with no trend against Re -- so it is Newton
    // diverging, not the preconditioner. The jump is what fails, so ramp it.
    //
    // Geometric in rho, which is geometric in Re since Re is linear in rho at fixed mu, U
    // and Ly. SFEM_RE_STEP is the ratio per stage; 4 gives Re = 1, 4, 16, 64, ... and seven
    // stages to reach 3200. Each stage starts from the previous stage's solution.
    //
    // A stage that fails is not fatal: the state is rolled back and a stage is inserted at
    // the geometric mean of the last success and the failure, up to SFEM_RE_MAX_RETRY times.
    // That is the cheap half of pseudo-transient continuation -- no timestep term in the
    // operator, just a smaller step in the parameter -- and it costs nothing when the ramp
    // is already fine enough, since no retry happens.
    const real_t re_step   = std::max(real_t(1.5), smesh::Env::read<real_t>("SFEM_RE_STEP", real_t(4)));
    const int    re_retry  = smesh::Env::read<int>("SFEM_RE_MAX_RETRY", 6);
    // Adaptive step control. The schedule above is only a starting guess: what actually has
    // to adapt during a run is the *increment*, not the destination. See the success and
    // failure blocks in the stage loop for why aiming each stage at the final target wastes
    // the retry budget.
    const bool   re_adapt  = smesh::Env::read<int>("SFEM_RE_ADAPT", 1) != 0;
    const real_t re_grow   = std::max(real_t(1), smesh::Env::read<real_t>("SFEM_RE_GROW", real_t(1.5)));
    const real_t re_shrink = smesh::Env::read<real_t>("SFEM_RE_SHRINK", real_t(0.5));
    const real_t re_fmin   = smesh::Env::read<real_t>("SFEM_RE_STEP_MIN", real_t(1.02));

    std::vector<real_t> rho_schedule;
    if (rho == real_t(0) || Re_phys <= real_t(1.5)) {
        rho_schedule.push_back(rho);
    } else {
        real_t r = rho_re1;
        rho_schedule.push_back(r);
        while (r * re_step < rho) {
            r *= re_step;
            rho_schedule.push_back(r);
        }
        rho_schedule.push_back(rho);
    }
    {
        std::printf("continuation: %d stages, Re =", (int)rho_schedule.size());
        for (size_t k = 0; k < rho_schedule.size(); ++k)
            std::printf(" %g", (double)(rho_schedule[k] * U * Ly / std::max(mu, real_t(1e-30))));
        std::printf("\n");
    }
    // SFEM_FD_CHECK: is the Jacobian action actually the derivative of the residual?
    //
    // cvfem_ns_op_gate compares the assembled matrix against the matrix-free action, which
    // catches a wiring mistake but not a modelling one: both encode the same linearisation, so
    // a term missing from both is invisible to it. This compares J*v against a central
    // difference of the residual, which has no such blind spot.
    //
    // Mode 2 evaluates at a random state, where every sub-control surface has mdot != 0.
    // Mode 1 evaluates at the current iterate, which for Poiseuille has mdot ~ 0 on the faces
    // perpendicular to the flow -- exactly where the upwind switch sgn(mdot) is not
    // differentiable. If the Jacobian is exact in mode 2 and inexact in mode 1, the upwind
    // kink is the cause of the linear Newton tail, and no solver tuning will remove it.
    // Fine-level Rhie-Chow scale. Setting this to 0 removes the stabilisation -- and with it
    // the only Jacobian term that is not exact -- which is how the missing pg derivative is
    // shown to be what caps Newton at a linear rate, rather than merely being inexact.
    if (const real_t rc_override = smesh::Env::read<real_t>("SFEM_RC_SCALE", real_t(-1));
        rc_override >= real_t(0)) {
        op->rhie_chow_scale = rc_override;
        op->update(x);
        std::printf("rhie_chow_scale overridden to %g\n", (double)rc_override);
    }

    if (const int fd_mode = smesh::Env::read<int>("SFEM_FD_CHECK", 0)) {
        std::vector<real_t> v((size_t)ndof), jv((size_t)ndof), rp((size_t)ndof),
                            rm((size_t)ndof), xt((size_t)ndof), xb(x, x + ndof);
        std::mt19937                       gen(12345u);
        std::uniform_real_distribution<real_t> dist(real_t(-1), real_t(1));
        for (ptrdiff_t i = 0; i < ndof; ++i) v[(size_t)i] = dist(gen);
        if (fd_mode == 2)
            for (ptrdiff_t i = 0; i < ndof; ++i) xb[(size_t)i] = real_t(0.1) * dist(gen);
        f->apply_zero_constraints(v.data());

        // SFEM_FD_NO_RC=1 switches Rhie-Chow off. The suspected missing term is the
        // derivative of the reconstructed nodal pressure gradient inside the Rhie-Chow
        // correction, so with rc off the Jacobian should be exact and the error collapse.
        if (smesh::Env::read<int>("SFEM_FD_NO_RC", 0)) {
            op->rhie_chow_scale = real_t(0);
            op->update(xb.data());
            std::printf("fd_check: Rhie-Chow DISABLED\n");
        }

        std::fill(jv.begin(), jv.end(), real_t(0));
        f->apply(xb.data(), v.data(), jv.data());
        f->apply_zero_constraints(jv.data());

        std::printf("fd_check: mode %d (%s), ndof %ld\n", fd_mode,
                    fd_mode == 2 ? "random state, mdot != 0" : "current iterate", (long)ndof);
        for (const real_t eps : {real_t(1e-3), real_t(1e-4), real_t(1e-5), real_t(1e-6),
                                 real_t(1e-7), real_t(1e-8)}) {
            for (ptrdiff_t i = 0; i < ndof; ++i) xt[(size_t)i] = xb[(size_t)i] + eps * v[(size_t)i];
            std::fill(rp.begin(), rp.end(), real_t(0));
            f->gradient(xt.data(), rp.data());
            f->apply_zero_constraints(rp.data());
            for (ptrdiff_t i = 0; i < ndof; ++i) xt[(size_t)i] = xb[(size_t)i] - eps * v[(size_t)i];
            std::fill(rm.begin(), rm.end(), real_t(0));
            f->gradient(xt.data(), rm.data());
            f->apply_zero_constraints(rm.data());
            // Split by field. A discrepancy confined to the continuity rows or to the
            // pressure columns is invisible in a global norm dominated by momentum, yet it is
            // exactly what would set Newton's asymptotic rate.
            real_t num = 0, den = 0, nmom = 0, dmom = 0, ncon = 0, dcon = 0;
            real_t worst = 0; ptrdiff_t worst_i = -1;
            for (ptrdiff_t i = 0; i < ndof; ++i) {
                const real_t fd = (rp[(size_t)i] - rm[(size_t)i]) / (real_t(2) * eps);
                const real_t d  = fd - jv[(size_t)i];
                num += d * d; den += fd * fd;
                if (i % 4 == 3) { ncon += d * d; dcon += fd * fd; }
                else            { nmom += d * d; dmom += fd * fd; }
                if (std::fabs(d) > worst) { worst = std::fabs(d); worst_i = i; }
            }
            std::printf("  eps %.1e  all %.4e  momentum %.4e  continuity %.4e"
                        "  worst |d| %.3e at dof %ld (field %ld)\n",
                        (double)eps,
                        (double)std::sqrt(num / std::max(den, real_t(1e-300))),
                        (double)std::sqrt(nmom / std::max(dmom, real_t(1e-300))),
                        (double)std::sqrt(ncon / std::max(dcon, real_t(1e-300))),
                        (double)worst, (long)worst_i, (long)(worst_i % 4));
        }
        return 0;
    }

    std::vector<real_t> x_stage_start((size_t)ndof, real_t(0));
    int                 re_retries = 0;
    real_t              step_f     = re_step;  // current continuation step factor

    double t_op    = 0;  // building the Jacobian operator (assembly, or nothing)
    double t_prec  = 0;  // building the block-Jacobi preconditioner
    double t_solve = 0;  // the Krylov solve itself

    // Matrix-free reads the state buffer on every apply, so one operator tracks Newton
    // for the whole solve. The assembled one is a snapshot and has to be rebuilt.
    std::shared_ptr<sfem::Operator<real_t>> mf_op;
    if (matrix_free) {
        const double t0 = smesh::time_seconds();
        mf_op = sfem::create_linear_operator(sfem::op_type::MATRIX_FREE, f, xbuf, sfem::EXECUTION_SPACE_HOST);
        t_op += smesh::time_seconds() - t0;
    }

    // Built once: the hierarchy and its transfer operators depend on the mesh, not the
    // state. The level states are refreshed per Newton step below, since they do.
    std::shared_ptr<GmgLevels> gmg;
    if (use_gmg == 1) {  // 2 is the no-hierarchy control and must not build one
        gmg = build_gmg(f, op, xbuf, gmg_smooth);
        if (gmg) build_state_weights(*gmg);
        if (gmg) build_transfer_matrices(*gmg);
        if (gmg && smesh::Env::read<int>("SFEM_GMG_CHECK", 0)) {
            refresh_gmg(*gmg);
            if (smesh::Env::read<int>("SFEM_GMG_CHECK", 0) == 4) {
                check_derefined_op(*gmg);
                check_cgc(*gmg, 0);
                check_cgc(*gmg, 3);
            } else {
                check_transfers(*gmg);
            }
        }
        if (!gmg) {
            std::fprintf(stderr, "SFEM_GMG=1 but the hierarchy could not be built "
                                 "(needs a semi-structured mesh with more than one level)\n");
            return EXIT_FAILURE;
        }
        std::printf("gmg: %zu levels, %d smoothing steps\n", gmg->ops.size(), gmg_smooth);
    }

    int  newton_it    = 0;
    int  lin_it_total = 0;
    // Newton steps summed over every continuation stage. `newton_it` below is the inner loop
    // variable and restarts at each stage, so on its own it reports only the last stage --
    // which, after a good ramp, is the cheapest one. Printing it beside the cumulative
    // lin_it_total invited exactly the wrong reading: "3 Newton steps, 1346 linear iterations".
    int    newton_total = 0;
    int    stages_run   = 0;
    real_t rho_solved   = 0;  // highest rho whose stage actually converged

    // Globalization. Two things the Newton loop was missing, both standard practice in the
    // implicit CFD codes surveyed in docs/CVFEM_SotA.tex:
    //
    //   * a residual merit function controlling the step, so an update that does not reduce
    //     ||R|| is backtracked rather than accepted. LAURA's Algorithm 1 does exactly this and
    //     treats a line-search parameter below 0.1 as the signal to give up on the step;
    //     HANIM discards any update failing its merit test and never re-linearises.
    //   * early divergence detection, so a stage that is going to fail fails FAST. At Re=1000
    //     the Re=1000 stage burned the full 40 Newton steps on each of four attempts while the
    //     residual grew monotonically -- 344 Newton steps and 183 s to discover the ramp could
    //     not get past 843.
    //
    // Both feed the continuation: a stage that reports failure early triggers the adaptive
    // bisection that much sooner.
    const bool   ls_on     = smesh::Env::read<int>("SFEM_NL_LINESEARCH", 1) != 0;
    const int    ls_max    = smesh::Env::read<int>("SFEM_NL_MAX_LS", 8);
    const real_t ls_armijo = smesh::Env::read<real_t>("SFEM_NL_ARMIJO", real_t(1e-4));
    const real_t div_grow  = smesh::Env::read<real_t>("SFEM_NL_DIVERGE", real_t(1e3));
    // Relative residual below which a line search that cannot improve means "converged", not
    // "failed": at the round-off floor there is no decrease left to find.
    // Ten times nl_rtol, not a hundred: a line search that cannot improve a residual already
    // this small has genuinely reached the floor, but anything looser starts accepting states
    // that simply have not converged.
    const real_t nl_ls_floor = smesh::Env::read<real_t>("SFEM_NL_LS_FLOOR", real_t(1e-7));
    std::vector<real_t> x_try((size_t)ndof, 0), r_try((size_t)ndof, 0);
    bool converged    = false;
    // Set once from the first nonzero residual and kept across stages, as in the
    // standalone driver: the continuation stage and the physical stage are measured
    // against the same reference.
    real_t r0 = 0;

    // ---------------------------------------------------------------- time stepping
    //
    // One time step is a full continuation-and-Newton solve of the transient residual, so
    // everything below -- the Reynolds ramp, the adaptive retry, the divergence detection --
    // works unchanged inside a step. With SFEM_DT = 0 the loop runs exactly once and the
    // operator has no time term, which is bit-for-bit the steady solve.
    std::vector<real_t> u_hist, u_hist2;
    if (dt_step > real_t(0)) {
        op->set_time_step(dt_step, bdf_order);
        u_hist.assign((size_t)nnodes * 3, real_t(0));
        // The initial condition is whatever the state holds when stepping starts, so the
        // first step is consistent with it rather than with an implied zero field.
        for (ptrdiff_t i = 0; i < nnodes; ++i)
            for (int c = 0; c < 3; ++c) u_hist[(size_t)i * 3 + (size_t)c] = x[(size_t)i * 4 + (size_t)c];
        op->set_velocity_history(u_hist.data(), nullptr);
        std::printf("transient: dt %g, %d steps, BDF%d\n", (double)dt_step, nsteps, bdf_order);
    }

    // The diaphragm waveform. A steady run leaves this at 1, which is a diaphragm held at
    // a constant displacement rate -- not a physical pump cycle, but the configuration in
    // which the swept-volume identity is easiest to read, and the one the verification
    // harness checks.
    //
    // With a timestep it becomes V sin(2 pi t / T). Note this does NOT rectify: the port is
    // an opening with no valve, so over a full cycle the chamber breathes in and out and
    // nets nothing. Rectification needs the port's condition to depend on the sign of its
    // own flux, which is a different and much less pleasant problem, and is out of scope.
    const real_t pump_period = smesh::Env::read<real_t>("SFEM_PUMP_PERIOD", real_t(1));
    real_t       pump_scale  = 1;

    for (int tstep = 0; tstep < (dt_step > real_t(0) ? nsteps : 1); ++tstep) {
    if (dt_step > real_t(0)) std::printf("=== step %d/%d  t = %g ===\n", tstep + 1, nsteps,
                                         (double)((tstep + 1) * dt_step));
    if (want_pump && dt_step > real_t(0) && dirichlet) {
        const real_t t_now = (tstep + 1) * dt_step;
        pump_scale         = std::sin(real_t(2) * real_t(M_PI) * t_now / pump_period);
        // set_time snapshots the base values on its first call and thereafter writes
        // scale * base, so passing the waveform as the global scale drives every prescribed
        // velocity together. In this case only the diaphragm is non-zero, so that is exactly
        // the diaphragm; the no-slip walls scale from zero to zero.
        dirichlet->set_time(t_now, pump_scale);
        // And put the new values into the state. set_time rewrites what the constraint
        // holds; it does not touch x, and the Newton loop only ever applies ZERO constraints
        // to its correction, so without this the diaphragm keeps whatever velocity the
        // initial apply_constraints gave it and the waveform is a number in a log line. It
        // read v_diaphragm = 0 at the end of a cycle while still carrying its full amplitude
        // of flux.
        f->apply_constraints(x);
        std::printf("pump: t = %g  v_diaphragm = %g\n", (double)t_now, (double)(U * pump_scale));
    }

    for (size_t stage = 0; stage < rho_schedule.size(); ++stage) {
    const real_t rho_use = rho_schedule[stage];
    // The manufactured forcing is a function of rho, so it must track the continuation. The
    // exact solution does not move -- u and p depend only on mu -- which is precisely why
    // the error measured at the final stage is still against the right reference.
    if (flow == cvfem_case::FlowCase::MMS) {
        const auto *const mx = mesh->points()->data()[0];
        const auto *const my = mesh->points()->data()[1];
        const auto *const mz = mesh->points()->data()[2];
        std::vector<real_t> bfx((size_t)nnodes), bfy((size_t)nnodes), bfz((size_t)nnodes);
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            cvfem_case::body_force(flow, rho_use, mu, (real_t)mx[i], (real_t)my[i], (real_t)mz[i],
                                   bfx[(size_t)i], bfy[(size_t)i], bfz[(size_t)i]);
        }
        op->set_body_force(bfx.data(), bfy.data(), bfz.data());
    }
    op->rho              = rho_use;
    std::copy(x, x + ndof, x_stage_start.begin());
    std::printf("stage %d/%d: rho: %g  Re: %g\n",
                (int)stage + 1,
                (int)rho_schedule.size(),
                (double)rho_use,
                (double)(rho_use * U * Ly / std::max(mu, real_t(1e-30))));

    converged = false;
    real_t prev_rnorm = 0; // previous Newton residual, for the adaptive band's rate test
    real_t r_stage0 = 0;   // this stage's initial residual, the divergence reference
    bool   diverged = false;
    int    lin_diverged_count = 0;
    for (newton_it = 0; newton_it <= max_newton; ++newton_it) {
        std::fill(r.begin(), r.end(), real_t(0));
        { const double t0 = smesh::time_seconds();
          f->gradient(x, r.data());
          phase_add("newton_residual", smesh::time_seconds() - t0); }
        // Measure the residual on the free dofs only. Function::gradient leaves the
        // boundary-condition residual (x - value) in the constrained rows, which is a
        // different quantity from the equation residual and never decays to zero the way
        // the Newton test expects -- it stalls the relative criterion near the solution.
        // The standalone driver zeroes them for the same reason.
        f->apply_zero_constraints(r.data());
        // Make the right-hand side compatible: remove the component along null(A^T).
        gauge.project(r.data());

        // SFEM_FD_AT_IT: run the finite-difference Jacobian check at *this* iterate.
        //
        // The check above the Newton loop can only ever evaluate the initial state, where
        // almost every sub-control surface has mdot near zero and the upwind switch sits on
        // its corner, so it cannot say whether the Jacobian is wrong where Newton actually
        // stalls. Evaluating it at a chosen iteration can. A central difference of a smooth
        // function has O(eps^2) truncation error; O(eps) is the signature of a corner being
        // crossed, and a plateau is the signature of a genuinely wrong derivative. The three
        // are distinguishable only by watching the error as eps shrinks.
        if (smesh::Env::read<int>("SFEM_FD_AT_IT", -1) == newton_it) {
            std::vector<real_t> v((size_t)ndof), jv((size_t)ndof), rp((size_t)ndof),
                    rm((size_t)ndof), xt((size_t)ndof);
            std::mt19937                           g2(12345u);
            std::uniform_real_distribution<real_t> d2(real_t(-1), real_t(1));
            for (ptrdiff_t i = 0; i < ndof; ++i) v[(size_t)i] = d2(g2);
            f->apply_zero_constraints(v.data());
            std::fill(jv.begin(), jv.end(), real_t(0));
            f->apply(x, v.data(), jv.data());
            f->apply_zero_constraints(jv.data());
            std::printf("fd_at_it %d: ndof %ld\n", newton_it, (long)ndof);
            for (const real_t eps : {real_t(1e-4), real_t(1e-5), real_t(1e-6), real_t(1e-7)}) {
                for (ptrdiff_t i = 0; i < ndof; ++i) xt[(size_t)i] = x[(size_t)i] + eps * v[(size_t)i];
                std::fill(rp.begin(), rp.end(), real_t(0));
                f->gradient(xt.data(), rp.data());
                f->apply_zero_constraints(rp.data());
                for (ptrdiff_t i = 0; i < ndof; ++i) xt[(size_t)i] = x[(size_t)i] - eps * v[(size_t)i];
                std::fill(rm.begin(), rm.end(), real_t(0));
                f->gradient(xt.data(), rm.data());
                f->apply_zero_constraints(rm.data());
                real_t nmom = 0, dmom = 0, ncon = 0, dcon = 0;
                for (ptrdiff_t i = 0; i < ndof; ++i) {
                    const real_t fd = (rp[(size_t)i] - rm[(size_t)i]) / (real_t(2) * eps);
                    const real_t d  = fd - jv[(size_t)i];
                    if (i % 4 == 3) { ncon += d * d; dcon += fd * fd; }
                    else            { nmom += d * d; dmom += fd * fd; }
                }
                std::printf("  eps %.1e  momentum %.4e  continuity %.4e\n", (double)eps,
                            (double)std::sqrt(nmom / std::max(dmom, real_t(1e-300))),
                            (double)std::sqrt(ncon / std::max(dcon, real_t(1e-300))));
            }
        }

        // Track the backflow guard's active set across Newton iterations.
        //
        // max(mdot, 0) is semismooth, and the Jacobian assembled for it is a genuine element
        // of the Clarke generalized Jacobian -- the mdot > 0 branch differentiated, zero
        // elsewhere -- so this is already a semismooth Newton method and should converge
        // superlinearly. When it does not, the usual cause is the active set failing to
        // settle: the iterates cycle between branch assignments and each linearisation solves
        // for a different problem. Counting the set and its changes per iteration is what
        // separates that from a merely inaccurate Jacobian, and the two want different
        // remedies -- an active-set or smoothing treatment for the first, a better derivative
        // for the second.
        if (smesh::Env::read<int>("SFEM_ACTIVE_SET_TRACE", 0) && !outlet_nodes.empty()) {
            ptrdiff_t n_pos = 0, n_flip = 0;
            for (size_t k = 0; k < outlet_nodes.size(); ++k) {
                const bool pos = x[(size_t)outlet_nodes[k] * 4 + 0] > real_t(0);
                if (pos) ++n_pos;
                if (newton_it > 0 && pos != (bool)outlet_active[k]) ++n_flip;
                outlet_active[k] = pos ? 1 : 0;
            }
            std::printf("  active set: %td of %zu outlet nodes have mdot>0, %td flipped\n",
                        n_pos, outlet_nodes.size(), n_flip);
        }

        real_t rnorm = 0;
        for (ptrdiff_t i = 0; i < ndof; ++i) rnorm += r[(size_t)i] * r[(size_t)i];
        rnorm = std::sqrt(rnorm);
        if (r0 == real_t(0) && rnorm > 0) r0 = rnorm;

        const real_t rel = (r0 > 0) ? rnorm / r0 : rnorm;
        std::printf("newton %d  ||R||: %.6e  rel: %.6e\n", newton_it, rnorm, rel);

        // Adaptive Harten band: smooth the upwind switch only while Newton is stalling.
        //
        // A band keyed on the flux magnitude cannot work, and the reason is worth stating
        // because it is not obvious. At mdot = 0 the smoothed |mdot| is eps/2, so the flux
        // picks up an artificial diffusion (eps/4)(u_i - u_j) on every face carrying no
        // flux. In a unidirectional flow that is most of the mesh -- which is why a global
        // band of 1e-3 takes Poiseuille from 25 linear iterations to no convergence at all,
        // while the same band restores quadratic convergence on the step. Both cases have
        // faces with mdot ~ 0 and an O(1) jump across them; no local flux measure separates
        // them.
        //
        // What separates them is stability, not magnitude. Poiseuille's near-zero fluxes sit
        // there stably -- the switch never flips, the assembled Jacobian is a perfectly good
        // subdifferential element, and Newton converges quadratically. The step's flip, and
        // that is what costs the rate. Newton's own convergence rate is therefore the honest
        // detector, and it needs no per-face state: engage the band only after the rate has
        // been poor for a step, and scale it by the current relative residual so it shrinks
        // to nothing as the iteration converges. A case that never stalls never sees it,
        // and a case that does converges to the unsmoothed equations rather than to the
        // smoothed ones.
        if (upwind_eps_ref > 0 && smesh::Env::read<int>("SFEM_UPWIND_ADAPT", 0)) {
            static const real_t bad_rate =
                    smesh::Env::read<real_t>("SFEM_UPWIND_ADAPT_RATE", real_t(0.5));
            const real_t rate = (prev_rnorm > 0) ? rnorm / prev_rnorm : real_t(0);
            const bool   stalling = newton_it >= 2 && rate > bad_rate;
            const real_t want = stalling ? upwind_eps_ref * std::min(real_t(1), rel) : real_t(0);
            if (want != op->upwind_eps) {
                op->upwind_eps = want;
                op->update(x);
                std::printf("  upwind band %s: eps = %.3e (rate %.3f, rel %.3e)\n",
                            want > 0 ? "ON" : "off", (double)want, (double)rate, (double)rel);
            }
        }
        prev_rnorm = rnorm;
        if (rnorm < nl_atol || rel < nl_rtol) {
            converged = true;
            break;
        }
        if (r_stage0 == real_t(0)) r_stage0 = rnorm;
        // Diverging: stop now and let the continuation bisect, instead of spending the rest of
        // the iteration budget watching the residual grow.
        if (rnorm > div_grow * r_stage0 || !std::isfinite((double)rnorm)) {
            std::printf("  diverging (||R|| grew %.3gx over the stage) -- abandoning this stage\n",
                        (double)(rnorm / std::max(r_stage0, real_t(1e-300))));
            diverged = true;
            break;
        }
        if (newton_it == max_newton) break;

        for (ptrdiff_t i = 0; i < ndof; ++i) rhs[(size_t)i] = -r[(size_t)i];
        std::fill(dx.begin(), dx.end(), real_t(0));

        // The assembled operator is a snapshot of the Jacobian at construction, so it is
        // rebuilt each step; the matrix-free one reads the live state and is not.
        std::shared_ptr<sfem::Operator<real_t>> linop = mf_op;
        {
            const double t0 = smesh::time_seconds();
            if (!matrix_free)
                linop = sfem::create_linear_operator(sfem::op_type::BSR, f, xbuf, sfem::EXECUTION_SPACE_HOST);
            t_op += smesh::time_seconds() - t0;
        }

        if (check_jv) {
            auto asm_op = sfem::create_linear_operator(sfem::op_type::BSR, f, xbuf, sfem::EXECUTION_SPACE_HOST);
            auto mf     = sfem::create_linear_operator(sfem::op_type::MATRIX_FREE, f, xbuf, sfem::EXECUTION_SPACE_HOST);
            std::vector<real_t> v((size_t)ndof), ya((size_t)ndof, 0), ym((size_t)ndof, 0);
            for (ptrdiff_t i = 0; i < ndof; ++i) v[(size_t)i] = std::sin(real_t(0.7) * real_t(i) + real_t(0.3));
            asm_op->apply(v.data(), ya.data());
            mf->apply(v.data(), ym.data());
            real_t dmax = 0, amax = 0, uinf = 0;
            for (ptrdiff_t i = 0; i < nnodes; ++i)
                for (int c = 0; c < 3; ++c) uinf = std::max(uinf, std::fabs(x[(size_t)i * 4 + c]));
            for (ptrdiff_t i = 0; i < ndof; ++i) {
                dmax = std::max(dmax, std::fabs(ya[(size_t)i] - ym[(size_t)i]));
                amax = std::max(amax, std::fabs(ya[(size_t)i]));
            }
            std::printf("check_jv[newton %d]: rel=%.6e  |u|_inf=%.3e\n", newton_it, (amax > 0) ? dmax / amax : dmax, uinf);
        }

        // A Krylov smoother makes the cycle vary between applications, and BiCGStab
        // assumes its preconditioner does not. It does not fail loudly when that is
        // violated -- it stagnates -- so the outer solver switches to FGMRES whenever the
        // preconditioner is not a fixed operator.
        // Still flexible: a Krylov smoother on any level makes the cycle vary.
        const int  ksmooth_outer = smesh::Env::read<int>("SFEM_GMG_KSMOOTH", 0);
        // Multigrid preconditions FGMRES, not BiCGStab.
        //
        // BiCGStab's short recurrence assumes the preconditioner is a fixed linear operator.
        // A multigrid cycle is not: its semi-structured restriction accumulates with
        // #pragma omp atomic update, so the cycle differs in the last bits between runs, and
        // BiCGStab has no theory for that. Flexible GMRES is built for a preconditioner that
        // varies between applications, and the difference is not subtle. On the 3D cavity at
        // 19,652 dof on 4 threads, over repeated identical runs:
        //
        //     BiCGStab   680 / 1662 / 748 / 2519 linear iterations, t_solve 40.1 s
        //     FGMRES     115 / 115 / 115         linear iterations, t_solve  1.78 s
        //
        // Reproducible, and 22x faster on the same 10 Newton steps. A model of the same
        // effect -- redrawing the preconditioner with relative noise on every application --
        // gives BiCGStab a spread and leaves FGMRES exactly invariant up to 1e-10 noise.
        const bool use_fgmres = smesh::Env::read<int>("SFEM_FGMRES", gmg ? 1 : 0) != 0;

        std::shared_ptr<sfem::FGMRES<real_t>>    fsolver;
        std::shared_ptr<sfem::BiCGStab<real_t>>  bsolver;
        std::function<void(const std::shared_ptr<sfem::Operator<real_t>> &)> set_prec;
        std::function<int()>                                                 get_its;
        std::function<bool()>                                                lin_failed;
        std::function<void(const real_t *, real_t *)>                        do_solve;

        // SFEM_ASSEMBLE_FINE=1 replaces the matrix-free fine operator with an assembled
        // BSR one, probed the same way the Galerkin levels are. The point is determinism:
        // the matrix-free apply accumulates through atomics, so its summation order follows
        // the thread schedule and neither solver is reproducible; a BSR apply is.
        if (smesh::Env::read<int>("SFEM_ASSEMBLE_FINE", 0)) {
            const double t0  = smesh::time_seconds();
            auto         src = gmg ? gmg->ops[0] : linop;
            auto         fine_bsr = assemble_galerkin(f, src, nullptr, nullptr,
                                                      f->space()->n_dofs(), nullptr);
            phase_add("fine_assembly", smesh::time_seconds() - t0);
            linop = fine_bsr;
        }
        // SFEM_JFNK=1: replace the OUTER operator with the exact Jacobian action, obtained by
        // differencing the residual, while the preconditioner keeps using the assembled
        // (inexact) Jacobian. This is the decisive test of whether the frozen Rhie-Chow pg
        // derivative is what caps Newton at a linear rate: if the tail disappears, it is.
        // It is also the cheap remedy -- one extra residual evaluation per Krylov iteration,
        // no new kernel and no change to the assembled sparsity pattern.
        if (smesh::Env::read<int>("SFEM_JFNK", 0)) {
            auto jfnk = std::make_shared<JFNKOperator>(f, ndof, cmask.data());
            jfnk->set_base(x);   // linearise about the current Newton iterate
            linop = jfnk;
        }
        auto linop_timed = timed("outer_op", linop);
        if (use_fgmres) {
            fsolver = std::make_shared<sfem::FGMRES<real_t>>(linop_timed);
            fsolver->set_max_it(lin_max_it);
            fsolver->set_rtol(lin_rtol);
            fsolver->set_atol(lin_atol);
            fsolver->set_restart(smesh::Env::read<int>("SFEM_FGMRES_RESTART", 30));
            fsolver->set_dtol(smesh::Env::read<real_t>("SFEM_LSOLVE_DTOL", real_t(1e4)));
            set_prec = [fsolver, &gauge](const std::shared_ptr<sfem::Operator<real_t>> &p) {
                fsolver->set_preconditioner_op(
                        gauge.active() ? std::static_pointer_cast<sfem::Operator<real_t>>(
                                                 std::make_shared<GaugedPreconditioner>(p, &gauge))
                                       : p);
            };
            get_its  = [fsolver]() { return fsolver->iterations(); };
            do_solve = [fsolver](const real_t *b, real_t *x) { fsolver->apply(b, x); };
            lin_failed = [fsolver]() { return fsolver->has_diverged(); };
        } else {
            bsolver = sfem::create_bcgs<real_t>(linop_timed, sfem::EXECUTION_SPACE_HOST);
            bsolver->set_max_it(lin_max_it);
            // A diverging Krylov solve otherwise burns the full iteration budget inside
            // every Newton step and hands back a correction that the line search will only
            // reject afterwards. Detecting it here stops the waste at its source.
            bsolver->set_dtol(smesh::Env::read<real_t>("SFEM_LSOLVE_DTOL", real_t(1e4)));
            bsolver->set_rtol(lin_rtol);
            bsolver->set_atol(lin_atol);
            set_prec = [bsolver, &gauge](const std::shared_ptr<sfem::Operator<real_t>> &p) {
                bsolver->set_preconditioner_op(
                        gauge.active() ? std::static_pointer_cast<sfem::Operator<real_t>>(
                                                 std::make_shared<GaugedPreconditioner>(p, &gauge))
                                       : p);
            };
            get_its  = [bsolver]() { return bsolver->iterations(); };
            do_solve = [bsolver](const real_t *b, real_t *x) { bsolver->apply(b, x); };
            lin_failed = [bsolver]() { return bsolver->has_diverged(); };
        }
        {
            const double t0 = smesh::time_seconds();
            if (gmg) {
                // The hierarchy is fixed but the linearisation is not.
                refresh_gmg(*gmg);

                // SFEM_GMG_CHECK=2: run the V-cycle standalone as a solver on this Newton
                // step's right-hand side and let it report its own convergence rate.
                //
                // Outer Krylov iteration counts cannot tell a broken coarse correction
                // from a weak smoother -- both just look like "many iterations". The
                // cycle's own rate can: a working V-cycle drops the residual by roughly
                // an order of magnitude per cycle at a rate independent of level, and one
                // whose coarse correction contributes nothing stalls near the rate of the
                // smoother alone.
                // SFEM_GMG_CHECK=3: the smoother, standalone, as the stationary iteration
                // it actually is inside the cycle.
                //
                // Its good showing as a BiCGStab preconditioner (SFEM_GMG=2) is no
                // evidence that it converges: a Krylov method tolerates a preconditioner
                // that would diverge if iterated. Inside a V-cycle it IS iterated, so a
                // divergent smoother makes the cycle diverge regardless of what the coarse
                // levels do -- and no coarse-grid fix can repair that.
                if (smesh::Env::read<int>("SFEM_GMG_CHECK", 0) == 3 && newton_it == 0) {
                    const std::string kind = smesh::Env::read<std::string>("SFEM_SMOOTHER", "vanka");
                    // Same source as the cycle's, or this check measures something else.
                    const real_t om = (kind == "vanka")
                                              ? smoother_omega()
                                              : smesh::Env::read<real_t>("SFEM_GMG_OMEGA", real_t(0.35));
                    std::shared_ptr<sfem::Operator<real_t>> prec;
                    if (kind == "vanka") {
                        // Diagonal Vanka: a coupled solve over each micro-element patch,
                        // measured through the same gate as block-Jacobi so the asymptotic
                        // rates are directly comparable.
                        std::vector<uint8_t> cb((size_t)ndof, 0);
                        for (ptrdiff_t k = 0; k < ndof; ++k)
                            cb[(size_t)k] = mask_get(k, cmask.data()) ? 1 : 0;
                        prec = cvfem_ss::make_diagonal_vanka(*op, f->space(), x, cb.data(), om);
                    } else if (kind == "simple")
                        prec = make_simple(*op, x, cmask.data(), nnodes, om,
                                           smesh::Env::read<int>("SFEM_SIMPLE_INNER", 1),
                                           smesh::Env::read<real_t>("SFEM_SIMPLE_DS", real_t(1)));
                    else
                        prec = make_block_jacobi(*op, x, cmask.data(), nnodes, om);
                    std::printf("smoother kind: %s\n", kind.c_str());
                    std::vector<real_t> xs((size_t)ndof, 0), r((size_t)ndof, 0), z((size_t)ndof, 0);
                    real_t prev = 0;
                    for (ptrdiff_t k = 0; k < ndof; ++k) prev += rhs[(size_t)k] * rhs[(size_t)k];
                    prev = std::sqrt(prev);
                    std::printf("smoother-only (omega=%g):\n", (double)om);
                    for (int it = 0; it < smesh::Env::read<int>("SFEM_GMG_CHECK_IT", 10); ++it) {
                        std::fill(r.begin(), r.end(), real_t(0));
                        linop->apply(xs.data(), r.data());
                        for (ptrdiff_t k = 0; k < ndof; ++k) r[(size_t)k] = rhs[(size_t)k] - r[(size_t)k];
                        std::fill(z.begin(), z.end(), real_t(0));
                        prec->apply(r.data(), z.data());
                        for (ptrdiff_t k = 0; k < ndof; ++k) xs[(size_t)k] += z[(size_t)k];
                        real_t nr = 0;
                        for (ptrdiff_t k = 0; k < ndof; ++k) nr += r[(size_t)k] * r[(size_t)k];
                        nr = std::sqrt(nr);
                        std::printf("  sweep %2d  |r| %.6e  rate %.6f\n", it, (double)nr,
                                    (double)(prev > 0 ? nr / prev : 0));
                        prev = nr;
                    }
                }

                // SFEM_GMG_CHECK=6: write the fine operator and the preconditioner out, so
                // the spectrum of what the Krylov method actually sees can be examined.
                //
                // A Krylov iteration count is a very indirect view of an operator. Whether the
                // iteration is fragile because the discretisation is badly conditioned,
                // because the preconditioned operator is strongly non-normal, or because
                // BiCGStab is simply erratic, are three different diagnoses with three
                // different remedies, and none of them can be told apart from the count. Both
                // matrices are recovered by probing with unit vectors, which is O(n) applies
                // and so only sensible for the small meshes used for this.
                // SFEM_GMG_CHECK=7: Brandt's regime diagnostic, localised.
                //
                // TME (NASA/CR-1998-207647) suggests running the relaxation of a non-elliptic
                // factor on its own to "produce a scalar sigma ~ 1 in regions of open
                // characteristics and sigma << 1 on closed characteristics (such as separated
                // flow zones)". The point is that the two regimes need different cures --
                // downstream-ordered marching for open, defect-correction or semicoarsening
                // for closed -- and the backward-facing step has both, so building either
                // without knowing which region is which is guesswork.
                //
                // What is computed here is the local decay of the error under the smoother
                // alone: set b = 0, start from a random error, relax, and measure per node
                //
                //     rate_i = ( |e_i^N| / |e_i^0| )^(1/N)
                //
                // Error is swept out of open-characteristic regions and lingers where the
                // characteristics close, so a rate near 1 marks the regions the smoother
                // cannot clear -- exactly the regions a coarse grid then has to handle, and
                // exactly what lam_min was telling us globally. This is that measurement
                // resolved in space rather than as one number. It is the local decay, not
                // Brandt's normalisation, so it is reported as a rate and not called sigma.
                if (smesh::Env::read<int>("SFEM_GMG_CHECK", 0) == 7 && newton_it == 0) {
                    const int nsweep = smesh::Env::read<int>("SFEM_SIGMA_SWEEPS", 20);
                    const std::string kind = smesh::Env::read<std::string>("SFEM_SMOOTHER", "vanka");
                    const real_t om = (kind == "vanka")
                                              ? smoother_omega()
                                              : smesh::Env::read<real_t>("SFEM_GMG_OMEGA", real_t(0.35));
                    std::shared_ptr<sfem::Operator<real_t>> prec;
                    if (kind == "vanka") {
                        std::vector<uint8_t> cb((size_t)ndof, 0);
                        for (ptrdiff_t k = 0; k < ndof; ++k)
                            cb[(size_t)k] = mask_get(k, cmask.data()) ? 1 : 0;
                        prec = cvfem_ss::make_diagonal_vanka(*op, f->space(), x, cb.data(), om);
                    } else {
                        prec = make_block_jacobi(*op, x, cmask.data(), nnodes, om);
                    }

                    std::vector<real_t> e((size_t)ndof), r((size_t)ndof), z((size_t)ndof);
                    std::mt19937                           g3(20260907u);
                    std::uniform_real_distribution<real_t> d3(real_t(-1), real_t(1));
                    for (ptrdiff_t k = 0; k < ndof; ++k) e[(size_t)k] = d3(g3);
                    f->apply_zero_constraints(e.data());
                    std::vector<real_t> e0((size_t)nnodes, 0);
                    for (ptrdiff_t i = 0; i < nnodes; ++i)
                        for (int c = 0; c < 3; ++c)
                            e0[(size_t)i] += e[(size_t)i * 4 + c] * e[(size_t)i * 4 + c];
                    for (auto &v : e0) v = std::sqrt(v);

                    for (int it = 0; it < nsweep; ++it) {
                        std::fill(r.begin(), r.end(), real_t(0));
                        linop->apply(e.data(), r.data());
                        for (ptrdiff_t k = 0; k < ndof; ++k) r[(size_t)k] = -r[(size_t)k];
                        std::fill(z.begin(), z.end(), real_t(0));
                        prec->apply(r.data(), z.data());
                        for (ptrdiff_t k = 0; k < ndof; ++k) e[(size_t)k] += z[(size_t)k];
                        f->apply_zero_constraints(e.data());
                    }

                    const auto *const pxs = mesh->points()->data()[0];
                    const auto *const pys = mesh->points()->data()[1];
                    // Bin the local rate by streamwise position; the recirculation sits just
                    // behind the step and the outlet is at the far end, so a rate profile
                    // along x separates them without needing a field dump.
                    const int    NB = 10;
                    std::vector<double> rsum(NB, 0), rmax(NB, 0);
                    std::vector<ptrdiff_t> cnt(NB, 0);
                    double slow_x = 0, slow_y = 0, slow_r = -1;
                    for (ptrdiff_t i = 0; i < nnodes; ++i) {
                        if (e0[(size_t)i] <= 0) continue;
                        double en = 0;
                        for (int c = 0; c < 3; ++c)
                            en += (double)e[(size_t)i * 4 + c] * (double)e[(size_t)i * 4 + c];
                        en = std::sqrt(en);
                        const double rate = std::pow(en / (double)e0[(size_t)i], 1.0 / (double)nsweep);
                        int b = (int)((double)pxs[i] / std::max(Lx, real_t(1e-30)) * NB);
                        b = std::min(NB - 1, std::max(0, b));
                        rsum[b] += rate; ++cnt[b];
                        rmax[b] = std::max(rmax[b], rate);
                        if (rate > slow_r) { slow_r = rate; slow_x = pxs[i]; slow_y = pys[i]; }
                    }
                    std::printf("regime diagnostic: %d smoother sweeps, local error decay by "
                                "streamwise band\n", nsweep);
                    std::printf("   x-band        nodes   mean rate   max rate\n");
                    for (int b = 0; b < NB; ++b) {
                        if (!cnt[b]) continue;
                        std::printf("   [%5.2f,%5.2f) %7td   %9.4f  %9.4f\n",
                                    (double)Lx * b / NB, (double)Lx * (b + 1) / NB,
                                    cnt[b], rsum[b] / (double)cnt[b], rmax[b]);
                    }
                    std::printf("   slowest node at (x %.3f, y %.3f) rate %.4f\n",
                                slow_x, slow_y, slow_r);
                }

                if (smesh::Env::read<int>("SFEM_GMG_CHECK", 0) == 6 && newton_it == 0) {
                    const std::string base =
                            smesh::Env::read_string("SFEM_DUMP_OP", std::string("/tmp/op"));
                    // Build the same preconditioner the solve would use, so the spectrum
                    // examined is the one the Krylov method is actually handed.
                    const std::string pkind =
                            smesh::Env::read<std::string>("SFEM_SMOOTHER", "vanka");
                    const real_t pom = (pkind == "vanka")
                                               ? smoother_omega()
                                               : smesh::Env::read<real_t>("SFEM_GMG_OMEGA", real_t(0.35));
                    std::shared_ptr<sfem::Operator<real_t>> prec;
                    if (pkind == "vanka") {
                        std::vector<uint8_t> cb((size_t)ndof, 0);
                        for (ptrdiff_t k = 0; k < ndof; ++k)
                            cb[(size_t)k] = mask_get(k, cmask.data()) ? 1 : 0;
                        prec = cvfem_ss::make_diagonal_vanka(*op, f->space(), x, cb.data(), pom);
                    } else {
                        prec = make_block_jacobi(*op, x, cmask.data(), nnodes, pom);
                    }
                    std::printf("dump: preconditioner kind %s (omega %g)\n", pkind.c_str(), (double)pom);
                    std::vector<real_t> col((size_t)ndof), out((size_t)ndof);
                    std::vector<real_t> dA((size_t)ndof * (size_t)ndof, 0),
                            dM((size_t)ndof * (size_t)ndof, 0);
                    for (ptrdiff_t j = 0; j < ndof; ++j) {
                        std::fill(col.begin(), col.end(), real_t(0));
                        std::fill(out.begin(), out.end(), real_t(0));
                        col[(size_t)j] = 1;
                        linop->apply(col.data(), out.data());
                        for (ptrdiff_t i = 0; i < ndof; ++i) dA[(size_t)i * ndof + j] = out[(size_t)i];
                        std::fill(out.begin(), out.end(), real_t(0));
                        prec->apply(col.data(), out.data());
                        for (ptrdiff_t i = 0; i < ndof; ++i) dM[(size_t)i * ndof + j] = out[(size_t)i];
                    }
                    dump_dense((base + "_A.txt").c_str(), ndof, dA);
                    dump_dense((base + "_M.txt").c_str(), ndof, dM);
                }

                if (smesh::Env::read<int>("SFEM_GMG_CHECK", 0) == 2 && newton_it == 0) {
                    std::vector<real_t> probe((size_t)ndof, 0);
                    gmg->mg->verbose = true;
                    // Multigrid::debug prints, per level per cycle, the coarse residual after
                    // the coarse solve, the coarse correction before prolongation, the
                    // prolonged correction, and the fine residual after the correction. Those
                    // are exactly the intermediate norms needed to localise where a cycle
                    // first produces a non-finite value; the residual monitor alone only
                    // reports once per cycle, which localises no further than "somewhere
                    // between the restriction of one cycle and the residual of the next".
                    gmg->mg->debug = smesh::Env::read<int>("SFEM_GMG_DEBUG", 0) != 0;
                    gmg->mg->set_max_it(smesh::Env::read<int>("SFEM_GMG_CHECK_IT", 20));
                    gmg->mg->apply(rhs.data(), probe.data());
                    gmg->mg->verbose = false;
                    gmg->mg->debug   = false;
                    gmg->mg->set_max_it(1);

                    // Where does the stalled error live?
                    //
                    // The cycle's rate decays to the smoother's own, which means the coarse
                    // correction stops contributing once the smoother has cleared the high
                    // frequencies. What is left is the smooth error the coarse grid exists
                    // to remove, and splitting it by component says which equation's smooth
                    // modes are being missed.
                    std::vector<real_t> rr((size_t)ndof, 0);
                    linop->apply(probe.data(), rr.data());
                    for (ptrdiff_t k = 0; k < ndof; ++k) rr[(size_t)k] = rhs[(size_t)k] - rr[(size_t)k];
                    real_t n0[N_FIELDS] = {0}, n1[N_FIELDS] = {0};
                    for (ptrdiff_t k = 0; k < ndof; ++k) {
                        const int c = (int)(k % N_FIELDS);
                        n0[c] += rhs[(size_t)k] * rhs[(size_t)k];
                        n1[c] += rr[(size_t)k] * rr[(size_t)k];
                    }
                    const char *nm[N_FIELDS] = {"ux", "uy", "uz", "p"};
                    std::printf("residual by component  (start -> after cycles, and reduction)\n");
                    for (int c = 0; c < N_FIELDS; ++c)
                        std::printf("  %s  %.4e -> %.4e   x%.3e\n", nm[c], std::sqrt(n0[c]),
                                    std::sqrt(n1[c]), (n0[c] > 0) ? std::sqrt(n1[c] / n0[c]) : 0.0);
                }
                set_prec(timed("precond_total", gmg->mg));
            } else if (use_gmg == 2) {
                // Cost-matched control for the V-cycle. The same damped block-Jacobi, run
                // as a stationary iteration on the fine level for the same number of
                // sweeps a V-cycle spends smoothing, with no hierarchy under it.
                //
                // Worth having as its own arm because "more smoothing steps help" says
                // nothing on its own: a damped smoother converges by itself, so a V-cycle
                // whose coarse-grid correction did nothing at all would still improve as
                // the smoothing count rose. This is the arm that separates the two. If the
                // V-cycle cannot beat it, the hierarchy is only an expensive smoother and
                // the fault is in the transfers or the coarse operator, not the smoother.
                const real_t om = smesh::Env::read<real_t>("SFEM_GMG_OMEGA", real_t(0.35));
                auto prec = make_block_jacobi(*op, x, cmask.data(), nnodes, om);
                auto sm   = sfem::create_stationary<real_t>(linop, prec, sfem::EXECUTION_SPACE_HOST);
                sm->set_max_it(2 * gmg_smooth);
                set_prec(sm);
            } else {
                // No hierarchy. SFEM_PRECOND chooses what preconditions the fine level, and
                // the default stays block-Jacobi so every existing SFEM_GMG=0 run is
                // untouched.
                //
                // It exists because block-Jacobi is the wrong preconditioner for a saddle
                // point and the backward-facing step is where that bites. Point-block
                // Jacobi has nothing to say about the pressure coupling -- A_pp is
                // structurally zero without Rhie-Chow -- so on the step the linear residual
                // wanders (0.0034 -> 0.070 -> 0.016 over 255 iterations at Re=1) and the
                // next Newton step diverges outright. Poiseuille and the cavity are
                // velocity-dominated enough not to care, which is why this went unnoticed.
                //
                // The semi-structured operator behaves the same as the flat one under every
                // one of these, which is the property that matters when choosing between the
                // two discretisations. The backward-facing step at Re=20, same 7,060-dof fine
                // mesh reached both ways -- flat 40x8x4, and 20x4x2 macro at level 2:
                //
                //     set-up                 flat                       semi-structured
                //     direct                 20/20, sum 1.90e-17        20/20, sum 4.15e-17
                //     fgmres r=480 bjacobi   20/20, 2688 its, 4.87e-15  20/20, 2689 its, 6.26e-15
                //     bcgs bjacobi           0/20, diverges             0/20, diverges
                //
                // Including the failure: block-Jacobi is too weak for this saddle point on
                // either path and gives out after 254 and 244 iterations respectively. That
                // is the equivalence being claimed -- not that semi-structured is better, but
                // that it is the same operator and answers to the same solvers.
                //
                //   bjacobi  (default) damped 4x4 point-block Jacobi
                //   simple   SIMPLE: velocity predictor, pressure Schur correction. Built
                //            for exactly this and already here, needing a semi-structured
                //            mesh -- apply_blocks has no flat path -- but no hierarchy.
                //   vanka    coupled solve per micro-element patch, likewise
                //   direct   dense LU of the fine Jacobian, probed column by column. O(n)
                //            applies and O(n^2) memory, so it is capped and is a
                //            verification instrument, not a solver: it removes the linear
                //            solve from the question entirely, which is what you want when
                //            asking whether Newton and the conservation property are sound.
                const std::string pc = smesh::Env::read_string("SFEM_PRECOND", "bjacobi");
                const real_t      om = smesh::Env::read<real_t>("SFEM_GMG_OMEGA", real_t(0.35));
                // simple and vanka need apply_blocks and a micro-element lattice, neither of
                // which a flat mesh has. Both do say so further down, but from inside the
                // construction and under MPI_Abort's noise; saying it here names the knob the
                // caller typed and costs nothing.
                if ((pc == "simple" || pc == "vanka") && !fs->has_semi_structured_mesh()) {
                    std::fprintf(stderr,
                                 "SFEM_PRECOND=%s needs a semi-structured mesh (set "
                                 "SFEM_ELEMENT_REFINE_LEVEL > 1). On a flat mesh the choices are "
                                 "bjacobi and direct.\n",
                                 pc.c_str());
                    return EXIT_FAILURE;
                }
                if (pc == "direct") {
                    const ptrdiff_t cap = (ptrdiff_t)smesh::Env::read<int>("SFEM_DIRECT_MAX_DOF", 20000);
                    if (ndof > cap) {
                        std::fprintf(stderr,
                                     "SFEM_PRECOND=direct refuses %td dofs (cap %td, raise with "
                                     "SFEM_DIRECT_MAX_DOF): it builds a dense %td x %td matrix.\n",
                                     ndof, cap, ndof, ndof);
                        return EXIT_FAILURE;
                    }
                    set_prec(timed("precond_total", make_dense_lu(linop, ndof)));
                } else if (pc == "simple") {
                    set_prec(timed("precond_total",
                                   make_simple(*op, x, cmask.data(), nnodes, om,
                                               smesh::Env::read<int>("SFEM_SIMPLE_INNER", 1),
                                               smesh::Env::read<real_t>("SFEM_SIMPLE_DS", real_t(1)))));
                } else if (pc == "vanka") {
                    std::vector<uint8_t> cb((size_t)ndof, 0);
                    for (ptrdiff_t k = 0; k < ndof; ++k) cb[(size_t)k] = mask_get(k, cmask.data()) ? 1 : 0;
                    set_prec(timed("precond_total",
                                   cvfem_ss::make_diagonal_vanka(*op, f->space(), x, cb.data(), om)));
                } else {
                    if (pc != "bjacobi") {
                        std::fprintf(stderr, "SFEM_PRECOND='%s' is not one of bjacobi|simple|vanka|direct\n",
                                     pc.c_str());
                        return EXIT_FAILURE;
                    }
                    set_prec(timed("precond_total", make_block_jacobi(*op, x, cmask.data(), nnodes)));
                }
            }
            t_prec += smesh::time_seconds() - t0;
        }
        {
            const double t0 = smesh::time_seconds();
            do_solve(rhs.data(), dx.data());
            t_solve += smesh::time_seconds() - t0;
        }
        // The correction carries no constant-pressure component; the gauge is fixed, so
        // whatever multiple of the null vector the solve happened to leave in is noise.
        gauge.project(dx.data());
        lin_it_total += get_its();

        // React to a divergent linear solve immediately. The correction it returns cannot be
        // trusted, and continuing would only spend a line search discovering that. Abandoning
        // here lets the continuation shrink its step while the failure is still cheap.
        if (lin_failed && lin_failed()) {
            // A Krylov solve handed a right-hand side that is already round-off cannot do
            // anything sensible with it, and its "divergence" says nothing about the state.
            // This is the normal case once the Jacobian is exact: the first stage converges to
            // ~1e-15 and, for a solution that barely changes with the continuation parameter,
            // every later stage starts solved. Abandoning here discards converged stages --
            // the same floor the line search already respects applies.
            if (rel < nl_ls_floor) {
                std::printf("  linear solve gave up on a rel=%.3e right-hand side -- residual"
                            " floor, accepting as converged\n", (double)rel);
                converged = true;
                ++newton_total;
                break;
            }
            std::printf("  linear solve diverged after %d iterations -- abandoning stage\n",
                        get_its());
            ++lin_diverged_count;
            diverged = true;
            break;
        }

        real_t dxinf = 0;
        for (ptrdiff_t i = 0; i < ndof; ++i) dxinf = std::max(dxinf, std::fabs(dx[(size_t)i]));

        {
            real_t xinf = 0;
            for (ptrdiff_t i = 0; i < ndof; ++i) xinf = std::max(xinf, std::fabs(x[(size_t)i]));
            // nl_stol > 0 guards the disabled case: with nl_stol == 0 the comparison reduces
            // to dxinf <= 0, which is *true* for an exactly-zero correction -- so a linear
            // solve that returned nothing would be reported as a converged Newton step.
            if (nl_stol > real_t(0) && dxinf <= nl_stol * std::max(xinf, real_t(1))) {
                for (ptrdiff_t i = 0; i < ndof; ++i) x[(size_t)i] += dx[(size_t)i];
                gauge.project(x);
                std::printf("  lin_it: %d  |dx|_inf: %.6e  converged on step size\n",
                            get_its(), dxinf);
                converged = true;
                ++newton_total;
                break;
            }
        }

        // Backtracking line search on ||R||. Accept the first step that reduces the residual by
        // the Armijo margin; halve otherwise. A step that cannot reduce it at all is not
        // accepted -- the stage is abandoned so the continuation can bisect.
        real_t alpha = 1;
        if (ls_on) {
            bool ok = false;
            for (int ls = 0; ls < ls_max; ++ls) {
                for (ptrdiff_t i = 0; i < ndof; ++i)
                    x_try[(size_t)i] = x[(size_t)i] + alpha * dx[(size_t)i];
                std::fill(r_try.begin(), r_try.end(), real_t(0));
                f->gradient(x_try.data(), r_try.data());
                f->apply_zero_constraints(r_try.data());
                gauge.project(r_try.data());   // same measure as the residual it is compared to
                real_t rt = 0;
                for (ptrdiff_t i = 0; i < ndof; ++i) rt += r_try[(size_t)i] * r_try[(size_t)i];
                rt = std::sqrt(rt);
                if (std::isfinite((double)rt) && rt < (real_t(1) - ls_armijo * alpha) * rnorm) {
                    ok = true;
                    break;
                }
                alpha *= real_t(0.5);
            }
            if (!ok) {
                // No step reduces ||R||. That is a genuine failure only if the residual is
                // still large; at the round-off floor it just means there is nothing left to
                // reduce, and treating it as failure discards a converged stage.
                if (rel < nl_ls_floor) {
                    std::printf("  line search found no decrease at rel=%.3e -- residual floor,"
                                " accepting as converged\n", (double)rel);
                    converged = true;
                    ++newton_total;
                    break;
                }
                std::printf("  line search failed (no decrease down to alpha=%.3g, rel=%.3e)"
                            " -- abandoning stage\n", (double)alpha, (double)rel);
                diverged = true;
                break;
            }
            for (ptrdiff_t i = 0; i < ndof; ++i) x[(size_t)i] = x_try[(size_t)i];
            gauge.project(x);
        } else {
            for (ptrdiff_t i = 0; i < ndof; ++i) x[(size_t)i] += dx[(size_t)i];
            gauge.project(x);
        }

        std::printf("  lin_it: %d  |dx|_inf: %.6e  alpha: %.4g\n", get_its(), dxinf, (double)alpha);
        ++newton_total;
    }
    (void)diverged;
    ++stages_run;
    if (converged) rho_solved = std::max(rho_solved, rho_use);
    if (converged && re_adapt && rho_solved > real_t(0) &&
        rho_solved < rho * (real_t(1) - real_t(1e-12))) {
        // The stage converged, so widen the step -- but re-plan from the state just solved,
        // never straight at the final target. Aiming every stage at the target is what made
        // the fixed schedule squander its retries: having solved Re 6400 it would attempt
        // 10000, fail, bisect to 8000, fail, bisect to 7155, fail, ... paying a full stage
        // for each. Stepping by a factor that has *just been shown to work* costs one stage
        // per success and lets the ramp refine itself only where the problem is actually
        // hard.
        step_f = std::min(step_f * re_grow, re_step);
        rho_schedule.erase(rho_schedule.begin() + (ptrdiff_t)stage + 1, rho_schedule.end());
        real_t r = rho_solved;
        while (r * step_f < rho) { r *= step_f; rho_schedule.push_back(r); }
        rho_schedule.push_back(rho);
    }
    if (!converged) {
        // Roll back and halve the step in log space rather than giving up. The stage that
        // failed is retried from the last state known to be good, via an intermediate Re.
        if ((stage > 0 || rho_solved > real_t(0)) && re_retries < re_retry) {
            std::copy(x_stage_start.begin(), x_stage_start.end(), x);
            real_t next;
            if (re_adapt) {
                // Shrink the increment towards 1 and re-step from the last solved state.
                // The old rule bisected between the previous schedule entry and the failure,
                // which converges on the ceiling from above and spends a stage per probe;
                // shrinking the factor instead keeps every subsequent step small enough to
                // stand a chance, so the budget buys progress rather than measurement.
                const real_t base = rho_solved > real_t(0) ? rho_solved : rho_re1;
                step_f = real_t(1) + (step_f - real_t(1)) * re_shrink;
                // Cap the adaptive step by the geometric mean of the last success and the
                // failure. Without this the step can overshoot the very target that just
                // failed -- having solved 2749 with a factor of 1.59, the "next" attempt
                // computes 4367 against a failing target of 3200, which is not a smaller
                // step at all. The mean keeps the attempt strictly inside the bracket, so
                // adaptive stepping degrades gracefully into bisection near a hard limit.
                next   = std::min(base * step_f, std::sqrt(base * rho_use));
                if (step_f < re_fmin || next <= base * (real_t(1) + real_t(1e-9))) {
                    std::printf("  stage failed; step factor collapsed to %.4f -- stopping\n",
                                (double)step_f);
                    break;
                }
            } else {
                next = std::sqrt(rho_schedule[stage - 1] * rho_use);
            }
            rho_schedule.insert(rho_schedule.begin() + (ptrdiff_t)stage, next);
            ++re_retries;
            std::printf("  stage failed; retrying via Re = %g (step x%.3f, retry %d/%d)\n",
                        (double)(next * U * Ly / std::max(mu, real_t(1e-30))),
                        (double)step_f, re_retries, re_retry);
            --stage;  // the for-increment lands back on the inserted stage
            continue;
        }
        break;
    }
    }

    // Shift the history: u^{n-1} <- u^n, u^n <- the state just solved for. Done after the
    // step rather than before the next one so a run that stops early leaves the history
    // consistent with the state it reports.
    if (dt_step > real_t(0)) {
        if (bdf_order >= 2) u_hist2 = u_hist;
        for (ptrdiff_t i = 0; i < nnodes; ++i)
            for (int c = 0; c < 3; ++c) u_hist[(size_t)i * 3 + (size_t)c] = x[(size_t)i * 4 + (size_t)c];
        op->set_velocity_history(u_hist.data(), bdf_order >= 2 && tstep >= 1 ? u_hist2.data() : nullptr);
    }

    // A transient run's whole point is the sequence, and the writer at the end of this file
    // only ever sees the last state. SFEM_WRITE_STEPS writes each one into its own
    // step_NNNN/ under the output folder, which python/create_xdmf.py turns into a temporal
    // XDMF that ParaView animates.
    //
    // Off by default: it is one full field dump per step, and the runs that gave the
    // verification numbers want none of it. The mesh is written once at the top rather than
    // per step -- this is transpiration on a FIXED mesh, so there is exactly one geometry
    // for every frame to share.
    if (dt_step > real_t(0) && smesh::Env::read<int>("SFEM_WRITE_STEPS", 0)) {
        char sub[64];
        std::snprintf(sub, sizeof(sub), "step_%04d", tstep);
        const smesh::Path step_dir = smesh::Path(out_folder) / sub;
        smesh::create_directory(smesh::Path(out_folder));
        smesh::create_directory(step_dir);
        if (tstep == 0) {
            if (fs->has_semi_structured_mesh())
                smesh::semistructured_export_as_standard(fs->mesh_ptr(), smesh::Path(out_folder) / "mesh");
            else
                mesh->write(smesh::Path(out_folder) / "mesh");
        }
        auto so = f->output();
        so->enable_AoS_to_SoA(true);
        so->set_output_dir(step_dir);
        so->write("vel", x);
        const char *const sext = sizeof(real_t) == 8 ? "float64" : "float32";
        const std::string sfrom = std::string(step_dir.c_str()) + "/vel.3." + sext;
        const std::string sto   = std::string(step_dir.c_str()) + "/p." + sext;
        std::remove(sto.c_str());
        (void)std::rename(sfrom.c_str(), sto.c_str());
        // The time each frame is AT, so the XDMF can carry real times rather than indices.
        FILE *tf = std::fopen((std::string(step_dir.c_str()) + "/time.txt").c_str(), "w");
        if (tf) {
            std::fprintf(tf, "%.17g\n", (double)((tstep + 1) * dt_step));
            std::fclose(tf);
        }
    }
    }  // time step


    // Report the Reynolds number actually reached, not just the one asked for.
    //
    // With continuation the run can stall partway up the ramp and still look healthy: for
    // Poiseuille the exact solution is a parabolic profile *independent of Re*, so u_linf
    // measures agreement with the same analytic answer whatever Re the state is at. A run
    // that stalled at Re=75 on the way to Re=200 reports the same u_linf as one that
    // arrived. Only this line distinguishes them.
    {
        // Report the highest Re whose stage actually CONVERGED, not the last one attempted.
        //
        // The first version of this printed op->rho, i.e. wherever the ramp had got to, and so
        // announced "reached Re = 1000 of 1000 (AT TARGET)" for a run that tried Re=1000 four
        // times, failed every time, and whose solution had blown up to u_linf 3.4e+06. The
        // highest Re it had actually solved was 843.
        const real_t re_solved = rho_solved * U * Ly / std::max(mu, real_t(1e-30));
        const bool   at_target = rho_solved > 0 && std::fabs(re_solved - Re_phys) <= real_t(1e-6) * Re_phys;
        std::printf("continuation: highest Re SOLVED = %g of %g target  %s\n",
                    (double)re_solved, (double)Re_phys,
                    at_target ? "(AT TARGET)" : "(SHORT OF TARGET)");
        if (!at_target) converged = false;
    }
    std::printf("newton_converged: %d  newton_it: %d (last stage)  newton_total: %d over %d stage(s)  "
                "lin_it_total: %d\n",
                converged ? 1 : 0, newton_it, newton_total, stages_run, lin_it_total);
    std::printf("matrix_free: %d  t_operator: %.4f s  t_precond: %.4f s  t_solve: %.4f s  us_per_lin_it: %.2f\n",
                matrix_free,
                t_op,
                t_prec,
                t_solve,
                lin_it_total ? 1e6 * t_solve / lin_it_total : 0.0);

    // Write the mesh and the solution so a run can actually be looked at afterwards.
    // argv[1] has always been taken as an output folder but was discarded, so every
    // verification case ran blind: a converged number and nothing to inspect. The layout is
    // the one the rest of the tree uses -- mesh/ plus SoA field files -- so
    // external/smesh/python/smesh/raw_to_db.py converts it without special-casing.
    //
    // Semi-structured meshes need semistructured_export_as_standard: the macro-element mesh
    // on its own describes only the corners, and writing that would silently show a
    // level-1 mesh with the fine solution attached to it.
    if (smesh::Env::read<int>("SFEM_ENABLE_OUTPUT", 1)) {
        const smesh::Path out_dir(out_folder);
        smesh::create_directory(out_dir);
        if (fs->has_semi_structured_mesh()) {
            mesh->write(out_dir / "coarse_mesh");
            smesh::semistructured_export_as_standard(fs->mesh_ptr(), out_dir / "mesh");
        } else {
            mesh->write(out_dir / "mesh");
        }
        auto output = f->output();
        // block_size is 4 (ux, uy, uz, p), and Output::write with AoS_to_SoA appends .0 .. .3
        // to the given name -- so it cannot by itself produce three velocity components plus a
        // differently named pressure. Write the block as "vel", then rename the fourth
        // component to "p": the files are plain nodal arrays, so this is a rename and not a
        // conversion. The result is vel.0 vel.1 vel.2 and p, named as the fields actually are
        // rather than after the state vector they happen to be packed in.
        output->enable_AoS_to_SoA(true);
        output->set_output_dir(out_dir);
        output->write("vel", x);

        const char *const ext  = sizeof(real_t) == 8 ? "float64" : "float32";
        const std::string from = std::string(out_folder) + "/vel.3." + ext;
        const std::string to   = std::string(out_folder) + "/p." + ext;
        std::remove(to.c_str());
        if (std::rename(from.c_str(), to.c_str()) != 0) {
            std::fprintf(stderr, "output: could not rename %s -> %s; pressure stays as vel.3\n",
                         from.c_str(), to.c_str());
        }
        std::printf("output: wrote mesh and solution to %s (vel.0 vel.1 vel.2 = u, p = pressure)\n",
                    out_folder.c_str());
    }

    // Verification against the analytic profile, on the free nodes only, matching what
    // the standalone driver reports.
    {
        const auto *const px = mesh->points()->data()[0];
        const auto *const py = mesh->points()->data()[1];
        const auto *const pz = mesh->points()->data()[2];
        real_t            u_linf = 0, p_linf = 0;
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            real_t ux, uy, uz, p;
            cvfem_case::exact_state(
                    flow, mu, U, Lx, Ly, (real_t)px[i], (real_t)py[i], (real_t)pz[i], ux, uy, uz, p);
            u_linf = std::max(u_linf, std::fabs(x[(size_t)i * 4 + 0] - ux));
            u_linf = std::max(u_linf, std::fabs(x[(size_t)i * 4 + 1] - uy));
            u_linf = std::max(u_linf, std::fabs(x[(size_t)i * 4 + 2] - uz));
            p_linf = std::max(p_linf, std::fabs(x[(size_t)i * 4 + 3] - p));
        }

        // Volume-weighted L2 norms, and the pressure additionally compared up to a constant.
        //
        // Both matter for a convergence study and neither is available from L-infinity. L-inf
        // is set by a single worst node, so it reports the worst corner rather than the field,
        // and it is the noisiest possible basis for an observed order. The mean shift matters
        // more: the pressure is fixed by a pin at one node rather than by a zero-mean
        // constraint, so a discrete solution that is right everywhere but offset by a constant
        // is penalised at every node. Subtracting mean(p_h - p_exact) is the standard MMS
        // treatment and separates "the pressure field is wrong" from "the gauge is offset".
        if (flow == cvfem_case::FlowCase::MMS) {
            std::vector<real_t> vol((size_t)nnodes, 0);
            op->node_volume(vol.data());
            long double vtot = 0, u_l2 = 0, p_l2 = 0, p_l2s = 0, dp_mean = 0;
            for (ptrdiff_t i = 0; i < nnodes; ++i) {
                real_t ux, uy, uz, p;
                cvfem_case::exact_state(flow, mu, U, Lx, Ly, (real_t)px[i], (real_t)py[i],
                                        (real_t)pz[i], ux, uy, uz, p);
                const long double v = vol[(size_t)i];
                vtot += v;
                dp_mean += v * (long double)(x[(size_t)i * 4 + 3] - p);
            }
            dp_mean /= (vtot > 0 ? vtot : 1);
            for (ptrdiff_t i = 0; i < nnodes; ++i) {
                real_t ux, uy, uz, p;
                cvfem_case::exact_state(flow, mu, U, Lx, Ly, (real_t)px[i], (real_t)py[i],
                                        (real_t)pz[i], ux, uy, uz, p);
                const long double v  = vol[(size_t)i];
                const long double ex = x[(size_t)i * 4 + 0] - ux;
                const long double ey = x[(size_t)i * 4 + 1] - uy;
                const long double ez = x[(size_t)i * 4 + 2] - uz;
                const long double ep = x[(size_t)i * 4 + 3] - p;
                u_l2  += v * (ex * ex + ey * ey + ez * ez);
                p_l2  += v * ep * ep;
                p_l2s += v * (ep - dp_mean) * (ep - dp_mean);
            }
            std::printf("mms_err: u_l2 %.6e  p_l2 %.6e  p_l2_shifted %.6e  (dp_mean %.6e, vol %.6f)\n",
                        (double)std::sqrt((double)u_l2), (double)std::sqrt((double)p_l2),
                        (double)std::sqrt((double)p_l2s), (double)dp_mean, (double)vtot);

            // Is the pressure pin polluting its neighbourhood?
            //
            // The pin fixes p at a single node instead of constraining the mean, which acts
            // as a point constraint on the pressure equation and can drive a spurious local
            // velocity in a colocated scheme. If that is happening, the worst errors sit on
            // top of the pin and excluding a few cells around it should collapse them. If the
            // errors are spread over the domain instead, the pin is exonerated and the
            // convergence rate is telling us about the discretisation.
            {
                const real_t hh   = Lx / (real_t)std::max<ptrdiff_t>(1, (ptrdiff_t)std::lround(
                                            std::cbrt((double)nnodes) - 1));
                const real_t pinx = (real_t)px[pin_node], piny = (real_t)py[pin_node],
                             pinz = (real_t)pz[pin_node];
                real_t    wu = 0, wp = 0, wux = 0, wuy = 0, wuz = 0, wpx = 0, wpy = 0, wpz = 0;
                real_t    fu[4] = {0, 0, 0, 0}, fp[4] = {0, 0, 0, 0};  // excluding r <= k*h, k=0,1,2,4
                for (ptrdiff_t i = 0; i < nnodes; ++i) {
                    real_t ux, uy, uz, p;
                    cvfem_case::exact_state(flow, mu, U, Lx, Ly, (real_t)px[i], (real_t)py[i],
                                            (real_t)pz[i], ux, uy, uz, p);
                    const real_t eu = std::max(std::max(std::fabs(x[(size_t)i * 4 + 0] - ux),
                                                        std::fabs(x[(size_t)i * 4 + 1] - uy)),
                                               std::fabs(x[(size_t)i * 4 + 2] - uz));
                    const real_t ep = std::fabs(x[(size_t)i * 4 + 3] - p);
                    if (eu > wu) { wu = eu; wux = (real_t)px[i]; wuy = (real_t)py[i]; wuz = (real_t)pz[i]; }
                    if (ep > wp) { wp = ep; wpx = (real_t)px[i]; wpy = (real_t)py[i]; wpz = (real_t)pz[i]; }
                    const real_t dx0 = (real_t)px[i] - pinx, dy0 = (real_t)py[i] - piny,
                                 dz0 = (real_t)pz[i] - pinz;
                    const real_t rr  = std::sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
                    const real_t ks[4] = {0, 1, 2, 4};
                    for (int k = 0; k < 4; ++k)
                        if (rr > ks[k] * hh) {
                            fu[k] = std::max(fu[k], eu);
                            fp[k] = std::max(fp[k], ep);
                        }
                }
                std::printf("mms_pin: pin at (%.4f,%.4f,%.4f)  h=%.4f\n",
                            (double)pinx, (double)piny, (double)pinz, (double)hh);
                std::printf("mms_pin: worst u err %.4e at (%.4f,%.4f,%.4f), dist_to_pin %.4f (%.1f h)\n",
                            (double)wu, (double)wux, (double)wuy, (double)wuz,
                            (double)std::sqrt((wux-pinx)*(wux-pinx)+(wuy-piny)*(wuy-piny)+(wuz-pinz)*(wuz-pinz)),
                            (double)(std::sqrt((wux-pinx)*(wux-pinx)+(wuy-piny)*(wuy-piny)+(wuz-pinz)*(wuz-pinz))/hh));
                std::printf("mms_pin: worst p err %.4e at (%.4f,%.4f,%.4f), dist_to_pin %.4f (%.1f h)\n",
                            (double)wp, (double)wpx, (double)wpy, (double)wpz,
                            (double)std::sqrt((wpx-pinx)*(wpx-pinx)+(wpy-piny)*(wpy-piny)+(wpz-pinz)*(wpz-pinz)),
                            (double)(std::sqrt((wpx-pinx)*(wpx-pinx)+(wpy-piny)*(wpy-piny)+(wpz-pinz)*(wpz-pinz))/hh));
                std::printf("mms_pin: u_linf excluding r<=0h %.4e  1h %.4e  2h %.4e  4h %.4e\n",
                            (double)fu[0], (double)fu[1], (double)fu[2], (double)fu[3]);
                std::printf("mms_pin: p_linf excluding r<=0h %.4e  1h %.4e  2h %.4e  4h %.4e\n",
                            (double)fp[0], (double)fp[1], (double)fp[2], (double)fp[3]);
            }
        }
        phase_report();

        // ------------------------------------------------------------------- the pump
        //
        // What the diaphragm displaces must leave through the port. The chamber is fixed and
        // the flow incompressible, so the flux through its closed boundary is zero; the walls
        // carry none, the diaphragm's velocity is prescribed, and the port is the only other
        // opening. So the port must carry exactly what the diaphragm sweeps:
        //
        //     Q_port  ==  rho * V * Lx * Lz
        //
        // with no closed-form solution anywhere in it. This is the check transpiration has to
        // pass -- it says the prescribed normal velocity moved the mass it claimed to -- and
        // it fails by the size of the lie if the boundary control volumes on either surface
        // are not closed the way the masks think they are.
        //
        // Both fluxes come from Op::sideset_mass_flux, which integrates on the operator's own
        // sub-control surfaces. The second line is the weaker but independent statement that
        // the two openings balance each other, which holds even if the amplitude is wrong.
        if (flow == cvfem_case::FlowCase::Pump) {
            real_t q_port = 0, q_diaphragm = 0;
            if (op->sideset_mass_flux(x, "port", q_port) == SFEM_SUCCESS &&
                op->sideset_mass_flux(x, "diaphragm", q_diaphragm) == SFEM_SUCCESS) {
                // The area vectors point out of the domain: a positive flux leaves. The
                // diaphragm moves in -y against an outward +y, so it carries -V*area, and
                // the port carries the opposite. pump_scale is the waveform at this instant,
                // 1 for a steady run.
                const real_t swept = rho * U * Lx * Lz * pump_scale;
                std::printf("pump: swept %.12f  port %.12f  diaphragm %.12f\n",
                            (double)swept, (double)q_port, (double)q_diaphragm);
                std::printf("pump: |port - swept| %.6e   |port + diaphragm| %.6e\n",
                            (double)std::fabs(q_port - swept),
                            (double)std::fabs(q_port + q_diaphragm));
            }
        }

        if (flow == cvfem_case::FlowCase::Step) {
            // Global mass balance. This is the check that detects an unclosed control volume
            // along the step: if a step face is missing its boundary sub-control-surface
            // term, mass leaks there, and the imbalance is of the order of (step area) x
            // (a velocity) -- large and obvious, not subtle.
            //
            // Fluxes are integrated on the inlet and outlet planes with trapezoidal nodal
            // weights, which are exact for the bilinear variation the mesh carries there.
            // Spacing is the FINE spacing: the macro mesh is refined by refine_level, so
            // Ly/ny is the macro cell size and using it overstates every area by
            // refine_level^2. Also, the two planes have different extents -- the inlet spans
            // y in [step_y, Ly] because the notch removes the rest, so its half-weight edge
            // is at y = step_y and not at y = 0.
            const int    Lref = std::max(1, refine_level);
            const real_t hy   = Ly / (real_t)(ny * Lref), hz = Lz / (real_t)(nz * Lref);
            const real_t step_y = smesh::Env::read<real_t>("SFEM_STEP_Y", 1);
            auto wgt = [](real_t c, real_t lo, real_t hi, real_t h) {
                return (std::fabs(c - lo) < 1e-9 || std::fabs(c - hi) < 1e-9) ? real_t(0.5) * h : h;
            };
            long double q_in = 0, q_out = 0;
            for (ptrdiff_t i = 0; i < nnodes; ++i) {
                const real_t xx = (real_t)px[i], yy = (real_t)py[i], zz = (real_t)pz[i];
                const real_t wz = wgt(zz, 0, Lz, hz);
                if (cvfem_case::on_plane(xx, real_t(0), Lx))
                    q_in += (long double)(wgt(yy, step_y, Ly, hy) * wz) * x[(size_t)i * 4 + 0];
                if (cvfem_case::on_plane(xx, Lx, Lx))
                    q_out += (long double)(wgt(yy, real_t(0), Ly, hy) * wz) * x[(size_t)i * 4 + 0];
            }
            // Quadrature-free mass balance. The continuity residual at a node is the net mass
            // flux out of its control volume; summed over every node the interior faces
            // cancel in pairs and what remains is the net flux through the domain boundary.
            // For an incompressible solution with all control volumes closed that is zero, so
            // this separates "the discretisation leaks mass" from "my trapezoidal rule on the
            // inlet and outlet planes disagrees with itself".
            {
                std::vector<real_t> rr((size_t)ndof, 0);
                f->gradient(x, rr.data());
                long double net = 0, absnet = 0;
                for (ptrdiff_t i = 0; i < nnodes; ++i) {
                    net += (long double)rr[(size_t)i * 4 + 3];
                    absnet += std::fabs((long double)rr[(size_t)i * 4 + 3]);
                }
                std::printf("step: sum of continuity residual %.6Le  (sum |.| %.6Le, ratio %.3Le)\n",
                            net, absnet, absnet > 0 ? std::fabs(net) / absnet : 0.0L);
            }

            // Is the backflow guard active, and is anything sitting on its kink?
            //
            // The do-nothing outflow convects with max(mdot, 0), which is piecewise linear
            // with a corner at mdot = 0. Newton has no quadratic rate across such a corner and
            // a differenced Jacobian action is meaningless there, so whether any outflow face
            // is near it decides whether the non-smoothness is a real problem on this case or
            // a theoretical one. mdot = rho u.a and a points along +x on the outlet, so the
            // nodal u_x carries the sign.
            {
                ptrdiff_t n_out = 0, n_back = 0, n_near = 0;
                real_t    umin = 1e300, umax = -1e300, amin = 1e300;
                for (ptrdiff_t i = 0; i < nnodes; ++i) {
                    if (!cvfem_case::on_plane((real_t)px[i], Lx, Lx)) continue;
                    const real_t u = x[(size_t)i * 4 + 0];
                    ++n_out;
                    if (u <= 0) ++n_back;
                    umin = std::min(umin, u);
                    umax = std::max(umax, u);
                    amin = std::min(amin, std::fabs(u));
                }
                // "Near" measured against the outlet's own scale, not an absolute number.
                for (ptrdiff_t i = 0; i < nnodes; ++i) {
                    if (!cvfem_case::on_plane((real_t)px[i], Lx, Lx)) continue;
                    if (std::fabs(x[(size_t)i * 4 + 0]) < real_t(1e-3) * std::fabs(umax)) ++n_near;
                }
                std::printf("step: outlet nodes %td   backflow (u_x<=0) %td   within 1e-3 of the "
                            "kink %td   u_x in [%.6e, %.6e]  min|u_x| %.3e\n",
                            n_out, n_back, n_near, (double)umin, (double)umax, (double)amin);
            }

            const long double exact_in = 1.0L / 9.0L;
            std::printf("step: inflow flux %.9Lf  (exact %.9Lf, err %.3Le)\n",
                        q_in, exact_in, std::fabs(q_in - exact_in));
            std::printf("step: outflow flux %.9Lf   imbalance (out-in) %.3Le  relative %.3Le\n",
                        q_out, q_out - q_in, std::fabs((q_out - q_in) / (q_in != 0 ? q_in : 1)));
            std::printf("cvfem_hex8_ns_ssgmg: %g seconds\n", smesh::time_seconds() - tick);
            if (!converged) {
                std::fprintf(stderr, "verification failed (step did not converge)\n");
                return EXIT_FAILURE;
            }
            (void)out_folder;
            return EXIT_SUCCESS;
        }
        if (flow == cvfem_case::FlowCase::Cavity ||
            flow == cvfem_case::FlowCase::CavityRegularized) {
            // No closed form to compare against. Report what a cavity run is actually judged
            // on: the velocity extrema, and u_x down the vertical centreline, which is the
            // profile tabulated by Ghia, Ghia & Shin (1982) for the square cavity.
            real_t umin = 1e300, umax = -1e300, vmin = 1e300, vmax = -1e300;
            for (ptrdiff_t i = 0; i < nnodes; ++i) {
                umin = std::min(umin, x[(size_t)i * 4 + 0]);
                umax = std::max(umax, x[(size_t)i * 4 + 0]);
                vmin = std::min(vmin, x[(size_t)i * 4 + 1]);
                vmax = std::max(vmax, x[(size_t)i * 4 + 1]);
            }
            std::printf("cavity: ux in [%.6f, %.6f]   uy in [%.6f, %.6f]\n",
                        (double)umin, (double)umax, (double)vmin, (double)vmax);
            std::printf("cavity: u_x on the vertical centreline (x=%.3g, z=%.3g)\n",
                        (double)(0.5 * Lx), (double)(0.5 * Lz));
            const real_t xtol = real_t(1e-6) * std::max(Lx, real_t(1));
            const real_t ztol = real_t(1e-6) * std::max(Lz, real_t(1));
            for (ptrdiff_t i = 0; i < nnodes; ++i) {
                if (std::fabs((real_t)px[i] - real_t(0.5) * Lx) > xtol) continue;
                if (std::fabs((real_t)pz[i] - real_t(0.5) * Lz) > ztol) continue;
                std::printf("   y/Ly %.4f   ux %+.6f   uy %+.6f\n",
                            (double)((real_t)py[i] / Ly), (double)x[(size_t)i * 4 + 0],
                            (double)x[(size_t)i * 4 + 1]);
            }
            std::printf("cvfem_hex8_ns_ssgmg: %g seconds\n", smesh::time_seconds() - tick);
            if (!converged) {
                std::fprintf(stderr, "verification failed (cavity did not converge)\n");
                return EXIT_FAILURE;
            }
            (void)out_folder;
            return EXIT_SUCCESS;
        }
    std::printf("u_linf: %.6e  p_linf: %.6e\n", u_linf, p_linf);
        std::printf("cvfem_hex8_ns_ssgmg: %g seconds\n", smesh::time_seconds() - tick);

        // The pump has no closed-form solution, so u_linf against exact_state -- which
        // returns its boundary data -- measures nothing and would fail every correct run.
        // Its verification is the swept-volume identity printed above, which is exact and is
        // what the report checks; convergence is still required.
        if (flow == cvfem_case::FlowCase::Pump) {
            if (!converged) {
                std::fprintf(stderr, "verification failed (pump did not converge)\n");
                return EXIT_FAILURE;
            }
        } else if (!converged || u_linf > verify_tol) {
            std::fprintf(stderr, "verification failed (converged=%d, u_linf=%.6e, tol=%g)\n", converged ? 1 : 0, u_linf, verify_tol);
            return EXIT_FAILURE;
        }
    }

    (void)out_folder;
    return EXIT_SUCCESS;
}
