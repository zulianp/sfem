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
#include "cvfem_ns_channel_case.hpp"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_GeometricMultigrid.hpp"
#include "sfem_Multigrid.hpp"
#include "sfem_context.hpp"
#include "sfem_mask.hpp"

#include "smesh_env.hpp"
#include "smesh_glob.hpp"
#include "smesh_buffer.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using cvfem_case::FlowCase;

namespace {

    constexpr int N_FIELDS = 4;

    void usage(const char *argv0) {
        std::fprintf(stderr,
                     "usage: %s <output_folder>\n"
                     "\n"
                     "HEX8 CVFEM Navier-Stokes channel, driven through the SFEM frontend.\n"
                     "Same problem and defaults as cvfem_hex8_ns_steady, so the two are\n"
                     "directly comparable.\n"
                     "\n"
                     "Environment:\n"
                     "  SFEM_CASE            poiseuille | couette (required)\n"
                     "  SFEM_N               cells in y (default 8)\n"
                     "  SFEM_NX SFEM_NY SFEM_NZ   override cells per direction\n"
                     "  SFEM_LX SFEM_LY SFEM_LZ   channel size (default 4, 1, 1)\n"
                     "  SFEM_RHO SFEM_MU SFEM_U   density, viscosity, velocity scale\n"
                     "  SFEM_GEOM            affine | isoparam (default affine)\n"
                     "  SFEM_NL_MAX_IT SFEM_NL_RTOL SFEM_NL_ATOL\n"
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

    // `mask` marks the constrained dofs. It is needed because hessian_block_diag reports
    // the operator's own diagonal and knows nothing about boundary conditions, while the
    // matrix the Krylov method actually sees has identity rows there -- Function applies
    // the constraints to it after the operators. A preconditioner built from the
    // unconstrained diagonal scales those rows by something unrelated to 1.
    std::shared_ptr<BlockJacobi> make_block_jacobi(sfem::CVFEMNavierStokes &op,
                                                   const real_t *const      x,
                                                   const mask_t *const      mask,
                                                   const ptrdiff_t          nnodes,
                                                   const real_t             omega = real_t(1)) {
        std::vector<real_t> blocks((size_t)nnodes * 16, 0);
        op.hessian_block_diag(x, blocks.data());

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

            if (std::fabs(det) > real_t(1e-30)) {
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

            const real_t pp = b[15];
            m[15]           = (std::fabs(pp) > real_t(1e-30)) ? real_t(1) / pp : real_t(1);

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
    };

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

    void refresh_gmg(GmgLevels &g) {
        const int nlevels = (int)g.ops.size();
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
        const real_t pscale = smesh::Env::read<real_t>("SFEM_GMG_PSCALE", real_t(1));
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
            auto prec = make_block_jacobi(*g.level_ops[i], g.states[i]->data(), mask.data(), nn, omega);

            // SFEM_GMG_PSCALE scales the continuity rows of every coarse level.
            //
            // Left-scaling a row does not change what the coarse system solves; it changes
            // the correction the cycle takes from it. Unlike SFEM_GMG_RC_DECAY this leaves
            // Df alone, so the balance between divergence and stabilisation inside the
            // continuity row is untouched and the coarse operator keeps the stabilisation
            // its own mesh needs. The value to use is not tuned: it is the pressure/velocity
            // ratio of the best-fit scales printed by SFEM_GMG_CHECK=1.
            auto lop = g.ops[i];
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
                auto sm = sfem::create_stationary<real_t>(lop, prec, sfem::EXECUTION_SPACE_HOST);
                sm->set_max_it(g.smoothing_steps);
                g.mg->add_level(lop, sm, i == 0 ? nullptr : wrap_p(i), g.data->restrictions[i]);
            } else {
                // Coarse solve. BiCGStab, not CG: the operator is not symmetric.
                auto cs = sfem::create_bcgs<real_t>(lop, sfem::EXECUTION_SPACE_HOST);
                cs->set_max_it(smesh::Env::read<int>("SFEM_GMG_COARSE_MAX_IT", 200));
                cs->set_rtol(1e-8);
                cs->set_atol(1e-14);
                // The coarse level is solved, not smoothed, so the cycle takes its answer
                // at face value. If that solve stagnates the correction is noise, and a
                // stagnating BiCGStab reports success by exhausting its iterations.
                cs->verbose = smesh::Env::read<int>("SFEM_GMG_COARSE_VERBOSE", 0) != 0;
                cs->set_preconditioner_op(prec);
                g.mg->add_level(lop, cs, wrap_p(i), nullptr);
            }
        }
        g.mg->set_max_it(1);  // one V-cycle per preconditioner application
    }

}  // namespace

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
    const real_t      Lx         = smesh::Env::read<real_t>("SFEM_LX", 4);
    const real_t      Ly         = smesh::Env::read<real_t>("SFEM_LY", 1);
    const real_t      Lz         = smesh::Env::read<real_t>("SFEM_LZ", 1);
    const real_t      rho        = smesh::Env::read<real_t>("SFEM_RHO", 1);
    const real_t      mu         = smesh::Env::read<real_t>("SFEM_MU", 0.01);
    const real_t      U          = smesh::Env::read<real_t>("SFEM_U", 1);
    const std::string geom_name  = smesh::Env::read_string("SFEM_GEOM", "affine");
    const int         max_newton = smesh::Env::read<int>("SFEM_NL_MAX_IT", 40);
    const real_t      nl_rtol    = smesh::Env::read<real_t>("SFEM_NL_RTOL", 1e-8);
    const real_t      nl_atol    = smesh::Env::read<real_t>("SFEM_NL_ATOL", 1e-12);
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
        std::fprintf(stderr, "SFEM_CASE is required (poiseuille or couette)\n");
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

    auto mesh = smesh::Mesh::create_hex8_cube(ctx->communicator(), nx, ny, nz, 0, 0, 0, Lx, Ly, Lz);
    // SFEM_ELEMENT_REFINE_LEVEL > 1 turns the mesh semi-structured: the cells above become
    // macro-elements, each holding a level^3 lattice, and the operator switches to the
    // sshex8 kernels on its own from what the space carries. The requested cell counts then
    // describe macro-elements, so the problem is level^3 times larger than the flat run of
    // the same SFEM_N -- which is the point, but worth knowing when comparing.
    const int refine_level = smesh::Env::read<int>("SFEM_ELEMENT_REFINE_LEVEL", 1);
    if (refine_level > 1) {
        mesh = smesh::to_semistructured(refine_level, mesh, true, false);
        if (!mesh) {
            std::fprintf(stderr, "to_semistructured failed for level %d\n", refine_level);
            return EXIT_FAILURE;
        }
    }
    auto fs   = sfem::FunctionSpace::create(mesh, N_FIELDS);
    auto f    = sfem::Function::create(fs);

    auto op  = std::make_shared<sfem::CVFEMNavierStokes>(fs);
    op->rho  = rho;
    op->mu   = mu;
    op->geom = (geom_name == "isoparam") ? sfem::CVFEMGeometry::Isoparam : sfem::CVFEMGeometry::Affine;
    op->pack_size = pack_size;
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

        std::vector<idx_t>  uvw_nodes, uz_nodes;
        std::vector<real_t> uvw_ux, uvw_uy, uvw_uz, uz_vals;
        p_exact.assign((size_t)nnodes, real_t(0));
        ptrdiff_t           pin  = 0;
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

            if (wall_y || inlet || outlet) {
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
        conds.push_back(make_cond({(idx_t)pin}, {pp}, 3));

        f->add_constraint(sfem::DirichletConditions::create(fs, conds));

        std::printf("constraints: uvw_nodes=%td  uz_nodes=%td  p_pin=%td\n",
                    (ptrdiff_t)uvw_nodes.size(),
                    (ptrdiff_t)uz_nodes.size(),
                    pin);
    }

    std::printf("case: %s  geom: %s  refine_level: %d  semi_structured: %d\n",
                case_name.c_str(), geom_name.c_str(), refine_level, op->is_semi_structured() ? 1 : 0);
    std::printf("channel: L=(%g,%g,%g)  cells=(%d,%d,%d)\n", Lx, Ly, Lz, nx, ny, nz);
    std::printf("nnodes: %td  nelements: %td  ndof: %td\n", nnodes, mesh->n_elements(0), ndof);
    std::printf("rho: %g  mu: %g  U: %g  Re: %g\n", rho, mu, U, rho * U * Ly / mu);

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

    std::vector<mask_t> cmask(mask_count(ndof), 0);
    f->constraints_mask(cmask.data());

    // Reynolds continuation, matching the standalone driver. Newton from a zero state
    // does not converge at Re=100: the first stage solves the same geometry at Re=1 by
    // taking rho = mu / (U Ly), and the second continues from that solution at the
    // physical density. Without it this diverges to inf, which is how its absence was
    // found rather than reasoned about.
    const real_t Re_phys = rho * U * Ly / std::max(mu, real_t(1e-30));
    const real_t rho_re1 = mu / std::max(U * Ly, real_t(1e-30));
    const int    n_stages = (rho == real_t(0) || Re_phys <= real_t(1.5)) ? 1 : 2;

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
        if (gmg && smesh::Env::read<int>("SFEM_GMG_CHECK", 0)) {
            refresh_gmg(*gmg);
            check_transfers(*gmg);
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
    bool converged    = false;
    // Set once from the first nonzero residual and kept across stages, as in the
    // standalone driver: the continuation stage and the physical stage are measured
    // against the same reference.
    real_t r0 = 0;

    for (int stage = 0; stage < n_stages; ++stage) {
    const real_t rho_use = (n_stages == 1 || stage == 1) ? rho : rho_re1;
    op->rho              = rho_use;
    std::printf("stage: %s  rho: %g  Re: %g\n",
                (n_stages == 1 || stage == 1) ? "navier-stokes" : "re1",
                rho_use,
                rho_use * U * Ly / std::max(mu, real_t(1e-30)));

    converged = false;
    for (newton_it = 0; newton_it <= max_newton; ++newton_it) {
        std::fill(r.begin(), r.end(), real_t(0));
        f->gradient(x, r.data());
        // Measure the residual on the free dofs only. Function::gradient leaves the
        // boundary-condition residual (x - value) in the constrained rows, which is a
        // different quantity from the equation residual and never decays to zero the way
        // the Newton test expects -- it stalls the relative criterion near the solution.
        // The standalone driver zeroes them for the same reason.
        f->apply_zero_constraints(r.data());

        real_t rnorm = 0;
        for (ptrdiff_t i = 0; i < ndof; ++i) rnorm += r[(size_t)i] * r[(size_t)i];
        rnorm = std::sqrt(rnorm);
        if (r0 == real_t(0) && rnorm > 0) r0 = rnorm;

        const real_t rel = (r0 > 0) ? rnorm / r0 : rnorm;
        std::printf("newton %d  ||R||: %.6e  rel: %.6e\n", newton_it, rnorm, rel);
        if (rnorm < nl_atol || rel < nl_rtol) {
            converged = true;
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

        auto solver = sfem::create_bcgs<real_t>(linop, sfem::EXECUTION_SPACE_HOST);
        solver->set_max_it(lin_max_it);
        solver->set_rtol(lin_rtol);
        solver->set_atol(lin_atol);
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
                    const real_t om   = smesh::Env::read<real_t>("SFEM_GMG_OMEGA", real_t(0.35));
                    auto         prec = make_block_jacobi(*op, x, cmask.data(), nnodes, om);
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

                if (smesh::Env::read<int>("SFEM_GMG_CHECK", 0) == 2 && newton_it == 0) {
                    std::vector<real_t> probe((size_t)ndof, 0);
                    gmg->mg->verbose = true;
                    gmg->mg->set_max_it(smesh::Env::read<int>("SFEM_GMG_CHECK_IT", 20));
                    gmg->mg->apply(rhs.data(), probe.data());
                    gmg->mg->verbose = false;
                    gmg->mg->set_max_it(1);
                }
                solver->set_preconditioner_op(gmg->mg);
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
                solver->set_preconditioner_op(sm);
            } else {
                solver->set_preconditioner_op(make_block_jacobi(*op, x, cmask.data(), nnodes));
            }
            t_prec += smesh::time_seconds() - t0;
        }
        {
            const double t0 = smesh::time_seconds();
            solver->apply(rhs.data(), dx.data());
            t_solve += smesh::time_seconds() - t0;
        }
        lin_it_total += solver->iterations();

        real_t dxinf = 0;
        for (ptrdiff_t i = 0; i < ndof; ++i) {
            x[(size_t)i] += dx[(size_t)i];
            dxinf = std::max(dxinf, std::fabs(dx[(size_t)i]));
        }
        std::printf("  lin_it: %d  |dx|_inf: %.6e\n", solver->iterations(), dxinf);
    }
    if (!converged) break;
    }

    std::printf("newton_converged: %d  newton_it: %d  lin_it_total: %d\n", converged ? 1 : 0, newton_it, lin_it_total);
    std::printf("matrix_free: %d  t_operator: %.4f s  t_precond: %.4f s  t_solve: %.4f s  us_per_lin_it: %.2f\n",
                matrix_free,
                t_op,
                t_prec,
                t_solve,
                lin_it_total ? 1e6 * t_solve / lin_it_total : 0.0);

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
        std::printf("u_linf: %.6e  p_linf: %.6e\n", u_linf, p_linf);
        std::printf("cvfem_hex8_ns_ssgmg: %g seconds\n", smesh::time_seconds() - tick);

        if (!converged || u_linf > verify_tol) {
            std::fprintf(stderr, "verification failed (converged=%d, u_linf=%.6e, tol=%g)\n", converged ? 1 : 0, u_linf, verify_tol);
            return EXIT_FAILURE;
        }
    }

    (void)out_folder;
    return EXIT_SUCCESS;
}
