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
#include "sfem_context.hpp"
#include "sfem_mask.hpp"

#include "smesh_env.hpp"
#include "smesh_glob.hpp"
#include "smesh_buffer.hpp"
#include "smesh_mesh.hpp"

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
                     "  SFEM_PACK_SIZE       affine packed SIMD (default 0 = atomic; see note in source)\n"
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
                                                   const ptrdiff_t          nnodes) {
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
    // Defaults to the atomic path, unlike the standalone driver, because the packed one
    // segfaults from here. Known and unresolved: the same operator with the same pack
    // size runs clean in cvfem_ns_op_gate, so it is not the packed kernels themselves.
    // The crash is inside cvfem_hex8_fill_pack_xyz_pgrad on the first residual, reading
    // through the pack's node lists. Set SFEM_PACK_SIZE=2048 to reproduce.
    const int         pack_size  = smesh::Env::read<int>("SFEM_PACK_SIZE", 0);
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
    auto fs   = sfem::FunctionSpace::create(mesh, N_FIELDS);
    auto f    = sfem::Function::create(fs);

    auto op  = std::make_shared<sfem::CVFEMNavierStokes>(fs);
    op->rho  = rho;
    op->mu   = mu;
    op->geom = (geom_name == "isoparam") ? sfem::CVFEMGeometry::Isoparam : sfem::CVFEMGeometry::Affine;
    op->pack_size = pack_size;
    if (op->initialize() != SFEM_SUCCESS) return EXIT_FAILURE;
    f->add_operator(op);

    const ptrdiff_t nnodes = mesh->n_nodes();
    const ptrdiff_t ndof   = nnodes * N_FIELDS;

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

    std::printf("case: %s  geom: %s\n", case_name.c_str(), geom_name.c_str());
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

        // Rebuilt each step: the operator assembles at construction, so it would
        // otherwise hold the Jacobian at the initial state for the whole solve.
        auto linop  = sfem::create_linear_operator(sfem::op_type::BSR, f, xbuf, sfem::EXECUTION_SPACE_HOST);
        auto solver = sfem::create_bcgs<real_t>(linop, sfem::EXECUTION_SPACE_HOST);
        solver->set_max_it(lin_max_it);
        solver->set_rtol(lin_rtol);
        solver->set_atol(lin_atol);
        solver->set_preconditioner_op(make_block_jacobi(*op, x, cmask.data(), nnodes));
        solver->apply(rhs.data(), dx.data());
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
