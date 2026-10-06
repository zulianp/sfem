#pragma once

// What the scheme did to the flow: a kinetic-energy budget, and the quantities it is
// assembled from.
//
// The reason this exists is the convection operator. The sub-control-surface flux is
// first-order upwind -- mdot (phi_i + phi_j)/2 + |mdot| (phi_i - phi_j)/2, no limiter, no
// higher-order reconstruction, no blending -- so the |mdot| term is the entire numerical
// dissipation of the discretisation, and nothing measured it. On a laminar case that is a
// detail; on a turbulent one it is the result, because it is what decides whether the energy
// leaving the resolved scales left through the viscous term or through the truncation error.
//
// The budget, per unit time, for an incompressible flow in a fixed domain:
//
//     dE/dt  =  P_in - P_out  -  eps_visc  -  eps_num
//
// with E the resolved kinetic energy, P the rate at which the open boundaries do work on the
// fluid, and eps_visc the resolved viscous dissipation. Every term but the last is measured
// independently; eps_num is what is left over. It is therefore a MEASUREMENT and not an error
// bar, and the sign convention is chosen so that a dissipative scheme reports it positive.
//
// Nothing here touches a kernel header. Everything is a contraction of two things the
// operator exposes -- the nodal velocity gradient and the control volume -- so the same code
// serves any driver that holds a CVFEMNavierStokes.

#include <cmath>
#include <cstddef>
#include <vector>

namespace cvfem_diag {

    // One instant. Rates are per unit time; E is an energy.
    struct FlowState {
        double E{0};           // resolved kinetic energy, sum of 1/2 rho |u|^2 V
        double eps_visc{0};    // resolved viscous dissipation, sum of 2 mu S:S V
        double enstrophy{0};   // sum of 1/2 |omega|^2 V
        double omega_max{0};   // peak vorticity magnitude
        double div_l2{0};      // volume-weighted L2 norm of div u
        double div_inf{0};     // worst nodal divergence
        double cfl_max{0};     // max |u| dt / h, zero when dt <= 0
        double u_max{0};
    };

    // The budget across one step. Populated from two FlowStates and the boundary powers.
    struct EnergyBudget {
        double dEdt{0};
        double p_in{0};        // rate of work by the inflow surface, positive into the domain
        double p_out{0};       // ... by the outflow surface, positive out of the domain
        double eps_visc{0};    // averaged across the step, to match dEdt's centring
        double eps_num{0};     // the residual: what the upwind term removed
        double closure{0};     // |eps_num| relative to the largest term, the honesty check
                               // (gauge-invariant: see close() on why p_in - p_out, not p_in)
    };

    // Contract a nodal velocity gradient into the volume quantities.
    //
    // `g` is row-major per node, g[i*9 + r*3 + c] = d u_r / d x_c, which is what
    // CVFEMNavierStokes::nodal_velocity_gradient writes. `x` is the interleaved state and
    // `vol` the control volumes. dt <= 0 leaves cfl_max at zero rather than dividing by it.
    //
    // The sums are plain serial accumulations in node order. That is deliberate: the operator
    // was made bit-reproducible across thread counts, and a diagnostic reduced in a
    // thread-dependent order would put the irreproducibility back into the one number the
    // case is judged on. These run once per step over a nodal array, which is nothing beside
    // the solve that produced it.
    template <typename Real>
    inline FlowState contract(const std::ptrdiff_t nnodes, const Real *const x, const Real *const g,
                              const Real *const vol, const double rho, const double mu,
                              const double dt) {
        FlowState s;
        long double E = 0, eps = 0, ens = 0, d2 = 0, voltot = 0;
        for (std::ptrdiff_t i = 0; i < nnodes; ++i) {
            const double V  = (double)vol[i];
            const double ux = (double)x[i * 4 + 0];
            const double uy = (double)x[i * 4 + 1];
            const double uz = (double)x[i * 4 + 2];
            const double um = std::sqrt(ux * ux + uy * uy + uz * uz);
            if (um > s.u_max) s.u_max = um;
            E += (long double)(0.5 * rho * um * um * V);

            const Real *const G = g + i * 9;
            // S:S with S the symmetric part. Written out rather than looped because the
            // off-diagonal terms are counted twice and a loop invites getting that wrong.
            const double s00 = (double)G[0], s11 = (double)G[4], s22 = (double)G[8];
            const double s01 = 0.5 * ((double)G[1] + (double)G[3]);
            const double s02 = 0.5 * ((double)G[2] + (double)G[6]);
            const double s12 = 0.5 * ((double)G[5] + (double)G[7]);
            const double SS  = s00 * s00 + s11 * s11 + s22 * s22 +
                              2.0 * (s01 * s01 + s02 * s02 + s12 * s12);
            eps += (long double)(2.0 * mu * SS * V);

            const double wx = (double)G[7] - (double)G[5];  // du_z/dy - du_y/dz
            const double wy = (double)G[2] - (double)G[6];  // du_x/dz - du_z/dx
            const double wz = (double)G[3] - (double)G[1];  // du_y/dx - du_x/dy
            const double wm = std::sqrt(wx * wx + wy * wy + wz * wz);
            if (wm > s.omega_max) s.omega_max = wm;
            ens += (long double)(0.5 * wm * wm * V);

            const double dv = s00 + s11 + s22;
            if (std::fabs(dv) > s.div_inf) s.div_inf = std::fabs(dv);
            d2 += (long double)(dv * dv * V);
            voltot += (long double)V;

            if (dt > 0 && V > 0) {
                // h from the control volume rather than from the element, so a graded mesh is
                // measured where the flow is rather than on a nominal spacing.
                const double h = std::cbrt(V);
                const double c = um * dt / h;
                if (c > s.cfl_max) s.cfl_max = c;
            }
        }
        s.E         = (double)E;
        s.eps_visc  = (double)eps;
        s.enstrophy = (double)ens;
        s.div_l2    = voltot > 0 ? std::sqrt((double)(d2 / voltot)) : 0.0;
        return s;
    }

    // The nodal weight whose flux is the rate of work on an open boundary:
    //     w = 1/2 |u|^2 + p / rho     so that   w * (rho u.n) = (1/2 rho |u|^2 + p) u.n
    // Fed to CVFEMNavierStokes::sideset_flux_weighted, which integrates it over the very
    // sub-control surfaces the residual used.
    template <typename Real>
    inline void energy_flux_weight(const std::ptrdiff_t nnodes, const Real *const x, const double rho,
                                   std::vector<Real> &w) {
        w.resize((size_t)nnodes);
        for (std::ptrdiff_t i = 0; i < nnodes; ++i) {
            const double ux = (double)x[i * 4 + 0];
            const double uy = (double)x[i * 4 + 1];
            const double uz = (double)x[i * 4 + 2];
            const double p  = (double)x[i * 4 + 3];
            w[(size_t)i]    = (Real)(0.5 * (ux * ux + uy * uy + uz * uz) + p / rho);
        }
    }

    // Close the budget over a step. `q_in`/`q_out` are the weighted fluxes as
    // sideset_flux_weighted returns them: the continuity rows are positive for flux OUT of a
    // control volume, so an inflow surface returns a negative number and the sign is undone
    // here rather than at every call site.
    inline EnergyBudget close(const FlowState &a, const FlowState &b, const double dt,
                              const double q_in, const double q_out) {
        EnergyBudget e;
        e.dEdt     = dt > 0 ? (b.E - a.E) / dt : 0.0;
        e.p_in     = -q_in;
        e.p_out    = q_out;
        // Centred, because dEdt is a centred difference of the two states and pairing it with
        // one end's dissipation biases the residual by half the change across the step.
        e.eps_visc = 0.5 * (a.eps_visc + b.eps_visc);
        e.eps_num  = e.p_in - e.p_out - e.eps_visc - e.dEdt;
        // THE LARGEST TERM OF THE BUDGET, and p_in and p_out are not terms of it -- their
        // DIFFERENCE is. Taking the max over them separately is not merely imprecise, it
        // makes the ratio depend on the pressure gauge: p_in carries integral p u.n over the
        // inlet, so shifting the pressure level by a constant shifts it, while eps_visc and
        // dEdt do not move at all. The difference is invariant whenever the prescribed fluxes
        // balance, because the shift contributes c (mdot_in + mdot_out).
        //
        // Measured on the outflow A/B at 47,268 dof, where the two arms differ only in the
        // outlet condition and therefore in the gauge: the denominator silently changed from
        // p_in to eps_visc between them, and the convective arm read closure 0.1931 against
        // the natural arm's 0.1405 -- a 37% gap reported for two states whose gauge-invariant
        // num_frac is 0.1618 and 0.1585, two percent apart. The conclusion drawn from it, that
        // the convective outflow closed the budget worse, was an artifact of this line.
        const double scale = std::fmax(std::fabs(e.p_in - e.p_out),
                                       std::fmax(std::fabs(e.eps_visc), std::fabs(e.dEdt)));
        e.closure  = scale > 0 ? std::fabs(e.eps_num) / scale : 0.0;
        return e;
    }

}  // namespace cvfem_diag
