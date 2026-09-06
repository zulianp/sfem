#pragma once

// The analytic channel case: which flow, its exact solution, and the plane test that
// decides where the boundary conditions go.
//
// This is case setup, not operator internals, and both the standalone driver and any
// frontend driver need it -- to impose the boundary values and to verify against. It is
// kept deliberately free of everything else in this directory: no MeshData, no kernels,
// no file-scope scalar_t, and none of the sixteen names the solver core and the
// benchmark layouts fight over. So it is safe to include next to either family, or next
// to cvfem_hex8_ns_op.hpp alone.

#include "cvfem_ns_mms_case.hpp"

#include <cmath>
#include <string>

namespace cvfem_case {

    enum class FlowCase { Poiseuille, Couette, Cavity, CavityRegularized, MMS };

    inline bool parse_case(const std::string &name, FlowCase &out) {
        if (name == "poiseuille") {
            out = FlowCase::Poiseuille;
            return true;
        }
        if (name == "couette" || name == "coutte") {
            out = FlowCase::Couette;
            return true;
        }
        if (name == "cavity" || name == "lid_driven_cavity" || name == "lid") {
            out = FlowCase::Cavity;
            return true;
        }
        if (name == "cavity_reg" || name == "regularized_cavity" || name == "cavity_regularized") {
            out = FlowCase::CavityRegularized;
            return true;
        }
        if (name == "mms" || name == "manufactured") {
            out = FlowCase::MMS;
            return true;
        }
        return false;
    }

    // Relative tolerance, so it does not become meaningless on a large domain and does
    // not admit interior nodes on a small one. Must match the boundary sub-control-
    // surface test in cvfem_hex8_boundary_scs.hpp: the two decide the same thing, and if
    // they disagree a node gets a closed control volume without a boundary condition,
    // or the reverse.
    template <typename T>
    inline bool on_plane(const T c, const T value, const T L) {
        const T tol = T(1e-8) * std::max(L, T(1));
        return std::fabs(c - value) <= tol;
    }

    // Fully developed flow between plates at y = 0 and y = Ly. Couette is driven by the
    // lid, Poiseuille by the pressure gradient G = 8 mu U / Ly^2 that produces peak
    // velocity U.
    //
    // The lid-driven cavity has no closed form, so for that case this returns the boundary
    // data rather than a solution: u = (U,0,0) on the lid y = Ly and zero on the other three
    // walls. That is all the driver asks of it when imposing constraints -- and the reason a
    // cavity run must not be verified against it, because in the interior it is not the
    // answer. Unlike Poiseuille and Couette, whose profiles are independent of Reynolds
    // number, the cavity's solution genuinely changes with Re, which is what makes it a real
    // test of the continuation rather than a formality.
    template <typename T>
    inline void exact_state(const FlowCase flow,
                            const T        mu,
                            const T        U,
                            const T        Lx,
                            const T        Ly,
                            const T        x,
                            const T        y,
                            const T z,
                            T &ux,
                            T &uy,
                            T &uz,
                            T &p) {
        uy = T(0);
        uz = T(0);
        if (flow == FlowCase::MMS) {
            // Their equation carries 1/Re on the viscous term and ours carries mu, so the
            // manufactured case mandates rho = 1 and mu = 1/Re. Re is therefore recoverable
            // from mu alone and need not be threaded through this signature -- but that
            // mandate is load-bearing: at any other (rho, mu) the pressure below is not the
            // exact pressure and the measured convergence rate would be meaningless.
            const T Re = T(1) / mu;
            cvfem_mms::velocity(x, y, z, ux, uy, uz);
            cvfem_mms::pressure(x, y, z, Re, p);
            return;
        }
        if (flow == FlowCase::CavityRegularized) {
            // Farrell, Mitchell & Wechsung section 5.5: the cube [0,2]^3, no-slip everywhere
            // except the top y = Ly, where u = (x^2 (2-x)^2 z^2 (2-z)^2, 0, 0).
            //
            // The lid velocity vanishes at the edges, which removes the corner singularity of
            // the constant-lid cavity. That is not cosmetic: the singularity is what limits
            // the attainable Reynolds number, so this and FlowCase::Cavity are different
            // problems and only this one is comparable with their Table 5.6.
            //
            // Peak value is exactly 1 at (x,z) = (1,1), so U scales it directly. Note
            // x^2 (2-x)^2 is exactly the manufactured solution's u1(x, y=2) -- a free
            // cross-check between the two cases that share this paper.
            if (on_plane(y, Ly, Ly)) {
                const T sx = x * x * (T(2) - x) * (T(2) - x);
                const T sz = z * z * (T(2) - z) * (T(2) - z);
                ux = U * sx * sz;
            } else {
                ux = T(0);
            }
            p = T(0);
            return;
        }
        if (flow == FlowCase::Cavity) {
            ux = on_plane(y, Ly, Ly) ? U : T(0);
            p  = T(0);
            return;
        }
        if (flow == FlowCase::Couette) {
            ux = U * (y / Ly);
            p  = T(0);
            return;
        }
        const T G = T(8) * mu * U / (Ly * Ly);
        ux        = T(4) * U * y * (Ly - y) / (Ly * Ly);
        p         = G * (T(0.5) * Lx - x);
    }

    // Body force for the manufactured case; zero for every other case, which is what makes
    // the force array empty and the residual post-pass a no-op elsewhere.
    // rho is a parameter because the forcing depends on it: f = rho (u.grad)u - mu lap(u)
    // + grad p. That is what makes continuation legitimate here -- ramping rho and
    // recomputing f leaves the exact solution (u, p) untouched, since those depend only on
    // mu. Without recomputing, every intermediate stage would be solving a problem whose
    // solution is not the one being differenced against.
    template <typename T>
    inline void body_force(const FlowCase flow, const T rho, const T mu, const T x, const T y,
                           const T z, T &fx, T &fy, T &fz) {
        if (flow != FlowCase::MMS) {
            fx = fy = fz = T(0);
            return;
        }
        cvfem_mms::body_force(x, y, z, rho, mu, T(1) / mu, fx, fy, fz);
    }

}  // namespace cvfem_case
