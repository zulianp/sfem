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

    enum class FlowCase { Poiseuille, Couette, Cavity, CavityRegularized, MMS, Step, StepTurb, Pump, Nozzle };

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
        if (name == "step" || name == "backward_facing_step" || name == "bfs") {
            out = FlowCase::Step;
            return true;
        }
        // A separate case from `step` rather than a Reynolds number for it. The step is a
        // verification case: its 1/9 flux oracle, its dense-LU gate and its no-slip span are
        // what make that verification mean something, and every one of them has to change
        // here. Sharing the enum would put a turbulent run's spanwise slip and its perturbed
        // initial state into the case the matrix checks conservation on.
        if (name == "step_turb" || name == "turbulent_step" || name == "bfs_turb") {
            out = FlowCase::StepTurb;
            return true;
        }
        if (name == "mms" || name == "manufactured") {
            out = FlowCase::MMS;
            return true;
        }
        if (name == "pump" || name == "diaphragm" || name == "diaphragm_pump") {
            out = FlowCase::Pump;
            return true;
        }
        if (name == "nozzle" || name == "fda_nozzle") {
            out = FlowCase::Nozzle;
            return true;
        }
        return false;
    }

    // ------------------------------------------------------------- FDA benchmark nozzle
    //
    // The nozzle of FDA's first computational round robin: Hariharan et al., J. Biomech. Eng.
    // 133(4) 041002 (2011) for the experiment, Stewart et al., Cardiovasc. Eng. Technol. 3(2)
    // 139-160 (2012) for the 28 simulations it was compared against. Data and CAD are at
    // github.com/OSEL-DAM/CFD-and-Blood-Damage-Benchmarks/tree/main/Nozzle.
    //
    //   inlet pipe     radius 6 mm
    //   cone           20 degrees included, 22.685 mm long, 6 mm down to 2 mm
    //   throat         radius 2 mm, 40 mm long, ending at x = 0
    //   expansion      sudden, back to radius 6 mm at x = 0
    //
    // The axis is x and the origin is the expansion, which is the convention of the published
    // data ("geometry-sudden-z 0.0" in every PIV file), so a station read from those files is
    // a coordinate here without translation. SI units throughout, because the data are.
    //
    // Flow is +x: contraction through the cone, a jet out of the sudden expansion -- the
    // files' "Sudden Expansion" orientation. The reverse orientation, the "Conical Diffuser"
    // data, is the same mesh with inlet and outlet exchanged and is not wired up here.
    //
    // The inlet and outlet are placed where the domain is cut, not where the experiment's
    // tubing ended (150 D upstream, 60 D downstream): the inflow is the fully developed pipe
    // profile the experiment's entrance length produced, and the outlet only has to be far
    // enough downstream that the jet has decayed and nothing re-enters.
    template <typename T>
    struct NozzleGeometry {
        T r_inlet     = T(0.006);
        T r_throat    = T(0.002);
        T x_cone      = T(-0.062685);  // start of the cone; the throat starts at x_throat
        T x_throat    = T(-0.04);
        T x_expansion = T(0);
        T x_in        = T(-0.1);       // domain inlet; the first PIV station is at -0.088
        T x_out       = T(0.16);       // domain outlet; the last PIV station is at +0.08
    };

    // Bulk throat velocity for a throat Reynolds number, which is how the benchmark is
    // specified: Re_t = rho u_t d_t / mu.
    template <typename T>
    inline T nozzle_throat_velocity(const NozzleGeometry<T> &g, const T Re_t, const T rho, const T mu) {
        return Re_t * mu / (rho * T(2) * g.r_throat);
    }

    // The fully developed inflow, u = 2 u_in (1 - r^2 / R^2) with u_in the bulk inlet velocity,
    // u_in = u_t (r_t / r_in)^2 by continuity. Written in terms of the THROAT velocity, because
    // that is the scale the Reynolds number and the continuation are expressed in.
    //
    // Clamped at zero outside the pipe: the mesh's wall nodes sit on an inscribed polygon whose
    // vertices lie exactly on r = r_in, so float rounding can put one a hair outside, and a
    // negative inflow there would be a backwards jet out of the wall.
    template <typename T>
    inline T nozzle_inflow_ux(const NozzleGeometry<T> &g, const T y, const T z, const T u_throat) {
        const T u_in = u_throat * (g.r_throat * g.r_throat) / (g.r_inlet * g.r_inlet);
        const T s    = (y * y + z * z) / (g.r_inlet * g.r_inlet);
        return s < T(1) ? T(2) * u_in * (T(1) - s) : T(0);
    }

    // The volumetric flux of that profile over the smooth disc, Q = pi r_t^2 u_t. It is the
    // oracle a mass balance is scaled by, not one it is compared to exactly: the mesh's inlet
    // is an inscribed polygon, whose area is short of the disc by a relative O(h^2).
    template <typename T>
    inline T nozzle_flow_rate(const NozzleGeometry<T> &g, const T u_throat) {
        return T(M_PI) * g.r_throat * g.r_throat * u_throat;
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

    // ------------------------------------------------------------------ diaphragm pump
    //
    // A closed chamber with two openings in it: a diaphragm that moves, and a port that
    // lets fluid in or out. It is the smallest thing that is recognisably a pump, and it
    // exists to exercise the boundary conditions on something they were built for rather
    // than on a channel.
    //
    //   the chamber   [0,Lx] x [0,Ly] x [0,Lz], every wall no-slip except the two below
    //   the diaphragm the whole face y = Ly, prescribed u = (0, -V, 0)
    //   the port      a centred square patch of the floor y = 0, held at a pressure
    //
    // The diaphragm is TRANSPIRATION on a fixed mesh, not a moving boundary: the mesh does
    // not deform and the wall does not move, the velocity is simply prescribed through it.
    // That is exact for the mass it carries -- which is the whole point here -- and wrong
    // about the geometric nonlinearity of a real diaphragm, which an ALE formulation would
    // capture and this deliberately does not. Amplitudes small against Ly are the regime
    // where the difference does not matter.
    //
    // What makes it worth having is that it is falsifiable in one line. The domain is fixed
    // and the flow incompressible, so the flux through the whole boundary is zero; the walls
    // carry none and the diaphragm carries exactly -V times its area, because its velocity
    // is prescribed. So
    //
    //     flux through the port  ==  V * Lx * Lz
    //
    // exactly, with no closed-form solution needed anywhere. If transpiration is not moving
    // the mass it claims to, that identity breaks, and it breaks by the amount of the lie.
    //
    // Valves are out of scope, so this does not rectify: the port is an opening, and over a
    // full sinusoidal cycle the pump moves fluid back and forth and nets nothing. Adding a
    // valve means making the port's condition depend on the sign of its own flux, which is
    // a different and much less pleasant problem.
    template <typename T>
    inline bool pump_on_diaphragm(const T y, const T Ly) {
        return on_plane(y, Ly, Ly);
    }

    // The port patch, by the same predicate the sideset and the nodeset are both built
    // from, so the two cannot disagree about which faces are open.
    // STRICTLY inside the patch, and the strictness is load-bearing.
    //
    // This one predicate answers two questions -- which faces are the opening, and which
    // nodes are free of no-slip -- and the rim is where they part company. A face is in the
    // port if its centroid is inside, and a centroid is never on the rim. A NODE on the rim
    // is shared between a port face and a wall face, and if it is left free the wall face
    // beside it carries a velocity and leaks: measured, a rim admitted this way sent 0.196
    // of a swept 1.000 out through the walls instead of the port, while the diaphragm
    // carried exactly its -1.000. So the rim belongs to the wall, which a strict inequality
    // says and a tolerant one does not.
    //
    // Subtracting the tolerance rather than adding it is what makes the difference: a node
    // sitting exactly on the rim, which is where they sit whenever the patch aligns with the
    // grid, must fall outside.
    template <typename T>
    inline bool pump_on_port(const T x, const T y, const T z, const T Lx, const T Ly, const T Lz,
                             const T port_frac) {
        if (!on_plane(y, T(0), Ly)) return false;
        const T hx  = T(0.5) * port_frac * Lx;
        const T hz  = T(0.5) * port_frac * Lz;
        const T tol = T(1e-8) * std::max(std::max(Lx, Lz), T(1));
        return std::fabs(x - T(0.5) * Lx) < hx - tol && std::fabs(z - T(0.5) * Lz) < hz - tol;
    }


    // ------------------------------------------------------- the step inflow profile
    //
    // The parabolic-by-parabolic inlet the step is fed, written in terms of the geometry
    // instead of in terms of the one geometry it used to assume.
    //
    // The `step` case spells this inline as 4(2-y)(y-1) z(1-z), which is correct only for
    // Ly = 2, step_y = 1, Lz = 1 -- the domain that case hard-defaults to. A turbulent run
    // wants a span several step heights wide, and at Lz = 4 those literals do not give a
    // profile that is wrong at the edges, they give one that is negative across most of the
    // span and silently drives flow backwards. Hence the parameters.
    //
    // Normalised so the PEAK is U, which is the quantity a Reynolds number is usually quoted
    // on. The bulk velocity follows from the shape: the mean of 4 s(1-s) over [0,1] is 2/3,
    // so U_bulk = (4/9) U and the volumetric flux is (4/9) U (Ly - step_y) Lz. Keeping that
    // relation here, next to the profile, is what lets a mass-balance oracle be derived for
    // any geometry rather than only for the one whose answer was 1/9.
    template <typename T>
    inline T step_inflow_ux(const T y, const T z, const T step_y, const T Ly, const T Lz, const T U) {
        if (y < step_y || y > Ly || z < T(0) || z > Lz) return T(0);
        const T sy = (y - step_y) / (Ly - step_y);
        const T sz = z / Lz;
        const T fy = T(4) * sy * (T(1) - sy);
        const T fz = T(4) * sz * (T(1) - sz);
        return (fy > T(0) && fz > T(0)) ? U * fy * fz : T(0);
    }

    // The exact volumetric flux of that profile, the oracle a mass balance is checked
    // against. Mean of 4 s(1-s) is 2/3 in each direction, so (2/3)^2 = 4/9 of the peak.
    template <typename T>
    inline T step_inflow_flux(const T step_y, const T Ly, const T Lz, const T U) {
        return (T(4) / T(9)) * U * (Ly - step_y) * Lz;
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
        if (flow == FlowCase::Pump) {
            // No closed form. Like the cavity, this returns the boundary data rather than a
            // solution: the diaphragm's normal velocity where the diaphragm is, and zero
            // elsewhere, which is the no-slip the rest of the chamber wants. U is the
            // AMPLITUDE -- the driver scales the whole set by the waveform through
            // DirichletConditions::set_time, so this stays a function of position alone.
            ux = T(0);
            uy = pump_on_diaphragm(y, Ly) ? -U : T(0);
            uz = T(0);
            p  = T(0);
            return;
        }
        if (flow == FlowCase::Step) {
            // Farrell, Mitchell & Wechsung section 5.5. Inflow on {x = 0}:
            //     u = ( 4(2-y)(y-1) z(1-z), 0, 0 )
            // supported on y in [1,2], which is the whole inlet face. No-slip everywhere
            // else, including the two step faces; natural outflow at x = xmax, imposed by
            // leaving those nodes unconstrained rather than by any value here.
            //
            // Peak is 1/4, not 1: 4(2-y)(y-1) peaks at 1 and z(1-z) at 1/4. The exact
            // volumetric flux is 1/9, which is the oracle the mass-balance check uses.
            if (on_plane(x, T(0), Lx)) {
                const T sy = T(4) * (T(2) - y) * (y - T(1));
                const T sz = z * (T(1) - z);
                ux = U * ((sy > T(0)) ? sy * sz : T(0));
            } else {
                ux = T(0);
            }
            p = T(0);
            return;
        }
        if (flow == FlowCase::Nozzle) {
            // Boundary data, not a solution. The inflow needs the nozzle geometry, which this
            // signature does not carry, so the driver evaluates nozzle_inflow_ux itself -- the
            // split StepTurb and the pump use. Zero is the no-slip every wall wants, and without
            // this branch the case would fall through to the Poiseuille formula below.
            ux = uy = uz = T(0);
            p            = T(0);
            return;
        }
        if (flow == FlowCase::StepTurb) {
            // Boundary data, not a solution -- there is none. The inflow itself is set by the
            // driver through step_inflow_ux, because it needs step_y and Lz and this
            // signature carries neither; the same split the pump uses for its port geometry.
            // Returning zero here is what every no-slip face wants, and the verification
            // block exempts this case from the u_linf gate for the reason the pump is
            // exempt: there is nothing to compare against.
            ux = uy = uz = T(0);
            p            = T(0);
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
