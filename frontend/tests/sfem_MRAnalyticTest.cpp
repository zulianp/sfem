#include <stdio.h>
#include <math.h>
#include <memory>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_MooneyRivlinVisco.hpp"
#include "sfem_test.h"

// Simple pointwise analytic verification for Mooney-Rivlin hyperelastic core (no visco)
// Tests: uniaxial, equibiaxial, pure shear, simple shear (Cauchy stress comparisons)

static double analytic_cauchy_uniaxial(double lambda, double C10, double C01) {
    // sigma11 = 2*C10*(lambda^2 - lambda^-1) + 2*C01*(lambda - lambda^-2)
    double l2 = lambda * lambda;
    double lm1 = 1.0 / lambda;
    double lm2 = 1.0 / (lambda * lambda);
    return 2.0 * C10 * (l2 - lm1) + 2.0 * C01 * (lambda - lm2);
}

static double analytic_cauchy_equibiaxial(double lambda, double C10, double C01) {
    // lambda1=lambda2=lambda, lambda3=lambda^-2
    // sigma11 = 2*C10*(lambda^2 - lambda^-4) + 2*C01*(lambda^4 - lambda^-2)
    double l2 = lambda * lambda;
    double l4 = l2 * l2;
    double lm2 = 1.0 / (lambda * lambda);
    return 2.0 * C10 * (l2 - 1.0 / (l4)) + 2.0 * C01 * (l4 - lm2);
}

static double analytic_cauchy_pureshear(double lambda, double C10, double C01) {
    // plane strain/pure shear: lambda1=lambda, lambda2=1, lambda3=1/lambda
    // sigma11 = 2*(C10 + C01)*(lambda^2 - lambda^-2)
    double l2 = lambda * lambda;
    double lm2 = 1.0 / (lambda * lambda);
    return 2.0 * (C10 + C01) * (l2 - lm2);
}

static double analytic_cauchy_simpleshear(double gamma, double C10, double C01) {
    // simple shear small gamma: sigma12 = 2*(C10 + C01) * gamma (exact for MR)
    return 2.0 * (C10 + C01) * gamma;
}

int test_mr_pointwise_analytic() {
    MPI_Comm comm = MPI_COMM_WORLD;

    // Create single hex element (1x1x1)
    auto mesh = sfem::Mesh::create_hex8_cube(
        sfem::Communicator::wrap(comm),
        1, 1, 1,
        0, 0, 0,
        1, 1, 1
    );

    auto fs = sfem::FunctionSpace::create(mesh, 3);

    // Operator (hyperelastic core)
    auto op = std::make_shared<sfem::MooneyRivlinVisco>(fs);
    double C10 = 0.499;
    double C01 = 0.577;
    double K = 1e6;
    op->set_C10((real_t)C10);
    op->set_C01((real_t)C01);
    op->set_K((real_t)K);
    op->set_use_flexible(false);
    op->set_dt(0.01);
    // Ensure no Prony terms used
    // Do not call set_prony_terms (num_prony_terms default 0)

    op->initialize();
    op->initialize_history();

    const ptrdiff_t ndofs = fs->n_dofs();
    auto x = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto f_int = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);

    // Node coordinates (points returns SharedBuffer<geom_t *>; access as array per coord)
    auto pts_buf = mesh->points();
    geom_t **pts = pts_buf->data();

    // Helper: assemble displacement field u(X) = (F - I) * X
    auto set_affine = [&](const double F[3][3]) {
        for (ptrdiff_t n = 0; n < fs->mesh_ptr()->n_nodes(); ++n) {
            const double X = (double)pts[0][n];
            const double Y = (double)pts[1][n];
            const double Z = (double)pts[2][n];
            double ux = (F[0][0] - 1.0) * X + F[0][1] * Y + F[0][2] * Z;
            double uy = F[1][0] * X + (F[1][1] - 1.0) * Y + F[1][2] * Z;
            double uz = F[2][0] * X + F[2][1] * Y + (F[2][2] - 1.0) * Z;
            x->data()[3*n + 0] = (real_t)ux;
            x->data()[3*n + 1] = (real_t)uy;
            x->data()[3*n + 2] = (real_t)uz;
        }
    };

    // Helper: compute traction (Cauchy) on face x = 1 by summing internal forces
    auto compute_reaction_on_right = [&]() -> double {
        // call operator gradient -> f_int (internal nodal forces)
        for (ptrdiff_t i = 0; i < ndofs; ++i) f_int->data()[i] = 0.0;
        op->gradient(x->data(), f_int->data());
        // sum x-components of nodes with X > 0.999 (right face)
        double reaction = 0.0;
        for (ptrdiff_t n = 0; n < fs->mesh_ptr()->n_nodes(); ++n) {
            double X = (double)pts[0][n];
            if (X > 0.999) {
                reaction += (double)f_int->data()[3*n + 0];
            }
        }
        return reaction;
    };

    printf("lambda, sfem_cauchy, analytic_cauchy, rel_err\n");
    // Uniaxial test
    double lambdas[] = {1.0, 1.01, 1.1, 1.5, 2.0};
    for (double lam : lambdas) {
        double F[3][3] = {{lam,0,0},{0,1.0/sqrt(lam),0},{0,0,1.0/sqrt(lam)}};
        set_affine(F);
        double reaction = compute_reaction_on_right();
        double sigma_sim = -reaction; // traction = -reaction by convention here
        double sigma_analytic = analytic_cauchy_uniaxial(lam, C10, C01);
        double rel = fabs(sigma_sim - sigma_analytic) / (fabs(sigma_analytic) > 1e-12 ? fabs(sigma_analytic) : 1.0);
        double sigma_sim_eng = sigma_sim / lam;
        double sigma_analytic_eng = sigma_analytic / lam;
        double ratio = (fabs(sigma_analytic) > 1e-12) ? sigma_sim / sigma_analytic : 0.0;

        // Tensor-level analytic (standard MR) for debugging:
        // compute b = F * F^T
        double b[3][3] = {0};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                b[i][j] = 0.0;
                for (int k = 0; k < 3; ++k) b[i][j] += F[i][k] * F[j][k];
            }
        double I1 = b[0][0] + b[1][1] + b[2][2];
        double bb[3][3] = {0};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                bb[i][j] = 0.0;
                for (int k = 0; k < 3; ++k) bb[i][j] += b[i][k] * b[k][j];
            }
        double sigma_tensor_xx = 2.0 * C10 * b[0][0] + 2.0 * C01 * (I1 * b[0][0] - bb[0][0]);

        printf("uniaxial, %g, %g, %g, sigma_eng(sim)=%g, sigma_eng(ana)=%g, ratio=%g, rel_err=%g, sign_sim=%d, sign_ana=%d, sigma_tensor_anal_xx=%g, reaction=%g\n",
               lam, sigma_sim, sigma_analytic, sigma_sim_eng, sigma_analytic_eng, ratio, rel,
               (sigma_sim>0.0)?1:((sigma_sim<0.0)?-1:0), (sigma_analytic>0.0)?1:((sigma_analytic<0.0)?-1:0),
               sigma_tensor_xx);
    }

    // Equibiaxial test
    for (double lam : lambdas) {
        double F[3][3] = {{lam,0,0},{0,lam,0},{0,0,1.0/(lam*lam)}};
        set_affine(F);
        double reaction = compute_reaction_on_right();
        double sigma_sim = -reaction;
        double sigma_analytic = analytic_cauchy_equibiaxial(lam, C10, C01);
        double rel = fabs(sigma_sim - sigma_analytic) / (fabs(sigma_analytic) > 1e-12 ? fabs(sigma_analytic) : 1.0);
        double ratio = (fabs(sigma_analytic) > 1e-12) ? sigma_sim / sigma_analytic : 0.0;
        printf("equibiaxial, %g, %g, %g, ratio=%g, rel_err=%g, sign_sim=%d, sign_ana=%d, reaction=%g\n",
               lam, sigma_sim, sigma_analytic, ratio, rel,
               (sigma_sim>0.0)?1:((sigma_sim<0.0)?-1:0), (sigma_analytic>0.0)?1:((sigma_analytic<0.0)?-1:0));
    }

    // Pure shear (plane strain)
    double shear_lams[] = {1.0, 1.01, 1.1, 1.5, 2.0};
    for (double lam : shear_lams) {
        double F[3][3] = {{lam,0,0},{0,1.0,0},{0,0,1.0/lam}};
        set_affine(F);
        double reaction = compute_reaction_on_right();
        double sigma_sim = -reaction;
        double sigma_analytic = analytic_cauchy_pureshear(lam, C10, C01);
        double rel = fabs(sigma_sim - sigma_analytic) / (fabs(sigma_analytic) > 1e-12 ? fabs(sigma_analytic) : 1.0);
        double ratio = (fabs(sigma_analytic) > 1e-12) ? sigma_sim / sigma_analytic : 0.0;
        printf("pureshear, %g, %g, %g, ratio=%g, rel_err=%g, sign_sim=%d, sign_ana=%d, reaction=%g\n",
               lam, sigma_sim, sigma_analytic, ratio, rel,
               (sigma_sim>0.0)?1:((sigma_sim<0.0)?-1:0), (sigma_analytic>0.0)?1:((sigma_analytic<0.0)?-1:0));
    }

    // Simple shear: use gamma small
    double gammas[] = {1e-6, 1e-4, 1e-2};
    for (double g : gammas) {
        double F[3][3] = {{1.0, g, 0},{0,1.0,0},{0,0,1.0}};
        set_affine(F);
        // For simple shear, traction on right face is not directly sigma12; approximate via integrated nodal forces in x on face with normal x?
        // We'll compute shear stress by summing y-direction forces on face x=1 and dividing by area -> approximate tau_xy
        for (ptrdiff_t i = 0; i < ndofs; ++i) f_int->data()[i] = 0.0;
        op->gradient(x->data(), f_int->data());
        double reaction_y = 0.0;
        for (ptrdiff_t n = 0; n < fs->mesh_ptr()->n_nodes(); ++n) {
            double X = (double)pts[0][n];
            if (X > 0.999) reaction_y += (double)f_int->data()[3*n + 1];
        }
        double tau_sim = -reaction_y;
        double tau_analytic = analytic_cauchy_simpleshear(g, C10, C01);
        double rel = fabs(tau_sim - tau_analytic) / (fabs(tau_analytic) > 1e-12 ? fabs(tau_analytic) : 1.0);
        double ratio = (fabs(tau_analytic) > 1e-12) ? tau_sim / tau_analytic : 0.0;
        printf("simpleshear, %g, %g, %g, ratio=%g, rel_err=%g, sign_sim=%d, sign_ana=%d, reaction=%g\n",
               g, tau_sim, tau_analytic, ratio, rel,
               (tau_sim>0.0)?1:((tau_sim<0.0)?-1:0), (tau_analytic>0.0)?1:((tau_analytic<0.0)?-1:0));
    }

    // -----------------------
    // Finite-difference check: dW/du ~ -f_int
    // Use uniaxial case lambda=1.1 as representative
    // -----------------------
    {
        double lam = 1.1;
        double F[3][3] = {{lam,0,0},{0,1.0/sqrt(lam),0},{0,0,1.0/sqrt(lam)}};
        set_affine(F);

        // compute internal forces at base
        for (ptrdiff_t i = 0; i < ndofs; ++i) f_int->data()[i] = 0.0;
        op->gradient(x->data(), f_int->data());

        // energy density function (unimodular dev + volumetric penalty)
        auto energy_density = [&](const double Fm[3][3]) {
            // compute C = F^T * F and b = F * F^T
            double Cc[3][3] = {0}, b[3][3] = {0};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    Cc[i][j] = 0.0;
                    b[i][j] = 0.0;
                    for (int k = 0; k < 3; ++k) {
                        Cc[i][j] += Fm[k][i] * Fm[k][j];
                        b[i][j] += Fm[i][k] * Fm[j][k];
                    }
                }
            double I1 = Cc[0][0] + Cc[1][1] + Cc[2][2];
            double trC2 = 0.0;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) trC2 += Cc[i][j] * Cc[j][i];
            double I2 = 0.5 * (I1 * I1 - trC2);
            // J = det(F)
            double J = Fm[0][0]*(Fm[1][1]*Fm[2][2]-Fm[1][2]*Fm[2][1])
                     - Fm[0][1]*(Fm[1][0]*Fm[2][2]-Fm[1][2]*Fm[2][0])
                     + Fm[0][2]*(Fm[1][0]*Fm[2][1]-Fm[1][1]*Fm[2][0]);
            double Jm23 = pow(fabs(J), -2.0/3.0);
            double Jm43 = pow(fabs(J), -4.0/3.0);
            double I1b = Jm23 * I1;
            double I2b = Jm43 * I2;
            double Wdev = C10 * (I1b - 3.0) + C01 * (I2b - 3.0);
            double Wvol = 0.5 * K * (J - 1.0) * (J - 1.0);
            return Wdev + Wvol;
        };

        // element volume (single unit cube)
        double vol = 1.0;
        // pick a DOF to perturb: node at (1,1,1) corner index (last node)
        ptrdiff_t node_idx = fs->mesh_ptr()->n_nodes() - 1;
        int dof_x = (int)(node_idx * 3 + 0);

        double h = 1e-6;
        // backup original
        double u_backup = x->data()[dof_x];

        // compute W_plus
        x->data()[dof_x] = u_backup + h;
        // need to compute F from nodal u: since affine, we can recompute F directly by updating matrix F perturbed by h on that node only.
        // Simpler: reconstruct F from the mapping we used: perturbing single node slightly modifies F slightly.
        // For robustness, approximate perturbation by finite-differencing energy via set_affine with perturbed mapping:
        // Recompute F_pert via perturbing displacement field and estimating gradient numerically using nodal positions.
        // But since u = (F-I)X, changing a single nodal displacement breaks exact affine; instead we approximate derivative of total W wrt that nodal u by recomputing W via element-level evaluation using current nodal displacements:
        // Compute deformation gradient by least-squares from nodal positions: here for a hex8 on unit cube, analytic mapping exists; we'll approximate by averaging nodal gradients via shape function derivatives at element center.

        // Helper: compute F from nodal displacements using central difference at element center
        auto compute_F_from_u = [&]() {
            double Fnum[3][3] = {{0}};
            // reference mapping for unit cube: dN/dX at center for linear hex is zero except using simple finite diff
            // Use nodal positions and displacements to compute approximate gradient via differences:
            // nodes ordering for create_hex8_cube is standard: we can sample displacement at x=0 and x=1 faces and compute derivative along x by difference.
            // compute average d/dX by averaging nodes at x=1 minus x=0 over 4 nodes.
            double ux_x1 = 0, ux_x0 = 0;
            int cnt = 0;
            for (ptrdiff_t n = 0; n < fs->mesh_ptr()->n_nodes(); ++n) {
                double X = (double)pts[0][n];
                if (X > 0.999) { ux_x1 += (double)x->data()[3*n + 0]; cnt++; }
                if (X < 1e-6) { ux_x0 += (double)x->data()[3*n + 0]; }
            }
            ux_x1 /= (cnt>0?cnt:1);
            ux_x0 /= (cnt>0?cnt:1);
            Fnum[0][0] = 1.0 + (ux_x1 - ux_x0) / 1.0;
            // similarly for y and z using faces y=1/y=0 and z=1/z=0
            double uy_y1 = 0, uy_y0 = 0; cnt = 0;
            for (ptrdiff_t n = 0; n < fs->mesh_ptr()->n_nodes(); ++n) {
                double Y = (double)pts[1][n];
                if (Y > 0.999) { uy_y1 += (double)x->data()[3*n + 1]; cnt++; }
                if (Y < 1e-6) { uy_y0 += (double)x->data()[3*n + 1]; }
            }
            uy_y1 /= (cnt>0?cnt:1);
            uy_y0 /= (cnt>0?cnt:1);
            Fnum[1][1] = 1.0 + (uy_y1 - uy_y0) / 1.0;
            double uz_z1 = 0, uz_z0 = 0; cnt = 0;
            for (ptrdiff_t n = 0; n < fs->mesh_ptr()->n_nodes(); ++n) {
                double Z = (double)pts[2][n];
                if (Z > 0.999) { uz_z1 += (double)x->data()[3*n + 2]; cnt++; }
                if (Z < 1e-6) { uz_z0 += (double)x->data()[3*n + 2]; }
            }
            uz_z1 /= (cnt>0?cnt:1);
            uz_z0 /= (cnt>0?cnt:1);
            Fnum[2][2] = 1.0 + (uz_z1 - uz_z0) / 1.0;
            // off-diagonals (approx) from average shear displacements differences
            // compute dux/dy approx
            double ux_y1 = 0, ux_y0 = 0; cnt = 0;
            for (ptrdiff_t n = 0; n < fs->mesh_ptr()->n_nodes(); ++n) {
                double Y = (double)pts[1][n];
                if (Y > 0.999) { ux_y1 += (double)x->data()[3*n + 0]; cnt++; }
                if (Y < 1e-6) { ux_y0 += (double)x->data()[3*n + 0]; }
            }
            ux_y1 /= (cnt>0?cnt:1); ux_y0 /= (cnt>0?cnt:1);
            Fnum[0][1] = (ux_y1 - ux_y0);
            // other off-diagonals set to zero for simplicity
            Fnum[0][2] = Fnum[1][0] = Fnum[1][2] = Fnum[2][0] = Fnum[2][1] = 0.0;
            return std::array<double,9>{Fnum[0][0],Fnum[0][1],Fnum[0][2],Fnum[1][0],Fnum[1][1],Fnum[1][2],Fnum[2][0],Fnum[2][1],Fnum[2][2]};
        };

        // compute W+ and W- by reconstructing F from nodal displacements (approx) and evaluating energy
        auto Fplus = compute_F_from_u();
        double Fm_plus[3][3] = {{Fplus[0],Fplus[1],Fplus[2]},{Fplus[3],Fplus[4],Fplus[5]},{Fplus[6],Fplus[7],Fplus[8]}};
        double W_plus = energy_density(Fm_plus) * vol;

        // minus
        x->data()[dof_x] = u_backup - h;
        auto Fminus = compute_F_from_u();
        double Fm_minus[3][3] = {{Fminus[0],Fminus[1],Fminus[2]},{Fminus[3],Fminus[4],Fminus[5]},{Fminus[6],Fminus[7],Fminus[8]}};
        double W_minus = energy_density(Fm_minus) * vol;

        // restore
        x->data()[dof_x] = u_backup;

        double dW_du_numeric = (W_plus - W_minus) / (2.0 * h);
        // internal force at dof_x
        for (ptrdiff_t i = 0; i < ndofs; ++i) f_int->data()[i] = 0.0;
        op->gradient(x->data(), f_int->data());
        double finternal = (double)f_int->data()[dof_x];

        printf(\"FD check (node %ld dof %d): dW/du_numeric=%g, -f_int=%g, rel_err=%g\\n\",\n+               (long)node_idx, dof_x, dW_du_numeric, -finternal,\n+               fabs(dW_du_numeric + finternal) / (fabs(dW_du_numeric)>1e-12?fabs(dW_du_numeric):1.0));\n+    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_mr_pointwise_analytic);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}


