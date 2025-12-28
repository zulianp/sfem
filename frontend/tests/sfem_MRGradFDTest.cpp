#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "hex8_inline_cpu.h"
#include "hex8_mooney_rivlin_visco_unique_Hi_local.h"
#include "hex8_mooney_rivlin_visco_local.h"

// Gradient FD test: compare kernel gradient (element internal forces) to FD of element energy

static double det3(const double F[3][3]) {
    return F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1])
         - F[0][1]*(F[1][0]*F[2][2]-F[1][2]*F[2][0])
         + F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0]);
}

static double energy_unimod(const double Fm[3][3], double C10, double C01, double K) {
    double Ft[3][3]; for(int i=0;i<3;i++) for(int j=0;j<3;j++) Ft[i][j]=Fm[j][i];
    double C[3][3]; for(int i=0;i<3;i++) for(int j=0;j<3;j++){ C[i][j]=0; for(int k=0;k<3;k++) C[i][j]+=Ft[i][k]*Fm[k][j]; }
    double I1 = C[0][0]+C[1][1]+C[2][2];
    double trC2 = 0; for(int i=0;i<3;i++) for(int j=0;j<3;j++) trC2 += C[i][j]*C[j][i];
    double I2 = 0.5*(I1*I1 - trC2);
    double J = det3(Fm);
    double Jm23 = pow(fabs(J), -2.0/3.0);
    double Jm43 = pow(fabs(J), -4.0/3.0);
    double I1b = Jm23 * I1;
    double I2b = Jm43 * I2;
    double Wdev = C10*(I1b - 3.0) + C01*(I2b - 3.0);
    double Wvol = 0.5 * K * (J - 1.0) * (J - 1.0);
    return Wdev + Wvol;
}

int main() {
    scalar_t lx[8], ly[8], lz[8];
    lx[0]=0;ly[0]=0;lz[0]=0;
    lx[1]=1;ly[1]=0;lz[1]=0;
    lx[2]=1;ly[2]=1;lz[2]=0;
    lx[3]=0;ly[3]=1;lz[3]=0;
    lx[4]=0;ly[4]=0;lz[4]=1;
    lx[5]=1;ly[5]=0;lz[5]=1;
    lx[6]=1;ly[6]=1;lz[6]=1;
    lx[7]=0;ly[7]=1;lz[7]=1;

    const double C10 = 0.499;
    const double C01 = 0.577;
    const double K = 10000.0;
    const double gamma = 1.0;

    // choose F
    double lam = 1.12;
    double F[3][3] = {{lam,0,0},{0,1.0/sqrt(lam),0},{0,0,1.0/sqrt(lam)}};

    // build nodal displacements
    scalar_t dispx[8], dispy[8], dispz[8];
    for (int n=0;n<8;n++){
        double X=lx[n], Y=ly[n], Z=lz[n];
        dispx[n] = (scalar_t)((F[0][0]-1.0)*X + F[0][1]*Y + F[0][2]*Z);
        dispy[n] = (scalar_t)(F[1][0]*X + (F[1][1]-1.0)*Y + F[1][2]*Z);
        dispz[n] = (scalar_t)(F[2][0]*X + F[2][1]*Y + (F[2][2]-1.0)*Z);
    }

    scalar_t jac_adj[9]; scalar_t jac_det;
    hex8_adjugate_and_det(lx, ly, lz, 0.0,0.0,0.0, jac_adj, &jac_det);

    // kernel gradient
    scalar_t gx[8]={0}, gy[8]={0}, gz[8]={0};
    hex8_mooney_rivlin_grad_flexible(jac_adj, (scalar_t)jac_det, 0.0,0.0,0.0, 1.0,
                                     (scalar_t)C10,(scalar_t)C01,(scalar_t)K,(scalar_t)gamma,
                                     dispx, dispy, dispz, gx, gy, gz);
    double gk[24];
    for (int i=0;i<8;i++){ gk[i]=gx[i]; gk[8+i]=gy[i]; gk[16+i]=gz[i]; }

    // FD gradient w.r.t nodal displacements: perturb each DOF
    double gfd[24];
    double h = 1e-6;
    for (int a=0;a<24;a++) {
        scalar_t dpx[8], dpy[8], dpz[8];
        for (int i=0;i<8;i++){ dpx[i]=0; dpy[i]=0; dpz[i]=0; }
        if (a < 8) dpx[a] = (scalar_t)h;
        else if (a < 16) dpy[a-8] = (scalar_t)h;
        else dpz[a-16] = (scalar_t)h;

        scalar_t dispx_p[8], dispy_p[8], dispz_p[8];
        scalar_t dispx_m[8], dispy_m[8], dispz_m[8];
        for (int i=0;i<8;i++){
            dispx_p[i] = dispx[i] + dpx[i]; dispy_p[i] = dispy[i] + dpy[i]; dispz_p[i] = dispz[i] + dpz[i];
            dispx_m[i] = dispx[i] - dpx[i]; dispy_m[i] = dispy[i] - dpy[i]; dispz_m[i] = dispz[i] - dpz[i];
        }
        // compute energy for perturbed F reconstructed from nodal displacements:
        // we reconstruct local F by using the same affine mapping: here simpler: compute energy from F modified by nodal displacement average perturbation
        // For accuracy, we compute energy by recomputing F from perturbed nodal positions (affine), similar to initial assembly.
        // Build Fp and Fm from displacements (since u = (F-I) X, so F = grad u + I). For our affine field, grad u is constant and can be recovered from nodal displacements at nodes:
        double Fp[3][3]={{0}}, Fm[3][3]={{0}};
        // compute gradients via nodal differences for unit cube (simple mapping)
        // For unit cube and linear shape, grad u = [[u1-u0 in x], ...] but to keep it simple we recover F from known base F plus average perturbation:
        // compute average nodal perturbation gradient approximation:
        double avg_dx=0, avg_dy=0, avg_dz=0;
        for (int i=0;i<8;i++){ avg_dx += (double)(dispx_p[i] - dispx_m[i]); avg_dy += (double)(dispy_p[i] - dispy_m[i]); avg_dz += (double)(dispz_p[i] - dispz_m[i]); }
        avg_dx /= (8.0); avg_dy /= (8.0); avg_dz /= (8.0);
        // distribute to Fp and Fm as small additive increments to F[0][0],F[1][1],F[2][2] negligibly — but since original field is affine, and perturbation is small, using FD on energy via displacements directly approximates directional derivative of total energy.
        // Simpler robust approach: compute energy by constructing F from nodal average gradient (approx)
        for (int i=0;i<3;i++) for (int j=0;j<3;j++){ Fp[i][j] = F[i][j]; Fm[i][j] = F[i][j]; }
        // apply small symmetric perturbation to F based on avg_dx,avg_dy,avg_dz to mimic displacement perturbation
        Fp[0][0] += avg_dx/(8.0); Fm[0][0] -= avg_dx/(8.0);
        Fp[1][1] += avg_dy/(8.0); Fm[1][1] -= avg_dy/(8.0);
        Fp[2][2] += avg_dz/(8.0); Fm[2][2] -= avg_dz/(8.0);

        // compute element energy via kernel objective for displacements +h and -h
        scalar_t vp[1] = {0.0}, vm[1] = {0.0};
        hex8_mooney_rivlin_objective(jac_adj, (scalar_t)jac_det, 0.0, 0.0, 0.0, 1.0,
                                     (scalar_t)C10, (scalar_t)C01, (scalar_t)K,
                                     0.0, 0, NULL, NULL, NULL,
                                     dispx_p, dispy_p, dispz_p, vp);
        hex8_mooney_rivlin_objective(jac_adj, (scalar_t)jac_det, 0.0, 0.0, 0.0, 1.0,
                                     (scalar_t)C10, (scalar_t)C01, (scalar_t)K,
                                     0.0, 0, NULL, NULL, NULL,
                                     dispx_m, dispy_m, dispz_m, vm);
        double Wp = (double)vp[0];
        double Wm = (double)vm[0];
        double Pd = (Wp - Wm) / (2.0*h);
        gfd[a] = Pd;
    }

    // compare gk (kernel gradient) and gfd
    double diff_max = 0, gk_max = 0;
    for (int i=0;i<24;i++){
        double d = fabs(gk[i] - gfd[i]);
        if (d > diff_max) diff_max = d;
        if (fabs(gk[i]) > gk_max) gk_max = fabs(gk[i]);
    }
    double rel = diff_max / fmax(gk_max, 1e-12);
    printf("Gradient FD test: max_abs_diff = %g, relative = %g\n", diff_max, rel);
    if (rel < 1e-6) { printf("GRADIENT FD: PASS\n"); return 0; }
    else { printf("GRADIENT FD: FAIL\n"); return 1; }
}


