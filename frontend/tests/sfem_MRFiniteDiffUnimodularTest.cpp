#include <stdio.h>
#include <math.h>
#include "hex8_inline_cpu.h"
#include "hex8_mooney_rivlin_visco_unique_Hi_local.h"

// Unimodular finite-difference test: compare P_fd = dW_unimodular/dF with P_kernel = F * S_dev
// where S_dev is obtained from hex8_mooney_rivlin_S_dev_from_disp.

static double det3(const double F[3][3]) {
    return F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1])
         - F[0][1]*(F[1][0]*F[2][2]-F[1][2]*F[2][0])
         + F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0]);
}

int main() {
    // unit cube nodes
    scalar_t lx[8], ly[8], lz[8];
    lx[0]=0;ly[0]=0;lz[0]=0;
    lx[1]=1;ly[1]=0;lz[1]=0;
    lx[2]=1;ly[2]=1;lz[2]=0;
    lx[3]=0;ly[3]=1;lz[3]=0;
    lx[4]=0;ly[4]=0;lz[4]=1;
    lx[5]=1;ly[5]=0;lz[5]=1;
    lx[6]=1;ly[6]=1;lz[6]=1;
    lx[7]=0;ly[7]=1;lz[7]=1;

    // choose F (uniaxial)
    double lam = 1.1;
    double F[3][3] = {{lam,0,0},{0,1.0/sqrt(lam),0},{0,0,1.0/sqrt(lam)}};

    const double C10 = 0.499;
    const double C01 = 0.577;
    const double K = 10000.0;

    // compute energy W_unimodular(F) using isochoric invariants
    auto W_unimod = [&](const double Fm[3][3]) {
        double Ft[3][3]; for(int i=0;i<3;i++) for(int j=0;j<3;j++) Ft[i][j]=Fm[j][i];
        double C[3][3]; // C = Ft * Fm
        for(int i=0;i<3;i++) for(int j=0;j<3;j++){ C[i][j]=0.0; for(int k=0;k<3;k++) C[i][j]+=Ft[i][k]*Fm[k][j]; }
        double I1 = C[0][0]+C[1][1]+C[2][2];
        double trC2 = 0.0; for(int i=0;i<3;i++) for(int j=0;j<3;j++) trC2 += C[i][j]*C[j][i];
        double I2 = 0.5*(I1*I1 - trC2);
        double J = det3(Fm);
        double Jm23 = pow(fabs(J), -2.0/3.0);
        double Jm43 = pow(fabs(J), -4.0/3.0);
        double I1b = Jm23 * I1;
        double I2b = Jm43 * I2;
        double Wdev = C10*(I1b - 3.0) + C01*(I2b - 3.0);
        double Wvol = 0.5 * K * (J - 1.0) * (J - 1.0);
        return Wdev + Wvol;
    };

    // finite-difference P_fd
    double P_fd[3][3];
    double h = 1e-6;
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
        double Fp[3][3], Fm[3][3];
        for (int i=0;i<3;i++) for (int j=0;j<3;j++){ Fp[i][j]=F[i][j]; Fm[i][j]=F[i][j]; }
        Fp[a][b] += h; Fm[a][b] -= h;
        double Wp = W_unimod(Fp);
        double Wm = W_unimod(Fm);
        P_fd[a][b] = (Wp - Wm) / (2.0*h);
    }
    double Wval = W_unimod(F);
    printf("W_unimod = %g\n", Wval);

    // compute S_dev via kernel at element center
    scalar_t jac_adj[9]; scalar_t jac_det;
    hex8_adjugate_and_det(lx, ly, lz, 0.0, 0.0, 0.0, jac_adj, &jac_det);
    // build displacement arrays dispx/dispy/dispz from u = (F - I) X
    scalar_t dispx[8], dispy[8], dispz[8];
    for (int n=0;n<8;n++){
        double X = lx[n], Y = ly[n], Z = lz[n];
        double ux = (F[0][0]-1.0)*X + F[0][1]*Y + F[0][2]*Z;
        double uy = F[1][0]*X + (F[1][1]-1.0)*Y + F[1][2]*Z;
        double uz = F[2][0]*X + F[2][1]*Y + (F[2][2]-1.0)*Z;
        dispx[n] = (scalar_t)ux; dispy[n] = (scalar_t)uy; dispz[n] = (scalar_t)uz;
    }

    scalar_t S_dev[6];
    hex8_mooney_rivlin_S_dev_from_disp(jac_adj, (scalar_t)jac_det, 0.0,0.0,0.0, 1.0,
                                       (scalar_t)C10,(scalar_t)C01,(scalar_t)K,
                                       dispx, dispy, dispz, S_dev);
    printf("jac_det = %g\n", (double)jac_det);
    printf("S_dev (voigt) = [%g, %g, %g, %g, %g, %g]\n",
           (double)S_dev[0], (double)S_dev[1], (double)S_dev[2],
           (double)S_dev[3], (double)S_dev[4], (double)S_dev[5]);

    // assemble S_dev into full 3x3 matrix (Voigt: xx,yy,zz,xy,xz,yz)
    double Smat[3][3] = {{S_dev[0], S_dev[3], S_dev[4]},
                         {S_dev[3], S_dev[1], S_dev[5]},
                         {S_dev[4], S_dev[5], S_dev[2]}};

    // compute P_kernel = F * Smat
    double P_k[3][3] = {{0}};
    for (int i=0;i<3;i++) for (int j=0;j<3;j++){
        for (int k=0;k<3;k++) P_k[i][j] += F[i][k]*Smat[k][j];
    }
    // alternate: if Smat is Kirchhoff tau, P = tau * F^{-T}
    // compute F^{-1}
    double Finv[3][3];
    double detF = det3(F);
    // compute inverse via adjugate
    Finv[0][0] =  (F[1][1]*F[2][2]-F[1][2]*F[2][1]) / detF;
    Finv[0][1] = -(F[0][1]*F[2][2]-F[0][2]*F[2][1]) / detF;
    Finv[0][2] =  (F[0][1]*F[1][2]-F[0][2]*F[1][1]) / detF;
    Finv[1][0] = -(F[1][0]*F[2][2]-F[1][2]*F[2][0]) / detF;
    Finv[1][1] =  (F[0][0]*F[2][2]-F[0][2]*F[2][0]) / detF;
    Finv[1][2] = -(F[0][0]*F[1][2]-F[0][2]*F[1][0]) / detF;
    Finv[2][0] =  (F[1][0]*F[2][1]-F[1][1]*F[2][0]) / detF;
    Finv[2][1] = -(F[0][0]*F[2][1]-F[0][1]*F[2][0]) / detF;
    Finv[2][2] =  (F[0][0]*F[1][1]-F[0][1]*F[1][0]) / detF;
    // F^{-T}
    double FinvT[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) FinvT[i][j] = Finv[j][i];
    double P_k_alt[3][3] = {{0}};
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int k=0;k<3;k++) P_k_alt[i][j] += Smat[i][k] * FinvT[k][j];

    // print results
    printf("P_fd (unimodular) =\n");
    for (int i=0;i<3;i++) printf("%g %g %g\n", P_fd[i][0], P_fd[i][1], P_fd[i][2]);
    printf("\nP_kernel = F * S_dev =\n");
    for (int i=0;i<3;i++) printf("%g %g %g\n", P_k[i][0], P_k[i][1], P_k[i][2]);
    printf("\nP_kernel_alt = tau * F^{-T} =\n");
    for (int i=0;i<3;i++) printf("%g %g %g\n", P_k_alt[i][0], P_k_alt[i][1], P_k_alt[i][2]);

    // compute S_fd = F^{-1} * P_fd
    double S_fd[3][3] = {{0}};
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int k=0;k<3;k++) S_fd[i][j] += Finv[i][k] * P_fd[k][j];
    printf("\nS_fd (from P_fd via S = F^{-1} P) =\n");
    for (int i=0;i<3;i++) printf("%g %g %g\n", S_fd[i][0], S_fd[i][1], S_fd[i][2]);
    printf("\nS_kernel (from hex8 S_dev) =\n");
    for (int i=0;i<3;i++) printf("%g %g %g\n", Smat[i][0], Smat[i][1], Smat[i][2]);

    // relative diff
    printf("\nrel_error |P_fd - P_k| / max(|P_fd|,eps):\n");
    double eps = 1e-12;
    for (int i=0;i<3;i++){
        printf("%g %g %g\n",
               fabs(P_fd[i][0]-P_k[i][0]) / fmax(fabs(P_fd[i][0]), eps),
               fabs(P_fd[i][1]-P_k[i][1]) / fmax(fabs(P_fd[i][1]), eps),
               fabs(P_fd[i][2]-P_k[i][2]) / fmax(fabs(P_fd[i][2]), eps));
    }

    return 0;
}


