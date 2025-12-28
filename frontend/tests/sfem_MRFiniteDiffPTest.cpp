#include <stdio.h>
#include <math.h>

// Finite-difference check of P = dW/dF for Mooney-Rivlin (standard + volumetric)
// W(F) = C10*(I1-3) + C01*(I2-3) + 0.5*K*(J-1)^2

static double det3(const double F[3][3]) {
    return F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1])
         - F[0][1]*(F[1][0]*F[2][2]-F[1][2]*F[2][0])
         + F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0]);
}

static void mat_mul(const double A[3][3], const double B[3][3], double C[3][3]) {
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
        C[i][j]=0.0;
        for (int k=0;k<3;k++) C[i][j]+=A[i][k]*B[k][j];
    }
}

static void mat_transpose(const double A[3][3], double AT[3][3]) {
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) AT[i][j]=A[j][i];
}

int main() {
    // test F
    double lam = 1.1;
    double F[3][3] = {{lam,0,0},{0,1.0/sqrt(lam),0},{0,0,1.0/sqrt(lam)}};
    const double C10 = 0.499;
    const double C01 = 0.577;
    const double K = 1e6;

    // energy W(F)
    auto W = [&](const double Fm[3][3]) {
        double Ft[3][3]; mat_transpose(Fm, Ft);
        double Ct[3][3]; mat_mul(Ft, Fm, Ct); // C = F^T * F
        double I1 = Ct[0][0]+Ct[1][1]+Ct[2][2];
        double trC2 = 0.0;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) trC2 += Ct[i][j]*Ct[j][i];
        double I2 = 0.5*(I1*I1 - trC2);
        double J = det3(Fm);
        double Wdev = C10*(I1-3.0) + C01*(I2-3.0);
        double Wvol = 0.5*K*(J-1.0)*(J-1.0);
        return Wdev + Wvol;
    };

    // analytic P (standard MR): S = 2*(C10*I + C01*(I1*I - C)), P = F*S + K*(J-1)*J*F^{-T}
    double Ft[3][3]; mat_transpose(F, Ft);
    double Cmat[3][3]; mat_mul(Ft, F, Cmat);
    double I1 = Cmat[0][0]+Cmat[1][1]+Cmat[2][2];
    double trC2=0.0;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) trC2 += Cmat[i][j]*Cmat[j][i];
    double I2 = 0.5*(I1*I1 - trC2);
    double S[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
        double identity = (i==j)?1.0:0.0;
        S[i][j] = 2.0*(C10*identity + C01*(I1*identity - Cmat[i][j]));
    }
    double P_analytic[3][3]; mat_mul(F, S, P_analytic);
    double J = det3(F);
    // compute F^{-T}
    double invF[3][3];
    double detF = J;
    invF[0][0] =  (F[1][1]*F[2][2]-F[1][2]*F[2][1]) / detF;
    invF[0][1] = -(F[0][1]*F[2][2]-F[0][2]*F[2][1]) / detF;
    invF[0][2] =  (F[0][1]*F[1][2]-F[0][2]*F[1][1]) / detF;
    invF[1][0] = -(F[1][0]*F[2][2]-F[1][2]*F[2][0]) / detF;
    invF[1][1] =  (F[0][0]*F[2][2]-F[0][2]*F[2][0]) / detF;
    invF[1][2] = -(F[0][0]*F[1][2]-F[0][2]*F[1][0]) / detF;
    invF[2][0] =  (F[1][0]*F[2][1]-F[1][1]*F[2][0]) / detF;
    invF[2][1] = -(F[0][0]*F[2][1]-F[0][1]*F[2][0]) / detF;
    invF[2][2] =  (F[0][0]*F[1][1]-F[0][1]*F[1][0]) / detF;
    double FinvT[3][3];
    mat_transpose(invF, FinvT);
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
        P_analytic[i][j] += K*(J-1.0)*J*FinvT[i][j];
    }

    // numerical P via FD
    double P_fd[3][3];
    double h = 1e-6;
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
        double Fp[3][3], Fm[3][3];
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) { Fp[i][j]=F[i][j]; Fm[i][j]=F[i][j]; }
        Fp[a][b] += h;
        Fm[a][b] -= h;
        double Wp = W(Fp);
        double Wm = W(Fm);
        P_fd[a][b] = (Wp - Wm) / (2.0*h);
    }

    printf("Analytic P (standard MR):\n");
    for (int i=0;i<3;i++) {
        printf("%g %g %g\n", P_analytic[i][0], P_analytic[i][1], P_analytic[i][2]);
    }
    printf("\nFinite-difference P_fd:\n");
    for (int i=0;i<3;i++) {
        printf("%g %g %g\n", P_fd[i][0], P_fd[i][1], P_fd[i][2]);
    }
    printf("\nRelative error matrix (|P_fd - P_analytic| / max(|P_analytic|,eps)):\n");
    double max_abs = 0.0;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) max_abs = fmax(max_abs, fabs(P_analytic[i][j]));
    double eps = 1e-12;
    for (int i=0;i<3;i++) {
        printf("%g %g %g\n", fabs(P_fd[i][0]-P_analytic[i][0])/(fmax(fabs(P_analytic[i][0]),eps)),
                               fabs(P_fd[i][1]-P_analytic[i][1])/(fmax(fabs(P_analytic[i][1]),eps)),
                               fabs(P_fd[i][2]-P_analytic[i][2])/(fmax(fabs(P_analytic[i][2]),eps)));
    }

    return 0;
}