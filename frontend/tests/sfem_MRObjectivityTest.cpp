#include <stdio.h>
#include <math.h>
#include "hex8_inline_cpu.h"
#include "hex8_mooney_rivlin_visco_unique_Hi_local.h"

// Objectivity test: verify W(RF)=W(F) and P(RF)=R*P(F) using flexible kernels

static double det3(const double F[3][3]) {
    return F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1])
         - F[0][1]*(F[1][0]*F[2][2]-F[1][2]*F[2][0])
         + F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0]);
}

// build rotation matrix from axis (nx,ny,nz) and angle theta
static void rodrigues(const double n[3], double theta, double R[3][3]) {
    double c = cos(theta), s = sin(theta), C = 1.0 - c;
    double nx = n[0], ny = n[1], nz = n[2];
    R[0][0] = c + nx*nx*C; R[0][1] = nx*ny*C - nz*s;  R[0][2] = nx*nz*C + ny*s;
    R[1][0] = ny*nx*C + nz*s; R[1][1] = c + ny*ny*C; R[1][2] = ny*nz*C - nx*s;
    R[2][0] = nz*nx*C - ny*s; R[2][1] = nz*ny*C + nx*s; R[2][2] = c + nz*nz*C;
}

static double mat_norm_inf(const double A[3][3]) {
    double m = 0;
    for (int i=0;i<3;i++){
        double s = 0;
        for (int j=0;j<3;j++) s += fabs(A[i][j]);
        if (s > m) m = s;
    }
    return m;
}

int main() {
    // unit cube geometry
    scalar_t lx[8], ly[8], lz[8];
    lx[0]=0;ly[0]=0;lz[0]=0;
    lx[1]=1;ly[1]=0;lz[1]=0;
    lx[2]=1;ly[2]=1;lz[2]=0;
    lx[3]=0;ly[3]=1;lz[3]=0;
    lx[4]=0;ly[4]=0;lz[4]=1;
    lx[5]=1;ly[5]=0;lz[5]=1;
    lx[6]=1;ly[6]=1;lz[6]=1;
    lx[7]=0;ly[7]=1;lz[7]=1;

    // material params (use data-fit)
    const double C10 = 0.499;
    const double C01 = 0.577;
    const double K = 10000.0;

    // choose F (uniaxial)
    double lam = 1.2;
    double F[3][3] = {{lam,0,0},{0,1.0/sqrt(lam),0},{0,0,1.0/sqrt(lam)}};

    // choose rotation: axis (1,1,1) normalized, angle 0.7 rad
    double n[3] = {1.0,1.0,1.0};
    double nl = sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
    for (int i=0;i<3;i++) n[i] /= nl;
    double R[3][3];
    rodrigues(n, 0.7, R);

    // compute RF = R * F
    double RF[3][3] = {{0}};
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int k=0;k<3;k++) RF[i][j] += R[i][k]*F[k][j];

    // energy function (unimodular) reused
    auto W_unimod = [&](const double Fm[3][3]) {
        double Ft[3][3]; for(int i=0;i<3;i++) for(int j=0;j<3;j++) Ft[i][j]=Fm[j][i];
        double C[3][3]; for(int i=0;i<3;i++) for(int j=0;j<3;j++){ C[i][j]=0.0; for(int k=0;k<3;k++) C[i][j]+=Ft[i][k]*Fm[k][j]; }
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

    double Wf = W_unimod(F);
    double Wrf = W_unimod(RF);

    // build displacement arrays for F and RF: u = (F - I) X
    scalar_t dispx[8], dispy[8], dispz[8];
    scalar_t dispx_r[8], dispy_r[8], dispz_r[8];
    for (int nidx=0;nidx<8;nidx++){
        double X = lx[nidx], Y = ly[nidx], Z = lz[nidx];
        double ux = (F[0][0]-1.0)*X + F[0][1]*Y + F[0][2]*Z;
        double uy = F[1][0]*X + (F[1][1]-1.0)*Y + F[1][2]*Z;
        double uz = F[2][0]*X + F[2][1]*Y + (F[2][2]-1.0)*Z;
        dispx[nidx]=(scalar_t)ux; dispy[nidx]=(scalar_t)uy; dispz[nidx]=(scalar_t)uz;
        double ux2 = (RF[0][0]-1.0)*X + RF[0][1]*Y + RF[0][2]*Z;
        double uy2 = RF[1][0]*X + (RF[1][1]-1.0)*Y + RF[1][2]*Z;
        double uz2 = RF[2][0]*X + RF[2][1]*Y + (RF[2][2]-1.0)*Z;
        dispx_r[nidx]=(scalar_t)ux2; dispy_r[nidx]=(scalar_t)uy2; dispz_r[nidx]=(scalar_t)uz2;
    }

    scalar_t jac_adj[9]; scalar_t jac_det;
    hex8_adjugate_and_det(lx, ly, lz, 0.0,0.0,0.0, jac_adj, &jac_det);

    scalar_t S_dev[6], S_dev_r[6];
    // call flexible kernel S_dev_from_disp for F and RF
    hex8_mooney_rivlin_S_dev_from_disp(jac_adj, (scalar_t)jac_det, 0.0,0.0,0.0, 1.0,
                                       (scalar_t)C10,(scalar_t)C01,(scalar_t)K,
                                       dispx, dispy, dispz, S_dev);
    hex8_mooney_rivlin_S_dev_from_disp(jac_adj, (scalar_t)jac_det, 0.0,0.0,0.0, 1.0,
                                       (scalar_t)C10,(scalar_t)C01,(scalar_t)K,
                                       dispx_r, dispy_r, dispz_r, S_dev_r);

    // assemble S matrices
    double Smat[3][3] = {{S_dev[0], S_dev[3], S_dev[4]},
                         {S_dev[3], S_dev[1], S_dev[5]},
                         {S_dev[4], S_dev[5], S_dev[2]}};
    double Smat_r[3][3] = {{S_dev_r[0], S_dev_r[3], S_dev_r[4]},
                           {S_dev_r[3], S_dev_r[1], S_dev_r[5]},
                           {S_dev_r[4], S_dev_r[5], S_dev_r[2]}};

    // compute P = F * S and P_r = R F * S_r
    double P[3][3] = {{0}}, P_r[3][3] = {{0}}, RP[3][3] = {{0}};
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int k=0;k<3;k++){
        P[i][j] += F[i][k]*Smat[k][j];
        P_r[i][j] += RF[i][k]*Smat_r[k][j];
    }
    // compute R * P
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int k=0;k<3;k++) RP[i][j] += R[i][k]*P[k][j];

    // compare
    double eps = 1e-12;
    double W_rel = fabs(Wrf - Wf) / fmax(fabs(Wf), eps);
    double Pdiff[3][3]; for (int i=0;i<3;i++) for (int j=0;j<3;j++) Pdiff[i][j] = P_r[i][j] - RP[i][j];
    double Pnorm = mat_norm_inf(P_r);
    double P_rel = mat_norm_inf(Pdiff) / fmax(Pnorm, 1e-12);

    printf("W(F) = %g, W(RF) = %g, rel_diff = %g\n", Wf, Wrf, W_rel);
    printf("||P_r - R*P||_inf = %g, relative = %g\n", mat_norm_inf(Pdiff), P_rel);
    printf("P_r =\n");
    for (int i=0;i<3;i++) printf("%g %g %g\n", P_r[i][0], P_r[i][1], P_r[i][2]);
    printf("R*P =\n");
    for (int i=0;i<3;i++) printf("%g %g %g\n", RP[i][0], RP[i][1], RP[i][2]);

    if (W_rel < 1e-10 && P_rel < 1e-8) {
        printf("OBJECTIVITY TEST: PASS\n");
        return 0;
    } else {
        printf("OBJECTIVITY TEST: FAIL\n");
        return 1;
    }
}






