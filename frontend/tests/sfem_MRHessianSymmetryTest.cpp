#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "hex8_inline_cpu.h"
#include "hex8_mooney_rivlin_visco_unique_Hi_local.h"

static double inf_norm_mat(const double *A, int n) {
    double m = 0.0;
    for (int i=0;i<n;i++){
        double s=0;
        for (int j=0;j<n;j++) s += fabs(A[i*n+j]);
        if (s>m) m=s;
    }
    return m;
}

int main(){
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

    double lam = 1.1;
    double F[3][3] = {{lam,0,0},{0,1.0/sqrt(lam),0},{0,0,1.0/sqrt(lam)}};
    scalar_t dispx[8], dispy[8], dispz[8];
    for (int n=0;n<8;n++){
        double X=lx[n], Y=ly[n], Z=lz[n];
        dispx[n] = (scalar_t)((F[0][0]-1.0)*X + F[0][1]*Y + F[0][2]*Z);
        dispy[n] = (scalar_t)(F[1][0]*X + (F[1][1]-1.0)*Y + F[1][2]*Z);
        dispz[n] = (scalar_t)(F[2][0]*X + F[2][1]*Y + (F[2][2]-1.0)*Z);
    }

    scalar_t jac_adj[9]; scalar_t jac_det;
    hex8_adjugate_and_det(lx, ly, lz, 0.0,0.0,0.0, jac_adj, &jac_det);

    // compute Hk
    scalar_t Hk[24*24];
    for (int i=0;i<24*24;i++) Hk[i]=0;
    hex8_mooney_rivlin_hessian_algo_micro(jac_adj, (scalar_t)jac_det, 0.0,0.0,0.0, 1.0,
                                          (scalar_t)C10,(scalar_t)C01,(scalar_t)K,(scalar_t)gamma,
                                          dispx, dispy, dispz, Hk);
    double H[24*24];
    for (int i=0;i<24*24;i++) H[i] = (double)Hk[i];

    // compute symmetry norm
    double Hm[24*24];
    for (int i=0;i<24;i++) for (int j=0;j<24;j++) Hm[i*24+j] = H[i*24+j] - H[j*24+i];
    double norm_Hm = inf_norm_mat(Hm,24);
    double norm_H = inf_norm_mat(H,24);
    double rel_sym = norm_Hm / fmax(norm_H, 1e-12);

    printf("Hessian symmetry: ||H - H^T||_inf = %g, ||H||_inf = %g, relative = %g\n", norm_Hm, norm_H, rel_sym);

    // sample d^T H d for random directions
    int ntrial = 20;
    double min_q = 1e300;
    for (int t=0;t<ntrial;t++){
        double d[24];
        for (int i=0;i<24;i++) d[i] = ((double)rand()/RAND_MAX - 0.5);
        // normalize
        double m = 0; for (int i=0;i<24;i++) if (fabs(d[i])>m) m=fabs(d[i]);
        if (m<1e-12) m=1.0;
        for (int i=0;i<24;i++) d[i] /= m;
        double q=0;
        for (int i=0;i<24;i++){
            double s=0;
            for (int j=0;j<24;j++) s += H[i*24+j]*d[j];
            q += d[i]*s;
        }
        if (q < min_q) min_q = q;
    }
    printf("min_{sample d} d^T H d = %g\n", min_q);

    if (rel_sym < 1e-8) printf("HESSIAN SYMMETRY: PASS\n");
    else printf("HESSIAN SYMMETRY: WARN (not fully symmetric)\n");
    if (min_q >= -1e-8) printf("HESSIAN PD (sampled): PASS\n");
    else printf("HESSIAN PD (sampled): WARNING: negative directional curvature observed\n");

    return 0;
}






