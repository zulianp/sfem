#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "hex8_inline_cpu.h"
#include "hex8_mooney_rivlin_visco_unique_Hi_local.h"

// Hessian directional FD test: compare (grad(F+eps*d)-grad(F-eps*d))/(2eps) to H * d

static double det3(const double F[3][3]) {
    return F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1])
         - F[0][1]*(F[1][0]*F[2][2]-F[1][2]*F[2][0])
         + F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0]);
}

static double vec_norm_inf(const double *v, int n) {
    double m = 0.0;
    for (int i=0;i<n;i++) { double a = fabs(v[i]); if (a>m) m=a; }
    return m;
}

int main() {
    // geometry
    scalar_t lx[8], ly[8], lz[8];
    lx[0]=0;ly[0]=0;lz[0]=0;
    lx[1]=1;ly[1]=0;lz[1]=0;
    lx[2]=1;ly[2]=1;lz[2]=0;
    lx[3]=0;ly[3]=1;lz[3]=0;
    lx[4]=0;ly[4]=0;lz[4]=1;
    lx[5]=1;ly[5]=0;lz[5]=1;
    lx[6]=1;ly[6]=1;lz[6]=1;
    lx[7]=0;ly[7]=1;lz[7]=1;

    // material
    const double C10 = 0.499;
    const double C01 = 0.577;
    const double K = 10000.0;
    const double gamma = 1.0; // algorithmic part only

    // base deformation F
    double lam = 1.15;
    double F[3][3] = {{lam,0,0},{0,1.0/sqrt(lam),0},{0,0,1.0/sqrt(lam)}};

    // build base displacements u = (F-I) X
    scalar_t dispx[8], dispy[8], dispz[8];
    for (int n=0;n<8;n++){
        double X=lx[n], Y=ly[n], Z=lz[n];
        dispx[n] = (scalar_t)((F[0][0]-1.0)*X + F[0][1]*Y + F[0][2]*Z);
        dispy[n] = (scalar_t)(F[1][0]*X + (F[1][1]-1.0)*Y + F[1][2]*Z);
        dispz[n] = (scalar_t)(F[2][0]*X + F[2][1]*Y + (F[2][2]-1.0)*Z);
    }

    // random direction d (length 24)
    double d[24];
    srand(12345);
    for (int i=0;i<24;i++) d[i] = ((double)rand()/RAND_MAX - 0.5);
    // normalize
    double maxd = 0;
    for (int i=0;i<24;i++) { double a=fabs(d[i]); if (a>maxd) maxd=a; }
    if (maxd == 0) maxd = 1.0;
    for (int i=0;i<24;i++) d[i] /= maxd;

    // split d into nodal displacements
    scalar_t dpx[8], dpy[8], dpz[8];
    for (int i=0;i<8;i++) { dpx[i]=(scalar_t)d[i]; dpy[i]=(scalar_t)d[8+i]; dpz[i]=(scalar_t)d[16+i]; }

    // quadrature & jacobian
    scalar_t jac_adj[9]; scalar_t jac_det;
    hex8_adjugate_and_det(lx, ly, lz, 0.0,0.0,0.0, jac_adj, &jac_det);

    // compute gradient at base
    scalar_t gx0[8]={0}, gy0[8]={0}, gz0[8]={0};
    hex8_mooney_rivlin_grad_flexible(jac_adj, (scalar_t)jac_det, 0.0,0.0,0.0, 1.0,
                                     (scalar_t)C10,(scalar_t)C01,(scalar_t)K,(scalar_t)gamma,
                                     dispx, dispy, dispz, gx0, gy0, gz0);
    double g0[24];
    for (int i=0;i<8;i++) { g0[i]=gx0[i]; g0[8+i]=gy0[i]; g0[16+i]=gz0[i]; }

    // choose eps
    double eps = 1e-6;

    // compute gradient at +eps*d
    scalar_t dispx_p[8], dispy_p[8], dispz_p[8];
    scalar_t dispx_m[8], dispy_m[8], dispz_m[8];
    for (int i=0;i<8;i++){
        dispx_p[i] = dispx[i] + (scalar_t)(eps * dpx[i]);
        dispy_p[i] = dispy[i] + (scalar_t)(eps * dpy[i]);
        dispz_p[i] = dispz[i] + (scalar_t)(eps * dpz[i]);
        dispx_m[i] = dispx[i] - (scalar_t)(eps * dpx[i]);
        dispy_m[i] = dispy[i] - (scalar_t)(eps * dpy[i]);
        dispz_m[i] = dispz[i] - (scalar_t)(eps * dpz[i]);
    }

    scalar_t gx_p[8]={0}, gy_p[8]={0}, gz_p[8]={0};
    scalar_t gx_m[8]={0}, gy_m[8]={0}, gz_m[8]={0};
    hex8_mooney_rivlin_grad_flexible(jac_adj, (scalar_t)jac_det, 0.0,0.0,0.0, 1.0,
                                     (scalar_t)C10,(scalar_t)C01,(scalar_t)K,(scalar_t)gamma,
                                     dispx_p, dispy_p, dispz_p, gx_p, gy_p, gz_p);
    hex8_mooney_rivlin_grad_flexible(jac_adj, (scalar_t)jac_det, 0.0,0.0,0.0, 1.0,
                                     (scalar_t)C10,(scalar_t)C01,(scalar_t)K,(scalar_t)gamma,
                                     dispx_m, dispy_m, dispz_m, gx_m, gy_m, gz_m);

    double v_fd[24];
    for (int i=0;i<8;i++){
        v_fd[i] = (double)(gx_p[i] - gx_m[i]) / (2.0*eps);
        v_fd[8+i] = (double)(gy_p[i] - gy_m[i]) / (2.0*eps);
        v_fd[16+i] = (double)(gz_p[i] - gz_m[i]) / (2.0*eps);
    }

    // compute Hessian H (24x24) via kernel
    double H[24*24];
    // kernel writes scalar_t H array; allocate local and copy
    scalar_t Hk[24*24];
    for (int i=0;i<24*24;i++) Hk[i]=0;
    hex8_mooney_rivlin_hessian_algo_micro(jac_adj, (scalar_t)jac_det, 0.0,0.0,0.0, 1.0,
                                          (scalar_t)C10,(scalar_t)C01,(scalar_t)K,(scalar_t)gamma,
                                          dispx, dispy, dispz, Hk);
    for (int i=0;i<24;i++) for (int j=0;j<24;j++) H[i*24 + j] = (double)Hk[i*24 + j];

    // compute v_h = H * d
    double v_h[24];
    for (int i=0;i<24;i++){
        double s = 0.0;
        for (int j=0;j<24;j++) s += H[i*24 + j] * d[j];
        v_h[i] = s;
    }

    // compare v_fd and v_h
    double diff[24];
    for (int i=0;i<24;i++) diff[i] = v_fd[i] - v_h[i];
    double rel = vec_norm_inf(diff,24) / fmax(vec_norm_inf(v_fd,24), 1e-12);

    printf("Hessian directional FD test:\n");
    printf("eps = %g, ||v_fd - H*d||_inf = %g, relative = %g\n", eps, vec_norm_inf(diff,24), rel);
    if (rel < 1e-6) {
        printf("HESSIAN DIRECTIONAL TEST: PASS\n");
        return 0;
    } else {
        printf("HESSIAN DIRECTIONAL TEST: FAIL\n");
        return 1;
    }
}






