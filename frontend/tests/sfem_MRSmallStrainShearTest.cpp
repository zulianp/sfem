#include <stdio.h>
#include <math.h>
#include "hex8_inline_cpu.h"
#include "hex8_mooney_rivlin_visco_unique_Hi_local.h"

// Small-strain shear test: verify sigma12 ≈ mu * gamma for small gamma

static double det3(const double F[3][3]) {
    return F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1])
         - F[0][1]*(F[1][0]*F[2][2]-F[1][2]*F[2][0])
         + F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0]);
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

    const double C10 = 0.499;
    const double C01 = 0.577;
    const double K = 10000.0;
    const double gamma_alg = 1.0;

    double mus = 2.0 * (C10 + C01);

    double gammas[] = {1e-6, 1e-5, 1e-4};

    printf("mu (2*(C10+C01)) = %g\n", mus);

    for (int ig=0; ig<3; ++ig) {
        double g = gammas[ig];
        // simple shear F = [[1, g,0],[0,1,0],[0,0,1]]
        double F[3][3] = {{1.0, g, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

        // build nodal displacements u = (F - I) X
        scalar_t dispx[8], dispy[8], dispz[8];
        for (int n=0;n<8;n++){
            double X=lx[n], Y=ly[n], Z=lz[n];
            dispx[n] = (scalar_t)((F[0][0]-1.0)*X + F[0][1]*Y + F[0][2]*Z);
            dispy[n] = (scalar_t)(F[1][0]*X + (F[1][1]-1.0)*Y + F[1][2]*Z);
            dispz[n] = (scalar_t)(F[2][0]*X + F[2][1]*Y + (F[2][2]-1.0)*Z);
        }

        scalar_t jac_adj[9]; scalar_t jac_det;
        hex8_adjugate_and_det(lx, ly, lz, 0.0,0.0,0.0, jac_adj, &jac_det);

        scalar_t S_dev[6];
        hex8_mooney_rivlin_S_dev_from_disp(jac_adj, (scalar_t)jac_det, 0.0,0.0,0.0, 1.0,
                                           (scalar_t)C10,(scalar_t)C01,(scalar_t)K,
                                           dispx, dispy, dispz, S_dev);

        // assemble S
        double S[3][3] = {{S_dev[0], S_dev[3], S_dev[4]},
                          {S_dev[3], S_dev[1], S_dev[5]},
                          {S_dev[4], S_dev[5], S_dev[2]}};
        double J = det3(F);

        // compute Kirchhoff tau = F * S * F^T
        double temp[3][3] = {{0}};
        double tau[3][3] = {{0}};
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int k=0;k<3;k++) temp[i][j] += F[i][k]*S[k][j];
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) for (int k=0;k<3;k++) tau[i][j] += temp[i][k]*F[j][k];

        // Cauchy sigma = tau / J
        double sigma[3][3];
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) sigma[i][j] = tau[i][j] / J;

        double sigma12 = sigma[0][1];
        double expected = mus * g;
        double abs_err = fabs(sigma12 - expected);
        double rel_err = fabs(sigma12 - expected) / fmax(fabs(expected), 1e-16);

        printf("gamma=%g: sigma12=%g, expected=mu*gamma=%g, abs_err=%g, rel_err=%g\n",
               g, sigma12, expected, abs_err, rel_err);
    }

    return 0;
}






