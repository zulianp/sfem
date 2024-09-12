#include "sfem_base.h"

#include <cmath>
#include <complex>
#include <cassert>

static SFEM_INLINE void principal_strain_kernel(const double px0,
                                                const double px1,
                                                const double px2,
                                                const double px3,
                                                const double py0,
                                                const double py1,
                                                const double py2,
                                                const double py3,
                                                const double pz0,
                                                const double pz1,
                                                const double pz2,
                                                const double pz3,
                                                // Data
                                                const double *const SFEM_RESTRICT ux,
                                                const double *const SFEM_RESTRICT uy,
                                                const double *const SFEM_RESTRICT uz,
                                                // Output
                                                double *const SFEM_RESTRICT e) {
    //FLOATING POINT OPS!
    //      - Result: 3*ADD + 3*ASSIGNMENT + 6*MUL + 2*POW
    //      - Subexpressions: 64*ADD + 54*DIV + 121*MUL + 4*NEG + 19*POW + 58*SUB
    const double x0 = -py0 + py2;
    const double x1 = -pz0 + pz3;
    const double x2 = -py0 + py3;
    const double x3 = -pz0 + pz2;
    const double x4 = -px0 + px1;
    const double x5 = x0*x4;
    const double x6 = -pz0 + pz1;
    const double x7 = -px0 + px2;
    const double x8 = x2*x7;
    const double x9 = -px0 + px3;
    const double x10 = -py0 + py1;
    const double x11 = x2*x4;
    const double x12 = x10*x7;
    const double x13 = x0*x9;
    const double x14 = 1.0/(-x1*x12 + x1*x5 + x10*x3*x9 - x11*x3 - x13*x6 + x6*x8);
    const double x15 = x14*(x0*x1 - x2*x3);
    const double x16 = x14*(-x1*x10 + x2*x6);
    const double x17 = x14*(-x0*x6 + x10*x3);
    const double x18 = -x15 - x16 - x17;
    const double x19 = uy[0]*x18 + uy[1]*x15 + uy[2]*x16 + uy[3]*x17;
    const double x20 = pow(x19, 2);
    const double x21 = uz[0]*x18 + uz[1]*x15 + uz[2]*x16 + uz[3]*x17;
    const double x22 = pow(x21, 2);
    const double x23 = ux[0]*x18 + ux[1]*x15 + ux[2]*x16 + ux[3]*x17 + 1;
    const double x24 = pow(x23, 2);
    const double x25 = (1.0/2.0)*x20 + (1.0/2.0)*x22 + (1.0/2.0)*x24;
    const double x26 = x25 - 1.0/2.0;
    const double x27 = x14*(-x1*x7 + x3*x9);
    const double x28 = x14*(x1*x4 - x6*x9);
    const double x29 = x14*(-x3*x4 + x6*x7);
    const double x30 = -x27 - x28 - x29;
    const double x31 = ux[0]*x30 + ux[1]*x27 + ux[2]*x28 + ux[3]*x29;
    const double x32 = x14*(-x13 + x8);
    const double x33 = x14*(x10*x9 - x11);
    const double x34 = x14*(-x12 + x5);
    const double x35 = -x32 - x33 - x34;
    const double x36 = ux[0]*x35 + ux[1]*x32 + ux[2]*x33 + ux[3]*x34;
    const double x37 = (1.0/2.0)*x36;
    const double x38 = uy[0]*x30 + uy[1]*x27 + uy[2]*x28 + uy[3]*x29 + 1;
    const double x39 = uy[0]*x35 + uy[1]*x32 + uy[2]*x33 + uy[3]*x34;
    const double x40 = (1.0/2.0)*x39;
    const double x41 = uz[0]*x30 + uz[1]*x27 + uz[2]*x28 + uz[3]*x29;
    const double x42 = uz[0]*x35 + uz[1]*x32 + uz[2]*x33 + uz[3]*x34 + 1;
    const double x43 = (1.0/2.0)*x42;
    const double x44 = x31*x37 + x38*x40 + x41*x43;
    const double x45 = pow(x44, 2);
    const double x46 = x26*x45;
    const double x47 = pow(x31, 2);
    const double x48 = pow(x41, 2);
    const double x49 = pow(x38, 2);
    const double x50 = (1.0/2.0)*x47 + (1.0/2.0)*x48 + (1.0/2.0)*x49;
    const double x51 = x50 - 1.0/2.0;
    const double x52 = x19*x40 + x21*x43 + x23*x37;
    const double x53 = pow(x52, 2);
    const double x54 = x51*x53;
    const double x55 = pow(x36, 2);
    const double x56 = pow(x39, 2);
    const double x57 = pow(x42, 2);
    const double x58 = (1.0/2.0)*x55 + (1.0/2.0)*x56 + (1.0/2.0)*x57;
    const double x59 = x58 - 1.0/2.0;
    const double x60 = (1.0/2.0)*x19*x38 + (1.0/2.0)*x21*x41 + (1.0/2.0)*x23*x31;
    const double x61 = pow(x60, 2);
    const double x62 = x59*x61;
    const double x63 = -x25 - x50 - x58 + 3.0/2.0;
    const double x64 = pow(x63, 3);
    const double x65 = x51*x59;
    const double x66 = x26*x65;
    const double x67 = x44*x52*x60;
    const double x68 = (x26*x51 + x26*x59 - x45 + x51*x59 - x53 - x61)*(-9.0/2.0*x20 - 9.0/2.0*x22 - 9.0/2.0*x24 - 
    9.0/2.0*x47 - 9.0/2.0*x48 - 9.0/2.0*x49 - 9.0/2.0*x55 - 9.0/2.0*x56 - 9.0/2.0*x57 + 27.0/2.0);
    const double x69 = -3*x26*x51 - 3*x26*x59 + 3*x45 + 3*x53 + 3*x61 + pow(x63, 2) - 3*x65;
    
    // Complex numbers
    std::complex<double> sqrt_x(-4*pow(x69, 3) + pow(27*x46 + 27*x54 + 27*x62 + 2*x64 - 27*x66 - 54*x67 - x68, 2), 0);

    const auto x70 = pow((27.0/2.0)*x46 + (27.0/2.0)*x54 + (27.0/2.0)*x62 + x64 - 27.0/2.0*x66 - 27*x67 - 
    1.0/2.0*x68 + (1.0/2.0)
    * std::sqrt(sqrt_x), 1./3);

    const auto x71 = (1.0/3.0)*x70;
    const auto x72 = (1.0/3.0)*x69/x70;
    const double x73 = (1.0/6.0)*x20 + (1.0/6.0)*x22 + (1.0/6.0)*x24 + (1.0/6.0)*x47 + (1.0/6.0)*x48 + (1.0/6.0)*x49 + 
    (1.0/6.0)*x55 + (1.0/6.0)*x56 + (1.0/6.0)*x57 - 1.0/2.0;
    const auto x74 = (1.0/2.0)*std::sqrt(3)*std::complex<double>(0, 1);
    const auto x75 = x74 - 1.0/2.0;
    const auto x76 = -x74 - 1.0/2.0;
    
    auto e0 = -x71 - x72 + x73;
    auto e1 = -x71*x75 - x72/x75 + x73;
    auto e2 = -x71*x76 - x72/x76 + x73;

    e[0] = std::real(e0);
    e[1] = std::real(e1);
    e[2] = std::real(e2);

    assert(std::imag(e0) < 1e-10);
    assert(std::imag(e1) < 1e-10);
    assert(std::imag(e2) < 1e-10);

    if(e[2] < e[0]) {
    	double temp = e[0];
    	e[0] = e[2];
    	e[2] = temp;
    }

   	if(e[1] < e[0]) {
   		double temp = e[0];
   		e[0] = e[1];
   		e[1] = temp;
   	}

   	if(e[2] < e[1]) {
   		double temp = e[1];
   		e[1] = e[2];
   		e[2] = temp;
   	}

    assert(e[0] <= e[1]);
    assert(e[1] <= e[2]);
}


extern "C" void principal_strains(const ptrdiff_t nelements,
           const ptrdiff_t nnodes,
           idx_t **const SFEM_RESTRICT elems,
           geom_t **const SFEM_RESTRICT xyz,
           const real_t *const SFEM_RESTRICT ux,
           const real_t *const SFEM_RESTRICT uy,
           const real_t *const SFEM_RESTRICT uz,
           real_t *const SFEM_RESTRICT strain_e0,
           real_t *const SFEM_RESTRICT strain_e1,
           real_t *const SFEM_RESTRICT strain_e2) {
    SFEM_UNUSED(nnodes);

    idx_t ev[4];
    double element_vector[4];

    double element_ux[4];
    double element_uy[4];
    double element_uz[4];
    double element_strain[3];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            element_ux[v] = ux[ev[v]];
        }

        for (int v = 0; v < 4; ++v) {
            element_uy[v] = uy[ev[v]];
        }

        for (int v = 0; v < 4; ++v) {
            element_uz[v] = uz[ev[v]];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        principal_strain_kernel(
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            xyz[0][i3],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
            xyz[1][i3],
            // Z-coordinates
            xyz[2][i0],
            xyz[2][i1],
            xyz[2][i2],
            xyz[2][i3],
            // Data
            element_ux,
            element_uy,
            element_uz,
            // Output
            element_strain);

        strain_e0[i] = element_strain[0];
        strain_e1[i] = element_strain[1];
        strain_e2[i] = element_strain[2];
    }
}

