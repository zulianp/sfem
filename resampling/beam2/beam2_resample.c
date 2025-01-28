#include "beam2_resample.h"
#include "beam2_mass.h"

#include <math.h>
#include <stdio.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define SFEM_RESAMPLE_GAP_DUAL

static SFEM_INLINE real_t put_inside(const real_t v) { return MIN(MAX(1e-7, v), 1 - 1e-7); }

#define EDGE2_NQP 12
static real_t edge2_qx[EDGE2_NQP] = {0.009219682876640489244124410106451250612735748291015625000000,
                                     0.047941371814762490100036984586040489375591278076171875000000,
                                     0.115048662902847487199409215463674627244472503662109375000000,
                                     0.206341022856691480580337838546256534755229949951171875000000,
                                     0.316084250500909991199449677878874354064464569091796875000000,
                                     0.437383295744265487847712847724324092268943786621093750000000,
                                     0.562616704255734512152287152275675907731056213378906250000000,
                                     0.683915749499089953289399090863298624753952026367187500000000,
                                     0.793658977143308463908510930195916444063186645507812500000000,
                                     0.884951337097152457289439553278498351573944091796875000000000,
                                     0.952058628185237454388811784156132489442825317382812500000000,
                                     0.990780317123359566267026821151375770568847656250000000000000};

static real_t edge2_qw[EDGE2_NQP] = {0.023587668193255501014604647025407757610082626342773437500000,
                                     0.053469662997658998215833037193078780546784400939941406250000,
                                     0.080039164271672999517726054818922420963644981384277343750000,
                                     0.101583713361533004015946346498822094872593879699707031250000,
                                     0.116746268269177499998789926394238136708736419677734375000000,
                                     0.124573522906701497636738906749087618663907051086425781250000,
                                     0.124573522906701497636738906749087618663907051086425781250000,
                                     0.116746268269177499998789926394238136708736419677734375000000,
                                     0.101583713361533004015946346498822094872593879699707031250000,
                                     0.080039164271672999517726054818922420963644981384277343750000,
                                     0.053469662997658998215833037193078780546784400939941406250000,
                                     0.023587668193255501014604647025407757610082626342773437500000};

SFEM_INLINE static real_t beam2_measure(
        // X-coordinates
        const real_t px0,
        const real_t px1,
        // Y-coordinates
        const real_t py0,
        const real_t py1,
        // Z-coordinates
        const real_t pz0,
        const real_t pz1) {
    return sqrt(pow(-px0 + px1, 2) + pow(-py0 + py1, 2) + pow(-pz0 + pz1, 2));
}

SFEM_INLINE static void beam2_transform(
        // X-coordinates
        const real_t px0,
        const real_t px1,
        // Y-coordinates
        const real_t py0,
        const real_t py1,
        // Z-coordinates
        const real_t pz0,
        const real_t pz1,
        // Quadrature point
        const real_t x,
        // Output
        real_t* const SFEM_RESTRICT out_x,
        real_t* const SFEM_RESTRICT out_y,
        real_t* const SFEM_RESTRICT out_z) {
    *out_x = px0 + x * (-px0 + px1);
    *out_y = py0 + x * (-py0 + py1);
    *out_z = pz0 + x * (-pz0 + pz1);
}

SFEM_INLINE static void hex_aa_8_eval_fun(
        // Quadrature point (local coordinates)
        const real_t x,
        const real_t y,
        const real_t z,
        // Output
        real_t* const SFEM_RESTRICT f) {
    f[0] = (1.0 - x) * (1.0 - y) * (1.0 - z);
    f[1] = x * (1.0 - y) * (1.0 - z);
    f[2] = x * y * (1.0 - z);
    f[3] = (1.0 - x) * y * (1.0 - z);
    f[4] = (1.0 - x) * (1.0 - y) * z;
    f[5] = x * (1.0 - y) * z;
    f[6] = x * y * z;
    f[7] = (1.0 - x) * y * z;
}

SFEM_INLINE static void hex_aa_8_eval_grad(
        // Quadrature point (local coordinates)
        const real_t x,
        const real_t y,
        const real_t z,
        // Output
        real_t* const SFEM_RESTRICT gx,
        real_t* const SFEM_RESTRICT gy,
        real_t* const SFEM_RESTRICT gz) {
    // Transformation to ref element
    gx[0] = -(1.0 - y) * (1.0 - z);
    gy[0] = -(1.0 - x) * (1.0 - z);
    gz[0] = -(1.0 - x) * (1.0 - y);

    gx[1] = (1.0 - y) * (1.0 - z);
    gy[1] = -x * (1.0 - z);
    gz[1] = -x * (1.0 - y);

    gx[2] = y * (1.0 - z);
    gy[2] = x * (1.0 - z);
    gz[2] = -x * y;

    gx[3] = -y * (1.0 - z);
    gy[3] = (1.0 - x) * (1.0 - z);
    gz[3] = -(1.0 - x) * y;

    gx[4] = -(1.0 - y) * z;
    gy[4] = -(1.0 - x) * z;
    gz[4] = (1.0 - x) * (1.0 - y);

    gx[5] = (1.0 - y) * z;
    gy[5] = -x * z;
    gz[5] = x * (1.0 - y);

    gx[6] = y * z;
    gy[6] = x * z;
    gz[6] = x * y;

    gx[7] = -y * z;
    gy[7] = (1.0 - x) * z;
    gz[7] = (1.0 - x) * y;
}

SFEM_INLINE static void hex_aa_8_collect_coeffs(const ptrdiff_t* const SFEM_RESTRICT stride,
                                                const ptrdiff_t                      i,
                                                const ptrdiff_t                      j,
                                                const ptrdiff_t                      k,
                                                // Attention this is geometric data transformed to solver data!
                                                const geom_t* const SFEM_RESTRICT data,
                                                real_t* const SFEM_RESTRICT       out) {
    const ptrdiff_t i0 = i * stride[0] + j * stride[1] + k * stride[2];
    const ptrdiff_t i1 = (i + 1) * stride[0] + j * stride[1] + k * stride[2];
    const ptrdiff_t i2 = (i + 1) * stride[0] + (j + 1) * stride[1] + k * stride[2];
    const ptrdiff_t i3 = i * stride[0] + (j + 1) * stride[1] + k * stride[2];
    const ptrdiff_t i4 = i * stride[0] + j * stride[1] + (k + 1) * stride[2];
    const ptrdiff_t i5 = (i + 1) * stride[0] + j * stride[1] + (k + 1) * stride[2];
    const ptrdiff_t i6 = (i + 1) * stride[0] + (j + 1) * stride[1] + (k + 1) * stride[2];
    const ptrdiff_t i7 = i * stride[0] + (j + 1) * stride[1] + (k + 1) * stride[2];

    out[0] = data[i0];
    out[1] = data[i1];
    out[2] = data[i2];
    out[3] = data[i3];
    out[4] = data[i4];
    out[5] = data[i5];
    out[6] = data[i6];
    out[7] = data[i7];
}

int beam2_resample_gap_local(
        // Mesh
        const ptrdiff_t              nelements,
        const ptrdiff_t              nnodes,
        idx_t** const SFEM_RESTRICT  elems,
        geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,
        const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT    origin,
        const geom_t* const SFEM_RESTRICT    delta,
        const geom_t* const SFEM_RESTRICT    data,
        // Output
        real_t* const SFEM_RESTRICT wg,
        real_t* const SFEM_RESTRICT xnormal,
        real_t* const SFEM_RESTRICT ynormal,
        real_t* const SFEM_RESTRICT znormal) {
    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

#pragma omp parallel for  // nowait
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t  ev[3];
        geom_t x[3], y[3], z[3];

        real_t hex8_f[8];
        real_t hex8_grad_x[8];
        real_t hex8_grad_y[8];
        real_t hex8_grad_z[8];
        real_t coeffs[8];

        real_t beam2_f[2];
        real_t element_gap[2];
        real_t element_xnormal[2];
        real_t element_ynormal[2];
        real_t element_znormal[2];

#pragma unroll(2)
        for (int v = 0; v < 2; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 2; ++v) {
            x[v] = xyz[0][ev[v]];
            y[v] = xyz[1][ev[v]];
            z[v] = xyz[2][ev[v]];
        }

        memset(element_gap, 0, 2 * sizeof(real_t));
        memset(element_xnormal, 0, 2 * sizeof(real_t));
        memset(element_ynormal, 0, 2 * sizeof(real_t));
        memset(element_znormal, 0, 2 * sizeof(real_t));

        const real_t measure = beam2_measure(x[0], x[1], y[0], y[1], z[0], z[1]);

        assert(measure > 0);

        for (int q = 0; q < EDGE2_NQP; q++) {
            real_t g_qx, g_qy, g_qz;
            beam2_transform(x[0], x[1], y[0], y[1], z[0], z[1], edge2_qx[q], &g_qx, &g_qy, &g_qz);

#ifndef SFEM_RESAMPLE_GAP_DUAL
            // Standard basis function
            {
                beam2_f[0] = 1 - edge2_qx[q];
                beam2_f[1] = edge2_qx[q];
            }
#else
            // DUAL basis function
            {
                const real_t f0 = 1 - edge2_qx[q];
                const real_t f1 = edge2_qx[q];
                beam2_f[0]      = 2 * f0 - f1;
                beam2_f[1]      = -f0 + 2 * f1;
            }
#endif

            const real_t dV = measure * edge2_qw[q];

            const real_t grid_x = (g_qx - ox) / dx;
            const real_t grid_y = (g_qy - oy) / dy;
            const real_t grid_z = (g_qz - oz) / dz;

            const ptrdiff_t i = floor(grid_x);
            const ptrdiff_t j = floor(grid_y);
            const ptrdiff_t k = floor(grid_z);

            // If outside
            if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) || (k + 1 >= n[2])) {
                SFEM_ERROR(
                        "(%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, "
                        "%ld)!\n",
                        g_qx,
                        g_qy,
                        g_qz,
                        i,
                        j,
                        k,
                        n[0],
                        n[1],
                        n[2]);
                continue;
            }

            // Get the reminder [0, 1]
            real_t l_x = (grid_x - i);
            real_t l_y = (grid_y - j);
            real_t l_z = (grid_z - k);

            assert(l_x >= -1e-8);
            assert(l_y >= -1e-8);
            assert(l_z >= -1e-8);

            assert(l_x <= 1 + 1e-8);
            assert(l_y <= 1 + 1e-8);
            assert(l_z <= 1 + 1e-8);

            hex_aa_8_eval_fun(l_x, l_y, l_z, hex8_f);
            hex_aa_8_eval_grad(put_inside(l_x), put_inside(l_y), put_inside(l_z), hex8_grad_x, hex8_grad_y, hex8_grad_z);
            hex_aa_8_collect_coeffs(stride, i, j, k, data, coeffs);

            // Integrate gap function
            {
                real_t eval_gap = 0;

#pragma unroll(8)
                for (int edof_j = 0; edof_j < 8; edof_j++) {
                    eval_gap += hex8_f[edof_j] * coeffs[edof_j];
                }

#pragma unroll(2)
                for (int edof_i = 0; edof_i < 2; edof_i++) {
                    element_gap[edof_i] += eval_gap * beam2_f[edof_i] * dV;
                }
            }

            {
                real_t eval_xnormal = 0;
                real_t eval_ynormal = 0;
                real_t eval_znormal = 0;

#pragma unroll(8)
                for (int edof_j = 0; edof_j < 8; edof_j++) {
                    eval_xnormal += hex8_grad_x[edof_j] * coeffs[edof_j];
                    eval_ynormal += hex8_grad_y[edof_j] * coeffs[edof_j];
                    eval_znormal += hex8_grad_z[edof_j] * coeffs[edof_j];
                }

                {
                    // Normalize
                    real_t denom = sqrt(eval_xnormal * eval_xnormal + eval_ynormal * eval_ynormal + eval_znormal * eval_znormal);

                    assert(denom != 0);

                    eval_xnormal /= denom;
                    eval_ynormal /= denom;
                    eval_znormal /= denom;
                }

#pragma unroll(2)
                for (int edof_i = 0; edof_i < 2; edof_i++) {
                    element_xnormal[edof_i] += eval_xnormal * beam2_f[edof_i] * dV;
                    element_ynormal[edof_i] += eval_ynormal * beam2_f[edof_i] * dV;
                    element_znormal[edof_i] += eval_znormal * beam2_f[edof_i] * dV;
                }
            }
        }

#pragma unroll(2)
        for (int v = 0; v < 2; ++v) {
            // Invert sign since distance field is negative insdide and positive outside
#pragma omp critical
            {
                wg[ev[v]] -= element_gap[v];
                xnormal[ev[v]] += element_xnormal[v];
                ynormal[ev[v]] += element_ynormal[v];
                znormal[ev[v]] += element_znormal[v];
            }
        }
    }

    return SFEM_SUCCESS;
}


int beam2_resample_weight_local(
        // Mesh
        const ptrdiff_t              nelements,
        const ptrdiff_t              nnodes,
        idx_t** const SFEM_RESTRICT  elems,
        geom_t** const SFEM_RESTRICT xyz,
        // Output
        real_t* const SFEM_RESTRICT w)
{
	return beam2_assemble_lumped_mass(nelements, nnodes, elems, xyz, w);
}


