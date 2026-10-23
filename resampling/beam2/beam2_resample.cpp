#include "beam2_resample.hpp"
#include "beam2_mass.hpp"

#include "sfem_sdf_sampling.hpp"

#include <math.h>
#include <stdio.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define SFEM_RESAMPLE_GAP_DUAL

// The gap where a point lies outside the SDF grid: large, so the constraint stays inactive,
// as on the other shells.
static const real_t infty = SFEM_SDF_UNREACHABLE;
static const real_t SFEM_SDF_GRAD_EPS = 1e-12;

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

int beam2_resample_gap_local(
        // Mesh
        const ptrdiff_t              nelements,
        const ptrdiff_t              /*nnodes*/,
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

#pragma omp parallel for  // nowait
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t  ev[3];
        geom_t x[3], y[3], z[3];

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

        // The edge's own normal in the xy-plane, for points outside the grid and flat SDF cells.
        real_t face_nx, face_ny;
        const real_t face_nz = 0;
        {
            const real_t ux = x[1] - x[0], uy = y[1] - y[0];
            const real_t len = sqrt(ux * ux + uy * uy);
            face_nx          = uy / len;
            face_ny          = -ux / len;
        }

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

            real_t eval_value, eval_grad[3];
            // If outside
            if (!sdf_sample<1>(n, stride, origin, delta, data, g_qx, g_qy, g_qz, &eval_value, eval_grad, nullptr)) {
                for (int edof_i = 0; edof_i < 2; edof_i++) {
                    element_gap[edof_i] += -infty * beam2_f[edof_i] * dV;
                    element_xnormal[edof_i] += face_nx * beam2_f[edof_i] * dV;
                    element_ynormal[edof_i] += face_ny * beam2_f[edof_i] * dV;
                    element_znormal[edof_i] += face_nz * beam2_f[edof_i] * dV;
                }
                continue;
            }

            // Integrate gap function
            {
                const real_t eval_gap = eval_value;

#pragma unroll(2)
                for (int edof_i = 0; edof_i < 2; edof_i++) {
                    element_gap[edof_i] += eval_gap * beam2_f[edof_i] * dV;
                }
            }

            {
                real_t eval_xnormal = eval_grad[0];
                real_t eval_ynormal = eval_grad[1];
                real_t eval_znormal = eval_grad[2];

                const real_t denom = sqrt(eval_xnormal * eval_xnormal + eval_ynormal * eval_ynormal + eval_znormal * eval_znormal);

                if (denom >= SFEM_SDF_GRAD_EPS) {
                    eval_xnormal /= denom;
                    eval_ynormal /= denom;
                    eval_znormal /= denom;
                } else {
                    // Flat SDF cell: fall back to geometric face normal
                    eval_xnormal = face_nx;
                    eval_ynormal = face_ny;
                    eval_znormal = face_nz;
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

