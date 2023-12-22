#include "sfem_resample_field.h"

#include "mass.h"
#include "read_mesh.h"

#include "matrixio_array.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define SFEM_RESAMPLE_GAP_DUAL

static SFEM_INLINE real_t put_inside(const real_t v) { return MIN(MAX(1e-7, v), 1 - 1e-7); }

// TRI3 6th order quadrature rule
#define TRI3_NQP 12
static real_t tri3_qw[TRI3_NQP] = {0.050844906370206816920936809106869,
                                   0.050844906370206816920936809106869,
                                   0.050844906370206816920936809106869,
                                   0.11678627572637936602528961138558,
                                   0.11678627572637936602528961138558,
                                   0.11678627572637936602528961138558,
                                   0.082851075618373575193553456420442,
                                   0.082851075618373575193553456420442,
                                   0.082851075618373575193553456420442,
                                   0.082851075618373575193553456420442,
                                   0.082851075618373575193553456420442,
                                   0.082851075618373575193553456420442};

static real_t tri3_qx[TRI3_NQP] = {0.063089014491502228340331602870819,
                                   0.063089014491502228340331602870819,
                                   0.87382197101699554331933679425836,
                                   0.24928674517091042129163855310702,
                                   0.24928674517091042129163855310702,
                                   0.50142650965817915741672289378596,
                                   0.053145049844816947353249671631398,
                                   0.053145049844816947353249671631398,
                                   0.31035245103378440541660773395655,
                                   0.31035245103378440541660773395655,
                                   0.63650249912139864723014259441205,
                                   0.63650249912139864723014259441205};

static real_t tri3_qy[TRI3_NQP] = {0.063089014491502228340331602870819,
                                   0.87382197101699554331933679425836,
                                   0.063089014491502228340331602870819,
                                   0.24928674517091042129163855310702,
                                   0.50142650965817915741672289378596,
                                   0.24928674517091042129163855310702,
                                   0.31035245103378440541660773395655,
                                   0.63650249912139864723014259441205,
                                   0.053145049844816947353249671631398,
                                   0.63650249912139864723014259441205,
                                   0.053145049844816947353249671631398,
                                   0.31035245103378440541660773395655};

#define EDGE2_NQP 12
static real_t edge2_qx[EDGE2_NQP] = {
    0.009219682876640489244124410106451250612735748291015625000000,
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

static real_t edge2_qw[EDGE2_NQP] = {
    0.023587668193255501014604647025407757610082626342773437500000,
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

#define TET4_NQP 45
static real_t tet4_qx[TET4_NQP] = {
    0.2500000000000000, 0.6175871903000830, 0.1274709365666390, 0.1274709365666390,
    0.1274709365666390, 0.9037635088221031, 0.0320788303926323, 0.0320788303926323,
    0.0320788303926323, 0.4502229043567190, 0.0497770956432810, 0.0497770956432810,
    0.0497770956432810, 0.4502229043567190, 0.4502229043567190, 0.3162695526014501,
    0.1837304473985499, 0.1837304473985499, 0.1837304473985499, 0.3162695526014501,
    0.3162695526014501, 0.0229177878448171, 0.2319010893971509, 0.2319010893971509,
    0.5132800333608811, 0.2319010893971509, 0.2319010893971509, 0.2319010893971509,
    0.0229177878448171, 0.5132800333608811, 0.2319010893971509, 0.0229177878448171,
    0.5132800333608811, 0.7303134278075384, 0.0379700484718286, 0.0379700484718286,
    0.1937464752488044, 0.0379700484718286, 0.0379700484718286, 0.0379700484718286,
    0.7303134278075384, 0.1937464752488044, 0.0379700484718286, 0.7303134278075384,
    0.1937464752488044};

static real_t tet4_qy[TET4_NQP] = {
    0.2500000000000000, 0.1274709365666390, 0.1274709365666390, 0.1274709365666390,
    0.6175871903000830, 0.0320788303926323, 0.0320788303926323, 0.0320788303926323,
    0.9037635088221031, 0.0497770956432810, 0.4502229043567190, 0.0497770956432810,
    0.4502229043567190, 0.0497770956432810, 0.4502229043567190, 0.1837304473985499,
    0.3162695526014501, 0.1837304473985499, 0.3162695526014501, 0.1837304473985499,
    0.3162695526014501, 0.2319010893971509, 0.0229177878448171, 0.2319010893971509,
    0.2319010893971509, 0.5132800333608811, 0.2319010893971509, 0.0229177878448171,
    0.5132800333608811, 0.2319010893971509, 0.5132800333608811, 0.2319010893971509,
    0.0229177878448171, 0.0379700484718286, 0.7303134278075384, 0.0379700484718286,
    0.0379700484718286, 0.1937464752488044, 0.0379700484718286, 0.7303134278075384,
    0.1937464752488044, 0.0379700484718286, 0.1937464752488044, 0.0379700484718286,
    0.7303134278075384};

static real_t tet4_qz[TET4_NQP] = {
    0.2500000000000000, 0.1274709365666390, 0.1274709365666390, 0.6175871903000830,
    0.1274709365666390, 0.0320788303926323, 0.0320788303926323, 0.9037635088221031,
    0.0320788303926323, 0.0497770956432810, 0.0497770956432810, 0.4502229043567190,
    0.4502229043567190, 0.4502229043567190, 0.0497770956432810, 0.1837304473985499,
    0.1837304473985499, 0.3162695526014501, 0.3162695526014501, 0.3162695526014501,
    0.1837304473985499, 0.2319010893971509, 0.2319010893971509, 0.0229177878448171,
    0.2319010893971509, 0.2319010893971509, 0.5132800333608811, 0.5132800333608811,
    0.2319010893971509, 0.0229177878448171, 0.0229177878448171, 0.5132800333608811,
    0.2319010893971509, 0.0379700484718286, 0.0379700484718286, 0.7303134278075384,
    0.0379700484718286, 0.0379700484718286, 0.1937464752488044, 0.1937464752488044,
    0.0379700484718286, 0.7303134278075384, 0.7303134278075384, 0.1937464752488044,
    0.0379700484718286};

static real_t tet4_qw[TET4_NQP] = {
    -0.2359620398477559, 0.0244878963560563, 0.0244878963560563, 0.0244878963560563,
    0.0244878963560563,  0.0039485206398261, 0.0039485206398261, 0.0039485206398261,
    0.0039485206398261,  0.0263055529507371, 0.0263055529507371, 0.0263055529507371,
    0.0263055529507371,  0.0263055529507371, 0.0263055529507371, 0.0829803830550590,
    0.0829803830550590,  0.0829803830550590, 0.0829803830550590, 0.0829803830550590,
    0.0829803830550590,  0.0254426245481024, 0.0254426245481024, 0.0254426245481024,
    0.0254426245481024,  0.0254426245481024, 0.0254426245481024, 0.0254426245481024,
    0.0254426245481024,  0.0254426245481024, 0.0254426245481024, 0.0254426245481024,
    0.0254426245481024,  0.0134324384376852, 0.0134324384376852, 0.0134324384376852,
    0.0134324384376852,  0.0134324384376852, 0.0134324384376852, 0.0134324384376852,
    0.0134324384376852,  0.0134324384376852, 0.0134324384376852, 0.0134324384376852,
    0.0134324384376852};

SFEM_INLINE static int hex_aa_8_contains(
    // X-coordinates
    const real_t xmin,
    const real_t xmax,
    // Y-coordinates
    const real_t ymin,
    const real_t ymax,
    // Z-coordinates
    const real_t zmin,
    const real_t zmax,
    const real_t x,
    const real_t y,
    const real_t z) {
    int outside = (x < xmin) | (x > xmax) | (y < ymin) | (y > ymax) | (z < zmin) | (x > zmax);
    return !outside;
}

SFEM_INLINE static real_t tri_shell_3_measure(
    // X-coordinates
    const real_t px0,
    const real_t px1,
    const real_t px2,
    // Y-coordinates
    const real_t py0,
    const real_t py1,
    const real_t py2,
    // Z-coordinates
    const real_t pz0,
    const real_t pz1,
    const real_t pz2) {
    const real_t x0 = -px0 + px1;
    const real_t x1 = -px0 + px2;
    const real_t x2 = -py0 + py1;
    const real_t x3 = -py0 + py2;
    const real_t x4 = -pz0 + pz1;
    const real_t x5 = -pz0 + pz2;
    return (1.0 / 2.0) *
           sqrt((pow(x0, 2) + pow(x2, 2) + pow(x4, 2)) * (pow(x1, 2) + pow(x3, 2) + pow(x5, 2)) -
                pow(x0 * x1 + x2 * x3 + x4 * x5, 2));
}

SFEM_INLINE static void tri_shell_3_transform(
    // X-coordinates
    const real_t x0,
    const real_t x1,
    const real_t x2,
    // Y-coordinates
    const real_t y0,
    const real_t y1,
    const real_t y2,
    // Z-coordinates
    const real_t z0,
    const real_t z1,
    const real_t z2,
    // Quadrature point
    const real_t x,
    const real_t y,
    // Output
    real_t* const SFEM_RESTRICT out_x,
    real_t* const SFEM_RESTRICT out_y,
    real_t* const SFEM_RESTRICT out_z) {
    const real_t phi0 = 1 - x - y;
    const real_t phi1 = x;
    const real_t phi2 = y;

    *out_x = phi0 * x0 + phi1 * x1 + phi2 * x2;
    *out_y = phi0 * y0 + phi1 * y1 + phi2 * y2;
    *out_z = phi0 * z0 + phi1 * z1 + phi2 * z2;
}

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

SFEM_INLINE static real_t tet4_measure(
    // X-coordinates
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t px3,
    // Y-coordinates
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t py3,
    // Z-coordinates
    const real_t pz0,
    const real_t pz1,
    const real_t pz2,
    const real_t pz3) {
    const real_t x0 = -pz0 + pz3;
    const real_t x1 = -py0 + py2;
    const real_t x2 = -1.0 / 6.0 * px0 + (1.0 / 6.0) * px1;
    const real_t x3 = -py0 + py3;
    const real_t x4 = -pz0 + pz2;
    const real_t x5 = -py0 + py1;
    const real_t x6 = -1.0 / 6.0 * px0 + (1.0 / 6.0) * px2;
    const real_t x7 = -pz0 + pz1;
    const real_t x8 = -1.0 / 6.0 * px0 + (1.0 / 6.0) * px3;
    return x0 * x1 * x2 - x0 * x5 * x6 - x1 * x7 * x8 - x2 * x3 * x4 + x3 * x6 * x7 + x4 * x5 * x8;
}

SFEM_INLINE static void tet4_transform(
    // X-coordinates
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t px3,
    // Y-coordinates
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t py3,
    // Z-coordinates
    const real_t pz0,
    const real_t pz1,
    const real_t pz2,
    const real_t pz3,
    // Quadrature point
    const real_t qx,
    const real_t qy,
    const real_t qz,
    // Output
    real_t* const SFEM_RESTRICT out_x,
    real_t* const SFEM_RESTRICT out_y,
    real_t* const SFEM_RESTRICT out_z) {
    *out_x = px0 + qx * (-px0 + px1) + qy * (-px0 + px2) + qz * (-px0 + px3);
    *out_y = py0 + qx * (-py0 + py1) + qy * (-py0 + py2) + qz * (-py0 + py3);
    *out_z = pz0 + qx * (-pz0 + pz1) + qy * (-pz0 + pz2) + qz * (-pz0 + pz3);
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

SFEM_INLINE static void hex_aa_8_collect_coeffs(
    const ptrdiff_t* const SFEM_RESTRICT stride,
    const ptrdiff_t i,
    const ptrdiff_t j,
    const ptrdiff_t k,
    // Attention this is geometric data transformed to solver data!
    const real_t* const SFEM_RESTRICT data,
    real_t* const SFEM_RESTRICT out) {
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

int tet4_resample_field_local(
    // Mesh
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t** const SFEM_RESTRICT elems,
    geom_t** const SFEM_RESTRICT xyz,
    // SDF
    const ptrdiff_t* const SFEM_RESTRICT n,
    const ptrdiff_t* const SFEM_RESTRICT stride,
    const geom_t* const SFEM_RESTRICT origin,
    const geom_t* const SFEM_RESTRICT delta,
    const real_t* const SFEM_RESTRICT data,
    // Output
    real_t* const SFEM_RESTRICT weighted_field) {
    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[4];
            geom_t x[4], y[4], z[4];

            real_t hex8_f[8];
            real_t coeffs[8];

            real_t tet4_f[4];
            real_t element_field[4];

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

            for (int v = 0; v < 4; ++v) {
                x[v] = xyz[0][ev[v]];
                y[v] = xyz[1][ev[v]];
                z[v] = xyz[2][ev[v]];
            }

            memset(element_field, 0, 4 * sizeof(real_t));

            const real_t measure = tet4_measure(x[0],
                                                x[1],
                                                x[2],
                                                x[3],
                                                //
                                                y[0],
                                                y[1],
                                                y[2],
                                                y[3],
                                                //
                                                z[0],
                                                z[1],
                                                z[2],
                                                z[3]);

            assert(measure > 0);

            for (int q = 0; q < TET4_NQP; q++) {
                real_t g_qx, g_qy, g_qz;
                tet4_transform(x[0],
                               x[1],
                               x[2],
                               x[3],
                               //
                               y[0],
                               y[1],
                               y[2],
                               y[3],
                               //
                               z[0],
                               z[1],
                               z[2],
                               z[3],
                               //
                               tet4_qx[q],
                               tet4_qy[q],
                               tet4_qz[q],
                               &g_qx,
                               &g_qy,
                               &g_qz);

#ifndef SFEM_RESAMPLE_GAP_DUAL
                // Standard basis function
                {
                    tet4_f[0] = 1 - tet4_qx[q] - tet4_qy[q] - tet4_qz[q];
                    tet4_f[1] = tet4_qx[q];
                    tet4_f[2] = tet4_qy[q];
                    tet4_f[2] = tet4_qz[q];
                }
#else
                // DUAL basis function
                {
                    const real_t f0 = 1 - tet4_qx[q] - tet4_qy[q] - tet4_qz[q];
                    const real_t f1 = tet4_qx[q];
                    const real_t f2 = tet4_qy[q];
                    const real_t f3 = tet4_qz[q];

                    tet4_f[0] = 4 * f0 - f1 - f2 - f3;
                    tet4_f[1] = -f0 + 4 * f1 - f2 - f3;
                    tet4_f[2] = -f0 - f1 + 4 * f2 - f3;
                    tet4_f[3] = -f0 - f1 - f2 + 4 * f3;
                }
#endif
                const real_t dV = measure * tet4_qw[q];

                const real_t grid_x = (g_qx - ox) / dx;
                const real_t grid_y = (g_qy - oy) / dy;
                const real_t grid_z = (g_qz - oz) / dz;

                const ptrdiff_t i = floor(grid_x);
                const ptrdiff_t j = floor(grid_y);
                const ptrdiff_t k = floor(grid_z);

                // If outside
                if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) ||
                    (k + 1 >= n[2])) {
                    fprintf(
                        stderr,
                        "warning (%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, %ld)!\n",
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
                hex_aa_8_collect_coeffs(stride, i, j, k, data, coeffs);

                // Integrate gap function
                {
                    real_t eval_field = 0;
#pragma unroll(8)
                    for (int edof_j = 0; edof_j < 8; edof_j++) {
                        eval_field += hex8_f[edof_j] * coeffs[edof_j];
                    }

#pragma unroll(4)
                    for (int edof_i = 0; edof_i < 4; edof_i++) {
                        element_field[edof_i] += eval_field * tet4_f[edof_i] * dV;
                    }
                }
            }

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                // Invert sign since distance field is negative insdide and positive outside
#pragma omp critical
                { weighted_field[ev[v]] += element_field[v]; }
            }
        }
    }

    return 0;
}

int trishell3_resample_field_local(
    // Mesh
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t** const SFEM_RESTRICT elems,
    geom_t** const SFEM_RESTRICT xyz,
    // SDF
    const ptrdiff_t* const SFEM_RESTRICT n,
    const ptrdiff_t* const SFEM_RESTRICT stride,
    const geom_t* const SFEM_RESTRICT origin,
    const geom_t* const SFEM_RESTRICT delta,
    const real_t* const SFEM_RESTRICT data,
    // Output
    real_t* const SFEM_RESTRICT weighted_field) {
    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[3];
            geom_t x[3], y[3], z[3];

            real_t hex8_f[8];
            real_t coeffs[8];

            real_t tri3_f[3];
            real_t element_field[3];

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                ev[v] = elems[v][i];
            }

            for (int v = 0; v < 3; ++v) {
                x[v] = xyz[0][ev[v]];
                y[v] = xyz[1][ev[v]];
                z[v] = xyz[2][ev[v]];
            }

            memset(element_field, 0, 3 * sizeof(real_t));

            const real_t measure =
                tri_shell_3_measure(x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]);

            assert(measure > 0);

            for (int q = 0; q < TRI3_NQP; q++) {
                real_t g_qx, g_qy, g_qz;
                tri_shell_3_transform(x[0],
                                      x[1],
                                      x[2],
                                      y[0],
                                      y[1],
                                      y[2],
                                      z[0],
                                      z[1],
                                      z[2],
                                      tri3_qx[q],
                                      tri3_qy[q],
                                      &g_qx,
                                      &g_qy,
                                      &g_qz);

#ifndef SFEM_RESAMPLE_GAP_DUAL
                // Standard basis function
                {
                    tri3_f[0] = 1 - tri3_qx[q] - tri3_qy[q];
                    tri3_f[1] = tri3_qx[q];
                    tri3_f[2] = tri3_qy[q];
                }
#else
                // DUAL basis function
                {
                    const real_t f0 = 1 - tri3_qx[q] - tri3_qy[q];
                    const real_t f1 = tri3_qx[q];
                    const real_t f2 = tri3_qy[q];

                    tri3_f[0] = 3 * f0 - f1 - f2;
                    tri3_f[1] = -f0 + 3 * f1 - f2;
                    tri3_f[2] = -f0 - f1 + 3 * f2;
                }
#endif

                const real_t dV = measure * tri3_qw[q];

                const real_t grid_x = (g_qx - ox) / dx;
                const real_t grid_y = (g_qy - oy) / dy;
                const real_t grid_z = (g_qz - oz) / dz;

                const ptrdiff_t i = floor(grid_x);
                const ptrdiff_t j = floor(grid_y);
                const ptrdiff_t k = floor(grid_z);

                // If outside
                if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) ||
                    (k + 1 >= n[2])) {
                    fprintf(
                        stderr,
                        "warning (%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, %ld)!\n",
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
                hex_aa_8_collect_coeffs(stride, i, j, k, data, coeffs);

                // Integrate gap function
                {
                    real_t eval_field = 0;

#pragma unroll(8)
                    for (int edof_j = 0; edof_j < 8; edof_j++) {
                        eval_field += hex8_f[edof_j] * coeffs[edof_j];
                    }

#pragma unroll(3)
                    for (int edof_i = 0; edof_i < 3; edof_i++) {
                        element_field[edof_i] += eval_field * tri3_f[edof_i] * dV;
                    }
                }
            }

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                // Invert sign since distance field is negative insdide and positive outside
#pragma omp critical
                { weighted_field[ev[v]] += element_field[v]; }
            }
        }
    }

    return 0;
}

int beam2_resample_field_local(const ptrdiff_t nelements,
                               const ptrdiff_t nnodes,
                               idx_t** const SFEM_RESTRICT elems,
                               geom_t** const SFEM_RESTRICT xyz,
                               // SDF
                               const ptrdiff_t* const SFEM_RESTRICT n,
                               const ptrdiff_t* const SFEM_RESTRICT stride,
                               const geom_t* const SFEM_RESTRICT origin,
                               const geom_t* const SFEM_RESTRICT delta,
                               const real_t* const SFEM_RESTRICT data,
                               // Output
                               real_t* const SFEM_RESTRICT weighted_field) {
    printf("beam2_resample_field_local!\n");

    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[3];
            geom_t x[3], y[3], z[3];

            real_t hex8_f[8];
            real_t coeffs[8];

            real_t beam2_f[2];
            real_t element_field[2];

#pragma unroll(2)
            for (int v = 0; v < 2; ++v) {
                ev[v] = elems[v][i];
            }

            for (int v = 0; v < 2; ++v) {
                x[v] = xyz[0][ev[v]];
                y[v] = xyz[1][ev[v]];
                z[v] = xyz[2][ev[v]];
            }

            memset(element_field, 0, 2 * sizeof(real_t));

            const real_t measure = beam2_measure(x[0], x[1], y[0], y[1], z[0], z[1]);

            assert(measure > 0);

            for (int q = 0; q < EDGE2_NQP; q++) {
                real_t g_qx, g_qy, g_qz;
                beam2_transform(
                    x[0], x[1], y[0], y[1], z[0], z[1], edge2_qx[q], &g_qx, &g_qy, &g_qz);

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
                    beam2_f[0] = 2 * f0 - f1;
                    beam2_f[1] = -f0 + 2 * f1;
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
                if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) ||
                    (k + 1 >= n[2])) {
                    fprintf(
                        stderr,
                        "warning (%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, %ld)!\n",
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
                hex_aa_8_collect_coeffs(stride, i, j, k, data, coeffs);

                // Integrate gap function
                {
                    real_t eval_field = 0;

#pragma unroll(8)
                    for (int edof_j = 0; edof_j < 8; edof_j++) {
                        eval_field += hex8_f[edof_j] * coeffs[edof_j];
                    }

#pragma unroll(2)
                    for (int edof_i = 0; edof_i < 2; edof_i++) {
                        element_field[edof_i] += eval_field * beam2_f[edof_i] * dV;
                    }
                }
            }

#pragma unroll(2)
            for (int v = 0; v < 2; ++v) {
                // Invert sign since distance field is negative insdide and positive outside
#pragma omp critical
                { weighted_field[ev[v]] += element_field[v]; }
            }
        }
    }

    return 0;
}

int resample_field_local(
    // Mesh
    const enum ElemType element_type,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t** const SFEM_RESTRICT elems,
    geom_t** const SFEM_RESTRICT xyz,
    // SDF
    const ptrdiff_t* const SFEM_RESTRICT n,
    const ptrdiff_t* const SFEM_RESTRICT stride,
    const geom_t* const SFEM_RESTRICT origin,
    const geom_t* const SFEM_RESTRICT delta,
    const real_t* const SFEM_RESTRICT data,
    // Output
    real_t* const SFEM_RESTRICT weighted_field) {
    switch (element_type) {
        case TET4: {
            return tet4_resample_field_local(
                nelements, nnodes, elems, xyz, n, stride, origin, delta, data, weighted_field);
        }
        default:
            break;
    }

    enum ElemType st = shell_type(element_type);

    switch (st) {
        case TRISHELL3:
            return trishell3_resample_field_local(
                nelements, nnodes, elems, xyz, n, stride, origin, delta, data, weighted_field);
        case BEAM2:
            return beam2_resample_field_local(
                nelements, nnodes, elems, xyz, n, stride, origin, delta, data, weighted_field);

        default: {
            assert(0);
            fprintf(stderr, "Unknown element type %d\n", st);
            MPI_Abort(MPI_COMM_WORLD, -1);
            return EXIT_FAILURE;
        }
    }
}

int resample_field(
    // Mesh
    const enum ElemType element_type,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t** const SFEM_RESTRICT elems,
    geom_t** const SFEM_RESTRICT xyz,
    // SDF
    const ptrdiff_t* const SFEM_RESTRICT n,
    const ptrdiff_t* const SFEM_RESTRICT stride,
    const geom_t* const SFEM_RESTRICT origin,
    const geom_t* const SFEM_RESTRICT delta,
    const real_t* const SFEM_RESTRICT data,
    // Output
    real_t* const SFEM_RESTRICT g) {
    real_t* weighted_field = calloc(nnodes, sizeof(real_t));

    resample_field_local(element_type,
                         nelements,
                         nnodes,
                         elems,
                         xyz,
                         n,
                         stride,
                         origin,
                         delta,
                         data,
                         weighted_field);

    enum ElemType st = shell_type(element_type);

    if (INVALID == st) {
        // Removing the mass-contributions from the weighted gap function "weighted_field"
        apply_inv_lumped_mass(element_type, nelements, nnodes, elems, xyz, weighted_field, g);
    } else {
        apply_inv_lumped_mass(st, nelements, nnodes, elems, xyz, weighted_field, g);
    }

    free(weighted_field);
    return 0;
}

int interpolate_field(const ptrdiff_t nnodes,
                      geom_t** const SFEM_RESTRICT xyz,
                      // SDF
                      const ptrdiff_t* const SFEM_RESTRICT n,
                      const ptrdiff_t* const SFEM_RESTRICT stride,
                      const geom_t* const SFEM_RESTRICT origin,
                      const geom_t* const SFEM_RESTRICT delta,
                      const real_t* const SFEM_RESTRICT data,
                      // Output
                      real_t* const SFEM_RESTRICT g) {
    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            real_t hex8_f[8];
            real_t hex8_grad_x[8];
            real_t hex8_grad_y[8];
            real_t hex8_grad_z[8];
            real_t coeffs[8];

            const real_t x = xyz[0][node];
            const real_t y = xyz[1][node];
            const real_t z = xyz[2][node];

            const real_t grid_x = (x - ox) / dx;
            const real_t grid_y = (y - oy) / dy;
            const real_t grid_z = (z - oz) / dz;

            const ptrdiff_t i = floor(grid_x);
            const ptrdiff_t j = floor(grid_y);
            const ptrdiff_t k = floor(grid_z);

            // If outside
            if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) || (k + 1 >= n[2])) {
                int rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                fprintf(
                    stderr,
                    "[%d] warning (%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, %ld)!\n",
                    rank,
                    x,
                    y,
                    z,
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
            hex_aa_8_collect_coeffs(stride, i, j, k, data, coeffs);

            // Interpolate gap function
            {
                real_t eval_field = 0;

#pragma unroll(8)
                for (int edof_j = 0; edof_j < 8; edof_j++) {
                    eval_field += hex8_f[edof_j] * coeffs[edof_j];
                }

                g[node] = eval_field;
            }
        }
    }

    return 0;
}

SFEM_INLINE static void minmax(const ptrdiff_t n,
                               const geom_t* const SFEM_RESTRICT x,
                               geom_t* xmin,
                               geom_t* xmax) {
    *xmin = x[0];
    *xmax = x[0];
    for (ptrdiff_t i = 1; i < n; i++) {
        *xmin = MIN(*xmin, x[i]);
        *xmax = MAX(*xmax, x[i]);
    }
}

int field_view(MPI_Comm comm,
               const ptrdiff_t nnodes,
               const geom_t* SFEM_RESTRICT z_coordinate,
               const ptrdiff_t* const nlocal,
               const ptrdiff_t* const SFEM_RESTRICT nglobal,
               const ptrdiff_t* const SFEM_RESTRICT stride,
               const geom_t* const origin,
               const geom_t* const SFEM_RESTRICT delta,
               const real_t* const field,
               real_t** field_out,
               ptrdiff_t* z_nlocal_out,
               geom_t* const SFEM_RESTRICT z_origin_out) {
    return field_view_ensure_margin(comm,
                                    nnodes,
                                    z_coordinate,
                                    nlocal,
                                    nglobal,
                                    stride,
                                    origin,
                                    delta,
                                    field,
                                    0,
                                    field_out,
                                    z_nlocal_out,
                                    z_origin_out);
}

int field_view_ensure_margin(MPI_Comm comm,
                             const ptrdiff_t nnodes,
                             const geom_t* SFEM_RESTRICT z_coordinate,
                             const ptrdiff_t* const nlocal,
                             const ptrdiff_t* const SFEM_RESTRICT nglobal,
                             const ptrdiff_t* const SFEM_RESTRICT stride,
                             const geom_t* const origin,
                             const geom_t* const SFEM_RESTRICT delta,
                             const real_t* const field,
                             const ptrdiff_t z_margin,
                             real_t** field_out,
                             ptrdiff_t* z_nlocal_out,
                             geom_t* const SFEM_RESTRICT z_origin_out) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size == 1) {
        if (!rank) {
            fprintf(stderr, "[%d] resample_grid_view cannot be used in serial runs!\n", rank);
        }

        MPI_Abort(comm, -1);
        return 1;
    }

    double field_view_tick = MPI_Wtime();

    geom_t zmin, zmax;
    minmax(nnodes, z_coordinate, &zmin, &zmax);

    // Z is distributed
    ptrdiff_t zoffset = 0;
    MPI_Exscan(&nlocal[2], &zoffset, 1, MPI_LONG, MPI_SUM, comm);

    // // Compute Local z-tile
    ptrdiff_t field_start = (zmin - origin[2]) / delta[2];
    ptrdiff_t field_end = (zmax - origin[2]) / delta[2];

    // Make sure we are inside the grid and get also the required margin for resampling
    field_start = MAX(0, field_start - 1 - z_margin);
    field_end =
        MIN(nglobal[2],
            field_end + 2 + z_margin);  // 1 for the rightside of the cell 1 for the exclusive range

    ptrdiff_t pnlocal_z = (field_end - field_start);
    real_t* pfield = malloc(pnlocal_z * stride[2] * sizeof(real_t));

    array_range_select(comm,
                       SFEM_MPI_REAL_T,
                       (void*)field,
                       (void*)pfield,
                       // Size of z-slice
                       nlocal[2] * stride[2],
                       // starting offset
                       field_start * stride[2],
                       // ending offset
                       field_end * stride[2]);

    *field_out = pfield;
    *z_nlocal_out = pnlocal_z;
    *z_origin_out = origin[2] + field_start * delta[2];

    double field_view_tock = MPI_Wtime();

    if (!rank) {
        printf("[%d] field_view %g (seconds)\n", rank, field_view_tock - field_view_tick);
    }

    return 0;
}
