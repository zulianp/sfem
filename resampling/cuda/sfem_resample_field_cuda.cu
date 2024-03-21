

#include <stdio.h>

#define real_t double
#define real_type real_t

#define MY_RESTRICT __restrict__

#define SFEM_USE_RULE56
#ifdef SFEM_USE_RULE56
#define TET4_NQP 56
static __device__ real_t tet4_qw[TET4_NQP] = {
        0.0010373112336140, 0.0010373112336140, 0.0010373112336140, 0.0010373112336140,
        0.0096016645399480, 0.0096016645399480, 0.0096016645399480, 0.0096016645399480,
        0.0096016645399480, 0.0096016645399480, 0.0096016645399480, 0.0096016645399480,
        0.0096016645399480, 0.0096016645399480, 0.0096016645399480, 0.0096016645399480,
        0.0164493976798232, 0.0164493976798232, 0.0164493976798232, 0.0164493976798232,
        0.0164493976798232, 0.0164493976798232, 0.0164493976798232, 0.0164493976798232,
        0.0164493976798232, 0.0164493976798232, 0.0164493976798232, 0.0164493976798232,
        0.0153747766513310, 0.0153747766513310, 0.0153747766513310, 0.0153747766513310,
        0.0153747766513310, 0.0153747766513310, 0.0153747766513310, 0.0153747766513310,
        0.0153747766513310, 0.0153747766513310, 0.0153747766513310, 0.0153747766513310,
        0.0293520118375230, 0.0293520118375230, 0.0293520118375230, 0.0293520118375230,
        0.0293520118375230, 0.0293520118375230, 0.0293520118375230, 0.0293520118375230,
        0.0293520118375230, 0.0293520118375230, 0.0293520118375230, 0.0293520118375230,
        0.0366291366405108, 0.0366291366405108, 0.0366291366405108, 0.0366291366405108};

static __device__ real_t tet4_qx[TET4_NQP] = {
        0.0149520651530592, 0.9551438045408220, 0.0149520651530592, 0.0149520651530592,
        0.1518319491659370, 0.7799760084415400, 0.0340960211962615, 0.0340960211962615,
        0.0340960211962615, 0.0340960211962615, 0.7799760084415400, 0.1518319491659370,
        0.7799760084415400, 0.1518319491659370, 0.0340960211962615, 0.0340960211962615,
        0.5526556431060171, 0.3549340560639790, 0.0462051504150017, 0.0462051504150017,
        0.0462051504150017, 0.0462051504150017, 0.3549340560639790, 0.5526556431060171,
        0.3549340560639790, 0.5526556431060171, 0.0462051504150017, 0.0462051504150017,
        0.2281904610687610, 0.5381043228880020, 0.2281904610687610, 0.2281904610687610,
        0.5381043228880020, 0.2281904610687610, 0.0055147549744775, 0.0055147549744775,
        0.0055147549744775, 0.5381043228880020, 0.2281904610687610, 0.2281904610687610,
        0.3523052600879940, 0.1961837595745600, 0.3523052600879940, 0.3523052600879940,
        0.1961837595745600, 0.3523052600879940, 0.0992057202494530, 0.0992057202494530,
        0.0992057202494530, 0.1961837595745600, 0.3523052600879940, 0.3523052600879940,
        0.1344783347929940, 0.5965649956210171, 0.1344783347929940, 0.1344783347929940};

static __device__ real_t tet4_qy[TET4_NQP] = {
        0.0149520651530592, 0.0149520651530592, 0.9551438045408220, 0.0149520651530592,
        0.0340960211962615, 0.0340960211962615, 0.1518319491659370, 0.7799760084415400,
        0.0340960211962615, 0.0340960211962615, 0.1518319491659370, 0.7799760084415400,
        0.0340960211962615, 0.0340960211962615, 0.7799760084415400, 0.1518319491659370,
        0.0462051504150017, 0.0462051504150017, 0.5526556431060171, 0.3549340560639790,
        0.0462051504150017, 0.0462051504150017, 0.5526556431060171, 0.3549340560639790,
        0.0462051504150017, 0.0462051504150017, 0.3549340560639790, 0.5526556431060171,
        0.2281904610687610, 0.2281904610687610, 0.5381043228880020, 0.0055147549744775,
        0.0055147549744775, 0.0055147549744775, 0.2281904610687610, 0.5381043228880020,
        0.2281904610687610, 0.2281904610687610, 0.5381043228880020, 0.2281904610687610,
        0.3523052600879940, 0.3523052600879940, 0.1961837595745600, 0.0992057202494530,
        0.0992057202494530, 0.0992057202494530, 0.3523052600879940, 0.1961837595745600,
        0.3523052600879940, 0.3523052600879940, 0.1961837595745600, 0.3523052600879940,
        0.1344783347929940, 0.1344783347929940, 0.5965649956210171, 0.1344783347929940};

static __device__ real_t tet4_qz[TET4_NQP] = {
        0.0149520651530592, 0.0149520651530592, 0.0149520651530592, 0.9551438045408220,
        0.0340960211962615, 0.0340960211962615, 0.0340960211962615, 0.0340960211962615,
        0.1518319491659370, 0.7799760084415400, 0.0340960211962615, 0.0340960211962615,
        0.1518319491659370, 0.7799760084415400, 0.1518319491659370, 0.7799760084415400,
        0.0462051504150017, 0.0462051504150017, 0.0462051504150017, 0.0462051504150017,
        0.5526556431060171, 0.3549340560639790, 0.0462051504150017, 0.0462051504150017,
        0.5526556431060171, 0.3549340560639790, 0.5526556431060171, 0.3549340560639790,
        0.0055147549744775, 0.0055147549744775, 0.0055147549744775, 0.2281904610687610,
        0.2281904610687610, 0.5381043228880020, 0.2281904610687610, 0.2281904610687610,
        0.5381043228880020, 0.2281904610687610, 0.2281904610687610, 0.5381043228880020,
        0.0992057202494530, 0.0992057202494530, 0.0992057202494530, 0.3523052600879940,
        0.3523052600879940, 0.1961837595745600, 0.3523052600879940, 0.3523052600879940,
        0.1961837595745600, 0.3523052600879940, 0.3523052600879940, 0.1961837595745600,
        0.1344783347929940, 0.1344783347929940, 0.1344783347929940, 0.5965649956210171};

#endif  // SFEM_USE_RULE56

////////////////////////////////////////////////////////
// tet4_transform_v2
////////////////////////////////////////////////////////
__device__ void tet4_transform_cu(
        /**
         ****************************************************************************************
        \begin{bmatrix}
        out_x \\
        out_y \\
        out_z
        \end{bmatrix}
        =
        \begin{bmatrix}
        px_0 \\
        py_0 \\
        pz_0
        \end{bmatrix}
        +
        \begin{bmatrix}
        px_1 - px_0 & px_2 - px_0 & px_3 - px_0 \\
        py_1 - py_0 & py_2 - py_0 & py_3 - py_0 \\
        pz_1 - pz_0 & pz_2 - pz_0 & pz_3 - pz_0
        \end{bmatrix}
        \cdot
        \begin{bmatrix}
        qx \\
        qy \\
        qz
        \end{bmatrix}
        *************************************************************************************************
      */

        // X-coordinates
        const real_type px0,
        const real_type px1,
        const real_type px2,
        const real_type px3,
        // Y-coordinates
        const real_type py0,
        const real_type py1,
        const real_type py2,
        const real_type py3,
        // Z-coordinates
        const real_type pz0,
        const real_type pz1,
        const real_type pz2,
        const real_type pz3,
        // Quadrature point
        const real_type qx,
        const real_type qy,
        const real_type qz,
        // Output
        real_type* const out_x,
        real_type* const out_y,
        real_type* const out_z) {
    //
    //
    *out_x = px0 + qx * (-px0 + px1) + qy * (-px0 + px2) + qz * (-px0 + px3);
    *out_y = py0 + qx * (-py0 + py1) + qy * (-py0 + py2) + qz * (-py0 + py3);
    *out_z = pz0 + qx * (-pz0 + pz1) + qy * (-pz0 + pz2) + qz * (-pz0 + pz3);
}

////////////////////////////////////////////////////////
// tet4_measure_v2
////////////////////////////////////////////////////////
__device__ real_type tet4_measure_cu(
        // X-coordinates
        const real_type px0,
        const real_type px1,
        const real_type px2,
        const real_type px3,
        // Y-coordinates
        const real_type py0,
        const real_type py1,
        const real_type py2,
        const real_type py3,
        // Z-coordinates
        const real_type pz0,
        const real_type pz1,
        const real_type pz2,
        const real_type pz3) {
    //
    // determinant of the Jacobian
    // M = [px0, py0, pz0, 1]
    //     [px1, py1, pz1, 1]
    //     [px2, py2, pz2, 1]
    //     [px3, py3, pz3, 1]
    //
    // V = (1/6) * det(M)

    const real_type x0 = -pz0 + pz3;
    const real_type x1 = -py0 + py2;
    const real_type x2 = -(1.0 / 6.0) * px0 + (1.0 / 6.0) * px1;
    const real_type x3 = -py0 + py3;
    const real_type x4 = -pz0 + pz2;
    const real_type x5 = -py0 + py1;
    const real_type x6 = -(1.0 / 6.0) * px0 + (1.0 / 6.0) * px2;
    const real_type x7 = -pz0 + pz1;
    const real_type x8 = -(1.0 / 6.0) * px0 + (1.0 / 6.0) * px3;

    return x0 * x1 * x2 - x0 * x5 * x6 - x1 * x7 * x8 - x2 * x3 * x4 + x3 * x6 * x7 + x4 * x5 * x8;
}

////////////////////////////////////////////////////////
// hex_aa_8_eval_fun_V
////////////////////////////////////////////////////////
__device__ void hex_aa_8_eval_fun_cu(
        // Quadrature point (local coordinates)
        // With respect to the hat functions of a cube element
        // In a local coordinate system
        const real_t x,
        const real_t y,
        const real_t z,

        // Output
        real_t* const MY_RESTRICT f0,
        real_t* const MY_RESTRICT f1,
        real_t* const MY_RESTRICT f2,
        real_t* const MY_RESTRICT f3,
        real_t* const MY_RESTRICT f4,
        real_t* const MY_RESTRICT f5,
        real_t* const MY_RESTRICT f6,
        real_t* const MY_RESTRICT f7) {
    //
    *f0 = (1.0 - x) * (1.0 - y) * (1.0 - z);
    *f1 = x * (1.0 - y) * (1.0 - z);
    *f2 = x * y * (1.0 - z);
    *f3 = (1.0 - x) * y * (1.0 - z);
    *f4 = (1.0 - x) * (1.0 - y) * z;
    *f5 = x * (1.0 - y) * z;
    *f6 = x * y * z;
    *f7 = (1.0 - x) * y * z;
}

////////////////////////////////////////////////////////
// hex_aa_8_collect_coeffs_cu
////////////////////////////////////////////////////////
__device__ void hex_aa_8_collect_coeffs_cu(
        //
        const ptrdiff_t* const MY_RESTRICT stride,
        const ptrdiff_t i,
        const ptrdiff_t j,
        const ptrdiff_t k,
        // Attention this is geometric data transformed to solver data!
        const real_t* const MY_RESTRICT data,
        //
        real_t* MY_RESTRICT out0,
        real_t* MY_RESTRICT out1,
        real_t* MY_RESTRICT out2,
        real_t* MY_RESTRICT out3,
        real_t* MY_RESTRICT out4,
        real_t* MY_RESTRICT out5,
        real_t* MY_RESTRICT out6,
        real_t* MY_RESTRICT out7) {
    //
    const ptrdiff_t i0 = i * stride[0] + j * stride[1] + k * stride[2];
    const ptrdiff_t i1 = (i + 1) * stride[0] + j * stride[1] + k * stride[2];
    const ptrdiff_t i2 = (i + 1) * stride[0] + (j + 1) * stride[1] + k * stride[2];
    const ptrdiff_t i3 = i * stride[0] + (j + 1) * stride[1] + k * stride[2];
    const ptrdiff_t i4 = i * stride[0] + j * stride[1] + (k + 1) * stride[2];
    const ptrdiff_t i5 = (i + 1) * stride[0] + j * stride[1] + (k + 1) * stride[2];
    const ptrdiff_t i6 = (i + 1) * stride[0] + (j + 1) * stride[1] + (k + 1) * stride[2];
    const ptrdiff_t i7 = i * stride[0] + (j + 1) * stride[1] + (k + 1) * stride[2];

    *out0 = data[i0];
    *out1 = data[i1];
    *out2 = data[i2];
    *out3 = data[i3];
    *out4 = data[i4];
    *out5 = data[i5];
    *out6 = data[i6];
    *out7 = data[i7];
}

////////////////////////////////////////////////////////
// hex_aa_8_collect_coeffs_cu
////////////////////////////////////////////////////////
__device__ void hex_aa_8_collect_coeffs_cu_v2(
        //
        const ptrdiff_t MY_RESTRICT stride0,
        const ptrdiff_t MY_RESTRICT stride1,
        const ptrdiff_t MY_RESTRICT stride2,

        const ptrdiff_t i,
        const ptrdiff_t j,
        const ptrdiff_t k,
        // Attention this is geometric data transformed to solver data!
        const real_t* const MY_RESTRICT data,
        //
        real_t* MY_RESTRICT out0,
        real_t* MY_RESTRICT out1,
        real_t* MY_RESTRICT out2,
        real_t* MY_RESTRICT out3,
        real_t* MY_RESTRICT out4,
        real_t* MY_RESTRICT out5,
        real_t* MY_RESTRICT out6,
        real_t* MY_RESTRICT out7) {
    //
    const ptrdiff_t i0 = i * stride0 + j * stride1 + k * stride2;
    const ptrdiff_t i1 = (i + 1) * stride0 + j * stride1 + k * stride2;
    const ptrdiff_t i2 = (i + 1) * stride0 + (j + 1) * stride1 + k * stride2;
    const ptrdiff_t i3 = i * stride0 + (j + 1) * stride1 + k * stride2;
    const ptrdiff_t i4 = i * stride0 + j * stride1 + (k + 1) * stride2;
    const ptrdiff_t i5 = (i + 1) * stride0 + j * stride1 + (k + 1) * stride2;
    const ptrdiff_t i6 = (i + 1) * stride0 + (j + 1) * stride1 + (k + 1) * stride2;
    const ptrdiff_t i7 = i * stride0 + (j + 1) * stride1 + (k + 1) * stride2;

    *out0 = data[i0];
    *out1 = data[i1];
    *out2 = data[i2];
    *out3 = data[i3];
    *out4 = data[i4];
    *out5 = data[i5];
    *out6 = data[i6];
    *out7 = data[i7];
}

// Struct for the elements
typedef struct {
    int* elems_v0;
    int* elems_v1;
    int* elems_v2;
    int* elems_v3;
} elems_tet4_device;

void                                                              //
cuda_allocate_elems_tet4_device(elems_tet4_device* elems_device,  //
                                const ptrdiff_t nelements) {      //
    cudaMalloc((void**)&elems_device->elems_v0, nelements * sizeof(int));
    cudaMalloc((void**)&elems_device->elems_v1, nelements * sizeof(int));
    cudaMalloc((void**)&elems_device->elems_v2, nelements * sizeof(int));
    cudaMalloc((void**)&elems_device->elems_v3, nelements * sizeof(int));
}

void free_elems_tet4_device(elems_tet4_device* elems_device) {
    cudaFree(elems_device->elems_v0);
    cudaFree(elems_device->elems_v1);
    cudaFree(elems_device->elems_v2);
    cudaFree(elems_device->elems_v3);
}

// Struct for xyz
typedef struct {
    float* x;
    float* y;
    float* z;
} xyz_tet4_device;

void cuda_allocate_xyz_tet4_device(xyz_tet4_device* xyz_device, const ptrdiff_t nnodes) {
    cudaMalloc((void**)&xyz_device->x, nnodes * sizeof(float));
    cudaMalloc((void**)&xyz_device->y, nnodes * sizeof(float));
    cudaMalloc((void**)&xyz_device->z, nnodes * sizeof(float));
}

void free_xyz_tet4_device(xyz_tet4_device* xyz_device) {
    cudaFree(xyz_device->x);
    cudaFree(xyz_device->y);
    cudaFree(xyz_device->z);
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_kernel //////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__ void tet4_resample_field_local_kernel(
        // Mesh
        const ptrdiff_t start_element,
        const ptrdiff_t end_element,
        const ptrdiff_t nnodes,
        const elems_tet4_device MY_RESTRICT elems,
        const xyz_tet4_device MY_RESTRICT xyz,
        // SDF
        // const ptrdiff_t* const MY_RESTRICT n,
        const ptrdiff_t MY_RESTRICT stride0,
        const ptrdiff_t MY_RESTRICT stride1,
        const ptrdiff_t MY_RESTRICT stride2,

        const float* const MY_RESTRICT origin,
        const float* const MY_RESTRICT delta,
        const real_type* const MY_RESTRICT data,
        // Output
        real_type* const MY_RESTRICT weighted_field) {
    //
    // Thread index
    const ptrdiff_t element_i = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("element_i = %ld\n", element_i);

    if (element_i < start_element || element_i >= end_element) {
        return;
    }

    // weighted_field[element_i] = 11.0;

    ////////////////////////////////////////////////////

    const real_type ox = (real_type)origin[0];
    const real_type oy = (real_type)origin[1];
    const real_type oz = (real_type)origin[2];

    const real_type dx = (real_type)delta[0];
    const real_type dy = (real_type)delta[1];
    const real_type dz = (real_type)delta[2];

    ////////////////////////////////////////////////////

    real_type x0 = 0.0, x1 = 0.0, x2 = 0.0, x3 = 0.0;
    real_type y0 = 0.0, y1 = 0.0, y2 = 0.0, y3 = 0.0;
    real_type z0 = 0.0, z1 = 0.0, z2 = 0.0, z3 = 0.0;

    // real_type hex8_f[8];
    real_type hex8_f0 = 0.0, hex8_f1 = 0.0, hex8_f2 = 0.0, hex8_f3 = 0.0, hex8_f4 = 0.0,
              hex8_f5 = 0.0, hex8_f6 = 0.0, hex8_f7 = 0.0;

    // real_type coeffs[8];
    real_type coeffs0 = 0.0, coeffs1 = 0.0, coeffs2 = 0.0, coeffs3 = 0.0, coeffs4 = 0.0,
              coeffs5 = 0.0, coeffs6 = 0.0, coeffs7 = 0.0;

    // real_type tet4_f[4];
    real_type tet4_f0 = 0.0, tet4_f1 = 0.0, tet4_f2 = 0.0, tet4_f3 = 0.0;

    // real_type element_field[4];
    real_type element_field0 = 0.0, element_field1 = 0.0, element_field2 = 0.0,
              element_field3 = 0.0;

    // loop over the 4 vertices of the tetrahedron
    int ev[4];
    ev[0] = elems.elems_v0[element_i];
    ev[1] = elems.elems_v1[element_i];
    ev[2] = elems.elems_v2[element_i];
    ev[3] = elems.elems_v3[element_i];

    // for (int v = 0; v < 4; ++v) {
    //     ev[v] = elems[v][element_i];
    // }

    // copy the coordinates of the vertices
    // for (int v = 0; v < 4; ++v) {
    //     x[v] = xyz[0][ev[v]];  // x-coordinates
    //     y[v] = xyz[1][ev[v]];  // y-coordinates
    //     z[v] = xyz[2][ev[v]];  // z-coordinates
    // }
    {
        // x0 = xyz[0][ev[0]];
        // x1 = xyz[0][ev[1]];
        // x2 = xyz[0][ev[2]];
        // x3 = xyz[0][ev[3]];

        // y0 = xyz[1][ev[0]];
        // y1 = xyz[1][ev[1]];
        // y2 = xyz[1][ev[2]];
        // y3 = xyz[1][ev[3]];

        // z0 = xyz[2][ev[0]];
        // z1 = xyz[2][ev[1]];
        // z2 = xyz[2][ev[2]];
        // z3 = xyz[2][ev[3]];

        x0 = xyz.x[ev[0]];
        x1 = xyz.x[ev[1]];
        x2 = xyz.x[ev[2]];
        x3 = xyz.x[ev[3]];

        y0 = xyz.y[ev[0]];
        y1 = xyz.y[ev[1]];
        y2 = xyz.y[ev[2]];
        y3 = xyz.y[ev[3]];

        z0 = xyz.z[ev[0]];
        z1 = xyz.z[ev[1]];
        z2 = xyz.z[ev[2]];
        z3 = xyz.z[ev[3]];
    }

    // Volume of the tetrahedron
    const real_type theta_volume = tet4_measure_cu(x0,
                                                   x1,
                                                   x2,
                                                   x3,
                                                   //
                                                   y0,
                                                   y1,
                                                   y2,
                                                   y3,
                                                   //
                                                   z0,
                                                   z1,
                                                   z2,
                                                   z3);

    /////////////////////////////////////////////
    // loop over the quadrature points
    for (int quad_i = 0; quad_i < TET4_NQP; quad_i++) {  // loop over the quadrature points

        real_type g_qx, g_qy, g_qz;

        tet4_transform_cu(x0,
                          x1,
                          x2,
                          x3,

                          y0,
                          y1,
                          y2,
                          y3,

                          z0,
                          z1,
                          z2,
                          z3,

                          tet4_qx[quad_i],
                          tet4_qy[quad_i],
                          tet4_qz[quad_i],

                          &g_qx,
                          &g_qy,
                          &g_qz);

#ifdef SFEM_RESAMPLE_GAP_DUAL
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
            const real_type f0 = 1.0 - tet4_qx[quad_i] - tet4_qy[quad_i] - tet4_qz[quad_i];
            const real_type f1 = tet4_qx[quad_i];
            const real_type f2 = tet4_qy[quad_i];
            const real_type f3 = tet4_qz[quad_i];

            tet4_f0 = 4.0 * f0 - f1 - f2 - f3;
            tet4_f1 = -f0 + 4.0 * f1 - f2 - f3;
            tet4_f2 = -f0 - f1 + 4.0 * f2 - f3;
            tet4_f3 = -f0 - f1 - f2 + 4.0 * f3;
        }
#endif

        const real_type grid_x = (g_qx - ox) / dx;
        const real_type grid_y = (g_qy - oy) / dy;
        const real_type grid_z = (g_qz - oz) / dz;

        const ptrdiff_t i = floor(grid_x);
        const ptrdiff_t j = floor(grid_y);
        const ptrdiff_t k = floor(grid_z);

        // Get the reminder [0, 1]
        real_type l_x = (grid_x - (double)i);
        real_type l_y = (grid_y - (double)j);
        real_type l_z = (grid_z - (double)k);

        // Critical point
        hex_aa_8_eval_fun_cu(l_x,
                             l_y,
                             l_z,
                             &hex8_f0,
                             &hex8_f1,
                             &hex8_f2,
                             &hex8_f3,
                             &hex8_f4,
                             &hex8_f5,
                             &hex8_f6,
                             &hex8_f7);

        hex_aa_8_collect_coeffs_cu_v2(stride0,
                                      stride1,
                                      stride2,
                                      i,
                                      j,
                                      k,
                                      data,
                                      &coeffs0,
                                      &coeffs1,
                                      &coeffs2,
                                      &coeffs3,
                                      &coeffs4,
                                      &coeffs5,
                                      &coeffs6,
                                      &coeffs7);

        // Integrate gap function
        {
            real_type eval_field = 0.0;
            // UNROLL_ZERO
            // for (int edof_j = 0; edof_j < 8; edof_j++) {
            //     eval_field += hex8_f[edof_j] * coeffs[edof_j];
            // }
            eval_field += hex8_f0 * coeffs0;
            eval_field += hex8_f1 * coeffs1;
            eval_field += hex8_f2 * coeffs2;
            eval_field += hex8_f3 * coeffs3;
            eval_field += hex8_f4 * coeffs4;
            eval_field += hex8_f5 * coeffs5;
            eval_field += hex8_f6 * coeffs6;
            eval_field += hex8_f7 * coeffs7;

            // UNROLL_ZERO
            // for (int edof_i = 0; edof_i < 4; edof_i++) {
            //     element_field[edof_i] += eval_field * tet4_f[edof_i] * dV;
            // }  // end edof_i loop

            real_type dV = theta_volume * tet4_qw[quad_i];
            // dV = 1.0;

            element_field0 += eval_field * tet4_f0 * dV;
            element_field1 += eval_field * tet4_f1 * dV;
            element_field2 += eval_field * tet4_f2 * dV;
            element_field3 += eval_field * tet4_f3 * dV;

        }  // end integrate gap function

    }  // end loop over the quadrature points

    atomicAdd(&weighted_field[ev[0]], element_field0);
    atomicAdd(&weighted_field[ev[1]], element_field1);
    atomicAdd(&weighted_field[ev[2]], element_field2);
    atomicAdd(&weighted_field[ev[3]], element_field3);

}  // end kernel tet4_resample_field_local_CU

__global__ void mykernel() { printf("hello fron kernel\n"); }

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_v2 //////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
/*
nvcc --gpu-architecture=sm_75 --device-c sfem_resample_field_cuda.cu -o sfem_resample_field_cuda.o
nvcc -O3 --gpu-architecture=sm_75 --shared --compiler-options -fPIC -shared
sfem_resample_field_cuda.cu -o sfem_resample_field_cuda.so


nvcc -O3  --gpu-architecture=sm_75 --compiler-options -fPIC -shared sfem_resample_field_cuda.cu -o
sfem_resample_field_cuda.o ar -r sfem_resample_field_cuda.a sfem_resample_field_cuda.o

*/

extern "C" int tet4_resample_field_local_CUDA(
        // Mesh
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        int** const MY_RESTRICT elems,
        float** const MY_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const MY_RESTRICT n,
        const ptrdiff_t* const MY_RESTRICT stride,
        const float* const MY_RESTRICT origin,
        const float* const MY_RESTRICT delta,
        const real_type* const MY_RESTRICT data,
        // Output
        real_type* const MY_RESTRICT weighted_field) {
    //

    printf("=============================================\n");
    printf("nelements = %ld\n", nelements);
    printf("=============================================\n");

    // Allocate memory on the device

    // Allocate weighted_field on the device
    double* weighted_field_device;
    cudaMalloc((void**)&weighted_field_device, nnodes * sizeof(double));
    cudaMemset(weighted_field_device, 0, sizeof(double) * nnodes);

    // copy the elements to the device
    elems_tet4_device elems_device;
    cuda_allocate_elems_tet4_device(&elems_device, nelements);

    cudaMemcpy(elems_device.elems_v0, elems[0], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(elems_device.elems_v1, elems[1], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(elems_device.elems_v2, elems[2], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(elems_device.elems_v3, elems[3], nelements * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate xyz on the device
    xyz_tet4_device xyz_device;
    cudaMalloc((void**)&xyz_device, 3 * sizeof(float*));
    cuda_allocate_xyz_tet4_device(&xyz_device, nnodes);
    cudaMemcpy(xyz_device.x, xyz[0], nnodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xyz_device.y, xyz[1], nnodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xyz_device.z, xyz[2], nnodes * sizeof(float), cudaMemcpyHostToDevice);

    // ptrdiff_t* stride_device;
    // cudaMalloc((void**)&stride_device, 3 * sizeof(ptrdiff_t));
    // cudaMemcpy(stride_device, stride, 3 * sizeof(ptrdiff_t), cudaMemcpyHostToDevice);

    float* origin_device;
    cudaMalloc((void**)&origin_device, 3 * sizeof(float));
    cudaMemcpy(origin_device, origin, 3 * sizeof(float), cudaMemcpyHostToDevice);

    float* delta_device;
    cudaMalloc((void**)&delta_device, 3 * sizeof(float));
    cudaMemcpy(delta_device, delta, 3 * sizeof(float), cudaMemcpyHostToDevice);

    double* data_device;
    const ptrdiff_t size_data = n[0] * n[1] * n[2];
    cudaMalloc((void**)&data_device, size_data * sizeof(double));
    cudaMemcpy(data_device, data, size_data * sizeof(double), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    ///////////////////////////////////////////////////////////////////////////////
    // Call the kernel
    cudaEvent_t start, stop;

    // Number of threads
    const ptrdiff_t threadsPerBlock = 128;

    // Number of blocks
    const ptrdiff_t numBlocks = (nelements + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("============================================================================\n");
    printf("GPU:    Launching the kernel\n");
    printf("GPU:    Number of blocks:            %ld\n", numBlocks);
    printf("GPU:    Number of threads per block: %ld\n", threadsPerBlock);
    printf("GPU:    Total number of threads:     %ld\n", (numBlocks * threadsPerBlock));
    printf("============================================================================\n");

    cudaEventRecord(start);

    tet4_resample_field_local_kernel<<<numBlocks, threadsPerBlock>>>(0,             //
                                                                     nelements,     //
                                                                     nnodes,        //
                                                                     elems_device,  //
                                                                     xyz_device,
                                                                     //  NULL,
                                                                     stride[0],
                                                                     stride[1],
                                                                     stride[2],
                                                                     origin_device,
                                                                     delta_device,
                                                                     data_device,
                                                                     weighted_field_device);

    // Stop the timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // get cuda error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(error));
    }

    // end kernel
    ///////////////////////////////////////////////////////////////////////////////

    double time = milliseconds / 1000.0;

    const double elements_second = (double)nelements / time;

    printf("============================================================================\n");
    printf("GPU:    Elapsed time: %e s\n", time);
    printf("GPU:    Throughput:   %e elements/second\n", elements_second);
    printf("============================================================================\n");

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Free memory on the device
    free_elems_tet4_device(&elems_device);

    free_xyz_tet4_device(&xyz_device);

    // cudaFree(stride_device);
    cudaFree(origin_device);
    cudaFree(delta_device);

    // Copy the result back to the host
    cudaMemcpy(weighted_field,           //
               weighted_field_device,    //
               nnodes * sizeof(double),  //
               cudaMemcpyDeviceToHost);  //

    cudaFree(weighted_field_device);

    cudaFree(data_device);

    return 0;
}