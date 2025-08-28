#include <cuda_runtime.h>
#include <stddef.h>

template <typename FloatType, typename FloatType3>
__device__ FloatType3 make_FloatType3(FloatType x, FloatType y, FloatType z) {
    FloatType3 t;
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
}

template <>
__device__ double3 make_FloatType3<double, double3>(double x, double y, double z) {
    return make_double3(x, y, z);
}

template <>
__device__ float3 make_FloatType3<float, float3>(float x, float y, float z) {
    return make_float3(x, y, z);
}

template <typename FloatType, typename FloatType3>
__device__ bool get_category_Jacobian(const unsigned int category, const FloatType L, FloatType3 *Jacobian_c) {
    const FloatType invL = FloatType(1.0) / L;
    const FloatType zero = FloatType(0.0);

    switch (category) {
        case 0:
            Jacobian_c[0] = make_FloatType3<FloatType, FloatType3>(invL, zero, zero);
            Jacobian_c[1] = make_FloatType3<FloatType, FloatType3>(zero, invL, zero);
            Jacobian_c[2] = make_FloatType3<FloatType, FloatType3>(zero, zero, invL);
            break;

        case 1:
            Jacobian_c[0] = make_FloatType3<FloatType, FloatType3>(zero, -invL, -invL);
            Jacobian_c[1] = make_FloatType3<FloatType, FloatType3>(zero, invL, zero);
            Jacobian_c[2] = make_FloatType3<FloatType, FloatType3>(invL, invL, invL);
            break;

        case 2:
            Jacobian_c[0] = make_FloatType3<FloatType, FloatType3>(-invL, zero, zero);
            Jacobian_c[1] = make_FloatType3<FloatType, FloatType3>(invL, zero, invL);
            Jacobian_c[2] = make_FloatType3<FloatType, FloatType3>(invL, invL, zero);
            break;

        case 3:
            Jacobian_c[0] = make_FloatType3<FloatType, FloatType3>(-invL, -invL, -invL);
            Jacobian_c[1] = make_FloatType3<FloatType, FloatType3>(zero, invL, invL);
            Jacobian_c[2] = make_FloatType3<FloatType, FloatType3>(invL, invL, zero);
            break;

        case 4:
            Jacobian_c[0] = make_FloatType3<FloatType, FloatType3>(-invL, -invL, zero);
            Jacobian_c[1] = make_FloatType3<FloatType, FloatType3>(invL, invL, invL);
            Jacobian_c[2] = make_FloatType3<FloatType, FloatType3>(zero, invL, zero);
            break;

        case 5:
            Jacobian_c[0] = make_FloatType3<FloatType, FloatType3>(zero, zero, -invL);
            Jacobian_c[1] = make_FloatType3<FloatType, FloatType3>(zero, -invL, zero);
            Jacobian_c[2] = make_FloatType3<FloatType, FloatType3>(invL, invL, invL);
            break;

        default:
            __trap();
            return false;
            break;
    }

    return true;
}

template <typename FloatType, typename FloatType3>
__device__ FloatType                                                                 //
tet4_resample_tetrahedron_local_adjoint_category(const unsigned int     category,    //
                                                 const unsigned int     L,           // Refinement level
                                                 const FloatType3       bc,          // Fixed double const
                                                 const FloatType3       J_phys[9],   // Jacobian matrix
                                                 const FloatType3       J_ref[9],    // Jacobian matrix
                                                 const FloatType3       det_J_phys,  // Determinant of the Jacobian matrix
                                                 const FloatType3       fxyz,        // Tetrahedron vertices X-coordinates
                                                 const FloatType        wf0,         // Weighted field at the vertices
                                                 const FloatType        wf1,         //
                                                 const FloatType        wf2,         //
                                                 const FloatType        wf3,         //
                                                 const FloatType        ox,          // Origin of the grid
                                                 const FloatType        oy,          //
                                                 const FloatType        oz,          //
                                                 const FloatType        dx,          // Spacing of the grid
                                                 const FloatType        dy,          //
                                                 const FloatType        dz,          //
                                                 const ptrdiff_t *const stride,      // Stride
                                                 const ptrdiff_t *const n,           // Size of the grid
                                                 FloatType *const       data) {            // Output

    const FloatType N_micro_tet     = (FloatType)(L) * (FloatType)(L) * (FloatType)(L);
    const FloatType inv_N_micro_tet = 1.0 / N_micro_tet;  // Inverse of the number of micro-tetrahedra

    const FloatType theta_volume = det_J_phys / ((FloatType)(6.0));  // Volume of the mini-tetrahedron in the physical space

    FloatType cumulated_dV = 0.0;

    return cumulated_dV;
}

template <typename FloatType, typename FloatType3>
__device__ void main_tet_loop(const int L) {
    const FloatType zero = 0.0;

    int Ik = 0;

    FloatType3 Jacobian_c[6][3];
    FloatType3 bc = make_FloatType3<FloatType, FloatType3>(zero, zero, zero);  // Origin of the tetrahedron
    FloatType  h  = 1.0 / FloatType(L);

    for (int c = 0; c < 6; ++c) {
        bool status = get_category_Jacobian<FloatType, FloatType3>(c, FloatType(L), Jacobian_c[c]);
        if (!status) {
            // Handle error: invalid category
            // For example, you might want to set a default value or log an error
        }
    }

    for (int k = 0; k <= L; ++k) {  // Loop over z

        const int nodes_per_side  = (L - k) + 1;
        const int nodes_per_layer = nodes_per_side * (nodes_per_side + 1) / 2;
        // Removed unused variable Ns
        const int Nl = nodes_per_layer;

        // Layer loop info
        // printf("Layer %d: Ik = %d, Ns = %d, Nl = %d\n", k, Ik, Ns, Nl);

        for (int j = 0; j < nodes_per_side - 1; ++j) {          // Loop over y
            for (int i = 0; i < nodes_per_side - 1 - j; ++i) {  // Loop over x

                bc = make_FloatType3<FloatType, FloatType3>(FloatType(i) * h, FloatType(j) * h, FloatType(k) * h);

                // Category 0
                // ... category 0 logic here ...

                if (i >= 1) {
                    // Category 1
                    // ... category 1 logic here ...

                    // Category 2
                    // ... category 2 logic here ...

                    // Category 3
                    // ... category 3 logic here ...

                    // Category 4
                    // ... category 4 logic here ...
                }

                if (j >= 1 && i >= 1) {
                    // Category 5
                    // ... category 5 logic here ...
                }
            }
        }
        Ik = Ik + Nl;
    }
}

__global__ void sfem_adjoint_mini_tet_kernel(const int L) { main_tet_loop<double, double3>(L); }

#define __TESTING__
#ifdef __TESTING__

int main() {
    const int L = 4;  // Example refinement level
    sfem_adjoint_mini_tet_kernel<<<1, 1>>>(L);
    cudaDeviceSynchronize();
    return 0;
}

#endif