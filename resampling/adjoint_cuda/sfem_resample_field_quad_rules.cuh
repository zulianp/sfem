#ifndef __SFEM_RESAMPLE_FIELD_QUAD_RULES_CUH__
#define __SFEM_RESAMPLE_FIELD_QUAD_RULES_CUH__

// Gauss-Legendre Quadrature (N=1, D=1)
__constant__ double gl_quad_nodes_1_double[1]   = {0.5};
__constant__ double gl_quad_weights_1_double[1] = {1.0};

__constant__ float gl_quad_nodes_1_float[1]   = {0.5f};
__constant__ float gl_quad_weights_1_float[1] = {1.0f};

// Gauss-Legendre Quadrature (N=2, D=1)
__constant__ double gl_quad_nodes_2_double[2]   = {0.21132486540518708, 0.7886751345948129};
__constant__ double gl_quad_weights_2_double[2] = {0.5, 0.5};

__constant__ float gl_quad_nodes_2_float[2]   = {0.21132486540518708f, 0.7886751345948129f};
__constant__ float gl_quad_weights_2_float[2] = {0.5f, 0.5f};

// Gauss-Legendre Quadrature (N=3, D=1)
__constant__ double gl_quad_nodes_3_double[3]   = {0.1127016653792583, 0.5, 0.8872983346207417};
__constant__ double gl_quad_weights_3_double[3] = {0.2777777777777778, 0.4444444444444444, 0.2777777777777778};

__constant__ float gl_quad_nodes_3_float[3]   = {0.1127016653792583f, 0.5f, 0.8872983346207417f};
__constant__ float gl_quad_weights_3_float[3] = {0.2777777777777778f, 0.4444444444444444f, 0.2777777777777778f};

// Gauss-Legendre Quadrature (N=4, D=1)
__constant__ double gl_quad_nodes_4_double[4]   = {0.06943184420297371,
                                                   0.33000947820757187,
                                                   0.6699905217924281,
                                                   0.9305681557970262};
__constant__ double gl_quad_weights_4_double[4] = {0.17392742256872692,
                                                   0.3260725774312731,
                                                   0.3260725774312731,
                                                   0.17392742256872692};

__constant__ float gl_quad_nodes_4_float[4]   = {0.06943184420297371f,
                                                 0.33000947820757187f,
                                                 0.6699905217924281f,
                                                 0.9305681557970262f};
__constant__ float gl_quad_weights_4_float[4] = {0.17392742256872692f,
                                                 0.3260725774312731f,
                                                 0.3260725774312731f,
                                                 0.17392742256872692f};

// Template accessor functions
template <typename FloatType, int N>
__device__ __forceinline__ const FloatType* get_gl_quad_nodes();

template <typename FloatType, int N>
__device__ __forceinline__ const FloatType* get_gl_quad_weights();

// Specializations for N=2
template <>
__device__ __forceinline__ const double* get_gl_quad_nodes<double, 2>() {
    return gl_quad_nodes_2_double;
}  // END Function: get_gl_quad_nodes

template <>
__device__ __forceinline__ const float* get_gl_quad_nodes<float, 2>() {
    return gl_quad_nodes_2_float;
}  // END Function: get_gl_quad_nodes

template <>
__device__ __forceinline__ const double* get_gl_quad_weights<double, 2>() {
    return gl_quad_weights_2_double;
}  // END Function: get_gl_quad_weights

template <>
__device__ __forceinline__ const float* get_gl_quad_weights<float, 2>() {
    return gl_quad_weights_2_float;
}  // END Function: get_gl_quad_weights

// Specializations for N=3
template <>
__device__ __forceinline__ const double* get_gl_quad_nodes<double, 3>() {
    return gl_quad_nodes_3_double;
}  // END Function: get_gl_quad_nodes

template <>
__device__ __forceinline__ const float* get_gl_quad_nodes<float, 3>() {
    return gl_quad_nodes_3_float;
}  // END Function: get_gl_quad_nodes

template <>
__device__ __forceinline__ const double* get_gl_quad_weights<double, 3>() {
    return gl_quad_weights_3_double;
}  // END Function: get_gl_quad_weights

template <>
__device__ __forceinline__ const float* get_gl_quad_weights<float, 3>() {
    return gl_quad_weights_3_float;
}  // END Function: get_gl_quad_weights

// Specializations for N=4
template <>
__device__ __forceinline__ const double* get_gl_quad_nodes<double, 4>() {
    return gl_quad_nodes_4_double;
}  // END Function: get_gl_quad_nodes

template <>
__device__ __forceinline__ const float* get_gl_quad_nodes<float, 4>() {
    return gl_quad_nodes_4_float;
}  // END Function: get_gl_quad_nodes

template <>
__device__ __forceinline__ const double* get_gl_quad_weights<double, 4>() {
    return gl_quad_weights_4_double;
}  // END Function: get_gl_quad_weights

template <>
__device__ __forceinline__ const float* get_gl_quad_weights<float, 4>() {
    return gl_quad_weights_4_float;
}  // END Function: get_gl_quad_weights

template <typename FloatType, typename FloatType2>
__device__ __forceinline__ FloatType2 make_FloatType2(const FloatType val1, const FloatType val2);

template <>
__device__ __forceinline__ float2 make_FloatType2<float, float2>(const float val1, const float val2) {
    return make_float2(val1, val2);
}  // END Function: make_FloatType2

template <>
__device__ __forceinline__ double2 make_FloatType2<double, double2>(const double val1, const double val2) {
    return make_double2(val1, val2);
}  // END Function: make_FloatType2

template <typename FloatType>
class declare_FloatType2 {};

template <>
class declare_FloatType2<float> {
public:
    using type = float2;
};

template <>
class declare_FloatType2<double> {
public:
    using type = double2;
};

template <typename FloatType, typename FloatType2, typename IntType>
__device__ __forceinline__ FloatType2 Gauss_Legendre_quadrature_pairs(const int N, const int idx) {
    switch (N) {
        case 1:
            return make_FloatType2<FloatType, FloatType2>(get_gl_quad_nodes<FloatType, 1>()[idx],
                                                          get_gl_quad_weights<FloatType, 1>()[idx]);
            break;
        case 2:
            return make_FloatType2<FloatType, FloatType2>(get_gl_quad_nodes<FloatType, 2>()[idx],
                                                          get_gl_quad_weights<FloatType, 2>()[idx]);
            break;
        case 3:
            return make_FloatType2<FloatType, FloatType2>(get_gl_quad_nodes<FloatType, 3>()[idx],
                                                          get_gl_quad_weights<FloatType, 3>()[idx]);
            break;
        case 4:
            return make_FloatType2<FloatType, FloatType2>(get_gl_quad_nodes<FloatType, 4>()[idx],
                                                          get_gl_quad_weights<FloatType, 4>()[idx]);
            break;
        default:
            __trap();  // Unsupported number of points
    }  // END switch (N)

    return make_FloatType2<FloatType, FloatType2>(FloatType(0), FloatType(0));
}  // END Function: Gauss_Legendre_quadrature_pairs

#endif  // __SFEM_RESAMPLE_FIELD_QUAD_RULES_CUH__