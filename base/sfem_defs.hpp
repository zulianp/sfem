#ifndef SFEM_DEFS_H
#define SFEM_DEFS_H

#include "sfem_base.hpp"
#include "smesh_adjacency.hpp"
#include "smesh_elem_type.hpp"
#include "smesh_graph.hpp"

#include <assert.h>

enum RealType { SFEM_FLOAT16 = 2, SFEM_FLOAT32 = 4, SFEM_FLOAT64 = 8, SFEM_REAL_DEFAULT = 0 };
enum IntegerType { SFEM_INT16 = 20, SFEM_INT32 = 40, SFEM_INT64 = 80, SFEM_INT_DEFAULT = 0 };

typedef const char* OperatorType;
static OperatorType MATRIX_FREE = "MF";
static OperatorType CRS = "CRS";
static OperatorType SPLITCRS = "SPLITCRS";
static OperatorType ALIGNEDCRS = "ALIGNEDCRS";
static OperatorType CRS_SYM = "CRS_SYM";
static OperatorType BSR = "BSR";
static OperatorType BSR_SYM = "BSR_SYM";
static OperatorType COO_SYM = "COO_SYM";
static OperatorType SPLITDACRS = "SPLITDACRS";
static OperatorType SELL = "SELL";

#define SFEM_UNSUPPORTED_ELEMENT_ERROR(element_type) SFEM_ERROR("Unsupported element type %d\n", element_type);

typedef enum {
    SFEM_ACCELERATOR_TYPE_CPU = 0,
    SFEM_ACCELERATOR_TYPE_CUDA = 1,
    SFEM_ACCELERATOR_TYPE_OPENCL = 2,
    SFEM_ACCELERATOR_TYPE_OPENACC = 3,
    SFEM_ACCELERATOR_TYPE_HIP = 4
} AcceleratorsType;

typedef enum {
    ADJOINT_BASE = 0,
    ADJOINT_REFINE_ONE_STEP,
    ADJOINT_REFINE_ITERATIVE,
    ADJOINT_REFINE_HYTEG_REFINEMENT
} AdjointRefineType;

static void* SFEM_DEFAULT_STREAM = 0;

#ifdef __cplusplus
namespace sfem {
using smesh::crs_to_coo;
using smesh::elem_higher_order;
using smesh::elem_lower_order;
using smesh::elem_manifold_dim;
using smesh::elem_num_nodes;
using smesh::elem_num_sides;
using smesh::extract_skin_sideset;
using smesh::is_second_order_lagrange;
using smesh::is_semistructured_type;
using smesh::macro_base_elem;
using smesh::macro_type_variant;
using smesh::proteus_hex_micro_elements_per_dim;
using smesh::proteus_hex_type;
using smesh::semistructured_type;
using smesh::shell_type;
using smesh::side_type;
using smesh::type_from_string;
using smesh::type_to_string;

SFEM_INLINE static int real_type_size(enum RealType type) {
    switch (type) {
        case SFEM_FLOAT16:
            return 2;
        case SFEM_FLOAT32:
            return 4;
        case SFEM_FLOAT64:
            return 8;
        case SFEM_REAL_DEFAULT:
            return sizeof(real_t);
        default:
            assert(0);
            return SFEM_FAILURE;
    }
}

SFEM_INLINE static const char* real_type_to_string(enum RealType type) {
    switch (type) {
        case SFEM_FLOAT16:
            return "SFEM_FLOAT16";
        case SFEM_FLOAT32:
            return "SFEM_FLOAT32";
        case SFEM_FLOAT64:
            return "SFEM_FLOAT64";
        case SFEM_REAL_DEFAULT:
            return "SFEM_REAL_DEFAULT";
        default:
            return "SFEM_FLOAT_UNDEFINED";
    }
}

SFEM_INLINE static const char* integer_type_to_string(enum IntegerType type) {
    switch (type) {
        case SFEM_INT16:
            return "SFEM_INT16";
        case SFEM_INT32:
            return "SFEM_INT32";
        case SFEM_INT64:
            return "SFEM_INT64";
        case SFEM_INT_DEFAULT:
            return "SFEM_INT_DEFAULT";
        default:
            return "SFEM_INT_UNDEFINED";
    }
}
}  // namespace sfem
#else
SFEM_INLINE static int real_type_size(enum RealType type) {
    switch (type) {
        case SFEM_FLOAT16:
            return 2;
        case SFEM_FLOAT32:
            return 4;
        case SFEM_FLOAT64:
            return 8;
        case SFEM_REAL_DEFAULT:
            return sizeof(real_t);
        default:
            assert(0);
            return SFEM_FAILURE;
    }
}

SFEM_INLINE static const char* real_type_to_string(enum RealType type) {
    switch (type) {
        case SFEM_FLOAT16:
            return "SFEM_FLOAT16";
        case SFEM_FLOAT32:
            return "SFEM_FLOAT32";
        case SFEM_FLOAT64:
            return "SFEM_FLOAT64";
        case SFEM_REAL_DEFAULT:
            return "SFEM_REAL_DEFAULT";
        default:
            return "SFEM_FLOAT_UNDEFINED";
    }
}

SFEM_INLINE static const char* integer_type_to_string(enum IntegerType type) {
    switch (type) {
        case SFEM_INT16:
            return "SFEM_INT16";
        case SFEM_INT32:
            return "SFEM_INT32";
        case SFEM_INT64:
            return "SFEM_INT64";
        case SFEM_INT_DEFAULT:
            return "SFEM_INT_DEFAULT";
        default:
            return "SFEM_INT_UNDEFINED";
    }
}
#endif

#endif  // SFEM_DEFS_H
