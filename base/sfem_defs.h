#ifndef SFEM_DEFS_H
#define SFEM_DEFS_H

#include "sfem_base.h"

#include <assert.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

enum RealType { SFEM_FLOAT16 = 2, SFEM_FLOAT32 = 4, SFEM_FLOAT64 = 8, SFEM_REAL_DEFAULT = 0 };
enum IntegerType { SFEM_INT16 = 20, SFEM_INT32 = 40, SFEM_INT64 = 80, SFEM_INT_DEFAULT = 0 };

typedef const char* OperatorType;
static OperatorType MATRIX_FREE = "MF";
static OperatorType CRS = "CRS";
static OperatorType CRS_SYM = "CRS_SYM";
static OperatorType BSR = "BSR";
static OperatorType BSR_SYM = "BSR_SYM";
static OperatorType COO_SYM = "COO_SYM";


#define SFEM_UNSUPPORTED_ELEMENT_ERROR(element_type) SFEM_ERROR("Unsupported element type %d\n", element_type);

typedef enum {
    SFEM_ACCELERATOR_TYPE_CPU     = 0,  // CPU
    SFEM_ACCELERATOR_TYPE_CUDA    = 1,  // CUDA
    SFEM_ACCELERATOR_TYPE_OPENCL  = 2,  // OpenCL Not supported
    SFEM_ACCELERATOR_TYPE_OPENACC = 3,  // OpenACC Not supported
    SFEM_ACCELERATOR_TYPE_HIP     = 4   // HIP Not supported
} AcceleratorsType;                     //

typedef enum {
    ADJOINT_BASE = 0,                //
    ADJOINT_REFINE_ONE_STEP,         //
    ADJOINT_REFINE_ITERATIVE,        //
    ADJOINT_REFINE_HYTEG_REFINEMENT  //
} AdjointRefineType;                 //

static void* SFEM_DEFAULT_STREAM = 0;

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
        default: {
            assert(0);
            return SFEM_FAILURE;
        }
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

enum ElemType {
    NIL             = 0,
    NODE1           = 1,
    EDGE2           = 2,
    EDGE3           = 11,
    EDGESHELL2      = 101,
    BEAM2           = 100002,
    TRI3            = 3,
    TRI6            = 6,
    TRI10           = 1010,
    TRISHELL3       = 103,
    TRISHELL6       = 106,
    TRISHELL10      = 110,
    QUAD4           = 40,
    QUADSHELL4      = 140,
    TET4            = 4,
    TET10           = 10,
    TET20           = 20,
    HEX8            = 8,
    WEDGE6          = 1006,
    MACRO           = 200,
    MACRO_TRI3      = (MACRO + TRI3),
    MACRO_TRISHELL3 = (MACRO + TRISHELL3),
    MACRO_TET4      = (MACRO + TET4),
    SSTET4          = 4000,
    SSQUAD4         = 40000,
    SSQUADSHELL4    = 140000,
    SSHEX8          = 8000,
    INVALID         = -1
};

SFEM_INLINE static enum ElemType type_from_string(const char* str) {
    if (!strcmp(str, "NODE1")) return NODE1;
    if (!strcmp(str, "EDGE2")) return EDGE2;
    if (!strcmp(str, "EDGE3")) return EDGE3;
    if (!strcmp(str, "TRI3")) return TRI3;
    if (!strcmp(str, "TRI6")) return TRI6;
    if (!strcmp(str, "TRI10")) return TRI10;
    if (!strcmp(str, "TRISHELL3")) return TRISHELL3;
    if (!strcmp(str, "WEDGE6")) return WEDGE6;
    if (!strcmp(str, "QUAD4")) return QUAD4;
    if (!strcmp(str, "QUADSHELL4")) return QUADSHELL4;
    if (!strcmp(str, "SSQUAD4")) return SSQUAD4;
    if (!strcmp(str, "SSQUADSHELL4")) return SSQUADSHELL4;
    if (!strcmp(str, "TET4")) return TET4;
    if (!strcmp(str, "TET10")) return TET10;
    if (!strcmp(str, "TET20")) return TET20;
    if (!strcmp(str, "MACRO_TRI3")) return MACRO_TRI3;
    if (!strcmp(str, "MACRO_TET4")) return MACRO_TET4;
    if (!strcmp(str, "HEX8")) return HEX8;
    if (!strcmp(str, "SSHEX8")) return SSHEX8;

    assert(0);
    return INVALID;
}

SFEM_INLINE static const char* type_to_string(enum ElemType type) {
    switch (type) {
        case NODE1:
            return "NODE1";
        case EDGE2:
            return "EDGE2";
        case EDGE3:
            return "EDGE3";
        case BEAM2:
            return "BEAM2";
        case TRI3:
            return "TRI3";
        case TRISHELL3:
            return "TRISHELL3";
        case WEDGE6:
            return "WEDGE6";
        case QUAD4:
            return "QUAD4";
        case QUADSHELL4:
            return "QUADSHELL4";
        case SSQUAD4:
            return "SSQUAD4";
        case SSQUADSHELL4:
            return "SSQUADSHELL4";
        case TET4:
            return "TET4";
        case TRI6:
            return "TRI6";
        case TRISHELL6:
            return "TRISHELL6";
        case TRI10:
            return "TRI10";
        case MACRO_TRI3:
            return "MACRO_TRI3";
        case MACRO_TRISHELL3:
            return "MACRO_TRISHELL3";
        case MACRO_TET4:
            return "MACRO_TET4";
        case HEX8:
            return "HEX8";
        case SSHEX8:
            return "SSHEX8";
        case TET10:
            return "TET10";
        case TET20:
            return "TET20";
        default: {
            assert(0);
            return "INVALID";
        }
    }
}

SFEM_INLINE static enum ElemType side_type(const enum ElemType type) {
    switch (type) {
        case TRI3:
        case QUAD4:
            return EDGE2;
        case TRI6:
            return EDGE3;
        case TET4:
            return TRI3;
        case TET10:
            return TRI6;
        case TET20:
            return TRI10;
        case EDGE2:
            return NODE1;
        case TRISHELL3:
            return BEAM2;
        case QUADSHELL4:
            return BEAM2;
        case MACRO_TET4:
            return MACRO_TRI3;
        case HEX8:
            return QUAD4;
        case SSHEX8:
            return SSQUAD4;
        default: {
            assert(0);
            return INVALID;
        }
    }
}

SFEM_INLINE static enum ElemType shell_type(const enum ElemType type) {
    switch (type) {
        case TRI3:
            return TRISHELL3;
        case MACRO_TRI3:
            return MACRO_TRISHELL3;
        case TRI6:
            return TRISHELL6;
        case TRI10:
            return TRISHELL10;
        case TRISHELL3:
            return TRISHELL3;
        case TRISHELL6:
            return TRISHELL6;
        case TRISHELL10:
            return TRISHELL10;
        case EDGE2:
            return EDGESHELL2;
        case BEAM2:
            return BEAM2;
        case QUAD4:
            return QUADSHELL4;
        case QUADSHELL4:
            return QUADSHELL4;
        case SSQUAD4:
            return SSQUADSHELL4;
        case SSQUADSHELL4:
            return SSQUADSHELL4;
        default: {
            // assert(0);
            return INVALID;
        }
    }
}

SFEM_INLINE static enum ElemType elem_lower_order(const enum ElemType type) {
    switch (type) {
        case NIL:
            return NIL;
        case TRI6:
            return TRI3;
        case TET10:
            return TET4;
        case TET20:
            return TET10;
        case EDGE3:
            return EDGE2;
        default: {
            assert(0);
            return INVALID;
        }
    }
}

SFEM_INLINE static enum ElemType elem_higher_order(const enum ElemType type) {
    switch (type) {
        case NIL:
            return NIL;
        case TRI3:
            return TRI6;
        case TET4:
            return TET10;
        case TET10:
            return TET20;
        case EDGE2:
            return EDGE3;
        default: {
            assert(0);
            return INVALID;
        }
    }
}

SFEM_INLINE static int elem_num_nodes(const enum ElemType type) {
    switch (type) {
        case NIL:
            return 0;
        case NODE1:
            return 1;
        case EDGE2:
            return 2;
        case EDGE3:
            return 3;
        case TRI3:
            return 3;
        case TRISHELL3:
            return 3;
        case WEDGE6:
            return 6;
        case QUAD4:
            return 4;
        case QUADSHELL4:
            return 4;
        case TET4:
            return 4;
        case TRI6:
            return 6;
        case MACRO_TRI3:
            return 6;
        case MACRO_TET4:
            return 10;
        case HEX8:
            return 8;
        case TET10:
            return 10;
        case TET20:
            return 20;
        default: {
            assert(0);
            return 0;
        }
    }
}

SFEM_INLINE static int elem_num_sides(const enum ElemType type) {
    switch (type) {
        case NIL:
            return 0;
        case EDGE2:
            return 2;
        case TRI3:
            return 3;
        case TRISHELL3:
            return 3;
        case MACRO_TRI3:
            return 3;  // Really?
        case MACRO_TET4:
            return 4;
        case QUAD4:
            return 4;
        case QUADSHELL4:
            return 4;
        case TET4:
            return 4;
        case WEDGE6:
            return 5;
        case TRI6:
            return 3;
        case HEX8:
            return 6;
        case TET10:
            return 4;
        case TET20:
            return 4;
        default: {
            assert(0);
            return 0;
        }
    }
}

SFEM_INLINE static int elem_manifold_dim(const enum ElemType type) {
    switch (type) {
        case NIL:
            return 0;
        case EDGE2:
            return 1;
        case TRI3:
            return 2;
        case QUAD4:
            return 2;
        case QUADSHELL4:
            return 2;
        case TET4:
            return 3;
        case WEDGE6:
            return 3;
        case TRI6:
            return 2;
        case MACRO_TRI3:
            return 2;
        case MACRO_TET4:
            return 3;
        case HEX8:
            return 3;
        case TET10:
            return 3;
        case TET20:
            return 3;
        default: {
            assert(0);
            return INVALID;
        }
    }
}

SFEM_INLINE static enum ElemType macro_type_variant(const enum ElemType type) {
    switch (type) {
        case TET10:
            return MACRO_TET4;
        case TRI6:
            return MACRO_TRI3;
        default: {
            assert(0);
            return type;
        }
    }
}

SFEM_INLINE static enum ElemType macro_base_elem(const enum ElemType macro_type) {
    switch (macro_type) {
        case MACRO_TET4:
            return TET4;
        case MACRO_TRI3:
            return TRI3;
        case TET10:
            return TET4;
        case TRI6:
            return TRI3;
        case SSHEX8:
            return HEX8;
        default: {
            assert(0);
            return macro_type;
        }
    }
}

SFEM_INLINE static int is_second_order_lagrange(const enum ElemType type) {
    switch (type) {
        case TET10:
            return 1;
        case TRI6:
            return 1;
        default: {
            return 0;
        }
    }
}

enum HEX8_Sides { HEX8_LEFT = 3, HEX8_RIGHT = 1, HEX8_BOTTOM = 4, HEX8_TOP = 5, HEX8_FRONT = 0, HEX8_BACK = 2 };

#ifdef __cplusplus
}
#endif

#endif  // SFEM_DEFS_H
