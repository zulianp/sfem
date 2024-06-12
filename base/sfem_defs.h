#ifndef SFEM_DEFS_H
#define SFEM_DEFS_H

#include "sfem_base.h"

#include <assert.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ElemType {
    NIL = 0,
    NODE1 = 1,
    EDGE2 = 2,
    EDGE3 = 11,
    BEAM2 = 100002,
    TRI3 = 3,
    TRI6 = 6,
    TRI10 = 1010,
    TRISHELL3 = 103,
    TRISHELL6 = 106,
    TRISHELL10 = 110,
    QUAD4 = 40,
    TET4 = 4,
    TET10 = 10,
    TET20 = 20,
    HEX8 = 8,
    WEDGE6 = 1006,
    MACRO = 200,
    MACRO_TRI3 = (MACRO + TRI3),
    MACRO_TET4 = (MACRO + TET4),
    INVALID = -1
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
    if (!strcmp(str, "TET4")) return TET4;
    if (!strcmp(str, "TET10")) return TET10;
    if (!strcmp(str, "TET20")) return TET20;
    if (!strcmp(str, "MACRO_TRI3")) return MACRO_TRI3;
    if (!strcmp(str, "MACRO_TET4")) return MACRO_TET4;
    if (!strcmp(str, "HEX8")) return HEX8;

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
        case TRI3:
            return "TRI3";
        case TRISHELL3:
            return "TRISHELL3";
        case WEDGE6:
            return "WEDGE6";
        case QUAD4:
            return "QUAD4";
        case TET4:
            return "TET4";
        case TRI6:
            return "TRI6";
        case TRI10:
            return "TRI10";
        case MACRO_TRI3:
            return "MACRO_TRI3";
        case MACRO_TET4:
            return "MACRO_TET4";
        case HEX8:
            return "HEX8";
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
        case MACRO_TET4:
            return TRI6;  // FIXME
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
            return BEAM2;
        case BEAM2:
            return BEAM2;
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
            return 3;
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
        default: {
            assert(0);
            return macro_type;
        }
    }
}

SFEM_INLINE int is_second_order_lagrange(const enum ElemType type) {
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

#ifdef __cplusplus
}
#endif

#endif  // SFEM_DEFS_H
