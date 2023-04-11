#ifndef SFEM_DEFS_H
#define SFEM_DEFS_H

#include "sfem_base.h"

#include <assert.h>

enum ElemType { TRI3 = 3, TET4 = 4, TRI6 = 6, HEX8 = 8, TET10 = 10, INVALID = -1 };

SFEM_INLINE static int side_type(const enum ElemType type) {
    switch (type) {
        case TET4:
            return TRI3;
        case TET10:
            return TRI6;
        default: {
        	assert(false);
            return INVALID;
        }
    }
}

#endif  // SFEM_DEFS_H
