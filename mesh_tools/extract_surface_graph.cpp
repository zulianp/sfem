
#include "sfem_base.h"

#include <algorithm>
#include <vector>

extern "C" void extract_surface_connectivity(const ptrdiff_t n_elements,
                                  idx_t** const elems,
                                  ptrdiff_t* n_surf_elements,
                                  idx_t** surf_elems) {
    const ptrdiff_t n_sides = 4 * n_elements;
    std::vector<idx_t> buff(n_sides * 3);

    ptrdiff_t face_idx = 0;
    for (ptrdiff_t i = 0; i < n_elements; ++i) {
        buff[face_idx + 0] = elems[0][i];
        buff[face_idx + 1] = elems[1][i];
        buff[face_idx + 2] = elems[2][i];

        std::sort(&buff[face_idx], &buff[face_idx] + 3);
        face_idx += 3;

        buff[face_idx + 0] = elems[1][i];
        buff[face_idx + 1] = elems[2][i];
        buff[face_idx + 2] = elems[3][i];

        std::sort(&buff[face_idx], &buff[face_idx] + 3);
        face_idx += 3;

        buff[face_idx + 0] = elems[2][i];
        buff[face_idx + 1] = elems[3][i];
        buff[face_idx + 2] = elems[0][i];

        std::sort(&buff[face_idx], &buff[face_idx] + 3);
        face_idx += 3;

        buff[face_idx + 0] = elems[0][i];
        buff[face_idx + 1] = elems[1][i];
        buff[face_idx + 2] = elems[3][i];

        std::sort(&buff[face_idx], &buff[face_idx] + 3);
        face_idx += 3;
    }

    std::vector<ptrdiff_t> sideidx(n_sides);

    for (ptrdiff_t i = 0; i < n_sides; ++i) {
        sideidx[i] = i;
    }

    std::sort(sideidx.begin(), sideidx.end(), [&](const ptrdiff_t l, const ptrdiff_t r) {
        for (int d = 0; d < 3; ++d) {
            const idx_t lidx = buff[l * 3 + d];
            const idx_t ridx = buff[r * 3 + d];

            if (lidx < ridx) {
                return true;
            } else if (lidx > ridx) {
                return false;
            }
        }

        return false;
    });

    ptrdiff_t n_surface = 0;

    for (ptrdiff_t i = 0; i < n_sides - 1;) {
        const ptrdiff_t l = sideidx[i];
        const ptrdiff_t r = sideidx[i + 1];

        assert(l >= 0);
        assert(r >= 0);

        bool same = true;

        for (int d = 0; d < 3; ++d) {
            const idx_t lidx = buff[l * 3 + d];
            const idx_t ridx = buff[r * 3 + d];

            if (lidx < ridx) {
                same = false;
                break;
            } else if (lidx > ridx) {
                same = false;
                break;
            }
        }

        if (same) {
            // Not surface face remove
            sideidx[i] = -1;
            sideidx[i+1] = -1;
            i += 2;
        } else {
            i += 1;
            n_surface += 1;
        }
    }

    if(sideidx[n_sides - 1] >= 0) {
    	n_surface += 1;
    }

    for (int d = 0; d < 3; ++d) {
        surf_elems[d] = (idx_t*)malloc(n_surface * sizeof(idx_t));
    }

    face_idx = 0;
    for (ptrdiff_t i = 0; i < n_sides; i++) {
        if (sideidx[i] < 0) continue;

        for (int d = 0; d < 3; ++d) {
            surf_elems[d][face_idx] = buff[sideidx[i] * 3 + d];
        }

        face_idx++;
    }

    *n_surf_elements = n_surface;
}
