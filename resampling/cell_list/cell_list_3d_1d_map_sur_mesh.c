#include "cell_list_3d_1d_map_sur_mesh.h"

////////////////////////////////////////////////// 
// intersect_triangle_xy
//////////////////////////////////////////////////
bool                                       //
intersect_triangle_xy(const real_t v0[3],  //
                      const real_t v1[3],  //
                      const real_t v2[3],  //
                      const real_t x,      //
                      const real_t y) {    //
    // Signed area of each sub-triangle formed by an edge and the query point.
    // A point is inside iff all three have the same sign (all CW or all CCW).
    const real_t d0 = (v1[0] - v0[0]) * (y - v0[1]) - (v1[1] - v0[1]) * (x - v0[0]);
    const real_t d1 = (v2[0] - v1[0]) * (y - v1[1]) - (v2[1] - v1[1]) * (x - v1[0]);
    const real_t d2 = (v0[0] - v2[0]) * (y - v2[1]) - (v0[1] - v2[1]) * (x - v2[0]);

    const bool has_neg = (d0 < 0) | (d1 < 0) | (d2 < 0);
    const bool has_pos = (d0 > 0) | (d1 > 0) | (d2 > 0);
    return !(has_neg & has_pos);  // inside iff all same sign (or zero)
}