#include "sfem_resample_field.h"
#include "sfem_resample_field_tet4_math.h"
#include "sfem_stack.h"

#include "mass.h"
// #include "read_mesh.h"
#include "matrixio_array.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

// #define real_t real_type

#include "hyteg.h"
#include "quadratures_rule.h"

#define real_type real_t
#define SFEM_RESTRICT __restrict__

#define SFEM_RESAMPLE_GAP_DUAL

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
// alpha_to_hyteg_level //////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
int                                                       //
alpha_to_hyteg_level(const real_t alpha,                  //
                     const real_t alpha_min_threshold,    //
                     const real_t alpha_max_threshold) {  //

    if (alpha < alpha_min_threshold) return 1;                           // No refinement
    if (alpha > alpha_max_threshold) return HYTEG_MAX_REFINEMENT_LEVEL;  // Maximum refinement

    real_t alpha_x = alpha - alpha_min_threshold;  // Shift the alpha to start from 0
    real_t L_real  = alpha_x / (alpha_max_threshold - alpha_min_threshold) * (real_t)(HYTEG_MAX_REFINEMENT_LEVEL - 1);

    int L = L_real >= 1 ? (int)L_real : 1;                                    // Convert to integer
    L     = L > HYTEG_MAX_REFINEMENT_LEVEL ? HYTEG_MAX_REFINEMENT_LEVEL : L;  // Clamp to maximum level

    return L;  // Return the level of refinement
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_refine_adjoint_hyteg ////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                                                  //
tet4_resample_field_local_refine_adjoint_hyteg(const ptrdiff_t                      start_element,   // Mesh
                                               const ptrdiff_t                      end_element,     //
                                               const ptrdiff_t                      nnodes,          //
                                               const idx_t** const SFEM_RESTRICT    elems,           //
                                               const geom_t** const SFEM_RESTRICT   xyz,             //
                                               const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
                                               const ptrdiff_t* const SFEM_RESTRICT stride,          //
                                               const geom_t* const SFEM_RESTRICT    origin,          //
                                               const geom_t* const SFEM_RESTRICT    delta,           //
                                               const real_t* const SFEM_RESTRICT    weighted_field,  // Input weighted field
                                               const real_t                         alpha_th,        // Threshold for alpha
                                               real_t* const SFEM_RESTRICT          data) {                   //

    PRINT_CURRENT_FUNCTION;
    int ret = 0;

    real_t alpha_min_threshold = 2.5;   // Minimum threshold for alpha
    real_t alpha_max_threshold = 15.0;  // Maximum threshold for alpha

    // The minimum and maximum thresholds for alpha are used to determine the level of refinement.
    // If the alpha value is below the minimum threshold, no refinement is applied.
    // If the alpha value is above the maximum threshold, the maximum level of refinement is applied.

    const real_type ox = (real_type)origin[0];
    const real_type oy = (real_type)origin[1];
    const real_type oz = (real_type)origin[2];

    const real_type dx = (real_type)delta[0];
    const real_type dy = (real_type)delta[1];
    const real_type dz = (real_type)delta[2];

    const real_type d_min = dx < dy ? (dx < dz ? dx : dz) : (dy < dz ? dy : dz);

    const real_type hexahedron_volume = dx * dy * dz;

#if SFEM_LOG_LEVEL >= 5
    printf("============================================================\n");
    printf("Start: %s: %s:%d \n", __FUNCTION__, __FILE__, __LINE__);
    printf("Heaxahedron volume = %g\n", hexahedron_volume);
    printf("============================================================\n");
#endif

    int degenerated_tetrahedra_cnt = 0;
    int uniform_refine_cnt         = 0;

    for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++) {
        // loop over the 4 vertices of the tetrahedron
        idx_t ev[4];
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][element_i];
        }

        // Read the coordinates of the vertices of the tetrahedron
        const real_type x0 = xyz[0][ev[0]];
        const real_type x1 = xyz[0][ev[1]];
        const real_type x2 = xyz[0][ev[2]];
        const real_type x3 = xyz[0][ev[3]];

        const real_type y0 = xyz[1][ev[0]];
        const real_type y1 = xyz[1][ev[1]];
        const real_type y2 = xyz[1][ev[2]];
        const real_type y3 = xyz[1][ev[3]];

        const real_type z0 = xyz[2][ev[0]];
        const real_type z1 = xyz[2][ev[1]];
        const real_type z2 = xyz[2][ev[2]];
        const real_type z3 = xyz[2][ev[3]];

        // Compute the alpha_tet to decide if the tetrahedron is refined
        // Sides of the tetrahedron
        real_type edges_length[6];

        int vertex_a = -1;
        int vertex_b = -1;

        const real_type max_edges_length =          //
                tet_edge_max_length(x0,             //
                                    y0,             //
                                    z0,             //
                                    x1,             //
                                    y1,             //
                                    z1,             //
                                    x2,             //
                                    y2,             //
                                    z2,             //
                                    x3,             //
                                    y3,             //
                                    z3,             //
                                    &vertex_a,      // Output
                                    &vertex_b,      // Output
                                    edges_length);  // Output

        const real_t alpha_tet = max_edges_length / d_min;

        const int L = alpha_to_hyteg_level(alpha_tet,             //
                                           alpha_min_threshold,   //
                                           alpha_max_threshold);  //

        const int     hteg_num_tetrahedra = get_hyteg_num_tetrahedra(L);
        const real_t* x_hyteg             = get_hyteg_x(L);
        const real_t* y_hyteg             = get_hyteg_y(L);
        const real_t* z_hyteg             = get_hyteg_z(L);
        const int*    categories_hyteg    = get_hyteg_categories(L);

        for (int tet_i = 0; tet_i < hteg_num_tetrahedra; tet_i++) {
        }
    }

    RETURN_FROM_FUNCTION(ret);
}  // END OF FUNCTION tet4_resample_field_local_refine_adjoint_hyteg
//////////////////////////////////////////////////////////
