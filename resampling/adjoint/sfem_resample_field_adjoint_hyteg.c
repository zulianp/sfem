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

int                                                                                         //
tet4_resample_tetrahedron_local_hyteg_adjoint(const real_t                         x0,      // Tetrahedron vertices X-coordinates
                                              const real_t                         x1,      //
                                              const real_t                         x2,      //
                                              const real_t                         x3,      //
                                              const real_t                         y0,      // Tetrahedron vertices Y-coordinates
                                              const real_t                         y1,      //
                                              const real_t                         y2,      //
                                              const real_t                         y3,      //
                                              const real_t                         z0,      // Tetrahedron vertices Z-coordinates
                                              const real_t                         z1,      //
                                              const real_t                         z2,      //
                                              const real_t                         z3,      //
                                              const real_t                         detCat,  // determinant of the category.
                                              const real_t                         wf0,     // Weighted field at the vertices
                                              const real_t                         wf1,     //
                                              const real_t                         wf2,     //
                                              const real_t                         wf3,     //
                                              const real_t                         ox,      // Origin of the grid
                                              const real_t                         oy,      //
                                              const real_t                         oz,      //
                                              const real_t                         dx,      // Spacing of the grid
                                              const real_t                         dy,      //
                                              const real_t                         dz,      //
                                              const ptrdiff_t* const SFEM_RESTRICT stride,  // Stride
                                              const ptrdiff_t* const SFEM_RESTRICT n,       // Size of the grid
                                              real_t* const SFEM_RESTRICT          data) {           //
}

real_t                                                     //
calculate_jacobian_for_category(const int       category,  //
                                const real_type px0,       //
                                const real_type py0,       //
                                const real_type pz0,       //
                                const real_type px1,       //
                                const real_type py1,       //
                                const real_type pz1,       //
                                const real_type px2,       //
                                const real_type py2,       //
                                const real_type pz2,       //
                                const real_type px3,       //
                                const real_type py3,       //
                                const real_type pz3,       //
                                const real_t    L,         //
                                const int       tet_i,     //
                                int*            error_flag) {         //
    real_t det_jacobian = 0.0;
    *error_flag         = 0;

    switch (category) {
        case 0:  // Category 0
            det_jacobian = det_jacobian_cat0_real(px0, py0, pz0, px1, py1, pz1, px2, py2, pz2, px3, py3, pz3, L);
            break;
        case 1:  // Category 1
            det_jacobian = det_jacobian_cat1_real(px0, py0, pz0, px1, py1, pz1, px2, py2, pz2, px3, py3, pz3, L);
            break;
        case 2:  // Category 2
            det_jacobian = det_jacobian_cat2_real(px0, py0, pz0, px1, py1, pz1, px2, py2, pz2, px3, py3, pz3, L);
            break;
        case 3:  // Category 3
            det_jacobian = det_jacobian_cat3_real(px0, py0, pz0, px1, py1, pz1, px2, py2, pz2, px3, py3, pz3, L);
            break;
        case 4:  // Category 4
            det_jacobian = det_jacobian_cat4_real(px0, py0, pz0, px1, py1, pz1, px2, py2, pz2, px3, py3, pz3, L);
            break;
        case 5:  // Category 5
            det_jacobian = det_jacobian_cat5_real(px0, py0, pz0, px1, py1, pz1, px2, py2, pz2, px3, py3, pz3, L);
            break;
        default:  // Invalid category
            fprintf(stderr,
                    "calculate_jacobian_for_category: Invalid category %d for tetrahedron %d at level "
                    "%d\n",
                    category,
                    tet_i,
                    L);
            *error_flag = -1;
            // The decision to exit should be made by the caller
            break;
    }
    return det_jacobian;
}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
// alpha_to_hyteg_level //////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
int                                                     //
alpha_to_hyteg_level(const real_t alpha,                //
                     const real_t alpha_min_threshold,  //
                     const real_t alpha_max_threshold,  //
                     const int    max_refinement_L) {      //

    if (alpha < alpha_min_threshold) return 1;                           // No refinement
    if (alpha > alpha_max_threshold) return HYTEG_MAX_REFINEMENT_LEVEL;  // Maximum refinement

    real_t alpha_x = alpha - alpha_min_threshold;  // Shift the alpha to start from 0
    real_t L_real  = alpha_x / (alpha_max_threshold - alpha_min_threshold) * (real_t)(HYTEG_MAX_REFINEMENT_LEVEL - 1);

    int L = L_real >= 1 ? (int)L_real : 1;                                    // Convert to integer
    L     = L > HYTEG_MAX_REFINEMENT_LEVEL ? HYTEG_MAX_REFINEMENT_LEVEL : L;  // Clamp to maximum level

    return L >= max_refinement_L ? max_refinement_L : L;  // Return the level, clamped to max_refinement_L
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
    int    max_refinement_L    = 5;

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

        const int L = alpha_to_hyteg_level(alpha_tet,            //
                                           alpha_min_threshold,  //
                                           alpha_max_threshold,  //
                                           max_refinement_L);    //

        const int     hteg_num_tetrahedra = get_hyteg_num_tetrahedra(L);
        const real_t* x_hyteg             = get_hyteg_x(L);
        const real_t* y_hyteg             = get_hyteg_y(L);
        const real_t* z_hyteg             = get_hyteg_z(L);
        const int*    categories_hyteg    = get_hyteg_categories(L);
        const int*    v0_array_hyteg      = get_hyteg_v0(L);
        const int*    v1_array_hyteg      = get_hyteg_v1(L);
        const int*    v2_array_hyteg      = get_hyteg_v2(L);
        const int*    v3_array_hyteg      = get_hyteg_v3(L);

        for (int tet_i = 0; tet_i < hteg_num_tetrahedra; tet_i++) {  //
            //
            const int category = categories_hyteg[tet_i];

            const int v0 = v0_array_hyteg[tet_i];
            const int v1 = v1_array_hyteg[tet_i];
            const int v2 = v2_array_hyteg[tet_i];
            const int v3 = v3_array_hyteg[tet_i];

            // Coordinates of the tetrahedron in the reference space
            const real_type px0 = x_hyteg[v0];
            const real_type px1 = x_hyteg[v1];
            const real_type px2 = x_hyteg[v2];
            const real_type px3 = x_hyteg[v3];

            const real_type py0 = y_hyteg[v0];
            const real_type py1 = y_hyteg[v1];
            const real_type py2 = y_hyteg[v2];
            const real_type py3 = y_hyteg[v3];

            const real_type pz0 = z_hyteg[v0];
            const real_type pz1 = z_hyteg[v1];
            const real_type pz2 = z_hyteg[v2];
            const real_type pz3 = z_hyteg[v3];

            const real_t fx0, fx1, fx2, fx3;
            const real_t fy0, fy1, fy2, fy3;
            const real_t fz0, fz1, fz2, fz3;

            // Transform the vertices of the sub-tetrahedron to the physical space
            tet4_transform_v2(x0,     // x-coordinates of the vertices
                              x1,     //
                              x2,     //
                              x3,     //
                              y0,     // y-coordinates of the vertices
                              y1,     //
                              y2,     //
                              y3,     //
                              z0,     // z-coordinates of the vertices
                              z1,     //
                              z2,     //
                              z3,     //
                              px0,    // Vertex of the sub-tetrahedron
                              py0,    //
                              pz0,    //
                              &fx0,   // Output coordinates
                              &fy0,   //
                              &fz0);  //

            tet4_transform_v2(x0,     // x-coordinates of the vertices
                              x1,     //
                              x2,     //
                              x3,     //
                              y0,     // y-coordinates of the vertices
                              y1,     //
                              y2,     //
                              y3,     //
                              z0,     // z-coordinates of the vertices
                              z1,     //
                              z2,     //
                              z3,     //
                              px1,    // Vertex of the sub-tetrahedron
                              py1,    //
                              pz1,    //
                              &fx1,   // Output coordinates
                              &fy1,   //
                              &fz1);  //

            tet4_transform_v2(x0,     // x-coordinates of the vertices
                              x1,     //
                              x2,     //
                              x3,     //
                              y0,     // y-coordinates of the vertices
                              y1,     //
                              y2,     //
                              y3,     //
                              z0,     // z-coordinates of the vertices
                              z1,     //
                              z2,     //
                              z3,     //
                              px2,    // Vertex of the sub-tetrahedron
                              py2,    //
                              pz2,    //
                              &fx2,   // Output coordinates
                              &fy2,   //
                              &fz2);  //

            tet4_transform_v2(x0,     // x-coordinates of the vertices
                              x1,     //
                              x2,     //
                              x3,     //
                              y0,     // y-coordinates of the vertices
                              y1,     //
                              y2,     //
                              y3,     //
                              z0,     // z-coordinates of the vertices
                              z1,     //
                              z2,     //
                              z3,     //
                              px3,    // Vertex of the sub-tetrahedron
                              py3,    //
                              pz3,    //
                              &fx3,   // Output coordinates
                              &fy3,   //
                              &fz3);  //

            real_t det_jacobian = 0.0;

            // Calculate the volume of the HyTeg tetrahedron
            // In the physical space
            const real_type theta_volume = tet4_measure_v2(fx0,
                                                           fx1,
                                                           fx2,
                                                           fx3,
                                                           //
                                                           fy0,
                                                           fy1,
                                                           fy2,
                                                           fy3,
                                                           //
                                                           fz0,
                                                           fz1,
                                                           fz2,
                                                           fz3);

            tet4_resample_tetrahedron_local_hyteg_adjoint(fx0,           // Tetrahedron vertices X-coordinates
                                                          fx1,           //
                                                          fx2,           //
                                                          fx3,           //
                                                          fy0,           // Tetrahedron vertices Y-coordinates
                                                          fy1,           //
                                                          fy2,           //
                                                          fy3,           //
                                                          fz0,           // Tetrahedron vertices Z-coordinates
                                                          fz1,           //
                                                          fz2,           //
                                                          fz3,           //
                                                          theta_volume,  // Volume of the tetrahedron
                                                          wf0,           // Weighted field at the vertices
                                                          wf1,           //
                                                          wf2,           //
                                                          wf3,           //
                                                          ox,            // Origin of the grid
                                                          oy,            //
                                                          oz,            //
                                                          dx,            // Spacing of the grid
                                                          dy,            //
                                                          dz,            //
                                                          stride,        // Stride
                                                          n,             // Size of the grid
                                                          data);         // Output
        }
    }

    RETURN_FROM_FUNCTION(ret);
}  // END OF FUNCTION tet4_resample_field_local_refine_adjoint_hyteg
//////////////////////////////////////////////////////////
