#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// #define real_t  real_t

#include "hyteg.h"
#include "hyteg_Jacobian_matrix_real_t.h"
#include "mass.h"
#include "matrixio_array.h"
#include "quadratures_rule.h"
#include "sfem_base.h"
#include "sfem_resample_field.h"
#include "sfem_resample_field_tet4_math.h"
#include "sfem_stack.h"
#include "tet10_resample_field.h"
#include "tet10_resample_field_V2.h"
#include "tet10_weno.h"

/**
 * @brief Resamples a field from a 10-node tetrahedral mesh back to a structured hexahedral grid with adaptive refinement.
 *
 * @param start_element
 * @param end_element
 * @param nnodes
 * @param elems
 * @param xyz
 * @param n
 * @param stride
 * @param origin
 * @param delta
 * @param weighted_field
 * @param data
 * @return int
 */
int                                                                                                              //
hex8_to_isoparametric_tet10_resample_field_minitet_adjoint(const ptrdiff_t                      start_element,   // Mesh
                                                           const ptrdiff_t                      end_element,     //
                                                           const ptrdiff_t                      nnodes,          //
                                                           const idx_t** const SFEM_RESTRICT    elems,           //
                                                           const geom_t** const SFEM_RESTRICT   xyz,             //
                                                           const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
                                                           const ptrdiff_t* const SFEM_RESTRICT stride,          //
                                                           const geom_t* const SFEM_RESTRICT    origin,          //
                                                           const geom_t* const SFEM_RESTRICT    delta,           //
                                                           const real_t* const SFEM_RESTRICT    weighted_field,  // Input WF
                                                           real_t* const SFEM_RESTRICT          data) {                   // Output
    //
    PRINT_CURRENT_FUNCTION;

    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

    const real_t d_min             = dx < dy ? (dx < dz ? dx : dz) : (dy < dz ? dy : dz);
    const real_t hexahedron_volume = dx * dy * dz;

#if SFEM_LOG_LEVEL >= 5
    printf("============================================================\n");
    printf("= Start: %s: %s:%d \n", __FUNCTION__, __FILE__, __LINE__);
    printf("= Hexahedron volume = %g\n", hexahedron_volume);
    printf("============================================================\n");
#endif

    int degenerated_tetrahedra_cnt = 0;
    int uniform_refine_cnt         = 0;

    // Unit tetrahedron vertices
    const real_t x0_unit = 0.0;
    const real_t x1_unit = 1.0;
    const real_t x2_unit = 0.0;
    const real_t x3_unit = 0.0;

    const real_t y0_unit = 0.0;
    const real_t y1_unit = 0.0;
    const real_t y2_unit = 1.0;
    const real_t y3_unit = 0.0;

    const real_t z0_unit = 0.0;
    const real_t z1_unit = 0.0;
    const real_t z2_unit = 0.0;
    const real_t z3_unit = 1.0;

    real_t J_vec_mini[6][9];  // Jacobian matrices for the 6 categories of tetrahedra for the refined and reference element
    real_t J_phy[9];          // Jacobian matrices for the 6 categories of tetrahedra for the physical current

    real_t hex8_f[8];
    real_t tet10_f[10];

    for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++) {
        // ISOPARAMETRIC
        geom_t x[10], y[10], z[10];
        idx_t  ev[10];

        real_t hex8_f[8];
        real_t coeffs[8];

        real_t tet10_f[10];
        // real_t element_field[10];

        // loop over the 4 vertices of the tetrahedron
        for (int v = 0; v < 10; ++v) {
            ev[v] = elems[v][element_i];
        }

        // ISOPARAMETRIC
        for (int v = 0; v < 10; ++v) {
            x[v] = (geom_t)(xyz[0][ev[v]]);  // x-coordinates
            y[v] = (geom_t)(xyz[1][ev[v]]);  // y-coordinates
            z[v] = (geom_t)(xyz[2][ev[v]]);  // z-coordinates
        }

        // memset(element_field, 0, 10 * sizeof(real_t));

        // set to zero the element field
        // memset(element_field, 0, 10 * sizeof(real_t));

        const real_t wf_tet10[10] = {weighted_field[ev[0]],
                                     weighted_field[ev[1]],
                                     weighted_field[ev[2]],
                                     weighted_field[ev[3]],
                                     weighted_field[ev[4]],
                                     weighted_field[ev[5]],
                                     weighted_field[ev[6]],
                                     weighted_field[ev[7]],
                                     weighted_field[ev[8]],
                                     weighted_field[ev[9]]};
    }

    RETURN_FROM_FUNCTION(0);
}