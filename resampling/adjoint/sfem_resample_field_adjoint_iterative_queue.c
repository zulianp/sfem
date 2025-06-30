#include "sfem_queue.h"
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

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_iterative_refinement /////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                      //
tet4_iterative_refinement_queue(const real_type       x0,                // Tetrahedron vertices X-coordinates
                                const real_type       x1,                //
                                const real_type       x2,                //
                                const real_type       x3,                //
                                const real_type       y0,                // Tetrahedron vertices Y-coordinates
                                const real_type       y1,                //
                                const real_type       y2,                //
                                const real_type       y3,                //
                                const real_type       z0,                // Tetrahedron vertices Z-coordinates
                                const real_type       z1,                //
                                const real_type       z2,                //
                                const real_type       z3,                //
                                const real_type       dx,                // Spacing of the grid
                                const real_type       dy,                //
                                const real_type       dz,                //
                                const real_type       wf0,               // Weighted field at the vertices
                                const real_type       wf1,               //
                                const real_type       wf2,               //
                                const real_type       wf3,               //
                                const real_type       alpha_th,          // Threshold for alpha
                                const int             max_refined_tets,  // Maximum number of iterations
                                struct tet_vertices** tets_out) {        // Output

    if (&tets_out != NULL) {
        free(*tets_out);
        *tets_out = NULL;
    }

    int tets_size          = 0;
    int tets_capacity      = 8;
    int tet_delta_capacity = 8;

    *tets_out = malloc(sizeof(struct tet_vertices) * tets_capacity);

    int flag_loop          = 1;
    int n_tets             = 0;
    int total_refined_tets = 0;

    struct tet_vertices* tets_ref                   = malloc(sizeof(struct tet_vertices) * 8);
    int                  degenerated_tetrahedra_cnt = 0;

    sfem_queue_t* queue = sfem_queue_create(2 * max_refined_tets);

    struct tet_vertices* first_tet = malloc(sizeof(struct tet_vertices));
    first_tet->x0                  = x0;
    first_tet->x1                  = x1;
    first_tet->x2                  = x2;
    first_tet->x3                  = x3;
    first_tet->y0                  = y0;
    first_tet->y1                  = y1;
    first_tet->y2                  = y2;
    first_tet->y3                  = y3;
    first_tet->z0                  = z0;
    first_tet->z1                  = z1;
    first_tet->z2                  = z2;
    first_tet->z3                  = z3;
    first_tet->w0                  = wf0;
    first_tet->w1                  = wf1;
    first_tet->w2                  = wf2;
    first_tet->w3                  = wf3;

    sfem_queue_push(queue, first_tet);
}