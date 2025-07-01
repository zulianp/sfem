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
// add_tetrahedron_to_array //////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                   //
add_tetrahedron_to_array_q(const struct tet_vertices* tet_head,       //
                           struct tet_vertices**      tets_out,       //
                           int                        tets_size,      //
                           int*                       tets_capacity,  //
                           const int                  tet_delta_capacity) {            //

    // Check if we need to expand the array
    if (tets_size >= *tets_capacity) {
        *tets_capacity += tet_delta_capacity;

        struct tet_vertices* new_tets = realloc(*tets_out, sizeof(struct tet_vertices) * (*tets_capacity));

        // Check if realloc failed
        if (new_tets == NULL) {
            fprintf(stderr, "ERROR: realloc failed: %s:%d\n", __FILE__, __LINE__);
            return -1;  // Return error code
        }

        *tets_out = new_tets;  // Update the pointer with the new allocation
    }

    // Add the new tetrahedron
    // (*tets_out)[tets_size] = *tet_head;
    memcpy(&(*tets_out)[tets_size], tet_head, sizeof(struct tet_vertices));

    // Return the new size
    return tets_size + 1;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// push_tet_vertices /////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                       //
push_tet_vertices_q(struct sfem_queue*         queue,     //
                    const struct tet_vertices* tets_ref,  //
                    const int                  n_tets) {                   //

    // Push the tetrahedron on the stack
    for (int i = 0; i < n_tets; i++) {
        struct tet_vertices* tet = malloc(sizeof(struct tet_vertices));
        if (tet == NULL) {
            fprintf(stderr, "ERROR: malloc failed: %s:%d\n", __FILE__, __LINE__);
            exit(1);
        }

        tet->x0 = tets_ref[i].x0;
        tet->x1 = tets_ref[i].x1;
        tet->x2 = tets_ref[i].x2;
        tet->x3 = tets_ref[i].x3;
        tet->y0 = tets_ref[i].y0;
        tet->y1 = tets_ref[i].y1;
        tet->y2 = tets_ref[i].y2;
        tet->y3 = tets_ref[i].y3;
        tet->z0 = tets_ref[i].z0;
        tet->z1 = tets_ref[i].z1;
        tet->z2 = tets_ref[i].z2;
        tet->z3 = tets_ref[i].z3;
        tet->w0 = tets_ref[i].w0;
        tet->w1 = tets_ref[i].w1;
        tet->w2 = tets_ref[i].w2;
        tet->w3 = tets_ref[i].w3;

        // Push the tetrahedron on the stack
        sfem_queue_push(queue, tet);

    }  // END for loop over n_tets

}  // end push_tet_vertices

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

    while (flag_loop == 1 && sfem_queue_size(queue) > 0) {
        // Pop the tetrahedron from the queue
        struct tet_vertices* tet_tail = (struct tet_vertices*)sfem_queue_pop(queue);

        if (tet_tail == NULL) {
            fprintf(stderr, "ERROR: sfem_stack_pop failed\n");
            exit(1);
        }

        real_type edges_length[6];

        int vertex_a = -1;
        int vertex_b = -1;

        const real_type max_edges_length =          //
                tet_edge_max_length(tet_tail->x0,   //
                                    tet_tail->y0,   //
                                    tet_tail->z0,   //
                                    tet_tail->x1,   //
                                    tet_tail->y1,   //
                                    tet_tail->z1,   //
                                    tet_tail->x2,   //
                                    tet_tail->y2,   //
                                    tet_tail->z2,   //
                                    tet_tail->x3,   //
                                    tet_tail->y3,   //
                                    tet_tail->z3,   //
                                    &vertex_a,      // Output
                                    &vertex_b,      // Output
                                    edges_length);  // Output

        const real_t alpha_tet           = max_edges_length / dx;
        const real_t max_min_edges_ratio = ratio_abs_max_min(edges_length, 6);

        int degenerated_tet = 0;

        if (max_min_edges_ratio > 2.0 && alpha_tet > alpha_th) {
            degenerated_tetrahedra_cnt++;
            degenerated_tet = 1;
        }

        if (alpha_tet <= alpha_th) {  // The tetrahedron is not refined
            // Resample the tetrahedron
            // No other refinement is needed

            tets_size = add_tetrahedron_to_array_q(tet_tail,             // The tetrahedron to add
                                                   tets_out,             // The array of tetrahedra
                                                   tets_size,            // The current size of the array
                                                   &tets_capacity,       // The current capacity of the array
                                                   tet_delta_capacity);  // The delta capacity of the array

            total_refined_tets++;

            if (tets_size >= max_refined_tets) {
                while (sfem_queue_size(queue) > 0) {
                    struct tet_vertices* tet_loc = (struct tet_vertices*)sfem_queue_pop(queue);

                    tets_size = add_tetrahedron_to_array_q(tet_loc,              // The tetrahedron to add
                                                           tets_out,             // The array of tetrahedra
                                                           tets_size,            // The current size of the array
                                                           &tets_capacity,       // The current capacity of the array
                                                           tet_delta_capacity);  // The delta capacity of the array
                    free(tet_loc);
                    total_refined_tets++;
                }

                flag_loop = 0;
            }

        } else if (degenerated_tet == 1) {
            // Refine the tetrahedron
            n_tets = 2;

            tet_refine_two_edge_vertex(tet_tail->x0,  // Coordinates of the 1st vertex
                                       tet_tail->y0,
                                       tet_tail->z0,
                                       tet_tail->x1,
                                       tet_tail->y1,  // Coordinates of the 2nd vertex
                                       tet_tail->z1,
                                       tet_tail->x2,
                                       tet_tail->y2,
                                       tet_tail->z2,  // Coordinates of the 3rd vertex
                                       tet_tail->x3,
                                       tet_tail->y3,
                                       tet_tail->z3,
                                       tet_tail->w0,  // Weighted field at the vertices
                                       tet_tail->w1,
                                       tet_tail->w2,
                                       tet_tail->w3,
                                       vertex_a,  // The two vertices to refine
                                       vertex_b,
                                       tets_ref);  // Output

            // Push the two tetrahedra on the stack
            push_tet_vertices_q(queue, tets_ref, n_tets);

        } else {
            // Uniformly refine the tetrahedron
            n_tets = 8;

            tet_uniform_refinement(tet_tail->x0,  // Coordinates of the 1st vertex
                                   tet_tail->y0,
                                   tet_tail->z0,
                                   tet_tail->x1,
                                   tet_tail->y1,  // Coordinates of the 2nd vertex
                                   tet_tail->z1,
                                   tet_tail->x2,
                                   tet_tail->y2,
                                   tet_tail->z2,  // Coordinates of the 3rd vertex
                                   tet_tail->x3,
                                   tet_tail->y3,
                                   tet_tail->z3,
                                   tet_tail->w0,  // Weighted field at the vertices
                                   tet_tail->w1,
                                   tet_tail->w2,
                                   tet_tail->w3,
                                   tets_ref);  // Output

            // Push the eight tetrahedra on the stack
            push_tet_vertices_q(queue, tets_ref, n_tets);
        }

        free(tet_tail);
        tet_tail = NULL;

    }  // END while loop (flag_loop == 1 && sfem_queue_size(queue) > 0)

    sfem_queue_clear(queue);
    sfem_queue_destroy(queue);
    queue = NULL;

    free(tets_ref);
    tets_ref = NULL;

    return tets_size;
}