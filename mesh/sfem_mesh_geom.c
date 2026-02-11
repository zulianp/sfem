#include "sfem_defs.h"
#include "sfem_mesh.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
// mesh_tet_geometry_init
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
mesh_tet_geom_t mesh_tet_geometry_init(const mesh_t *mesh) {
    mesh_tet_geom_t geom = {0};

    geom.ref_mesh     = (mesh_t *)mesh;
    geom.inv_Jacobian = NULL;
    geom.vetices_zero = NULL;

    return geom;
}

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
// mesh_tet_geometry_alloc
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
mesh_tet_geom_t *mesh_tet_geometry_alloc(const mesh_t *mesh) {
    mesh_tet_geom_t *geom = malloc(sizeof(mesh_tet_geom_t));

    if (geom == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for mesh_tet_geom_t\n");
        return NULL;
    }

    *geom = mesh_tet_geometry_init(mesh);

    return geom;
}

////////////////////////////////////////////////
/// mesh_tet_geometry_free
////////////////////////////////////////////////
void mesh_tet_geometry_free(mesh_tet_geom_t *geom) {
    if (geom == NULL) {
        return;
    }

    if (geom->inv_Jacobian) {
        free(geom->inv_Jacobian);
        geom->inv_Jacobian = NULL;
    }

    if (geom->vetices_zero) {
        free(geom->vetices_zero);
        geom->vetices_zero = NULL;
    }

    free(geom);
}

/////////////////////////////////////////////////////////
// tet4_inv_Jacobian ////////////////////////////
/////////////////////////////////////////////////////////
static void                                   //
tet_inv_Jacobian_mesh_geom(const real_t px0,  //
                           const real_t px1,  //
                           const real_t px2,  //
                           const real_t px3,  //
                           const real_t py0,  //
                           const real_t py1,  //
                           const real_t py2,  //
                           const real_t py3,  //
                           const real_t pz0,  //
                           const real_t pz1,  //
                           const real_t pz2,  //
                           const real_t pz3,  //
                           real_t      *J_inv) {   //
    //
    //

    /**
     ****************************************************************************************
    J^{-1} =
    \begin{bmatrix}
    inv_J11 & inv_J12 & inv_J13 \\
    inv_J21 & inv_J22 & inv_J23 \\
    inv_J31 & inv_J32 & inv_J33
    \end{bmatrix}
    *************************************************************************************************
     */

    // Compute the Jacobian matrix components
    const real_t J11 = -px0 + px1;
    const real_t J12 = -px0 + px2;
    const real_t J13 = -px0 + px3;

    const real_t J21 = -py0 + py1;
    const real_t J22 = -py0 + py2;
    const real_t J23 = -py0 + py3;

    const real_t J31 = -pz0 + pz1;
    const real_t J32 = -pz0 + pz2;
    const real_t J33 = -pz0 + pz3;

    // Compute common subexpressions for cofactor matrix
    const real_t J22_J33 = J22 * J33;
    const real_t J23_J32 = J23 * J32;
    const real_t J21_J33 = J21 * J33;
    const real_t J23_J31 = J23 * J31;
    const real_t J21_J32 = J21 * J32;
    const real_t J22_J31 = J22 * J31;
    const real_t J11_J33 = J11 * J33;
    const real_t J13_J31 = J13 * J31;
    const real_t J11_J23 = J11 * J23;
    const real_t J13_J21 = J13 * J21;
    const real_t J11_J32 = J11 * J32;
    const real_t J12_J31 = J12 * J31;
    const real_t J11_J22 = J11 * J22;
    const real_t J12_J21 = J12 * J21;
    const real_t J12_J33 = J12 * J33;
    const real_t J13_J32 = J13 * J32;
    const real_t J12_J23 = J12 * J23;
    const real_t J13_J22 = J13 * J22;

    // Compute cofactor differences (reused in determinant and inverse)
    const real_t cof00 = J22_J33 - J23_J32;
    const real_t cof01 = J23_J31 - J21_J33;
    const real_t cof02 = J21_J32 - J22_J31;
    const real_t cof10 = J13_J32 - J12_J33;
    const real_t cof11 = J11_J33 - J13_J31;
    const real_t cof12 = J12_J31 - J11_J32;
    const real_t cof20 = J12_J23 - J13_J22;
    const real_t cof21 = J13_J21 - J11_J23;
    const real_t cof22 = J11_J22 - J12_J21;

    // Compute the determinant of the Jacobian using cofactors
    const real_t det_J     = J11 * cof00 + J12 * cof01 + J13 * cof02;
    const real_t inv_det_J = 1.0 / det_J;

    // Compute the inverse of the Jacobian matrix using precomputed cofactors
    J_inv[0] = cof00 * inv_det_J;
    J_inv[1] = cof10 * inv_det_J;
    J_inv[2] = cof20 * inv_det_J;
    J_inv[3] = cof01 * inv_det_J;
    J_inv[4] = cof11 * inv_det_J;
    J_inv[5] = cof21 * inv_det_J;
    J_inv[6] = cof02 * inv_det_J;
    J_inv[7] = cof12 * inv_det_J;
    J_inv[8] = cof22 * inv_det_J;
}  // END: tet_inv_Jacobian_mesh_geom

/////////////////////////////////////////////////////////
// tet4_inv_transform_J ////////////////////////////
/////////////////////////////////////////////////////////
void                                                                //
tet_inv_transform_J_mesh_geom(const real_t               *J_inv,    //
                              const real_t                pfx,      //
                              const real_t                pfy,      //
                              const real_t                pfz,      //
                              const real_t                px0,      //
                              const real_t                py0,      //
                              const real_t                pz0,      //
                              real_t *const SFEM_RESTRICT out_x,    //
                              real_t *const SFEM_RESTRICT out_y,    //
                              real_t *const SFEM_RESTRICT out_z) {  //
    //
    //

    /**
     ****************************************************************************************
    \begin{bmatrix}
    out_x \\
    out_y \\
    out_z
    \end{bmatrix}
    =
    J^{-1} \cdot
    \begin{bmatrix}
    pfx - px0 \\
    pfy - py0 \\
    pfz - pz0
    \end{bmatrix}
    *************************************************************************************************
  */

    // Compute the difference between the physical point and the origin
    const real_t dx = pfx - px0;
    const real_t dy = pfy - py0;
    const real_t dz = pfz - pz0;

    // Apply the inverse transformation
    *out_x = J_inv[0] * dx + J_inv[1] * dy + J_inv[2] * dz;
    *out_y = J_inv[3] * dx + J_inv[4] * dy + J_inv[5] * dz;
    *out_z = J_inv[6] * dx + J_inv[7] * dy + J_inv[8] * dz;
}  // END: sfem_resample_field_adjoint_hex_quad

/////////////////////////////////////////////////////////
// is_hex_out_of_tet ////////////////////////////
/////////////////////////////////////////////////////////
bool                                            //
is_point_out_of_tet(const real_t inv_J_tet[9],  //
                    const real_t tet_origin_x,  //
                    const real_t tet_origin_y,  //
                    const real_t tet_origin_z,  //
                    const real_t vertex_x,      //
                    const real_t vertex_y,      //
                    const real_t vertex_z) {    //

    // Precompute inverse Jacobian components for better cache utilization
    const real_t inv_J00 = inv_J_tet[0];
    const real_t inv_J01 = inv_J_tet[1];
    const real_t inv_J02 = inv_J_tet[2];
    const real_t inv_J10 = inv_J_tet[3];
    const real_t inv_J11 = inv_J_tet[4];
    const real_t inv_J12 = inv_J_tet[5];
    const real_t inv_J20 = inv_J_tet[6];
    const real_t inv_J21 = inv_J_tet[7];
    const real_t inv_J22 = inv_J_tet[8];

    // Track if all vertices violate each constraint
    bool negative_x      = true;  // All ref_x < 0
    bool negative_y      = true;  // All ref_y < 0
    bool negative_z      = true;  // All ref_z < 0
    bool all_outside_sum = true;  // All ref_x + ref_y + ref_z > 1

    // Transform hex vertex to tet reference space
    const real_t dx = vertex_x - tet_origin_x;
    const real_t dy = vertex_y - tet_origin_y;
    const real_t dz = vertex_z - tet_origin_z;

    const real_t ref_x = inv_J00 * dx + inv_J01 * dy + inv_J02 * dz;
    const real_t ref_y = inv_J10 * dx + inv_J11 * dy + inv_J12 * dz;
    const real_t ref_z = inv_J20 * dx + inv_J21 * dy + inv_J22 * dz;

    // Update flags - a vertex that satisfies a constraint breaks that flag
    negative_x = negative_x && (ref_x < 0.0);
    negative_y = negative_y && (ref_y < 0.0);
    negative_z = negative_z && (ref_z < 0.0);

    const real_t sum_ref = ref_x + ref_y + ref_z;
    all_outside_sum      = all_outside_sum && (sum_ref > 1.0);

    // If this vertex is inside, we can early exit and say the hex is not completely outside
    if (!negative_x && !negative_y && !negative_z && !all_outside_sum) {
        return false;
    } else {
        return true;
    }

}  // END Function: is_point_out_of_tet

/////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
// mesh_tet_geometry_compute_inv_Jacobian
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
void mesh_tet_geometry_compute_inv_Jacobian(mesh_tet_geom_t *geom) {
    // IMPLEMENT ME
    if (geom == NULL || geom->ref_mesh == NULL) {
        fprintf(stderr, "Error: Invalid input to mesh_tet_geometry_compute_inv_Jacobian\n");
        return;
    }

    geom->inv_Jacobian = malloc(geom->ref_mesh->nelements * 9 * sizeof(real_t));
    if (geom->inv_Jacobian == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for inv_Jacobian\n");
        return;
    }

    geom->vetices_zero = malloc(geom->ref_mesh->nelements * 3 * sizeof(real_t));
    if (geom->vetices_zero == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for vetices_zero\n");
        free(geom->inv_Jacobian);
        geom->inv_Jacobian = NULL;
        return;
    }

    const ptrdiff_t num_elements = (ptrdiff_t)(geom->ref_mesh->nelements);

    geom_t **xyz   = geom->ref_mesh->points;
    idx_t  **elems = geom->ref_mesh->elements;

    for (ptrdiff_t element_i = 0; element_i < num_elements; element_i++) {
        // Get the vertex indices for this element
        idx_t ev[4];

        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][element_i];
        }  // END: for vq

        // Read the coordinates of the vertices of the tetrahedron
        // In the physical space
        const real_t x0_n = xyz[0][ev[0]];
        const real_t x1_n = xyz[0][ev[1]];
        const real_t x2_n = xyz[0][ev[2]];
        const real_t x3_n = xyz[0][ev[3]];

        const real_t y0_n = xyz[1][ev[0]];
        const real_t y1_n = xyz[1][ev[1]];
        const real_t y2_n = xyz[1][ev[2]];
        const real_t y3_n = xyz[1][ev[3]];

        const real_t z0_n = xyz[2][ev[0]];
        const real_t z1_n = xyz[2][ev[1]];
        const real_t z2_n = xyz[2][ev[2]];
        const real_t z3_n = xyz[2][ev[3]];

        real_t *J_inv = &geom->inv_Jacobian[element_i * 9];

        tet_inv_Jacobian_mesh_geom(x0_n,  //
                                   x1_n,
                                   x2_n,
                                   x3_n,
                                   y0_n,
                                   y1_n,
                                   y2_n,
                                   y3_n,
                                   z0_n,
                                   z1_n,
                                   z2_n,
                                   z3_n,
                                   J_inv);

        // Store the coordinates of the first vertex (v0) for potential use in transformations
        geom->vetices_zero[element_i * 3 + 0] = x0_n;
        geom->vetices_zero[element_i * 3 + 1] = y0_n;
        geom->vetices_zero[element_i * 3 + 2] = z0_n;
    }  // END: for element_i
}  // END: mesh_tet_geometry_compute_inv_Jacobian

/////////////////////////////////////////////////////
// get_inv_Jacobian_geom
/////////////////////////////////////////////////////
real_t *get_inv_Jacobian_geom(const mesh_tet_geom_t *geom, ptrdiff_t element_i) {
    if (geom == NULL || geom->inv_Jacobian == NULL) {
        fprintf(stderr, "Error: Invalid input to get_inv_Jacobian_geom\n");
        return NULL;
    }

    if (element_i < 0 || element_i >= geom->ref_mesh->nelements) {
        fprintf(stderr, "Error: element_i out of bounds in get_inv_Jacobian_geom\n");
        return NULL;
    }

    return &geom->inv_Jacobian[element_i * 9];
}

/////////////////////////////////////////////////////
// get_vertices_zero_geom
/////////////////////////////////////////////////////
real_t *get_vertices_zero_geom(const mesh_tet_geom_t *geom, ptrdiff_t element_i) {
    if (geom == NULL || geom->vetices_zero == NULL) {
        fprintf(stderr, "Error: Invalid input to get_vertices_zero_geom\n");
        return NULL;
    }

    if (element_i < 0 || element_i >= geom->ref_mesh->nelements) {
        fprintf(stderr, "Error: element_i out of bounds in get_vertices_zero_geom\n");
        return NULL;
    }

    return &geom->vetices_zero[element_i * 3];
}
