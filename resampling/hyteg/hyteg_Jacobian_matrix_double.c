/**
 * Jacobian matrix calculation for tetrahedron categories
 * Auto-generated code - DO NOT EDIT MANUALLY
 * Data type: double
 */

#include "hyteg_Jacobian_matrix_double.h"
#include <math.h>

void jacobian_matrix_cat0_double(const double x0, const double y0, const double z0,  //
                                 const double x1, const double y1, const double z1,  //
                                 const double x2, const double y2, const double z2,  //
                                 const double x3, const double y3, const double z3,  //
                                 const double L,                                     //
                                 double       J[9]                                   //
) {
    // Calculate 3x3 Jacobian matrix for tetrahedron category 0
    // Matrix stored in row-major order: J[0]=J11, J[1]=J12, J[2]=J13, J[3]=J21, etc.
    J[0] = (-x0 + x1) / L;  // J11
    J[1] = (-x0 + x2) / L;  // J12
    J[2] = (-x0 + x3) / L;  // J13
    J[3] = (-y0 + y1) / L;  // J21
    J[4] = (-y0 + y2) / L;  // J22
    J[5] = (-y0 + y3) / L;  // J23
    J[6] = (-z0 + z1) / L;  // J31
    J[7] = (-z0 + z2) / L;  // J32
    J[8] = (-z0 + z3) / L;  // J33
}

void jacobian_matrix_cat1_double(const double x0, const double y0, const double z0,  //
                                 const double x1, const double y1, const double z1,  //
                                 const double x2, const double y2, const double z2,  //
                                 const double x3, const double y3, const double z3,  //
                                 const double L,                                     //
                                 double       J[9]                                   //
) {
    // Calculate 3x3 Jacobian matrix for tetrahedron category 1
    // Matrix stored in row-major order: J[0]=J11, J[1]=J12, J[2]=J13, J[3]=J21, etc.
    J[0] = (-x1 + x3) / L;            // J11
    J[1] = (-x0 + x3) / L;            // J12
    J[2] = (-x0 - x1 + x2 + x3) / L;  // J13
    J[3] = (-y1 + y3) / L;            // J21
    J[4] = (-y0 + y3) / L;            // J22
    J[5] = (-y0 - y1 + y2 + y3) / L;  // J23
    J[6] = (-z1 + z3) / L;            // J31
    J[7] = (-z0 + z3) / L;            // J32
    J[8] = (-z0 - z1 + z2 + z3) / L;  // J33
}

void jacobian_matrix_cat2_double(const double x0, const double y0, const double z0,  //
                                 const double x1, const double y1, const double z1,  //
                                 const double x2, const double y2, const double z2,  //
                                 const double x3, const double y3, const double z3,  //
                                 const double L,                                     //
                                 double       J[9]                                   //
) {
    // Calculate 3x3 Jacobian matrix for tetrahedron category 2
    // Matrix stored in row-major order: J[0]=J11, J[1]=J12, J[2]=J13, J[3]=J21, etc.
    J[0] = (-x0 + x2) / L;            // J11
    J[1] = (-x0 - x1 + x2 + x3) / L;  // J12
    J[2] = (-x0 + x3) / L;            // J13
    J[3] = (-y0 + y2) / L;            // J21
    J[4] = (-y0 - y1 + y2 + y3) / L;  // J22
    J[5] = (-y0 + y3) / L;            // J23
    J[6] = (-z0 + z2) / L;            // J31
    J[7] = (-z0 - z1 + z2 + z3) / L;  // J32
    J[8] = (-z0 + z3) / L;            // J33
}

void jacobian_matrix_cat3_double(const double x0, const double y0, const double z0,  //
                                 const double x1, const double y1, const double z1,  //
                                 const double x2, const double y2, const double z2,  //
                                 const double x3, const double y3, const double z3,  //
                                 const double L,                                     //
                                 double       J[9]                                   //
) {
    // Calculate 3x3 Jacobian matrix for tetrahedron category 3
    // Matrix stored in row-major order: J[0]=J11, J[1]=J12, J[2]=J13, J[3]=J21, etc.
    J[0] = (-x1 + x2) / L;            // J11
    J[1] = (-x1 + x3) / L;            // J12
    J[2] = (-x0 - x1 + x2 + x3) / L;  // J13
    J[3] = (-y1 + y2) / L;            // J21
    J[4] = (-y1 + y3) / L;            // J22
    J[5] = (-y0 - y1 + y2 + y3) / L;  // J23
    J[6] = (-z1 + z2) / L;            // J31
    J[7] = (-z1 + z3) / L;            // J32
    J[8] = (-z0 - z1 + z2 + z3) / L;  // J33
}

void jacobian_matrix_cat4_double(const double x0, const double y0, const double z0,  //
                                 const double x1, const double y1, const double z1,  //
                                 const double x2, const double y2, const double z2,  //
                                 const double x3, const double y3, const double z3,  //
                                 const double L,                                     //
                                 double       J[9]                                   //
) {
    // Calculate 3x3 Jacobian matrix for tetrahedron category 4
    // Matrix stored in row-major order: J[0]=J11, J[1]=J12, J[2]=J13, J[3]=J21, etc.
    J[0] = (-x2 + x3) / L;  // J11
    J[1] = (-x0 + x3) / L;  // J12
    J[2] = (-x1 + x3) / L;  // J13
    J[3] = (-y2 + y3) / L;  // J21
    J[4] = (-y0 + y3) / L;  // J22
    J[5] = (-y1 + y3) / L;  // J23
    J[6] = (-z2 + z3) / L;  // J31
    J[7] = (-z0 + z3) / L;  // J32
    J[8] = (-z1 + z3) / L;  // J33
}

void jacobian_matrix_cat5_double(const double x0, const double y0, const double z0,  //
                                 const double x1, const double y1, const double z1,  //
                                 const double x2, const double y2, const double z2,  //
                                 const double x3, const double y3, const double z3,  //
                                 const double L,                                     //
                                 double       J[9]                                   //
) {
    // Calculate 3x3 Jacobian matrix for tetrahedron category 5
    // Matrix stored in row-major order: J[0]=J11, J[1]=J12, J[2]=J13, J[3]=J21, etc.
    J[0] = (-x1 + x2) / L;            // J11
    J[1] = (-x0 - x1 + x2 + x3) / L;  // J12
    J[2] = (-x0 + x2) / L;            // J13
    J[3] = (-y1 + y2) / L;            // J21
    J[4] = (-y0 - y1 + y2 + y3) / L;  // J22
    J[5] = (-y0 + y2) / L;            // J23
    J[6] = (-z1 + z2) / L;            // J31
    J[7] = (-z0 - z1 + z2 + z3) / L;  // J32
    J[8] = (-z0 + z2) / L;            // J33
}

void jacobian_matrix_double(const int category, const double x0, const double y0, const double z0, const double x1,
                            const double y1, const double z1, const double x2, const double y2, const double z2, const double x3,
                            const double y3, const double z3, const double L, double J[9]) {
    switch (category) {
        case 0:
            jacobian_matrix_cat0_double(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, L, J);
            break;
        case 1:
            jacobian_matrix_cat1_double(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, L, J);
            break;
        case 2:
            jacobian_matrix_cat2_double(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, L, J);
            break;
        case 3:
            jacobian_matrix_cat3_double(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, L, J);
            break;
        case 4:
            jacobian_matrix_cat4_double(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, L, J);
            break;
        case 5:
            jacobian_matrix_cat5_double(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, L, J);
            break;
        default:
            // Invalid category, set J to zero matrix
            for (int i = 0; i < 9; i++) {
                J[i] = 0.0;
            }
            break;
    }
}
