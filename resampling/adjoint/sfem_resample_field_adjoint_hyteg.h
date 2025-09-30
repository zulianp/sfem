#ifndef __SFEM_RESAMPLE_FIELD_ADJOINT_HYTEG_H__
#define __SFEM_RESAMPLE_FIELD_ADJOINT_HYTEG_H__

typedef struct {
    real_t       alpha_min_threshold;
    real_t       alpha_max_threshold;
    unsigned int min_refinement_L;
    unsigned int max_refinement_L;
} mini_tet_parameters_t;

int                                                           //
alpha_to_hyteg_level(const real_t       alpha,                //
                     const real_t       alpha_min_threshold,  //
                     const real_t       alpha_max_threshold,  //
                     const unsigned int min_refinement_L,     //
                     const unsigned int max_refinement_L);    //

real_t                                      //
make_Jacobian_matrix_tet(const real_t fx0,  // Tetrahedron vertices X-coordinates
                         const real_t fx1,  //
                         const real_t fx2,  //
                         const real_t fx3,  //
                         const real_t fy0,  // Tetrahedron vertices Y-coordinates
                         const real_t fy1,  //
                         const real_t fy2,  //
                         const real_t fy3,  //
                         const real_t fz0,  // Tetrahedron vertices Z-coordinates
                         const real_t fz1,  //
                         const real_t fz2,  //
                         const real_t fz3,
                         real_t       J[9]);  // Jacobian matrix

#endif  // __SFEM_RESAMPLE_FIELD_ADJOINT_HYTEG_H__