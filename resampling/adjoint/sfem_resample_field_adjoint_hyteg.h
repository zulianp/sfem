#ifndef __SFEM_RESAMPLE_FIELD_ADJOINT_HYTEG_H__
#define __SFEM_RESAMPLE_FIELD_ADJOINT_HYTEG_H__

typedef struct {
    float        alpha_min_threshold;
    float        alpha_max_threshold;
    unsigned int min_refinement_L;
    unsigned int max_refinement_L;
} mini_tet_parameters_t;

int                                                           //
alpha_to_hyteg_level(const real_t       alpha,                //
                     const real_t       alpha_min_threshold,  //
                     const real_t       alpha_max_threshold,  //
                     const unsigned int min_refinement_L,     //
                     const unsigned int max_refinement_L);    //

#endif  // __SFEM_RESAMPLE_FIELD_ADJOINT_HYTEG_H__