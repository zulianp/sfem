#ifndef RESAMPLE_ADJOINT_MAIN_H
#define RESAMPLE_ADJOINT_MAIN_H

#include "cell_list_3d_map.h"
#include "cell_tet2box.h"
#include "sfem_mesh.h"

int main_adjoint(int argc, char* argv[]);

int main_test_ccell(int argc, char* argv[]);

int main_ccel_test(int argc, char **argv);

int benchmark_cdf_ratio_scan(const side_length_histograms_t *histograms,
                             const boxes_t                  *bounding_boxes_ptr,
                             const mesh_tet_geom_t          *geom,
                             const real_t                    min_grid_x,
                             const real_t                    max_grid_x,
                             const real_t                    min_grid_y,
                             const real_t                    max_grid_y,
                             const real_t                    min_grid_z,
                             const real_t                    max_grid_z,
                             const real_t                    min_ratio,
                             const real_t                    max_ratio,
                             const real_t                    step,
                             const int                       num_z,
                             const int                       num_queries,
                             const char                     *output_dir);

#endif  // RESAMPLE_ADJOINT_MAIN_H