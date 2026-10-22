#pragma once

#include "sfem_dss_bxyze.hpp"

/* Affine scalar-diffusion geometry, without material or quadrature weights:
 *   H_e = det(J_e) * J_e^{-1} * J_e^{-T}.
 *
 * Input elements use the same tensor-local ordering as global_to_bxyze.
 * Elements must be affine, nondegenerate and positively oriented. Affinity
 * and connectivity bounds are caller preconditions. The metric is evaluated
 * at the reference center, consistently with SpectralElementLaplacian.
 *
 * Output: geometry[(eb * 6 + c) * B + b], where
 *   c = 0,...,5 denotes H00, H01, H02, H11, H12, H22;
 *   B = layout.element_block_size, eb = e / B, b = e % B.
 * Padded lanes are zero. Returns failure for invalid metadata or nonfinite /
 * nonpositive geometry; output must be discarded on failure.
 */
ptrdiff_t sfem_dss_bxyze_affine_geometry_n_values(const sfem_dss_bxyze_layout_t *layout);

int sfem_dss_bxyze_affine_geometry(const sfem_dss_bxyze_layout_t *layout,
                                  const idx_t *const *elements,
                                  const geom_t *const *points,
                                  scalar_t *geometry);
