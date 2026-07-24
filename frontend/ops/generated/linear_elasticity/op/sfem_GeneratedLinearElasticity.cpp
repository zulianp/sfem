#include "sfem_GeneratedLinearElasticity.hpp"
#include "sfem_GeneratedLinearElasticity_c_abi.hpp"
#include "packed_thread_scratch.hpp"
#include "smesh_env.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_kernel_data.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

extern "C" {
int linear_elasticity_tri3_tri3_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, real_t *);
int linear_elasticity_tri3_tri3_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_tri3_tri3_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_tri3_tri3_objective_steps_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_tri3_tri3_objective_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, real_t *);
int linear_elasticity_tri3_tri3_gradient_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_tri3_tri3_gradient_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_tri3_tri3_apply_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_tri3_tri3_apply_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_tri3_tri3_objective_steps_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_tri6_tri6_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, real_t *);
int linear_elasticity_tri6_tri6_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_tri6_tri6_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_tri6_tri6_objective_steps_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_tri6_tri6_objective_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, real_t *);
int linear_elasticity_tri6_tri6_gradient_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_tri6_tri6_gradient_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_tri6_tri6_apply_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_tri6_tri6_apply_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_tri6_tri6_objective_steps_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_quad4_quad4_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, real_t *);
int linear_elasticity_quad4_quad4_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_quad4_quad4_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_quad4_quad4_objective_steps_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_quad4_quad4_objective_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, real_t *);
int linear_elasticity_quad4_quad4_gradient_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_quad4_quad4_gradient_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_quad4_quad4_apply_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_quad4_quad4_apply_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_quad4_quad4_objective_steps_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_proteus_quad4_proteus_quad4_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, real_t *);
int linear_elasticity_proteus_quad4_proteus_quad4_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_proteus_quad4_proteus_quad4_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_proteus_quad4_proteus_quad4_objective_steps_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_proteus_quad4_proteus_quad4_objective_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, real_t *);
int linear_elasticity_proteus_quad4_proteus_quad4_gradient_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_proteus_quad4_proteus_quad4_gradient_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_proteus_quad4_proteus_quad4_apply_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_proteus_quad4_proteus_quad4_apply_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int linear_elasticity_proteus_quad4_proteus_quad4_objective_steps_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_tet4_tet4_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int linear_elasticity_tet4_tet4_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_tet4_tet4_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_tet4_tet4_objective_steps_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_tet4_tet4_objective_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int linear_elasticity_tet4_tet4_gradient_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_tet4_tet4_gradient_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_tet4_tet4_apply_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_tet4_tet4_apply_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_tet4_tet4_objective_steps_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_tet10_tet10_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int linear_elasticity_tet10_tet10_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_tet10_tet10_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_tet10_tet10_objective_steps_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_tet10_tet10_objective_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int linear_elasticity_tet10_tet10_gradient_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_tet10_tet10_gradient_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_tet10_tet10_apply_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_tet10_tet10_apply_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_tet10_tet10_objective_steps_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_hex8_hex8_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int linear_elasticity_hex8_hex8_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_hex8_hex8_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_hex8_hex8_objective_steps_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_hex8_hex8_objective_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int linear_elasticity_hex8_hex8_gradient_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_hex8_hex8_gradient_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_hex8_hex8_apply_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_hex8_hex8_apply_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_hex8_hex8_objective_steps_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_hex27_hex27_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int linear_elasticity_hex27_hex27_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_hex27_hex27_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_hex27_hex27_objective_steps_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_hex27_hex27_objective_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int linear_elasticity_hex27_hex27_gradient_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_hex27_hex27_gradient_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_hex27_hex27_apply_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_hex27_hex27_apply_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_hex27_hex27_objective_steps_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_proteus_hex8_proteus_hex8_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int linear_elasticity_proteus_hex8_proteus_hex8_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex8_proteus_hex8_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex8_proteus_hex8_objective_steps_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_proteus_hex8_proteus_hex8_objective_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int linear_elasticity_proteus_hex8_proteus_hex8_gradient_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex8_proteus_hex8_gradient_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex8_proteus_hex8_apply_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex8_proteus_hex8_apply_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex8_proteus_hex8_objective_steps_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_proteus_hex27_proteus_hex27_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int linear_elasticity_proteus_hex27_proteus_hex27_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex27_proteus_hex27_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex27_proteus_hex27_objective_steps_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_proteus_hex27_proteus_hex27_objective_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int linear_elasticity_proteus_hex27_proteus_hex27_gradient_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex27_proteus_hex27_gradient_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex27_proteus_hex27_apply_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex27_proteus_hex27_apply_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex27_proteus_hex27_objective_steps_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_proteus_hex64_proteus_hex64_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int linear_elasticity_proteus_hex64_proteus_hex64_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex64_proteus_hex64_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex64_proteus_hex64_objective_steps_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_proteus_hex64_proteus_hex64_objective_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int linear_elasticity_proteus_hex64_proteus_hex64_gradient_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex64_proteus_hex64_gradient_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex64_proteus_hex64_apply_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex64_proteus_hex64_apply_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex64_proteus_hex64_objective_steps_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_proteus_hex125_proteus_hex125_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int linear_elasticity_proteus_hex125_proteus_hex125_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex125_proteus_hex125_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex125_proteus_hex125_objective_steps_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_proteus_hex125_proteus_hex125_objective_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int linear_elasticity_proteus_hex125_proteus_hex125_gradient_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex125_proteus_hex125_gradient_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex125_proteus_hex125_apply_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex125_proteus_hex125_apply_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex125_proteus_hex125_objective_steps_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_proteus_hex729_proteus_hex729_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int linear_elasticity_proteus_hex729_proteus_hex729_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex729_proteus_hex729_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex729_proteus_hex729_objective_steps_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
int linear_elasticity_proteus_hex729_proteus_hex729_objective_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int linear_elasticity_proteus_hex729_proteus_hex729_gradient_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex729_proteus_hex729_gradient_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex729_proteus_hex729_apply_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex729_proteus_hex729_apply_affine_mesh_soa_aos_unit(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int linear_elasticity_proteus_hex729_proteus_hex729_objective_steps_affine_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const geom_t *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, real_t *);
}

namespace sfem {
    namespace {
        void seed_parameters(Parameters &parameters) {
            parameters.set_value("mu", 1);
            parameters.set_value("lmbda", 1);
        }

        void seed_material(MultiDomainOp &domains) {
            for (auto &entry : domains.domains()) {
                seed_parameters(*entry.second.parameters);
            }
        }

        struct AffineOption {
            const char *name;
            bool       *flag;
        };

        inline bool set_affine_option(const std::string &name,
                                      const bool val,
                                      const AffineOption *const options,
                                      const int n_options) {
            if (name == "ASSUME_AFFINE" || name == "assume_affine") {
                for (int i = 0; i < n_options; ++i) {
                    *options[i].flag = val;
                }
                return true;
            }
            bool matched = false;
            for (int i = 0; i < n_options; ++i) {
                if (name == options[i].name) {
                    *options[i].flag = val;
                    matched = true;
                }
            }
            return matched;
        }

        void material_defaults(real_t *const values) {
            values[0] = 1;
            values[1] = 1;
        }

#ifdef SFEM_ENABLE_RYAML
        constexpr int N_DEFINED_MATERIAL_PARAMETERS = 2;
        constexpr int N_MATERIAL_PARAMETERS = 2;
        static const char *const MATERIAL_PARAMETER_NAMES[N_MATERIAL_PARAMETERS] = {"mu", "lmbda"};

        bool yaml_read_real(const ryml::ConstNodeRef &node,
                            const char *const key,
                            real_t &value) {
            if (!node.has_child(key)) {
                return false;
            }
            node[key] >> value;
            return true;
        }

        bool yaml_read_parameter(const ryml::ConstNodeRef &node,
                                 const char *const key,
                                 real_t &value) {
            if (yaml_read_real(node, key, value)) {
                return true;
            }
            if (node.has_child("parameters") &&
                yaml_read_real(node["parameters"], key, value)) {
                return true;
            }
            if (node.has_child("material") &&
                yaml_read_real(node["material"], key, value)) {
                return true;
            }
            return false;
        }

        std::string yaml_read_string(const ryml::ConstNodeRef &node) {
            const auto value = node.val();
            return std::string(value.str, value.len);
        }

        void copy_material_parameters(const real_t *const src,
                                      real_t *const dst) {
            for (int i = 0; i < N_MATERIAL_PARAMETERS; ++i) {
                dst[i] = src[i];
            }
        }

        bool material_from_yaml(const ryml::ConstNodeRef &node,
                                const real_t *const base,
                                real_t *const values) {
            copy_material_parameters(base, values);
            bool changed = false;
            for (int i = 0; i < N_DEFINED_MATERIAL_PARAMETERS; ++i) {
                changed |= yaml_read_parameter(node,
                                               MATERIAL_PARAMETER_NAMES[i],
                                               values[i]);
            }
            return changed;
        }

        void set_material(MultiDomainOp &domains,
                          const real_t *const values) {
            for (auto &entry : domains.domains()) {
                for (int i = 0; i < N_DEFINED_MATERIAL_PARAMETERS; ++i) {
                    entry.second.parameters->set_value(MATERIAL_PARAMETER_NAMES[i],
                                                       values[i]);
                }
            }
        }

        void set_material_in_block(MultiDomainOp &domains,
                                   const std::string &block_name,
                                   const real_t *const values) {
            for (int i = 0; i < N_DEFINED_MATERIAL_PARAMETERS; ++i) {
                domains.set_value_in_block(block_name,
                                           MATERIAL_PARAMETER_NAMES[i],
                                           values[i]);
            }
        }

        bool yaml_read_bool(const ryml::ConstNodeRef &node,
                            const char *const key,
                            bool &value) {
            if (!node.has_child(key)) {
                return false;
            }
            int raw = value ? 1 : 0;
            node[key] >> raw;
            value = raw != 0;
            return true;
        }

        inline void read_affine_options(const ryml::ConstNodeRef &node,
                                        const AffineOption *const options,
                                        const int n_options) {
            bool all = true;
            for (int i = 0; i < n_options; ++i) {
                all = all && *options[i].flag;
            }
            if (yaml_read_bool(node, "ASSUME_AFFINE", all) ||
                yaml_read_bool(node, "assume_affine", all)) {
                for (int i = 0; i < n_options; ++i) {
                    *options[i].flag = all;
                }
            }
            for (int i = 0; i < n_options; ++i) {
                yaml_read_bool(node, options[i].name, *options[i].flag);
            }
        }
#endif  // SFEM_ENABLE_RYAML

        smesh::block_idx_t block_id_for_domain(const smesh::Mesh &mesh,
                                               const smesh::Mesh::Block &block) {
            for (size_t i = 0; i < mesh.n_blocks(); ++i) {
                if (mesh.block(i).get() == &block) {
                    return static_cast<smesh::block_idx_t>(i);
                }
            }
            SFEM_ERROR("GeneratedLinearElasticity: mesh block pointer not found in mesh.blocks()\n");
            return 0;
        }

        int packed_block_id_for_domain(const FunctionSpace::PackedMesh &packed,
                                       const smesh::Mesh::Block &block) {
            for (ptrdiff_t i = 0; i < packed.n_blocks(); ++i) {
                if (packed.block_name(i) == block.name()) {
                    return static_cast<int>(i);
                }
            }
            return -1;
        }

        struct AffineGeometryCache {
            std::shared_ptr<smesh::JacobianAdjugateAndDeterminant> jacobian_soa;
            std::shared_ptr<smesh::JacobianAdjugateAndDeterminant> jacobian_aos;
        };

        int cache_affine_geometry(const std::shared_ptr<FunctionSpace> &space,
                                  MultiDomainOp &domains) {
            auto mesh = space->mesh_ptr();
            const bool needs_jacobian_aos =
                    true ||
                    true;
            for (auto &entry : domains.domains()) {
                const smesh::block_idx_t block_id =
                        block_id_for_domain(*mesh, *entry.second.block);
                auto cache = std::make_shared<AffineGeometryCache>();
                cache->jacobian_soa = smesh::JacobianAdjugateAndDeterminant::create_SoA(
                        mesh, smesh::MEMORY_SPACE_HOST, block_id);
                if (!cache->jacobian_soa) {
                    return SFEM_FAILURE;
                }
                if (needs_jacobian_aos) {
                    cache->jacobian_aos = smesh::JacobianAdjugateAndDeterminant::create_AoS(
                            mesh, smesh::MEMORY_SPACE_HOST, block_id);
                    if (!cache->jacobian_aos) {
                        return SFEM_FAILURE;
                    }
                }
                entry.second.user_data = std::static_pointer_cast<void>(cache);
            }
            return SFEM_SUCCESS;
        }
    }  // namespace

    class GeneratedLinearElasticity::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::unique_ptr<real_t[]> element_values;
        ptrdiff_t element_capacity{0};
        bool objective_uses_affine{false};
        bool gradient_uses_affine{false};
        bool apply_uses_affine{false};
        bool use_packed_two_pass{false};
        std::vector<SharedBuffer<real_t>> packed_ghost_buf;
    };

    std::unique_ptr<Op> GeneratedLinearElasticity::create(const std::shared_ptr<FunctionSpace> &space) {
        if (space->block_size() != space->mesh_ptr()->spatial_dimension()) {
            SFEM_ERROR("GeneratedLinearElasticity requires block_size=spatial_dimension\n");
            return nullptr;
        }
        auto op = std::make_unique<GeneratedLinearElasticity>(space);
        op->initialize();
        return op;
    }

    GeneratedLinearElasticity::GeneratedLinearElasticity(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    GeneratedLinearElasticity::~GeneratedLinearElasticity() = default;

    ptrdiff_t GeneratedLinearElasticity::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t GeneratedLinearElasticity::n_dofs_image() const { return impl_->space->n_dofs(); }

    double GeneratedLinearElasticity::flops_value() const {
        double total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
                case smesh::TRI3: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_tri3_tri3_objective_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_tri3_tri3_objective_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::TRI6: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_tri6_tri6_objective_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_tri6_tri6_objective_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_quad4_quad4_objective_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_quad4_quad4_objective_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_proteus_quad4_proteus_quad4_objective_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_proteus_quad4_proteus_quad4_objective_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::TET4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_tet4_tet4_objective_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_tet4_tet4_objective_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::TET10: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_tet10_tet10_objective_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_tet10_tet10_objective_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_hex8_hex8_objective_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_hex8_hex8_objective_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_hex27_hex27_objective_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_hex27_hex27_objective_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_proteus_hex8_proteus_hex8_objective_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_proteus_hex8_proteus_hex8_objective_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_proteus_hex27_proteus_hex27_objective_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_proteus_hex27_proteus_hex27_objective_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX64: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_proteus_hex64_proteus_hex64_objective_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_proteus_hex64_proteus_hex64_objective_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX125: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_proteus_hex125_proteus_hex125_objective_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_proteus_hex125_proteus_hex125_objective_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX729: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_proteus_hex729_proteus_hex729_objective_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_proteus_hex729_proteus_hex729_objective_soa_diagnostics(), nelements);
                    break;
                }
                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    size_t GeneratedLinearElasticity::memory_traffic_bytes_value() const {
        size_t total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
                case smesh::TRI3: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_tri3_tri3_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_tri3_tri3_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::TRI6: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_tri6_tri6_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_tri6_tri6_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_quad4_quad4_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_quad4_quad4_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_proteus_quad4_proteus_quad4_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_proteus_quad4_proteus_quad4_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::TET4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_tet4_tet4_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_tet4_tet4_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::TET10: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_tet10_tet10_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_tet10_tet10_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_hex8_hex8_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_hex8_hex8_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_hex27_hex27_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_hex27_hex27_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_proteus_hex8_proteus_hex8_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_proteus_hex8_proteus_hex8_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_proteus_hex27_proteus_hex27_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_proteus_hex27_proteus_hex27_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX64: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_proteus_hex64_proteus_hex64_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_proteus_hex64_proteus_hex64_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX125: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_proteus_hex125_proteus_hex125_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_proteus_hex125_proteus_hex125_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX729: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->objective_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_proteus_hex729_proteus_hex729_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_proteus_hex729_proteus_hex729_objective_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    double GeneratedLinearElasticity::flops_gradient() const {
        double total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
                case smesh::TRI3: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_tri3_tri3_gradient_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_tri3_tri3_gradient_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::TRI6: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_tri6_tri6_gradient_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_tri6_tri6_gradient_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_quad4_quad4_gradient_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_quad4_quad4_gradient_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_proteus_quad4_proteus_quad4_gradient_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_proteus_quad4_proteus_quad4_gradient_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::TET4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_tet4_tet4_gradient_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_tet4_tet4_gradient_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::TET10: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_tet10_tet10_gradient_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_tet10_tet10_gradient_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_hex8_hex8_gradient_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_hex8_hex8_gradient_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_hex27_hex27_gradient_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_hex27_hex27_gradient_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_proteus_hex8_proteus_hex8_gradient_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_proteus_hex8_proteus_hex8_gradient_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_proteus_hex27_proteus_hex27_gradient_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_proteus_hex27_proteus_hex27_gradient_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX64: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_proteus_hex64_proteus_hex64_gradient_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_proteus_hex64_proteus_hex64_gradient_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX125: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_proteus_hex125_proteus_hex125_gradient_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_proteus_hex125_proteus_hex125_gradient_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX729: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_proteus_hex729_proteus_hex729_gradient_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_proteus_hex729_proteus_hex729_gradient_soa_diagnostics(), nelements);
                    break;
                }
                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    size_t GeneratedLinearElasticity::memory_traffic_bytes_gradient() const {
        size_t total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
                case smesh::TRI3: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_tri3_tri3_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_tri3_tri3_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::TRI6: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_tri6_tri6_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_tri6_tri6_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_quad4_quad4_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_quad4_quad4_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_proteus_quad4_proteus_quad4_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_proteus_quad4_proteus_quad4_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::TET4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_tet4_tet4_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_tet4_tet4_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::TET10: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_tet10_tet10_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_tet10_tet10_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_hex8_hex8_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_hex8_hex8_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_hex27_hex27_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_hex27_hex27_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_proteus_hex8_proteus_hex8_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_proteus_hex8_proteus_hex8_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_proteus_hex27_proteus_hex27_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_proteus_hex27_proteus_hex27_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX64: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_proteus_hex64_proteus_hex64_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_proteus_hex64_proteus_hex64_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX125: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_proteus_hex125_proteus_hex125_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_proteus_hex125_proteus_hex125_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX729: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->gradient_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_proteus_hex729_proteus_hex729_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_proteus_hex729_proteus_hex729_gradient_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    double GeneratedLinearElasticity::flops_apply() const {
        double total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
                case smesh::TRI3: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_tri3_tri3_apply_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_tri3_tri3_apply_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::TRI6: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_tri6_tri6_apply_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_tri6_tri6_apply_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_quad4_quad4_apply_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_quad4_quad4_apply_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_proteus_quad4_proteus_quad4_apply_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_proteus_quad4_proteus_quad4_apply_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::TET4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_tet4_tet4_apply_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_tet4_tet4_apply_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::TET10: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_tet10_tet10_apply_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_tet10_tet10_apply_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_hex8_hex8_apply_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_hex8_hex8_apply_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_hex27_hex27_apply_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_hex27_hex27_apply_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_proteus_hex8_proteus_hex8_apply_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_proteus_hex8_proteus_hex8_apply_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_proteus_hex27_proteus_hex27_apply_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_proteus_hex27_proteus_hex27_apply_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX64: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_proteus_hex64_proteus_hex64_apply_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_proteus_hex64_proteus_hex64_apply_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX125: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_proteus_hex125_proteus_hex125_apply_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_proteus_hex125_proteus_hex125_apply_soa_diagnostics(), nelements);
                    break;
                }
                case smesh::PROTEUS_HEX729: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(linear_elasticity_proteus_hex729_proteus_hex729_apply_soa_diagnostics(), nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(linear_elasticity_proteus_hex729_proteus_hex729_apply_soa_diagnostics(), nelements);
                    break;
                }
                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    size_t GeneratedLinearElasticity::memory_traffic_bytes_apply() const {
        size_t total = 0;
        if (!impl_->domains) {
            return total;
        }

        impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
                case smesh::TRI3: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_tri3_tri3_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_tri3_tri3_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::TRI6: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_tri6_tri6_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_tri6_tri6_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_quad4_quad4_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_quad4_quad4_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_QUAD4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_proteus_quad4_proteus_quad4_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_proteus_quad4_proteus_quad4_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::TET4: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_tet4_tet4_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_tet4_tet4_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::TET10: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_tet10_tet10_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_tet10_tet10_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_hex8_hex8_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_hex8_hex8_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_hex27_hex27_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_hex27_hex27_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX8: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_proteus_hex8_proteus_hex8_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_proteus_hex8_proteus_hex8_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX27: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_proteus_hex27_proteus_hex27_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_proteus_hex27_proteus_hex27_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX64: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_proteus_hex64_proteus_hex64_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_proteus_hex64_proteus_hex64_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX125: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_proteus_hex125_proteus_hex125_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_proteus_hex125_proteus_hex125_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                case smesh::PROTEUS_HEX729: {
                    const ptrdiff_t nelements = domain.block->n_elements();
                    total += impl_->apply_uses_affine ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(linear_elasticity_proteus_hex729_proteus_hex729_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(linear_elasticity_proteus_hex729_proteus_hex729_apply_soa_diagnostics(), nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));
                    break;
                }
                default:
                    break;
            }
            return SFEM_SUCCESS;
        });

        return total;
    }

    int GeneratedLinearElasticity::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        auto mesh = impl_->space->mesh_ptr();
        const bool needs_affine_geometry =
                impl_->objective_uses_affine ||
                impl_->gradient_uses_affine ||
                impl_->apply_uses_affine;
        for (auto &entry : impl_->domains->domains()) {
            seed_parameters(*entry.second.parameters);
            impl_->element_capacity =
                    std::max(impl_->element_capacity, entry.second.block->n_elements());
            if (needs_affine_geometry) {
                const smesh::block_idx_t block_id =
                        block_id_for_domain(*mesh, *entry.second.block);
                auto cache = std::make_shared<AffineGeometryCache>();
                cache->jacobian_soa = smesh::JacobianAdjugateAndDeterminant::create_SoA(
                        mesh, smesh::MEMORY_SPACE_HOST, block_id);
                if (!cache->jacobian_soa) {
                    return SFEM_FAILURE;
                }
                if ((impl_->gradient_uses_affine && true) ||
                    (impl_->apply_uses_affine && true)) {
                    cache->jacobian_aos = smesh::JacobianAdjugateAndDeterminant::create_AoS(
                            mesh, smesh::MEMORY_SPACE_HOST, block_id);
                    if (!cache->jacobian_aos) {
                        return SFEM_FAILURE;
                    }
                }
                entry.second.user_data = std::static_pointer_cast<void>(cache);
            }
        }
        impl_->element_values.reset(new real_t[impl_->element_capacity]);
        impl_->use_packed_two_pass = smesh::Env::read("SFEM_PACKED_TWO_PASS", false);
        if (impl_->space->has_packed_mesh()) {
            auto packed = impl_->space->packed_mesh();
            const ptrdiff_t max_nodes_per_pack = packed->max_nodes_per_pack();
            const int dim = impl_->space->mesh_ptr()->spatial_dimension();
            const size_t scratch_size = (size_t)dim * (size_t)max_nodes_per_pack;
            sfem::codegen::prealloc_thread_scratch<real_t>(0, scratch_size);
            sfem::codegen::prealloc_thread_scratch<real_t>(1, scratch_size);
            sfem::codegen::prealloc_thread_scratch<real_t>(2, scratch_size);
            sfem::codegen::prealloc_thread_scratch<real_t>(3, scratch_size);
            impl_->packed_ghost_buf.resize((size_t)packed->n_blocks());
            for (int b = 0; b < packed->n_blocks(); ++b) {
                const ptrdiff_t n_ghost = packed->n_ghost_entries(b);
                const ptrdiff_t n_slots = (n_ghost > 0 ? n_ghost : 1) * (ptrdiff_t)dim;
                impl_->packed_ghost_buf[b] = create_host_buffer<real_t>(n_slots);
            }
        }
        return SFEM_SUCCESS;
    }

    int GeneratedLinearElasticity::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::gradient");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *adjugate_aos = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->gradient_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache || !cache->jacobian_soa) {
                    SFEM_ERROR("GeneratedLinearElasticity affine gradient requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        cache->jacobian_soa->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        cache->jacobian_soa->jacobian_determinant()->data());
                if (true) {
                    if (!cache->jacobian_aos) {
                        SFEM_ERROR("GeneratedLinearElasticity affine gradient requires cached AoS geometry\n");
                        return SFEM_FAILURE;
                    }
                    adjugate_aos = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_adjugate_AoS()->data());
                    determinant = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_determinant()->data());
                }
            }
            if (impl_->gradient_uses_affine && impl_->space->has_packed_mesh()) {
                auto packed = impl_->space->packed_mesh();
                const int packed_block = packed_block_id_for_domain(*packed, *domain.block);
                if (packed_block >= 0) {
                    auto packed_elements = packed->elements(packed_block);
                    auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);
                    auto n_shared_nodes = packed->n_shared(packed_block);
                    auto ghost_ptr = packed->ghost_ptr(packed_block);
                    auto ghost_idx = packed->ghost_idx(packed_block);
                    auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);
                    auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);
                    auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);
                    const int dim = mesh->spatial_dimension();
                    if (dim == 2) {
                        if (impl_->use_packed_two_pass) {
                            return linear_elasticity_gradient_packed_two_pass_2d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, out + 0, out + 1);
                        }
                        return linear_elasticity_gradient_packed_2d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, out + 0, out + 1);
                    }
                    else if (dim == 3) {
                        if (impl_->use_packed_two_pass) {
                            return linear_elasticity_gradient_packed_two_pass_3d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                        }
                        return linear_elasticity_gradient_packed_3d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                    }
                }
            }
            if (!impl_->gradient_uses_affine && impl_->space->has_packed_mesh()) {
                auto packed = impl_->space->packed_mesh();
                const int packed_block = packed_block_id_for_domain(*packed, *domain.block);
                if (packed_block >= 0) {
                    auto packed_elements = packed->elements(packed_block);
                    auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);
                    auto n_shared_nodes = packed->n_shared(packed_block);
                    auto ghost_ptr = packed->ghost_ptr(packed_block);
                    auto ghost_idx = packed->ghost_idx(packed_block);
                    auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);
                    auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);
                    auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);
                    const int dim = mesh->spatial_dimension();
                    if (dim == 2) {
                        if (impl_->use_packed_two_pass) {
                            return linear_elasticity_gradient_packed_two_pass_2d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, out + 0, out + 1);
                        }
                        return linear_elasticity_gradient_packed_2d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, out + 0, out + 1);
                    }
                    else if (dim == 3) {
                        if (impl_->use_packed_two_pass) {
                            return linear_elasticity_gradient_packed_two_pass_3d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                        }
                        return linear_elasticity_gradient_packed_3d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                    }
                }
            }
            switch (domain.element_type) {
                case smesh::TRI3:
                    return impl_->gradient_uses_affine ? linear_elasticity_tri3_tri3_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, out + 0, out + 1) : linear_elasticity_tri3_tri3_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, out + 0, out + 1);
                case smesh::TRI6:
                    return impl_->gradient_uses_affine ? linear_elasticity_tri6_tri6_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, out + 0, out + 1) : linear_elasticity_tri6_tri6_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, out + 0, out + 1);
                case smesh::QUAD4:
                    return impl_->gradient_uses_affine ? linear_elasticity_quad4_quad4_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, out + 0, out + 1) : linear_elasticity_quad4_quad4_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, out + 0, out + 1);
                case smesh::PROTEUS_QUAD4:
                    return impl_->gradient_uses_affine ? linear_elasticity_proteus_quad4_proteus_quad4_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, out + 0, out + 1) : linear_elasticity_proteus_quad4_proteus_quad4_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, out + 0, out + 1);
                case smesh::TET4:
                    if (impl_->gradient_uses_affine) {
                        return adjugate_aos ? linear_elasticity_tet4_tet4_gradient_affine_mesh_soa_aos_unit(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate_aos, determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_tet4_tet4_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                    }
                    return linear_elasticity_tet4_tet4_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                case smesh::TET10:
                    return impl_->gradient_uses_affine ? linear_elasticity_tet10_tet10_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_tet10_tet10_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                case smesh::HEX8:
                    return impl_->gradient_uses_affine ? linear_elasticity_hex8_hex8_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_hex8_hex8_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                case smesh::HEX27:
                    return impl_->gradient_uses_affine ? linear_elasticity_hex27_hex27_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_hex27_hex27_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                case smesh::PROTEUS_HEX8:
                    return impl_->gradient_uses_affine ? linear_elasticity_proteus_hex8_proteus_hex8_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_proteus_hex8_proteus_hex8_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                case smesh::PROTEUS_HEX27:
                    return impl_->gradient_uses_affine ? linear_elasticity_proteus_hex27_proteus_hex27_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_proteus_hex27_proteus_hex27_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                case smesh::PROTEUS_HEX64:
                    return impl_->gradient_uses_affine ? linear_elasticity_proteus_hex64_proteus_hex64_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_proteus_hex64_proteus_hex64_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                case smesh::PROTEUS_HEX125:
                    return impl_->gradient_uses_affine ? linear_elasticity_proteus_hex125_proteus_hex125_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_proteus_hex125_proteus_hex125_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                case smesh::PROTEUS_HEX729:
                    return impl_->gradient_uses_affine ? linear_elasticity_proteus_hex729_proteus_hex729_gradient_affine_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2) : linear_elasticity_proteus_hex729_proteus_hex729_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                default:
                    SFEM_ERROR("GeneratedLinearElasticity does not support element type %d\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

    int GeneratedLinearElasticity::apply(const real_t *const x,
                      const real_t *const h,
                      real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::apply");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *adjugate_aos = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->apply_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache || !cache->jacobian_soa) {
                    SFEM_ERROR("GeneratedLinearElasticity affine hessian action requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        cache->jacobian_soa->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        cache->jacobian_soa->jacobian_determinant()->data());
                if (true) {
                    if (!cache->jacobian_aos) {
                        SFEM_ERROR("GeneratedLinearElasticity affine hessian action requires cached AoS geometry\n");
                        return SFEM_FAILURE;
                    }
                    adjugate_aos = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_adjugate_AoS()->data());
                    determinant = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_determinant()->data());
                }
            }
            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                if (impl_->apply_uses_affine) {
                    if (impl_->space->has_packed_mesh()) {
                        auto packed = impl_->space->packed_mesh();
                        const int packed_block = packed_block_id_for_domain(*packed, *domain.block);
                        if (packed_block >= 0) {
                            auto packed_elements = packed->elements(packed_block);
                            auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);
                            auto n_shared_nodes = packed->n_shared(packed_block);
                            auto ghost_ptr = packed->ghost_ptr(packed_block);
                            auto ghost_idx = packed->ghost_idx(packed_block);
                            auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);
                            auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);
                            auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);
                            if (impl_->use_packed_two_pass) {
                                return linear_elasticity_apply_packed_two_pass_2d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, h + 0, h + 1, 2, out + 0, out + 1);
                            }
                            return linear_elasticity_apply_packed_2d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, h + 0, h + 1, 2, out + 0, out + 1);
                        }
                    }
                    return linear_elasticity_apply_2d_affine_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, h + 0, h + 1, 2, out + 0, out + 1);
                }
                if (impl_->space->has_packed_mesh()) {
                    auto packed = impl_->space->packed_mesh();
                    const int packed_block = packed_block_id_for_domain(*packed, *domain.block);
                    if (packed_block >= 0) {
                        auto packed_elements = packed->elements(packed_block);
                        auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);
                        auto n_shared_nodes = packed->n_shared(packed_block);
                        auto ghost_ptr = packed->ghost_ptr(packed_block);
                        auto ghost_idx = packed->ghost_idx(packed_block);
                        auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);
                        auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);
                        auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);
                        if (impl_->use_packed_two_pass) {
                            return linear_elasticity_apply_packed_two_pass_2d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, h + 0, h + 1, 2, out + 0, out + 1);
                        }
                        return linear_elasticity_apply_packed_2d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, h + 0, h + 1, 2, out + 0, out + 1);
                    }
                }
                return linear_elasticity_apply_2d_isoparametric_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, h + 0, h + 1, 2, out + 0, out + 1);
            }
            else if (dim == 3) {
                if (impl_->apply_uses_affine) {
                    if (impl_->space->has_packed_mesh()) {
                        auto packed = impl_->space->packed_mesh();
                        const int packed_block = packed_block_id_for_domain(*packed, *domain.block);
                        if (packed_block >= 0) {
                            auto packed_elements = packed->elements(packed_block);
                            auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);
                            auto n_shared_nodes = packed->n_shared(packed_block);
                            auto ghost_ptr = packed->ghost_ptr(packed_block);
                            auto ghost_idx = packed->ghost_idx(packed_block);
                            auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);
                            auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);
                            auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);
                            if (impl_->use_packed_two_pass) {
                                return linear_elasticity_apply_packed_two_pass_3d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                            }
                            return linear_elasticity_apply_packed_3d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                        }
                    }
                    return linear_elasticity_apply_3d_affine_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                }
                if (impl_->space->has_packed_mesh()) {
                    auto packed = impl_->space->packed_mesh();
                    const int packed_block = packed_block_id_for_domain(*packed, *domain.block);
                    if (packed_block >= 0) {
                        auto packed_elements = packed->elements(packed_block);
                        auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);
                        auto n_shared_nodes = packed->n_shared(packed_block);
                        auto ghost_ptr = packed->ghost_ptr(packed_block);
                        auto ghost_idx = packed->ghost_idx(packed_block);
                        auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);
                        auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);
                        auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);
                        if (impl_->use_packed_two_pass) {
                            return linear_elasticity_apply_packed_two_pass_3d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), packed->n_ghost_entries(packed_block), packed->n_ghost_reduce_rows(packed_block), ghost_reduce_ptr->data(), ghost_reduce_idx->data(), ghost_reduce_dest->data(), impl_->packed_ghost_buf[packed_block]->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                        }
                        return linear_elasticity_apply_packed_3d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                    }
                }
                return linear_elasticity_apply_3d_isoparametric_mesh_soa(domain.element_type, domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
            }
            SFEM_ERROR("linear_elasticity apply does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    int GeneratedLinearElasticity::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::value");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        *out = 0;
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->objective_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache || !cache->jacobian_soa) {
                    SFEM_ERROR("GeneratedLinearElasticity affine objective requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        cache->jacobian_soa->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        cache->jacobian_soa->jacobian_determinant()->data());
            }
            std::fill(impl_->element_values.get(),
                      impl_->element_values.get() + nelements,
                      0);
            int status = SFEM_FAILURE;
            switch (domain.element_type) {
                case smesh::TRI3:
                    status = impl_->objective_uses_affine ? linear_elasticity_tri3_tri3_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, impl_->element_values.get()) : linear_elasticity_tri3_tri3_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, impl_->element_values.get());
                    break;
                case smesh::TRI6:
                    status = impl_->objective_uses_affine ? linear_elasticity_tri6_tri6_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, impl_->element_values.get()) : linear_elasticity_tri6_tri6_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, impl_->element_values.get());
                    break;
                case smesh::QUAD4:
                    status = impl_->objective_uses_affine ? linear_elasticity_quad4_quad4_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, impl_->element_values.get()) : linear_elasticity_quad4_quad4_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_QUAD4:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_quad4_proteus_quad4_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, impl_->element_values.get()) : linear_elasticity_proteus_quad4_proteus_quad4_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, impl_->element_values.get());
                    break;
                case smesh::TET4:
                    status = impl_->objective_uses_affine ? linear_elasticity_tet4_tet4_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, impl_->element_values.get()) : linear_elasticity_tet4_tet4_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                case smesh::TET10:
                    status = impl_->objective_uses_affine ? linear_elasticity_tet10_tet10_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, impl_->element_values.get()) : linear_elasticity_tet10_tet10_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                case smesh::HEX8:
                    status = impl_->objective_uses_affine ? linear_elasticity_hex8_hex8_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, impl_->element_values.get()) : linear_elasticity_hex8_hex8_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                case smesh::HEX27:
                    status = impl_->objective_uses_affine ? linear_elasticity_hex27_hex27_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, impl_->element_values.get()) : linear_elasticity_hex27_hex27_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX8:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex8_proteus_hex8_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, impl_->element_values.get()) : linear_elasticity_proteus_hex8_proteus_hex8_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX27:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex27_proteus_hex27_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, impl_->element_values.get()) : linear_elasticity_proteus_hex27_proteus_hex27_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX64:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex64_proteus_hex64_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, impl_->element_values.get()) : linear_elasticity_proteus_hex64_proteus_hex64_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX125:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex125_proteus_hex125_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, impl_->element_values.get()) : linear_elasticity_proteus_hex125_proteus_hex125_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX729:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex729_proteus_hex729_objective_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, impl_->element_values.get()) : linear_elasticity_proteus_hex729_proteus_hex729_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                default:
                    SFEM_ERROR("GeneratedLinearElasticity does not support element type %d\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
            if (status != SFEM_SUCCESS) return status;
            real_t sum = 0;
#pragma omp simd reduction(+ : sum)
            for (ptrdiff_t element = 0; element < nelements; ++element) {
                sum += impl_->element_values[element];
            }
            *out += sum;
            return SFEM_SUCCESS;
        });
    }

    int GeneratedLinearElasticity::value_steps(const real_t *x,
                            const real_t *h,
                            const int nsteps,
                            const real_t *const steps,
                            real_t *const out) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::value_steps");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        if (nsteps <= 0) {
            return SFEM_SUCCESS;
        }
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            const ptrdiff_t nvalues = (ptrdiff_t)nsteps * nelements;
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->objective_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache || !cache->jacobian_soa) {
                    SFEM_ERROR("GeneratedLinearElasticity affine objective_steps requires cached geometry\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        cache->jacobian_soa->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        cache->jacobian_soa->jacobian_determinant()->data());
            }
            if (nvalues > impl_->element_capacity) {
                impl_->element_values.reset(new real_t[nvalues]);
                impl_->element_capacity = nvalues;
            }
            std::fill(impl_->element_values.get(),
                      impl_->element_values.get() + nvalues,
                      real_t(0));
            int status = SFEM_FAILURE;
            if (impl_->objective_uses_affine && impl_->space->has_packed_mesh()) {
                auto packed = impl_->space->packed_mesh();
                const int packed_block = packed_block_id_for_domain(*packed, *domain.block);
                if (packed_block >= 0) {
                    auto packed_elements = packed->elements(packed_block);
                    auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);
                    auto n_shared_nodes = packed->n_shared(packed_block);
                    auto ghost_ptr = packed->ghost_ptr(packed_block);
                    auto ghost_idx = packed->ghost_idx(packed_block);
                    const int dim = mesh->spatial_dimension();
                    if (dim == 2) {
                        status = linear_elasticity_objective_steps_packed_2d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get());
                    }
                    else if (dim == 3) {
                        status = linear_elasticity_objective_steps_packed_3d_affine_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    }
                }
            }
            if (!impl_->objective_uses_affine && impl_->space->has_packed_mesh()) {
                auto packed = impl_->space->packed_mesh();
                const int packed_block = packed_block_id_for_domain(*packed, *domain.block);
                if (packed_block >= 0) {
                    auto packed_elements = packed->elements(packed_block);
                    auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);
                    auto n_shared_nodes = packed->n_shared(packed_block);
                    auto ghost_ptr = packed->ghost_ptr(packed_block);
                    auto ghost_idx = packed->ghost_idx(packed_block);
                    const int dim = mesh->spatial_dimension();
                    if (dim == 2) {
                        status = linear_elasticity_objective_steps_packed_2d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get());
                    }
                    else if (dim == 3) {
                        status = linear_elasticity_objective_steps_packed_3d_isoparametric_mesh_soa(domain.element_type, packed->n_packs(packed_block), packed->n_elements_per_pack(packed_block), domain.block->n_elements(), mesh->n_nodes(), packed->max_nodes_per_pack(), packed_elements->data(), owned_nodes_ptr->data(), n_shared_nodes->data(), ghost_ptr->data(), ghost_idx->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    }
                }
            }
            if (status == SFEM_FAILURE) {
            switch (domain.element_type) {
                case smesh::TRI3:
                    status = impl_->objective_uses_affine ? linear_elasticity_tri3_tri3_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get()) : linear_elasticity_tri3_tri3_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::TRI6:
                    status = impl_->objective_uses_affine ? linear_elasticity_tri6_tri6_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get()) : linear_elasticity_tri6_tri6_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::QUAD4:
                    status = impl_->objective_uses_affine ? linear_elasticity_quad4_quad4_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get()) : linear_elasticity_quad4_quad4_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_QUAD4:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_quad4_proteus_quad4_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get()) : linear_elasticity_proteus_quad4_proteus_quad4_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 2, x + 0, x + 1, 2, h + 0, h + 1, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::TET4:
                    status = impl_->objective_uses_affine ? linear_elasticity_tet4_tet4_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get()) : linear_elasticity_tet4_tet4_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::TET10:
                    status = impl_->objective_uses_affine ? linear_elasticity_tet10_tet10_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get()) : linear_elasticity_tet10_tet10_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::HEX8:
                    status = impl_->objective_uses_affine ? linear_elasticity_hex8_hex8_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get()) : linear_elasticity_hex8_hex8_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::HEX27:
                    status = impl_->objective_uses_affine ? linear_elasticity_hex27_hex27_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get()) : linear_elasticity_hex27_hex27_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX8:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex8_proteus_hex8_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get()) : linear_elasticity_proteus_hex8_proteus_hex8_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX27:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex27_proteus_hex27_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get()) : linear_elasticity_proteus_hex27_proteus_hex27_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX64:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex64_proteus_hex64_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get()) : linear_elasticity_proteus_hex64_proteus_hex64_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX125:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex125_proteus_hex125_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get()) : linear_elasticity_proteus_hex125_proteus_hex125_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX729:
                    status = impl_->objective_uses_affine ? linear_elasticity_proteus_hex729_proteus_hex729_objective_steps_affine_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), adjugate[0], adjugate[1], adjugate[2], adjugate[3], adjugate[4], adjugate[5], adjugate[6], adjugate[7], adjugate[8], determinant, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get()) : linear_elasticity_proteus_hex729_proteus_hex729_objective_steps_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("lmbda"), domain.parameters->require_real_value("mu"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, nsteps, steps, impl_->element_values.get());
                    break;
                default:
                    SFEM_ERROR("GeneratedLinearElasticity does not support element type %d\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
            }
            if (status != SFEM_SUCCESS) return status;
            for (int step = 0; step < nsteps; ++step) {
                real_t sum = 0;
#pragma omp simd reduction(+ : sum)
                for (ptrdiff_t element = 0; element < nelements; ++element) {
                    sum += impl_->element_values[(ptrdiff_t)step * nelements + element];
                }
                out[step] += sum;
            }
            return SFEM_SUCCESS;
        });
    }

    int GeneratedLinearElasticity::hessian_crs(const real_t *const x,
                            const count_t *const rowptr,
                            const idx_t *const colidx,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::hessian_crs");
        const real_t *const current = x;
        if (!current) {
            SFEM_ERROR("GeneratedLinearElasticity::hessian_crs requires a current state\n");
            return SFEM_FAILURE;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                SFEM_ERROR("linear_elasticity hessian_crs 2d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            else if (dim == 3) {
                SFEM_ERROR("linear_elasticity hessian_crs 3d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            SFEM_ERROR("linear_elasticity hessian_crs does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    int GeneratedLinearElasticity::hessian_bsr(const real_t *const x,
                            const count_t *const rowptr,
                            const idx_t *const colidx,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::hessian_bsr");
        const real_t *const current = x;
        if (!current) {
            SFEM_ERROR("GeneratedLinearElasticity::hessian_bsr requires a current state\n");
            return SFEM_FAILURE;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                SFEM_ERROR("linear_elasticity hessian_bsr 2d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            else if (dim == 3) {
                SFEM_ERROR("linear_elasticity hessian_bsr 3d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            SFEM_ERROR("linear_elasticity hessian_bsr does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    int GeneratedLinearElasticity::hessian_dia(const real_t *const x,
                            const int *const diag_offsets,
                            const ptrdiff_t ndiag,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::hessian_dia");
        const real_t *const current = x;
        if (!current) {
            SFEM_ERROR("GeneratedLinearElasticity::hessian_dia requires a current state\n");
            return SFEM_FAILURE;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                SFEM_ERROR("linear_elasticity hessian_dia 2d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            else if (dim == 3) {
                SFEM_ERROR("linear_elasticity hessian_dia 3d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            SFEM_ERROR("linear_elasticity hessian_dia does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    int GeneratedLinearElasticity::hessian_coo(const real_t *const x,
                            const ptrdiff_t nnz,
                            const idx_t *const rows,
                            const idx_t *const cols,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::hessian_coo");
        const real_t *const current = x;
        if (!current) {
            SFEM_ERROR("GeneratedLinearElasticity::hessian_coo requires a current state\n");
            return SFEM_FAILURE;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                SFEM_ERROR("linear_elasticity hessian_coo 2d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            else if (dim == 3) {
                SFEM_ERROR("linear_elasticity hessian_coo 3d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            SFEM_ERROR("linear_elasticity hessian_coo does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    int GeneratedLinearElasticity::hessian_patch(const real_t *const x,
                              const count_t *const rowptr,
                              const idx_t *const colidx,
                              real_t *const values) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::hessian_patch");
        const real_t *const current = x;
        if (!current) {
            SFEM_ERROR("GeneratedLinearElasticity::hessian_patch requires a current state\n");
            return SFEM_FAILURE;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const int dim = mesh->spatial_dimension();
            if (dim == 2) {
                SFEM_ERROR("linear_elasticity hessian_patch 2d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            else if (dim == 3) {
                SFEM_ERROR("linear_elasticity hessian_patch 3d dispatch was not generated\n");
                return SFEM_FAILURE;
            }
            SFEM_ERROR("linear_elasticity hessian_patch does not support spatial dimension %d\n", dim);
            return SFEM_FAILURE;
        });
    }

    void GeneratedLinearElasticity::set_option(const std::string &name, const bool val) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::set_option");
        if (name == "PACKED_TWO_PASS" || name == "two_pass") {
            impl_->use_packed_two_pass = val;
            return;
        }
        AffineOption options[] = {
            {"ASSUME_AFFINE_OBJECTIVE", &impl_->objective_uses_affine},
            {"objective_assume_affine", &impl_->objective_uses_affine},
            {"ASSUME_AFFINE_GRADIENT", &impl_->gradient_uses_affine},
            {"gradient_assume_affine", &impl_->gradient_uses_affine},
            {"ASSUME_AFFINE_HESSIAN_ACTION", &impl_->apply_uses_affine},
            {"hessian_action_assume_affine", &impl_->apply_uses_affine},
            {"ASSUME_AFFINE_APPLY", &impl_->apply_uses_affine},
            {"apply_assume_affine", &impl_->apply_uses_affine},
        };
        const bool matched = set_affine_option(name, val, options, sizeof(options) / sizeof(options[0]));
        if (matched && val && impl_->domains) {
            if (cache_affine_geometry(impl_->space, *impl_->domains) != SFEM_SUCCESS) {
                SFEM_ERROR("GeneratedLinearElasticity failed to cache affine geometry\n");
            }
        }
    }

    void GeneratedLinearElasticity::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::set_value_in_block");
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> GeneratedLinearElasticity::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
        SFEM_TRACE_SCOPE("GeneratedLinearElasticity::create_from_yaml");
        auto ret = std::make_shared<GeneratedLinearElasticity>(space);

        std::vector<std::string> block_names;
        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (block.has_child("name")) {
                    block_names.push_back(yaml_read_string(block["name"]));
                }
            }
        }

        AffineOption options[] = {
            {"ASSUME_AFFINE_OBJECTIVE", &ret->impl_->objective_uses_affine},
            {"objective_assume_affine", &ret->impl_->objective_uses_affine},
            {"ASSUME_AFFINE_GRADIENT", &ret->impl_->gradient_uses_affine},
            {"gradient_assume_affine", &ret->impl_->gradient_uses_affine},
            {"ASSUME_AFFINE_HESSIAN_ACTION", &ret->impl_->apply_uses_affine},
            {"hessian_action_assume_affine", &ret->impl_->apply_uses_affine},
            {"ASSUME_AFFINE_APPLY", &ret->impl_->apply_uses_affine},
            {"apply_assume_affine", &ret->impl_->apply_uses_affine},
        };
        read_affine_options(node, options, sizeof(options) / sizeof(options[0]));

        if (ret->initialize(block_names) != SFEM_SUCCESS) {
            return nullptr;
        }

        real_t defaults[N_MATERIAL_PARAMETERS];
        material_defaults(defaults);
        real_t top_values[N_MATERIAL_PARAMETERS];
        copy_material_parameters(defaults, top_values);
        if (material_from_yaml(node, defaults, top_values)) {
            set_material(*ret->impl_->domains, top_values);
        }

        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (!block.has_child("name")) {
                    continue;
                }

                real_t block_values[N_MATERIAL_PARAMETERS];
                copy_material_parameters(top_values, block_values);
                if (!material_from_yaml(block, top_values, block_values)) {
                    continue;
                }

                const std::string block_name = yaml_read_string(block["name"]);
                set_material_in_block(*ret->impl_->domains, block_name, block_values);
            }
        }

        return ret;
    }
#endif  // SFEM_ENABLE_RYAML
}  // namespace sfem

