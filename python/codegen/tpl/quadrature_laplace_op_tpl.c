// quadrature_laplace_op_tpl.c

void trial_operand(
const real_t qx,
const real_t qy,
const real_t qz,
const real_t qw,
const geom_t * const SFEM_RESTRICT fff,
const real_t * const SFEM_RESTRICT u,
real_t * const SFEM_RESTRICT out
)
{{
{TRIAL_OPERAND_CODE}
}}

static inline __device__ __host__ void ref_shape_grad_x(
	const real_t qx,
	const real_t qy,
	const real_t qz,
	real_t * const out)
{{
{REF_SHAPE_GRAD_X_CODE}
}}

static inline __device__ __host__ void ref_shape_grad_y(
	const real_t qx,
	const real_t qy,
	const real_t qz,
	real_t * const out)
{{
{REF_SHAPE_GRAD_Y_CODE}
}}

static inline __device__ __host__ void ref_shape_grad_z(
	const real_t qx,
	const real_t qy,
	const real_t qz,
	real_t * const out)
{{
{REF_SHAPE_GRAD_Z_CODE}
}}

