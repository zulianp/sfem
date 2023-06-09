{CONSTANTS}

static inline __device__ __host__ void mk_jacobian(
{COORDINATES}
{QUADRATURE_POINT}
const count_t stride_jacobian,
real_t *jacobian
)
{{
{JACOBIAN}
}}

static inline __device__ __host__ void mk_jacobian_inverse(
{COORDINATES}
{QUADRATURE_POINT}
const count_t stride_jacobian_inverse,
real_t *jacobian_inverse
)
{{
{JACOBIAN_INVERSE}
}}

static inline __device__ __host__ void mk_jacobian_determinant(
{COORDINATES}
{QUADRATURE_POINT}
const count_t stride_jacobian_determinant,
real_t *jacobian_determinant
)
{{
{JACOBIAN_DETERMINANT}
}}

static inline __device__ __host__ void mk_fun(
{QUADRATURE_POINT}
real_t * SFEM_RESTRICT f
)
{{
{FUN}
}}

static inline __device__ __host__ void mk_partial_x(
{QUADRATURE_POINT}
const count_t stride_jacobian_inverse,
const real_t * SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * SFEM_RESTRICT gx
)
{{
{PARTIAL_X}
}}

static inline __device__ __host__ void mk_partial_y(
{QUADRATURE_POINT}
const count_t stride_jacobian_inverse,
const real_t * SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * SFEM_RESTRICT gy
)
{{
{PARTIAL_Y}
}}

static inline __device__ __host__ void mk_partial_z(
{QUADRATURE_POINT}
const count_t stride_jacobian_inverse,
const real_t * SFEM_RESTRICT jacobian_inverse,
const count_t stride_grad,
real_t * SFEM_RESTRICT gz
) 
{{
{PARTIAL_Z}
}}
