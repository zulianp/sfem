#include <math.h>
#include <stddef.h>

#ifndef SFEM_RESTRICT
typedef double real_t;
#define SFEM_INLINE inline
#define SFEM_RESTRICT restrict
#endif

#ifndef SFEM_STATIC
#define SFEM_STATIC static
#endif

SFEM_STATIC SFEM_INLINE void micro_kernel_energy(
	const real_t mu,
	const real_t lambda,
	const real_t Gc,
	const real_t ls,
	const real_t dV,
	const real_t * SFEM_RESTRICT u_grad,
	const real_t c,
	const real_t * SFEM_RESTRICT c_grad,
	real_t * SFEM_RESTRICT element_scalar
	)
{{
	// Generated code
	{MICRO_KERNEL_ENERGY}
}}

SFEM_STATIC SFEM_INLINE void micro_kernel_form1_u(
	const real_t mu,
	const real_t lambda,
	// const real_t Gc,
	// const real_t ls,
	const real_t dV,
	const real_t * SFEM_RESTRICT v_test_grad,
	const real_t * SFEM_RESTRICT u_grad,
	const real_t c,
	// const real_t * SFEM_RESTRICT c_grad,
	const ptrdiff_t stride,
	real_t * SFEM_RESTRICT element_vector
	)
{{
	// Generated code
	{MICRO_KERNEL_FORM1_U}
}}

SFEM_STATIC SFEM_INLINE void micro_kernel_form1_c(
	const real_t mu,
	const real_t lambda,
	const real_t Gc,
	const real_t ls,
	const real_t dV,
	const real_t v_test_fun,
	const real_t * SFEM_RESTRICT v_test_grad,
	const real_t * SFEM_RESTRICT u_grad,
	const real_t c,
	const real_t * SFEM_RESTRICT c_grad,
	const ptrdiff_t stride,
	real_t * SFEM_RESTRICT element_vector
	)
{{
	// Generated code
	{MICRO_KERNEL_FORM1_C}
}}

SFEM_STATIC SFEM_INLINE void micro_kernel_form2_uu(
	const real_t mu,
	const real_t lambda,
	// const real_t Gc,
	// const real_t ls,
	const real_t dV,
	const real_t * SFEM_RESTRICT v_trial_grad,
	const real_t * SFEM_RESTRICT v_test_grad,
	const real_t c,
	const ptrdiff_t stride,
	real_t * SFEM_RESTRICT element_matrix
	)
{{
	// Generated code
	{MICRO_KERNEL_FORM2_UU}
}}

SFEM_STATIC SFEM_INLINE void micro_kernel_form2_uc(
	const real_t mu,
	const real_t lambda,
	// const real_t Gc,
	// const real_t ls,
	const real_t dV,
	const real_t v_trial_fun,
	// const real_t * SFEM_RESTRICT v_trial_grad,
	const real_t * SFEM_RESTRICT v_test_grad,
	const real_t * SFEM_RESTRICT u_grad,
	const real_t c,
	const ptrdiff_t stride,
	real_t * SFEM_RESTRICT element_matrix
	)
{{
	// Generated code
	{MICRO_KERNEL_FORM2_UC}
}}

SFEM_STATIC SFEM_INLINE void micro_kernel_form2_cu(
	const real_t mu,
	const real_t lambda,
	// const real_t Gc,
	// const real_t ls,
	const real_t dV,
	const real_t * SFEM_RESTRICT v_trial_grad,
	const real_t v_test_fun,
	// const real_t * SFEM_RESTRICT v_test_grad,
	const real_t * SFEM_RESTRICT u_grad,
	const real_t c,
	const ptrdiff_t stride,
	real_t * SFEM_RESTRICT element_matrix
	)
{{
	// Generated code
	{MICRO_KERNEL_FORM2_CU}
}}

SFEM_STATIC SFEM_INLINE void micro_kernel_form2_cc(
	const real_t mu,
	const real_t lambda,
	const real_t Gc,
	const real_t ls,
	const real_t dV,
	const real_t v_trial_fun,
	const real_t * SFEM_RESTRICT v_trial_grad,
	const real_t v_test_fun,
	const real_t * SFEM_RESTRICT v_test_grad,
	const real_t * SFEM_RESTRICT u_grad,
	const ptrdiff_t stride,
	real_t * SFEM_RESTRICT element_matrix
	)
{{
	// Generated code
	{MICRO_KERNEL_FORM2_CC}
}}


