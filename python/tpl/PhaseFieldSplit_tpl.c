// PhaseFieldSplit_tpl.c



// Basic includes
#include "sfem_base.h"
#include <math.h>

///////////////////////////////
// Energy
///////////////////////////////
void {kernel_name}_energy({args_energy}) {{
{energy}
}}

///////////////////////////////
// Gradient
///////////////////////////////
void {kernel_name}_gradient_c({args_gradient_c}) {{
{gradient_c}
}}


void {kernel_name}_gradient_u({args_gradient_u}) {{
{gradient_u}
}}

///////////////////////////////
// Hessian
///////////////////////////////

void {kernel_name}_hessian_uu({args_hessian_uu}) {{
{hessian_uu}
}}

void {kernel_name}_hessian_uc({args_hessian_uc}) {{
{hessian_uc}
}}

void {kernel_name}_hessian_cu({args_hessian_cu}) {{
{hessian_cu}
}}

void {kernel_name}_hessian_cc({args_hessian_cc}) {{
{hessian_cc}
}}


///////////////////////////////
// Apply
///////////////////////////////
void {kernel_name}_apply_cc({args_apply_cc}) {{
{apply_cc}
}}


void {kernel_name}_apply_uu({args_apply_uu}) {{
{apply_uu}
}}


