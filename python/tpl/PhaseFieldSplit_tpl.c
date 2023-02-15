// PhaseFieldSplit_tpl.c



// Basic includes
#include "sfem_base.h"
#include <math.h>

///////////////////////////////
// Energy
///////////////////////////////
void {kernel_name}_energy({args_e}) {{
{energy}
}}

///////////////////////////////
// Gradient
///////////////////////////////
void {kernel_name}_gradient_c({args_g_c}) {{
{gradient_c}
}}


void {kernel_name}_gradient_u({args_g_u}) {{
{gradient_u}
}}

///////////////////////////////
// Hessian
///////////////////////////////

void {kernel_name}_hessian_uu({args_H_uu}) {{
{hessian_uu}
}}

void {kernel_name}_hessian_uc({args_H_uc}) {{
{hessian_uc}
}}

void {kernel_name}_hessian_cu({args_H_cu}) {{
{hessian_cu}
}}

void {kernel_name}_hessian_cc({args_H_cc}) {{
{hessian_cc}
}}

