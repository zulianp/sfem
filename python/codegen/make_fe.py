#!/usr/bin/env python3

from sfem_codegen import *


def shape_grad(q):
    d = len(q)
    assert d == 3

    f = fun(q[0], q[1], q[2])
    ret = [0] * 4

    for i in range(0, 4):
        fi = f[i]

        gix = sp.simplify(sp.diff(fi, q[0]))
        giy = sp.simplify(sp.diff(fi, q[1]))
        giz = sp.simplify(sp.diff(fi, q[2]))

        g = sp.Matrix(d, 1, [gix, giy, giz])
        ret[i] = g

    return ret


def fe_tgrad(q, coeff):
    d = len(q)
    assert d == 3

    z = [0] * (d * d)
    evalgrad = sp.Matrix(d, d, z)

    shapegrad = tgrad(q[0], q[1], q[2])

    n = len(coeff)
    assert n == 4 * 3

    for i in range(0, n):
        for d1 in range(0, d):
            for d2 in range(0, d):
                evalgrad[d1, d2] += shapegrad[i][d1, d2] * coeff[i]

    return evalgrad


def fe_grad(q, coeff):
    d = len(q)
    assert d == 3

    z = [0] * (d)
    evalgrad = sp.Matrix(d, 1, z)

    f = fun(q[0], q[1], q[2])
    n = len(coeff)
    assert n == 4

    for i in range(0, n):
        fi = f[i]
        gix = sp.simplify(sp.diff(fi, q[0]))
        giy = sp.simplify(sp.diff(fi, q[1]))
        giz = sp.simplify(sp.diff(fi, q[2]))

        g = [gix, giy, giz]

        for d1 in range(0, d):
            evalgrad[d1] += g[d1] * coeff[i]

    return evalgrad


def fe_fun(q, coeff):
    d = len(q)
    assert d == 3

    f = ref_fun(q[0], q[1], q[2])
    n = len(coeff)
    assert n == 4

    evalfun = 0.0
    for i in range(0, n):
        evalfun += f[i] * coeff[i]

    return evalfun


########################################################
# Vector gradient
########################################################

vec3_list = []
for i in range(0, 4 * 3):
    ci = sp.symbols(f"c[{i}]", real=True)
    vec3_list.append(ci)

vec3 = sp.Matrix(4 * 3, 1, vec3_list)
vec3_fg = fe_tgrad(q, vec3)

vec3_expr = []
for d1 in range(0, 3):
    for d2 in range(0, 3):
        vec3_expr.append(
            ast.Assignment(sp.symbols(f"vg_output[{d1 * 3 + d2}]"), vec3_fg[d1, d2])
        )

vec3_g_code = c_gen(vec3_expr)

########################################################
# Shape grad
########################################################

sg = shape_grad(q)
sg_list = []

for i in range(0, 4):
    sgi = sg[i]
    for d in range(0, 3):
        gi = sp.symbols(f"g[{i*3 + d}]", real=True)
        sg_list.append(ast.Assignment(gi, sgi[d]))


scalar_g_code = c_gen(sg_list)
print(scalar_g_code)

########################################################
# Scalar grad
########################################################
scalar_list = []

for i in range(0, 4):
    ci = sp.symbols(f"c[{i}]", real=True)
    scalar_list.append(ci)

scalar = sp.Matrix(4, 1, scalar_list)
scalar_fg = fe_grad(q, scalar)

scalar_expr = []
for d in range(0, 3):
    scalar_expr.append(ast.Assignment(sp.symbols(f"sg_output[{d}]"), scalar_fg[d]))

scalar_g_code = c_gen(scalar_expr)

########################################################
# Scalar fun
########################################################

scalar_expr = []
scalar_expr.append(ast.Assignment(sp.symbols(f"f_output[0]"), fe_fun(q, scalar)))
scalar_f_code = c_gen(scalar_expr)

tpl = """
//
SFEM_INLINE void {name}(
// x
const real_t px0,
const real_t px1,
const real_t px2,
const real_t px3,
// y
const real_t py0,
const real_t py1,
const real_t py2,
const real_t py3,
// z
const real_t pz0,
const real_t pz1,
const real_t pz2,
const real_t pz3,
// Coefficients
const real_t *c,
{extras}
// Out
real_t *output
) 
{{
{code}
}}

"""

qpstr = """const real_t qx,
const real_t qy,
const real_t qz,"""

fe_tgrad_code = tpl.format(name="tet4_fe_tgrad", code=vec3_g_code, extras="")
fe_grad_code = tpl.format(name="tet4_fe_grad", code=scalar_g_code, extras="")
fe_fun_code = tpl.format(name="tet4_fe_fun", code=scalar_f_code, extras=qpstr)

f = open("generated_tet4.c", "w")

f.write(
    """
#include "sfem_base.h"
#include <math.h>
"""
)

f.write(fe_tgrad_code)
f.write(fe_grad_code)
f.write(fe_fun_code)

f.close()

c_code(det3(A))
