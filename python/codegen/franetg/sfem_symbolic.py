import sys
import os

# FIXME
sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/..")
from sfem_codegen import *


def jacobian(rows, cols):
    return matrix_coeff("jacobian", rows, cols)


def jacobian_inverse(rows, cols):
    return matrix_coeff("jacobian_inverse", rows, cols)


def jacobian_determinant():
    return sp.symbols("jacobian_determinant")


def trial_function(var):
    return sp.symbols(f"{var}_trial_fun")


def test_function(var):
    return sp.symbols(f"{var}_test_fun")


def trial_gradient(var, rows):
    return coeffs(f"{var}_trial_grad", rows)


def test_gradient(var, rows):
    return coeffs(f"{var}_test_grad", rows)


def reference_measure():
    return sp.symbols("ref_measure")


def tensorize_vector(g):
    ret = []

    dim = len(g)
    for d1 in range(0, dim):
        G = sp.Matrix(dim, dim, [0] * (dim * dim))
        for d2 in range(0, dim):
            G[d1, d2] = g[d2]

        ret.append(G)
    return ret


def derivative1(expr, x):
    rows, cols = x.shape

    dx = sp.Matrix(rows, cols, [0] * (rows * cols))
    for d1 in range(0, rows):
        for d2 in range(0, cols):
            dx[d1, d2] = sp.diff(expr, x[d1, d2])
    return dx


def derivative0(expr, x):
    return sp.diff(expr, x)


def form0_assign(var, form0):
    var_i = sp.symbols(f"{var}[0]")
    return ast.AddAugmentedAssignment(var_i, form0)


def form1_assign(var, form1):
    dim = len(form1)

    ret = []
    for d in range(0, dim):
        var_i = sp.symbols(f"{var}[{d}*stride]")
        expr = ast.AddAugmentedAssignment(var_i, form1[d])
        ret.append(expr)
    return ret


def form2_assign(var, form2):
    rows, cols = form2.shape

    ret = []
    for d1 in range(0, rows):
        for d2 in range(0, cols):
            var_i = sp.symbols(f"{var}[{d1*cols+d2}*stride]")
            expr = ast.AddAugmentedAssignment(var_i, form2[d1, d2])
            ret.append(expr)
    return ret


def read_file(path):
    with open(path, "r") as f:
        tpl = f.read()
        return tpl
    assert False
    return ""


def str_to_file(path, mystr):
    with open(path, "w") as f:
        f.write(mystr)
        f.close()
