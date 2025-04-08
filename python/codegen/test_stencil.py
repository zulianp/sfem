#!/usr/bin/env python3

from tri6 import *
from quad4 import *
from mass_op import *
from laplace_op import *


def assign_matrix(name, mat):
    rows, cols = mat.shape
    expr = []
    for i in range(0, rows):
        for j in range(0, cols):
            var = sp.symbols(f"{name}[{i*cols + j}]")
            expr.append(ast.Assignment(var, mat[i, j]))
    return expr


def assign_tensor3(name, mat):
    rows, cols = mat.shape
    expr = []
    for zi in range(0, 3):
        for yi in range(0, 3):
            for xi in range(0, 3):
                var = sp.symbols(f"{name}[{xi}][{yi}][{zi}]")
                val = mat[zi * 9 + yi * 3 + xi]
                expr.append(ast.Assignment(var, val))
    return expr


if __name__ == "__main__":
    # fe = Quad4(True)
    fe = Hex8(False)
    dim = fe.spatial_dim()

    op = LaplaceOp(fe, True)
    f = Field(fe, coeffs("u", fe.n_nodes()))
    # op = MassOp(f, fe, True)

    # L = op.sym_matrix()

    L = matrix_coeff("A", fe.n_nodes(), fe.n_nodes())
    # L = sym_matrix_coeff('A', fe.n_nodes(), fe.n_nodes())
    S = fe.to_stencil(L)
    # S = fe.to_masked_stencil(L)
    # print(S.shape)

    # Su = S * coeffs('u', S.shape[1])
    # print('//DIFF = ', matrix_sum(S) - matrix_sum(L))
    # expr = []

    paramlist = "(\nconst ptrdiff_t xc,\nconst ptrdiff_t yc,\nconst ptrdiff_t zc,\n"
    paramlist += f"const ptrdiff_t xstride,\nconst ptrdiff_t ystride,\nconst ptrdiff_t zstride,\n"
    paramlist += f"const scalar_t * const SFEM_RESTRICT A,\n"
    paramlist += f"const scalar_t * const SFEM_RESTRICT input,\n"
    paramlist += f"scalar_t * const SFEM_RESTRICT output\n)\n"

    xstride, ystride, zstride = sp.symbols("xstride ystride zstride")
    stride = ["xstride"]
    for k, v in S.items():
        print(f"//===============\n//{k})\n//===============")
        # print(v)

        size = v["size"]
        inoffset = v["inoffset"]
        outoffset = v["outoffset"]
        extent = v["extent"]

        expr = assign_matrix(k, v["stencil"])
        # expr = assign_tensor3("S", S)
        # expr = assign_matrix("Su", Su)
        m2s = c_gen(expr, optimizations="basic")
        code = f"static void sshex8_apply_{k}"
        code += paramlist

        code += "{\n"
        code += f"scalar_t {k}[{size[0] * size[1] * size[2]}];"
        code += m2s
        code += "\n"
        code += "// buffs\n"

        for zi in range(0, size[2]):
            for yi in range(0, size[1]):
                for xi in range(0, size[0]):
                    code += f"const scalar_t *const in{xi + yi * size[0] + zi * size[1] * size[0]} = &input[{(inoffset[0] + xi) * xstride + (inoffset[1] + yi) * ystride + (inoffset[2] + zi) * zstride}];\n"

        code += f"scalar_t * const out = &output[{outoffset[0] * xstride + outoffset[1] * ystride + outoffset[2] * zstride}];\n"
        code += f"for(ptrdiff_t zi = 0; zi < ({extent[2]}); zi++)\n"
        code += "{\n"
        code += f"for(ptrdiff_t yi = 0; yi < ({extent[1]}); yi++)\n"
        code += "{\n"
        code += f"for(ptrdiff_t xi = 0; xi < ({extent[0]}); xi++)\n"
        code += "{\n"
        code += (
            f"const ptrdiff_t idx = xi * {xstride} + yi * {ystride} + zi * {zstride};\n"
        )

        for zi in range(0, size[2]):
            for yi in range(0, size[1]):
                for xi in range(0, size[0]):
                    ii = xi + yi * size[0] + zi * size[1] * size[0]
                    code += f"out[idx] += "
                    code += f"in{ii}[idx] * {k}[{ii}]"
                    code += ";\n"

        code += "}\n"
        code += "}\n"
        code += "}\n"
        code += "}\n"
        print(code)

    code = "static void sshex8_surface_stencil"
    code += paramlist
    code += "{\n"

    for k, v in S.items():
        if k == "stencil111":
            code += "// "

        code += f"sshex8_apply_{k}"
        code += "(xc, yc, zc, xstride, ystride, zstride, A, input, output);\n"
    code += "}\n"
    print(code)

    # c_code(op.hessian())
