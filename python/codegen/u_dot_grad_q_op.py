#!/usr/bin/env python3

from sfem_codegen import *


class UDotGradQOp:
    def __init__(self, u, q, qp):
        self.u = u
        self.q = q
        self.qp = qp

    def gradient(self):
        rf = ref_fun(qx, qy, qz)
        dV = det3(A) / 6

        ux = sp.symbols("ux")
        uy = sp.symbols("uy")
        uz = sp.symbols("uz")

        u = [ux, uy, uz]
        f = fun(qx, qy, qz)

        expr = []

        for i in range(0, 4):
            fi = f[i]
            dfdx = sp.diff(fi, qx)
            dfdy = sp.diff(fi, qy)
            dfdz = sp.diff(fi, qz)
            g = [dfdx, dfdy, dfdz]

            form = 0
            for d in range(0, 3):
                form += g[d] * u[d]

            integr = form * dV
            var = sp.symbols(f"element_vector[{i}]")
            expr.append(ast.Assignment(var, integr))

        c_code(expr)

    # def hessian(self):
    # 	# TODO
