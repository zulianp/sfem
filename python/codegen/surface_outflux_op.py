#!/usr/bin/env python3

from sfem_codegen import *

from tet4 import *
from tri3 import *
from tri6 import *
from tri1 import *

from fields import *


class SurfaceOutflux:
    def __init__(self, vector_field, fe_test, q):
        self.vector_field = vector_field
        self.fe_test = fe_test
        self.q = q

    def apply(self):
        vector_field = self.vector_field
        fe_test = self.fe_test
        q = self.q

        vfx = vector_field.eval(q)

        nnames = ["nx", "ny", "nz"]

        n = [0] * fe_test.spatial_dim()

        for d in range(0, 3):
            n[d] = sp.symbols(nnames[d])

        expr = []

        rf = fe_test.fun(q)

        for i in range(0, fe_test.n_nodes()):
            integr = 0
            for d in range(0, fe_test.spatial_dim()):
                integr += fe_test.integrate(
                    q, n[d] * vfx[d] * rf[i] * fe_test.jacobian_determinant(q)
                )

            var = sp.symbols(f"element_vector[{i}]")
            expr.append(ast.Assignment(var, integr))

        return expr


def main():
    shell_fe_field = [TriShell3(), TriShell6()]
    shell_fe_test = [TriShell3(), TriShell6()]

    q = [qx, qy]
    for sf_field in shell_fe_field:
        for sf_test in shell_fe_test:
            c = [
                coeffs("vx", sf_field.n_nodes()),
                coeffs("vy", sf_field.n_nodes()),
                coeffs("vz", sf_field.n_nodes()),
            ]

            vf = VectorField(sf_field, c)
            op = SurfaceOutflux(vf, sf_test, q)

            print("------------------------------------------------------")
            print(f"Vector field {sf_field.name()} test {sf_test.name()}")
            print("------------------------------------------------------")
            c_code(op.apply())
            print("------------------------------------------------------")


if __name__ == "__main__":
    main()
