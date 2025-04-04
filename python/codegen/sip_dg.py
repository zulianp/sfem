#!/usr/bin/env python3

import sys
from aahex8 import *
from hex8 import *
from hex27 import *
from quad4 import *
from sfem_codegen import *
from tet10 import *
from tet20 import *
from tet4 import *
from tri3 import *
from tri6 import *

from sympy import pprint


def simplify(x):
    # return sp.simplify(x)
    return x


class SIPDG:
    def __init__(self, fe, symbolic_integration=False):
        self.symbolic_integration = symbolic_integration
        self.fe = fe
        self.tau = sp.symbols("tau")
        self.qw = sp.symbols("qw")

    def apply(self):
        fe = self.fe
        tau = self.tau
        qw = self.qw

        side_fe = fe.trace_element()
        s_qp = side_fe.quadrature_point()

        nn = fe.n_nodes()
        u = coeffs("u", nn)

        sides = fe.sides()
        ns = len(sides)

        dim = fe.spatial_dim()

        v_qp_sym = coeffs("qp", dim)
        uh = fe.interpolate(v_qp_sym, u)
        guh = fe.grad_interpolate(v_qp_sym, u)
        J_inv = fe.symbol_jacobian_inverse()

        uh_sym = sp.symbols("uh")
        guh_sym = coeffs("guh", dim)

        gv = fe.grad(v_qp_sym)

        integrals = {}
        for si in range(0, ns):
            side = sides[si]
            coords = fe.select_coords(side)
            p = [side_fe.interpolate(s_qp, coords[d]) for d in range(0, dim)]

            g = sp.zeros(dim, dim - 1)
            for k in range(0, dim):
                for j in range(0, dim - 1):
                    g[k, j] = sp.simplify(sp.diff(p[k], s_qp[j]))

            if dim == 3:
                dS = cross(g[:, 0], g[:, 1])
            else:
                # IMPLEMENT ME
                assert False

            norm_dS = norm2(dS)
            v_qp = fe.map_to_side(si, s_qp)

            s_uh = subsmat(uh, v_qp_sym, v_qp)
            s_guh = subsmat(guh, v_qp_sym, v_qp)

            n_side_nodes = side_fe.n_nodes()

            primal_consistency_term = sp.zeros(n_side_nodes, 1)
            adjoint_consistency_term = sp.zeros(n_side_nodes, 1)
            penalty_term = sp.zeros(n_side_nodes, 1)

            v = side_fe.fun(s_qp)
            for j in range(0, n_side_nodes):
                primal_consistency_term[j] = v[j] * inner(dS / 2, guh_sym)
                adjoint_consistency_term[j] = uh_sym * inner(dS / 2, gv[j])
                penalty_term[j] = v[j] * uh_sym * norm_dS * tau

                if self.symbolic_integration:
                    primal_consistency_term[j] = side_fe.integrate(
                        s_qp, primal_consistency_term[j]
                    )
                    adjoint_consistency_term[j] = side_fe.integrate(
                        s_qp, adjoint_consistency_term[j]
                    )
                    penalty_term[j] = side_fe.integrate(s_qp, penalty_term[j])
                else:
                    primal_consistency_term[j] = simplify(
                        primal_consistency_term[j] * qw
                    )
                    adjoint_consistency_term[j] = simplify(
                        adjoint_consistency_term[j] * qw
                    )
                    penalty_term[j] = simplify(penalty_term[j] * qw)

            # split_ops = True
            split_ops = False

            integrals[f"uh_sym_{si}"] = assign_value("uh", uh)
            integrals[f"guh_sym_{si}"] = assign_matrix("guh", guh)

            if split_ops:
                integrals[f"primal_consistency_term_{si}"] = assign_matrix(
                    "element_vector", primal_consistency_term
                )
                integrals[f"adjoint_consistency_term_{si}"] = assign_matrix(
                    "element_vector", adjoint_consistency_term
                )
                integrals[f"penalty_term_{si}"] = assign_matrix(
                    "element_vector", penalty_term
                )
            else:
                lform = (
                    primal_consistency_term + adjoint_consistency_term + penalty_term
                )
                integrals[si] = assign_matrix("element_vector", lform)

        return integrals


def main():

    fes = {
        # "TRI6": Tri6(),
        # "TRI3": Tri3(),
        # "TET4": Tet4(),
        # "TET10": Tet10(),
        # "TET20": Tet20(),
        "HEX8": Hex8(),
        # "HEX27": Hex27(),
        # "AAHEX8": AAHex8(),
        # "AAQUAD4": AxisAlignedQuad4()
    }

    if len(sys.argv) >= 2:
        fe = fes[sys.argv[1]]
    else:
        print("Fallback with Hex8")
        fe = Hex8()

    symbolic_integration = False
    if len(sys.argv) >= 3:
        symbolic_integration = int(sys.argv[2])

    op = SIPDG(fe, symbolic_integration)

    for k, v in op.apply().items():
        print("-----------------------------------")
        print(f"{k})")
        c_code(v)
        print("-----------------------------------")


if __name__ == "__main__":
    main()
