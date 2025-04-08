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
from quad4 import *

from sympy import pprint


def simplify(x):
    # return sp.simplify(x)
    return x


class SIPDG:

    def __init__(self, fe):
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

        integrals = []
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
                dS = perp(g)

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
                s_gv = subsmat(gv[j], v_qp_sym, v_qp)
                primal_consistency_term[j] = v[j] * inner(dS / 2, guh_sym)
                adjoint_consistency_term[j] = uh_sym * inner(dS / 2, s_gv)
                penalty_term[j] = v[j] * uh_sym * norm_dS * tau

                primal_consistency_term[j] = simplify(primal_consistency_term[j] * qw)

                adjoint_consistency_term[j] = simplify(adjoint_consistency_term[j] * qw)

                penalty_term[j] = simplify(penalty_term[j] * qw)

            blocks = {}
            blocks["INTERP"] = assign_value("uh", s_uh)
            blocks["INTERP"].extend(assign_matrix("guh", s_guh))

            # split_ops = True
            split_ops = False
            if split_ops:
                blocks[f"PRIMAL_CONSISTENCY_TERM"] = add_assign_matrix(
                    "element_vector", primal_consistency_term
                )
                blocks[f"ADJOINT_CONSISTENCY_TERM"] = add_assign_matrix(
                    "element_vector", adjoint_consistency_term
                )
                blocks[f"PENALTY_TERM"] = add_assign_matrix(
                    "element_vector", penalty_term
                )
            else:
                lform = (
                    primal_consistency_term + adjoint_consistency_term + penalty_term
                )
                blocks[f"FORM"] = add_assign_matrix("element_vector", lform)

            integrals.append(blocks)

        return integrals

    def apply_code(self):
        tpl = """
static void SFEM_INLINE dg_{NAME}_sip_{FACE}(
    {COORDS}
    const scalar_t * const SFEM_RESTRICT jacobian_inverse,
    {QUAD_POINTS}
    const scalar_t qw,
    const scalar_t tau,
    const scalar_t * const SFEM_RESTRICT u,
    scalar_t * const SFEM_RESTRICT element_vector)
{{  
    scalar_t uh;
    scalar_t guh[{DIM}];
    {{
        {INTERP}
    }}

    {FORM}
}}
"""

        coordnames = ["x", "y", "z"]
        arg_coords = ""

        dim = self.fe.spatial_dim()
        for i in range(0, dim):
            arg_coords += f"const scalar_t * const SFEM_RESTRICT {coordnames[i]},"
            if i < dim - 1:
                arg_coords += "\n"

        arg_quad_points = ""
        for i in range(0, dim - 1):
            arg_quad_points += f"const scalar_t q{coordnames[i]},"

            if i < dim - 2:
                arg_coords += "\n"

        blocks = self.apply()
        for b in range(0, len(blocks)):
            block = blocks[b]

            code = tpl.format(
                NAME=self.fe.name().lower(),
                COORDS=arg_coords,
                QUAD_POINTS=arg_quad_points,
                FACE=b,
                DIM=dim,
                INTERP=c_gen(block["INTERP"]),
                FORM=c_gen(block["FORM"]),
            )

            print(code)


def main():

    fes = {
        # "TRI6": Tri6(),
        # "TRI3": Tri3(),
        # "TET4": Tet4(),
        # "TET10": Tet10(),
        # "TET20": Tet20(),
        "HEX8": Hex8(),
        "QUAD4": Quad4(),
        # "HEX27": Hex27(),
        # "AAHEX8": AAHex8(),
        # "AAQUAD4": AxisAlignedQuad4()
    }

    if len(sys.argv) >= 2:
        fe = fes[sys.argv[1]]
    else:
        print("// Fallback with Hex8")
        fe = Hex8()

    op = SIPDG(fe)
    op.apply_code()


if __name__ == "__main__":
    main()
