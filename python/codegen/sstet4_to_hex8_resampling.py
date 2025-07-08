#!/usr/bin/env python3

from sfem_codegen import *
import numpy as np

class SSTet4:
    def __init__(self):
        self.x = coeffs('x', 4)
        self.y = coeffs('y', 4)
        self.z = coeffs('z', 4)
        self.dim = 3
        self.sub_vol =  sp.symbols('sub_vol')

        self.sub_simplices = [
            [0, 4, 6, 7],
            [4, 1, 5, 8],
            [6, 5, 2, 9],
            [7, 8, 9, 3],
            [4, 5, 6, 8],
            [7, 4, 6, 8],
            [6, 5, 9, 8],
            [7, 6, 9, 8]
        ]

        self.edges = [[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]]
        self.unique = [0, 4, 5, 6, 7]

        self.ref_volume = sp.Rational(1, 6)
        self.J_symb = matrix_coeff("J", self.dim, self.dim)
        self.J_det_symb = sp.symbols("J_det")
        self.b_symb = coeffs("b", self.dim)

        ref_points = []
        ref_points.append(sp.zeros(self.dim, 1))
        for d1 in range(0, self.dim):
            p = sp.zeros(self.dim, 1)
            p[d1] = 1
            ref_points.append(p)

        for e in self.edges :
            p = (ref_points[e[0]] + ref_points[e[1]]) / 2
            ref_points.append(p)

        self.ref_points = np.array(ref_points)

    def jacobian(self):
        x = self.x
        y = self.y
        z = self.z
        J = sp.Matrix(3, 3, [
            x[1] - x[0], x[2] - x[0], x[3] - x[0],
            y[1] - y[0], y[2] - y[0], y[3] - y[0],
            z[1] - z[0], z[2] - z[0], z[3] - z[0]
        ])

        return J

    def sub_jacobians(self, J):
        u = J[:, 0]
        v = J[:, 1]
        w = J[:, 2]

        J0 = sp.zeros(self.dim, self.dim)
        J1 = sp.zeros(self.dim, self.dim)
        J2 = sp.zeros(self.dim, self.dim)
        J3 = sp.zeros(self.dim, self.dim)
        J4 = sp.zeros(self.dim, self.dim)
        J5 = sp.zeros(self.dim, self.dim)

        sv = self.sub_vol

        # Cat 0
        J0[:, 0] = u * sv 
        J0[:, 1] = v * sv
        J0[:, 2] = w * sv

        # Cat 1
        J1[:, 0] = -(u + w) * sv 
        J1[:, 1] = w * sv
        J1[:, 2] = (-u + v + w) * sv

        # Cat 2
        J2[:, 0] = v * sv 
        J2[:, 1] = (-u + v + w) * sv
        J2[:, 2] = w * sv

        # Cat 3
        J3[:, 0] = (-u + v) * sv 
        J3[:, 1] = (-u + w) * sv
        J3[:, 2] = (-u + v + w) * sv

        # Cat 4
        J4[:, 0] = (-v + w) * sv 
        J4[:, 1] = w * sv
        J4[:, 2] = (-u + w) * sv

        # Cat 5
        J5[:, 0] = (-u + v) * sv 
        J5[:, 1] = (-u + v + w) * sv
        J5[:, 2] = v * sv

        Js = [J0, J1, J2, J3, J4, J5]
        return Js

    def ref_sub_jacobian_expressions(self):
        Id = sp.eye(self.dim, self.dim)
        Js = self.sub_jacobians(Id)

        expr = []

        cat = 0
        for J in Js:
            expr.extend(assign_matrix(f"J_ref[{cat}]", J))
            cat += 1

        return expr

    def c_gen_ref_sub_jacobian_apply(self):
        Id = sp.eye(self.dim, self.dim)
        Js = self.sub_jacobians(Id)

        x = coeffs("x", self.dim)


        cat = 0
        for J in Js:
            Jx = J * x
            print("//--------------------------")
            print(c_gen(assign_matrix(f"y{cat}", Jx)))
            cat += 1


    def element_template(self):
        x = coeffs('x', self.dim)
        b_micro = coeffs("b_micro", self.dim)
        J_micro_symb = matrix_coeff("J_micro", self.dim, 1)
        JxJ_micro_symb = matrix_coeff("JxJ_micro", self.dim, self.dim)
        translate_symb = matrix_coeff("translate", self.dim, 1)

        J_symb = self.J_symb
        J_det_symb = self.J_det_symb
        b_symb = self.b_symb

        translate = J_symb * b_micro + b_symb

        JxJ_det_micro = J_det_symb * self.sub_vol

        JxJ_micro = J_symb * J_micro_symb
        affine_trafo = JxJ_micro_symb * x + translate_symb

        expr = []
        expr.extend(assign_matrix("translate", translate))
        expr.extend(assign_value("JxJ_det_micro", JxJ_det_micro))
        expr.extend(assign_matrix("p_global", affine_trafo))
        # expr.extend(assign_matrix("JxJ_micro", JxJ_micro))
        expr.extend(assign_matrix("J", self.jacobian()))
        expr.extend(assign_value(f"J_det", determinant(self.jacobian())))
        
        
        return expr
        

    def sub_jacobian_expressions(self):
        Js = self.sub_jacobians(self.J_symb)

        expr = []

        cat = 0
        for J in Js:
            expr.extend(assign_matrix(f"J_micro[{cat}]", J))
            cat += 1

        return expr



sstet4 = SSTet4()


print("-----------------------------")
print("Micro element template")
print("-----------------------------")
print(c_gen(sstet4.element_template()))

print("-----------------------------")
print("Jacobians per category")
print("-----------------------------")
print(c_gen(sstet4.sub_jacobian_expressions()))

print("-----------------------------")
print("Ref Jacobians per category")
print("-----------------------------")
sstet4.c_gen_ref_sub_jacobian_apply()
