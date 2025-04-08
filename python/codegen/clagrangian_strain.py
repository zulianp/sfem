#!/usr/bin/env python3

from sfem_codegen import *

ux = coeffs("ux", 4)
uy = coeffs("uy", 4)
uz = coeffs("uz", 4)

f = fun(qx, qy, qz)

gradu = sp.Matrix(3, 3, [0.0] * 9)
Id = sp.Matrix(3, 3, [1, 0, 0, 0, 1, 0, 0, 0, 1])

for i in range(0, 4):
    fi = f[i]
    dfdx = sp.diff(fi, qx)
    dfdy = sp.diff(fi, qy)
    dfdz = sp.diff(fi, qz)

    gradu[0, 0] += dfdx * ux[i]
    gradu[0, 1] += dfdy * ux[i]
    gradu[0, 2] += dfdz * ux[i]

    gradu[1, 0] += dfdx * uy[i]
    gradu[1, 1] += dfdy * uy[i]
    gradu[1, 2] += dfdz * uy[i]

    gradu[2, 0] += dfdx * uz[i]
    gradu[2, 1] += dfdy * uz[i]
    gradu[2, 2] += dfdz * uz[i]


# Nicer way
F = Id + gradu
strain = (F.T * F - Id) / 2

# Faster way?
# strain = (gradu.T + gradu + gradu.T * gradu) / 2

expr = []

strain0 = sp.symbols(f"strain[0]")
expr.append(ast.Assignment(strain0, strain[0, 0]))

strain1 = sp.symbols(f"strain[1]")
expr.append(ast.Assignment(strain1, strain[0, 1]))

strain2 = sp.symbols(f"strain[2]")
expr.append(ast.Assignment(strain2, strain[0, 2]))

strain3 = sp.symbols(f"strain[3]")
expr.append(ast.Assignment(strain3, strain[1, 1]))

strain4 = sp.symbols(f"strain[4]")
expr.append(ast.Assignment(strain4, strain[1, 2]))

strain5 = sp.symbols(f"strain[5]")
expr.append(ast.Assignment(strain5, strain[2, 2]))

c_code(expr)

# Principal strains

print("----------------------")
print("Principal strains")
print("----------------------")

s0, s1, s2 = sp.symbols("s0 s1 s2", real=True)
s3, s4, s5 = sp.symbols("s3 s4 s5", real=True)

S = sp.Matrix(3, 3, [s0, s1, s2, s1, s3, s4, s2, s4, s5])
# S.is_positive_semidefinite = True
# print(S.is_positive_semidefinite)

Se = S.eigenvals()

# V, Se = S.diagonalize()
# print(V)

# print("----------------------")
# print("Generic Eigen-decomposition")
# print("----------------------")

# d = 0
# for e in Se:
# 	print(f'{d})')
# 	c_code([sp.simplify(e)])
# 	d+=1

expr = []

d = 0
for e in Se:
    e_subs = e
    for i in range(0, 3):
        for j in range(i, 3):
            e_subs = e_subs.subs(S[i, j], strain[i, j])

    real_e = sp.symbols(f"e[{d}]")
    expr.append(ast.Assignment(real_e, e_subs))
    d += 1


print("----------------------")
print("Specific")
print("----------------------")

c_code(expr)
