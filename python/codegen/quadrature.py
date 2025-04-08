#!/usr/bin/env python3

import sys
from sympy.integrals.quadrature import gauss_lobatto, gauss_legendre

order = int(sys.argv[1])
qp_type = "gauss_legendre"

if len(sys.argv) > 2:
    qp_type = sys.argv[2]

if qp_type == "gauss_legendre":
    x, w = gauss_legendre(order, 10)
else:
    x, w = gauss_lobatto(order, 10)

n = len(w)

print("// ------------------------------")
print("// 1D")
print("// ------------------------------")

# Points
for i in range(0, n):
    x[i] = (x[i] + 1) / 2

sum_w = 0
for i in range(0, n):
    sum_w += w[i]

for i in range(0, n):
    w[i] = w[i] / sum_w

print(f"#define line_q{n}_n {n}")
print(f"static const scalar_t line_q{n}_x[line_q{n}_n] = ", end="{")
print(*x, sep=", ", end="};\n")
print(f"static const scalar_t line_q{n}_w[line_q{n}_n] = ", end="{")
print(*w, sep=", ", end="};\n")

print("// ------------------------------")
print("// 3D")
print("// ------------------------------")

x3 = []
y3 = []
z3 = []
w3 = []

sum_w3 = 0
for i in range(0, n):
    for j in range(0, n):
        for k in range(0, n):
            ww = w[i] * w[j] * w[k]
            sum_w3 += ww
            w3.append(ww)
            x3.append(x[i])
            y3.append(x[j])
            z3.append(x[k])

print(f"#define hex_q{n}_n {n}")
print(f"static const scalar_t hex_q{n}_x[hex_q{n}_n] = ", end="{")
print(*x3, sep=", ", end="};\n")

print(f"static const scalar_t hex_q{n}_y[hex_q{n}_n] = ", end="{")
print(*y3, sep=", ", end="};\n")

print(f"static const scalar_t hex_q{n}_z[hex_q{n}_n] = ", end="{")
print(*z3, sep=", ", end="};\n")

print(f"static const scalar_t hex_q{n}_w[hex_q{n}_n] = ", end="{")
print(*w3, sep=", ", end="};\n")
