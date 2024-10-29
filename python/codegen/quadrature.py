#!/usr/bin/env python3

from sympy.integrals.quadrature import gauss_lobatto, gauss_legendre

order = 3

x, w = gauss_legendre(order, 10)


print("1D")
# Points
print("Points")
for xi in x:
	xi = (xi+1)/2
	print(xi)

print("Weights")
sum_w = 0
for wi in w:
	sum_w += wi

for wi in w:
	wi = wi	/ sum_w
	print(wi)



n = len(w)

print("3D")

x3 = []
y3 = []
z3 = []
w3 = []

sum_w3 = 0
for i in range(0, n):
	for j in range(0, n):
		for k in range(0, n):
			ww = w[i] * w[j] * w[k] / (sum_w*sum_w*sum_w)
			sum_w3 += ww
			w3.append(ww)
			x3.append(x[i])
			y3.append(x[j])
			z3.append(x[k])


print("\nx: ", x3)
print("\ny: ", y3)
print("\nz: ", z3)
print("\nw: ", w3)




