#!/usr/bin/env python3

from fe import FE
import sympy as sp
from sfem_codegen import *

def lagrange_triangle(m, x, y):
	l = [1 - x - y, x, y]

	def L0(x, y):
		return 1 - x - y

	def L1(x, y):
		return x

	def L2(x, y):
		return y

	def lagr(k, x):
		num = 1
		denom = 1

		for i in range(0, k):
			num *= x - sp.Rational(i, m)
			denom *= sp.Rational((k - i), m)
		return num / denom

	def f(i, j, k, x, y):
		return sp.nsimplify(lagr(i, L0(x, y)) * lagr(j, L1(x, y)) * lagr(k, L2(x, y)))

	def lf(i, j, k):
		l0, l1, l2 = sp.symbols('l0 l1 l2', positive=True, real=True)
		return sp.nsimplify(lagr(i, l0) * lagr(j, l1) * lagr(k, l2))

	f = []

	# Corners
	f.append(sp.nsimplify(lf(m, 0, 0)))
	f.append(sp.nsimplify(lf(0, m, 0)))
	f.append(sp.nsimplify(lf(0, 0, m)))
	idx = 3

	# Edges

	for i in range(1, m):
		f.append(sp.nsimplify(lf(m-i, i, 0)))
	idx += (m-1)

	for i in range(1, m):
		f.append(sp.nsimplify(lf(0, m-i, i)))
	idx += (m-1)

	for i in range(1, m):
		f.append(sp.nsimplify(lf(i, 0, m-i)))
	idx += (m-1)

	# Face
	for i in range(1, m-1):
		for j in range(1, m-1):
			for k in range(1, m-1):
				if i + j + k != m:
					continue

				f.append(sp.nsimplify(lf(i, j, k)))

	return f

if __name__ == '__main__':
	x, y = sp.symbols('x y', postive=True, real=True)
	m = 3

	f = lagrange_triangle(m, x, y)

	for i in range(0, len(f)):
		print(f'f[{i}] = {f[i]}')
