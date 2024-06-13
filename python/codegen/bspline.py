#!/usr/bin/env python3

from sfem_codegen import *

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def bspline(i, order, t):
	if order == 1:
		return sp.Piecewise((1, (t >= i)), (0, True)) * sp.Piecewise((1, (t < i+1)), (0, True))
	else:
		return (t - i)/(i+order-1 - i) * bspline(i, order-1, t) + (i+order - t)/(i+order - (i+1)) * bspline(i+1, order-1, t) 

order = 5
extras = 0
nctrl = int((order + 1) + extras)

# if False:
if True:	
	x = np.linspace(0, nctrl-1, nctrl)
	y = np.sin(x * (2*3.14/nctrl)) + np.sin(x * (4.4*3.14/nctrl))

	n = 100
	t = np.linspace(0, nctrl-1, n)
	yt = np.zeros(t.shape)

	b = [0] * nctrl

	for j in range(0, nctrl):
		b[j] = np.zeros(t.shape)

	for i in range(0, n):
		ti = t[i]

		f = 0
		for j in range(0, nctrl):
			bj = bspline(j-(order+1)/2, order+1, ti)
			b[j][i] = bj
			f += y[j] * bj

		yt[i] = f


	fig, axs = plt.subplots(2)
	fig.suptitle('Quintic Uniform B-Splines')
	axs[0].plot(x, y)
	axs[0].axvspan(2, 3, color='green', alpha=0.2)

	axs[0].plot(x, y, marker='8')
	axs[0].plot(t, yt)
	axs[0].set_title("Evalutation (valid interval is [2-3])")

	for j in range(0, nctrl):
		axs[1].plot(t, b[j])

	axs[1].set_title("Basis functions")
	fig.tight_layout()
	plt.savefig("bspline.pdf")

expr = []
t = sp.symbols('t', positive=True)
for j in range(0, nctrl):
	s = bspline(j-(order+1)/2,order+1, t)
	var = sp.symbols(f'w[{j}]') 
	expr.append(ast.Assignment(var, s))

# Generated code is garbage with this solution
# c_code(expr)
