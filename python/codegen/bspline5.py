#!/usr/bin/env python3

from sfem_codegen import *

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Reference implementation
def bspline(i, order, t):
	if order == 1:
		return sp.Piecewise((1, (t >= i)), (0, True)) * sp.Piecewise((1, (t < i+1)), (0, True))
	else:
		return (t - i)/(i+order-1 - i) * bspline(i, order-1, t) + (i+order - t)/(i+order - (i+1)) * bspline(i+1, order-1, t) 

# Optimized for interval [2-3]
def bspline5(i, order, t):
	if order == 1:
		if i == 2:
			return 1
		else:
			return 0
	else:
		return (t - i)/(i+order-1 - i) * bspline5(i, order-1, t) + (i+order - t)/(i+order - (i+1)) * bspline5(i+1, order-1, t) 

order = 5
extras = 0
nctrl = int((order + 1) + extras)

x = np.linspace(0, nctrl-1, nctrl)
y = np.sin(x * (2*3.14/nctrl)) + np.sin(x * (4.4*3.14/nctrl))

n = 10
t = np.linspace(2, 3, n)
yt = np.zeros(t.shape)
ytref = np.zeros(t.shape)

b = [0] * nctrl
bref = [0] * nctrl

for j in range(0, nctrl):
	b[j] = np.zeros(t.shape)
	bref[j] = np.zeros(t.shape)

for i in range(0, n):
	ti = t[i]

	f = 0
	fref = 0
	for j in range(0, nctrl):
		b[j][i] = bspline5(j-(order+1)/2, order+1, ti)
		f += y[j] * b[j][i] 

		bref[j][i] = bspline(j-(order+1)/2, order+1, ti)
		fref += y[j] * bref[j][i]

	yt[i] = f
	ytref[i] = fref


fig, axs = plt.subplots(2)
fig.suptitle('Quintic Uniform B-Splines')
axs[0].plot(x, y)
axs[0].axvspan(2, 3, color='green', alpha=0.2)

axs[0].plot(x, y, marker='8')
axs[0].plot(t, yt, label='Optimized')
axs[0].plot(t, ytref,  linestyle='None', marker='.', label='Reference')
axs[0].set_title("Evaluation (assumed in interval is [2-3])")
axs[0].legend()

for j in range(0, nctrl):
	axs[1].plot(t, b[j],  label=f'b{j}')
	axs[1].plot(t, bref[j],  linestyle='None', marker='.', label=f'ref{j}')

axs[1].set_title("Basis functions")
fig.tight_layout()
plt.savefig("bspline5.pdf")

expr = []
t = sp.symbols('t', positive=True)
for j in range(0, nctrl):
	s = bspline5(j-(order+1)/2,order+1, t)
	var = sp.symbols(f'w[{j}]') 
	expr.append(ast.Assignment(var, s))

c_code(expr)
