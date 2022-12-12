#!/usr/bin/env python3

import numpy as np
from numpy import dot

xfactor = 2.
yfactor = 1.
zfactor = 1.

dx = (xfactor*yfactor*zfactor) / 6.0

grad0 = np.array([-1/xfactor, -1/yfactor, -1/zfactor])
grad1 = np.array([1/xfactor,  0,  0])
grad2 = np.array([0,  1/yfactor,  0])
grad3 = np.array([0,  0,  1/zfactor])

A = np.array([
	[dot(grad0, grad0) * dx, dot(grad0, grad1) * dx, dot(grad0, grad2) * dx, dot(grad0, grad3) * dx],
	[dot(grad1, grad0) * dx, dot(grad1, grad1) * dx, dot(grad1, grad2) * dx, dot(grad1, grad3) * dx],
	[dot(grad2, grad0) * dx, dot(grad2, grad1) * dx, dot(grad2, grad2) * dx, dot(grad2, grad3) * dx],
	[dot(grad3, grad0) * dx, dot(grad3, grad1) * dx, dot(grad3, grad2) * dx, dot(grad3, grad3) * dx]]
)

print(A)
