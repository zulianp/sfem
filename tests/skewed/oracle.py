#!/usr/bin/env python3

import numpy as np
from numpy import dot

xfactor = 2.
yfactor = 1.
zfactor = 1.

# Patrick
dx = (xfactor*yfactor*zfactor) / 6.0

# Diego
# dx = 1/(xfactor*yfactor*zfactor * 6.0)

# Derivation in 1D
# T(x_hat) = A*x_hat + b
# Reference space (.)_hat

# Geometric transformation
# x is in physical space
# x_hat is in the reference space 
# x_hat = T^{-1}(x)
# x = T(x_hat)

# Example for 1D element of size 2
# x_hat = T^{-1}(x) = x / 2.0
# T(x_hat) = 2 * x_hat

# Basis function
# phi_hat(T^{-1}(x)) = phi(x)
# phi_hat(x_hat) 1 - x_hat

# Gradient for 1D element of size 2
# grad_x phi_hat(x/2) = -1/2

# Bilinear form
# integral()
# -1/2 * -1/2 * 2 = 1/2
# -1 * -1 * 1/2 = 1/2
# dx = det(A) * ref_volume = 2
# grad_x phi(x) = -1/2
# dot( grad_x phi(x), grad_x phi(x) ) = 1/4

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
