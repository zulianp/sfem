#!/usr/bin/env python3

import numpy as np

grad0 = np.array([-1, -1, -1])
grad1 = np.array([1,  0,  0])
grad2 = np.array([0,  1,  0])
grad3 = np.array([0,  0,  1])

A = np.array([
	[np.dot(grad0, grad0) / 6, np.dot(grad0, grad1) / 6, np.dot(grad0, grad2) / 6, np.dot(grad0, grad3) / 6],
	[np.dot(grad1, grad0) / 6, np.dot(grad1, grad1) / 6, np.dot(grad1, grad2) / 6, np.dot(grad1, grad3) / 6],
	[np.dot(grad2, grad0) / 6, np.dot(grad2, grad1) / 6, np.dot(grad2, grad2) / 6, np.dot(grad2, grad3) / 6],
	[np.dot(grad3, grad0) / 6, np.dot(grad3, grad1) / 6, np.dot(grad3, grad2) / 6, np.dot(grad3, grad3) / 6]]
)

print(A)