#!/usr/bin/env python3

import numpy as np
import subprocess

x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
removed = np.array([0, 2, 3], dtype=np.int32)

testx = []
for i in range(0, len(x)):
	if not i in removed:
		testx.append(x[i])

np.array(testx, dtype=np.float64).tofile("testx.raw")
removed.tofile("removed.raw")

subprocess.run("../../remap_vector testx.raw removed.raw testout.raw", shell=True, check=True, capture_output=True)

expected = np.array([0, 2, 0, 0, 5])
actual = np.fromfile("testout.raw")

diff = np.sum(np.abs(expected - actual))

if diff > 0:
	print(f'expected: {expected}')
	print(f'actual:   {actual}')
