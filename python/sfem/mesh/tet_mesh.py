#!/usr/bin/env python3

import numpy as np

x = np.array([0, 1, 0, 0], dtype=np.float32)
y = np.array([0, 0, 1, 0], dtype=np.float32)
z = np.array([0, 0, 0, 1], dtype=np.float32)

i0 = np.array([0], dtype=np.int32)
i1 = np.array([1], dtype=np.int32)
i2 = np.array([2], dtype=np.int32)
i3 = np.array([3], dtype=np.int32)

path = "./mesh"

x.tofile(f'{path}/x.raw')
y.tofile(f'{path}/y.raw')
z.tofile(f'{path}/z.raw')

i0.tofile(f'{path}/i0.raw')
i1.tofile(f'{path}/i1.raw')
i2.tofile(f'{path}/i2.raw')
i3.tofile(f'{path}/i3.raw')
