#!/usr/bin/env python3


# Pseudo code for CUDA kernel. Tiled linear interpolation on semi-structured meshes
# (TILE_SIZE={4,8} for {2,3}D)

import numpy as np

from_level = 1
to_level = 3
padding = (to_level % 2) == 0

step_factor = int(to_level / from_level)
nsteps = int(to_level / step_factor)

n_points = step_factor + 1
phi0 = np.linspace(1.0, 0.0, num=n_points)
phi1 = np.linspace(0.0, 1.0, num=n_points)

# Padding for simpler logic
if padding:
    phi0 = np.append(phi0, 0)
    phi1 = np.append(phi1, 0)

# phi is generated on the host and passed to the kernel
phi = np.zeros((2, len(phi0)))
phi[0, :] = phi0[:]
phi[1, :] = phi1[:]

# Padding, bounds check in CUDA
uH = np.ones((from_level + 1 + padding, from_level + 1 + padding))

for i in range(0, from_level + 1):
    for j in range(0, from_level + 1):
        uH[i, j] = i * (from_level + 1) + j

# Padding, bounds check in CUDA
uh = np.zeros((to_level + 1 + padding, to_level + 1 + padding))

# Pseudo code for cuda tile level workload
for i in range(0, from_level + padding):
    for j in range(0, from_level + padding):

        # Parallel read copy to shared mem (bounds check)
        c = [uH[i, j], uH[i, j + 1], uH[i + 1, j], uH[i + 1, j + 1]]

        l_start = i * step_factor
        m_start = j * step_factor

        l_odd = l_start % 2 == 1
        m_odd = m_start % 2 == 1

        l_end = step_factor + l_odd
        m_end = step_factor + m_odd

        if i == from_level:
            l_end = l_odd + 1

        if j == from_level:
            m_end = m_odd + 1

        for l in range(l_odd, l_end, 2):
            for m in range(m_odd, m_end, 2):
                ll = l_start + l
                mm = m_start + m

                # Parallel execution on different threads
                for d1 in range(0, 2):
                    for d2 in range(0, 2):

                        acc = 0
                        for k1 in range(0, 2):
                            for k2 in range(0, 2):
                                acc += (
                                    c[k1 * 2 + k2] * phi[k1, l + d1] * phi[k2, m + d2]
                                )

                        uh[ll + d1, mm + d2] += acc

# Remove padding for output (bounds check in CUDA)
uh = uh[0 : (to_level + 1), 0 : (to_level + 1)]
print(np.round(uh, 2))
