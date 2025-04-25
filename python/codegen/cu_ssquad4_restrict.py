#!/usr/bin/env python3

# Pseudo code for CUDA kernel. Tiled linear interpolation on semi-structured meshes
# (TILE_SIZE={4,8} for {2,3}D)

import numpy as np

from_level = 8
to_level = 4
padding = (from_level % 2) == 0

step_factor = int(from_level / to_level)
nsteps = int(from_level / step_factor)

n_points = step_factor + 1
phi0 = np.linspace(1.0, 0.0, num=n_points)
phi1 = np.linspace(0.0, 1.0, num=n_points)

# Padding for simpler logic
if padding:
    phi0 = np.append(phi0, 0)
    phi1 = np.append(phi1, 0)

# phi is generated on the host and passed to the kernel to expose parallelism
phi = np.zeros((2, len(phi0)))
phi[0, :] = phi0[:]
phi[1, :] = phi1[:]

# Padding, bounds check in CUDA
rH = np.zeros((to_level + 1 + padding, to_level + 1 + padding))

# Padding, bounds check in CUDA
rh = np.zeros((from_level + 1 + padding, from_level + 1 + padding))

for i in range(0, from_level + 1):
    for j in range(0, from_level + 1):
        # rh[i, j] = i * (to_level + 1) + j
        # rh[i, j] = i
        rh[i, j] = 1

# Pseudo code for cuda tile level workload
for i in range(0, to_level + padding):
    for j in range(0, to_level + padding):

        # Parallel execution on different threads ?
        for k1 in range(0, 2):
            for k2 in range(0, 2):

                l_start = i * step_factor
                m_start = j * step_factor

                l_odd = l_start % 2 == 1
                m_odd = m_start % 2 == 1

                l_end = step_factor + l_odd
                m_end = step_factor + m_odd

                if i == to_level:
                    l_end = l_odd + 1

                if j == to_level:
                    m_end = m_odd + 1

                for l in range(l_odd, l_end, 2):
                    for m in range(m_odd, m_end, 2):
                        ll = l_start + l
                        mm = m_start + m

                        # Round robin execution
                        for d1 in range(0, 2):
                            for d2 in range(0, 2):
                                c = rh[ll + d1, mm + d2]  # Divide by count factor
                                # Register add?
                                rH[i + k1, j + k2] += (
                                    c * phi[k1, l + d1] * phi[k2, m + d2]
                                )

                # Atomic add to global memory ...


# Remove padding for output (bounds check in CUDA)
rH = rH[0 : (to_level + 1), 0 : (to_level + 1)]
print(np.round(rH, 2))
