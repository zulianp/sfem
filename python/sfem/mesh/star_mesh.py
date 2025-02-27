#!/usr/bin/env python3

import os
import numpy as np
import sys


# Parameters for the star
num_points = 5  # Number of points in the star
outer_radius = 0.5  # Outer radius
inner_radius = 0.15  # Inner radius
center = np.array([0.5, 0.5])  # Center of the star

if len(sys.argv) > 1:
     num_points = int(sys.argv[1])


# Generate angles for all points (outer and inner vertices)
angles = np.linspace(0, 2*np.pi, num_points*2, endpoint=False)

# Initialize arrays for x,y coordinates
x_coords = np.zeros(num_points*2, dtype=np.float32)
y_coords = np.zeros(num_points*2, dtype=np.float32)

# Generate coordinates
for i in range(num_points*2):
    # Alternate between outer and inner radius
    current_radius = outer_radius if i % 2 == 0 else inner_radius
    x_coords[i] = center[0] + current_radius * np.cos(angles[i])
    y_coords[i] = center[1] + current_radius * np.sin(angles[i])

# Add the first point at the end to close the shape
x_coords = np.append(x_coords, x_coords[0])
y_coords = np.append(y_coords, y_coords[0])

n = len(x_coords) - 1  # Subtract 1 since last point equals first
i0 = np.arange(n, dtype=np.int32)
i1 = np.arange(1, n+1, dtype=np.int32)

os.path.exists('star') or os.makedirs('star')

# Save arrays to binary files
x_coords.astype(np.float32).tofile('star/x.raw')
y_coords.astype(np.float32).tofile('star/y.raw')
i0.astype(np.int32).tofile('star/i0.raw')
i1.astype(np.int32).tofile('star/i1.raw')

