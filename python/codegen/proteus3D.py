#!/usr/bin/env python3

# proteus3D.py (from exodusII manual)

from fe import FE
from sfem_codegen import *
from weighted_fe import *

idx_offset = 0	# or 1 for fortran

def plot_points(points):
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D

	# Sample array of points
	

	# Extract x, y, and z coordinates
	x, y, z = zip(*points)
	print(x)
	print(y)
	print(z)

	# Create the plot
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	for i in range(0, len(x)):
	    # ax.scatter(x[i], y[i], z[i], label=f"{i+1}")
	    ax.scatter(x[i], y[i], z[i])

	    offset = (0.01, 0.01, 0.01)  # adjust offset as needed

	       # Add text label near the point
	    ax.text(x[i] + offset[0], y[i] + offset[1], z[i] + offset[2], f"{i+idx_offset}", zdir=(1, 1, 1))  # adjust zdir for better visibility


	# Set labels and title
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title('3D Scatter Plot with Point Labels')

	# Add legend
	# ax.legend()

	plt.show()

class ProteusTet(FE):
	def __init__(self, order):
		super().__init__()
		self.order = order

		points = []
		
		for k in range(0, order + 1):
			for j in range(0, order - k + 1):
				for i in range(0, order - j - k + 1):
					x = i / (order)
					y = j / (order)
					z = k / (order)
					p = [x, y, z]
					points.append(p)
		

		
		idx = 0
		for p in points:
			print(f'{idx+idx_offset})\t{p[0]},{p[1]},{p[2]}')
			idx += 1

		plot_points(points)
		self.points = points


tet = ProteusTet(3)
