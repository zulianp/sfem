#!/usr/bin/env python3

for level in range(1, 6):
	corner_nodes = 8
	edge_nodes = 12 * (level - 1);
	face_nodes = 6 * (level - 1) * (level - 1);
	vol_nodes = (level - 1) * (level - 1) * (level - 1);

	surface_nodes = corner_nodes + edge_nodes + face_nodes

	print("----------------------------------")
	print(f"Level\t{level}")
	print(f"corners\t{corner_nodes}")
	print(f"edges\t{edge_nodes}")
	print(f"faces\t{face_nodes}")
	print(f"surf\t{surface_nodes}")
	print(f"vols\t{vol_nodes}")
	print("--")
	print("Vol/Surf ratio")
	print(vol_nodes, " / ", surface_nodes, " = ", float(vol_nodes)/surface_nodes)
