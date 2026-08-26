#!/usr/bin/env python3
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np


def usage():
    print(
        "usage: sample_point_displacement.py <output.xdmf> <x> <y> <z> [output.csv]",
        file=sys.stderr,
    )


def hdf_dataset_name(text):
    match = re.match(r"[^:]+:/([^\\s]+)", text.strip())
    if not match:
        raise RuntimeError(f"Could not parse HDF data item: {text}")
    return match.group(1)


def main():
    if len(sys.argv) not in (5, 6):
        usage()
        return 1

    xdmf_path = Path(sys.argv[1])
    target = np.array([float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])])
    csv_path = Path(sys.argv[5]) if len(sys.argv) == 6 else xdmf_path.with_name("sampled_displacement.csv")
    h5_path = xdmf_path.with_suffix(".h5")

    tree = ET.parse(xdmf_path)
    root = tree.getroot()
    temporal_grids = []
    for grid in root.findall(".//Grid"):
        time = grid.find("Time")
        if time is not None:
            temporal_grids.append(grid)

    if not temporal_grids:
        raise RuntimeError(f"No temporal grids found in {xdmf_path}")

    geometry_item = root.find('.//Grid[@Name="mesh"]/Geometry/DataItem')
    if geometry_item is None or geometry_item.text is None:
        raise RuntimeError(f"No mesh geometry found in {xdmf_path}")

    times = []
    disp_datasets = []
    for grid in temporal_grids:
        time = grid.find("Time")
        times.append(float(time.attrib["Value"]))

        disp_attr = None
        for attr in grid.findall("Attribute"):
            if attr.attrib.get("Name") == "disp":
                disp_attr = attr
                break

        if disp_attr is None:
            raise RuntimeError("Temporal grid is missing disp attribute")

        data_item = disp_attr.find("DataItem")
        if data_item is None or data_item.text is None:
            raise RuntimeError("disp attribute is missing a DataItem")
        disp_datasets.append(hdf_dataset_name(data_item.text))

    with h5py.File(h5_path, "r") as h5:
        points = h5[hdf_dataset_name(geometry_item.text)][:]
        node = int(np.argmin(np.linalg.norm(points - target, axis=1)))
        displacement = np.empty((len(times), 3), dtype=np.float64)
        for i, dataset in enumerate(disp_datasets):
            displacement[i, :] = h5[dataset][node, :]

    times = np.asarray(times)
    uy = displacement[:, 1]

    minima = []
    maxima = []
    for i in range(1, len(uy) - 1):
        if uy[i - 1] > uy[i] <= uy[i + 1]:
            minima.append(i)
        if uy[i - 1] < uy[i] >= uy[i + 1]:
            maxima.append(i)

    crossings = np.where(uy[:-1] * uy[1:] <= 0)[0]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "ux", "uy", "uz"])
        for t, u in zip(times, displacement):
            writer.writerow([f"{t:.17g}", f"{u[0]:.17g}", f"{u[1]:.17g}", f"{u[2]:.17g}"])

    print(f"CSV: {csv_path}")
    print(f"Nearest node: {node}")
    print(f"Node position: [{points[node, 0]}, {points[node, 1]}, {points[node, 2]}]")
    print(f"Frames: {len(times)}")
    print(f"Time range: [{times[0]}, {times[-1]}]")
    print(f"u_y start/end/min/max: {uy[0]} {uy[-1]} {uy.min()} {uy.max()}")
    print("Zero crossings:")
    for i in crossings[:16]:
        print(f"  [{times[i]}, {times[i + 1]}]")
    print("Local maxima:")
    for i in maxima[:8]:
        print(f"  t={times[i]} u_y={uy[i]}")
    print("Local minima:")
    for i in minima[:8]:
        print(f"  t={times[i]} u_y={uy[i]}")
    if len(maxima) > 1:
        periods = np.diff(times[maxima])
        print("Period estimates from maxima:")
        for p in periods[:8]:
            print(f"  {p}")
    if len(minima) > 1:
        periods = np.diff(times[minima])
        print("Period estimates from minima:")
        for p in periods[:8]:
            print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
