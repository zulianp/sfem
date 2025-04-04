#!/usr/bin/env python3

import getopt
import glob
import os
import sys
import numpy as np


def mkdir(output_folder):
    """
    Create a directory if it does not already exist.

    Args:
        output_folder (str): The path of the directory to create.

    Returns:
        None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


try:
    geom_t
except NameError:
    print("submesh.py: self contained mode")
    geom_t = np.float32
    idx_t = np.int32
    element_idx_t = np.int32

if __name__ == "__main__":
    usage = f"usage: {sys.argv[0]} <input_folder> <element_indices> <output_folder>"

    if len(sys.argv) < 4:
        print(usage)
        sys.exit(1)

    input_folder = sys.argv[1]
    element_indices = sys.argv[2]
    output_folder = sys.argv[3]
    verbose = False
    points_folder = input_folder

    try:
        opts, args = getopt.getopt(sys.argv[4:], "hv", ["verbose"])
    except getopt.GetoptError as err:
        print(err)
        print(usage)
        sys.exit(1)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(usage)
            sys.exit()
        elif opt in ("-v", "--verbose"):
            verbose = True

    if verbose:
        print(f"Input folder: {input_folder}")
        print(f"Element_indices: {element_indices}")
        print(f"Output mesh: {output_folder}")

    # Avoid ensure increasing order and avoid duplicates
    selection = np.unique(np.sort(np.fromfile(element_indices, dtype=element_idx_t)))

    mkdir(output_folder)

    ifiles = glob.glob(f"{input_folder}/i*.raw", recursive=False)
    for path in ifiles:
        if verbose:
            print(f"Reading {path}...")

        ii = np.fromfile(path, dtype=idx_t)
        iis = ii[selection]
        iis.tofile(f"{output_folder}/{os.path.basename(path)}")

    selection.tofile(
        f"{output_folder}/element_mapping.{str(element_idx_t.__name__)}.raw"
    )
