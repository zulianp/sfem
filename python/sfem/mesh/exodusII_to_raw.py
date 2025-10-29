#!/usr/bin/env python3

import netCDF4
import numpy as np
import sys
import os
import getopt

# import pdb

try:
    geom_t
except NameError:
    print("exodusII_to_raw: self contained mode")
    geom_t = np.float32
    idx_t = np.int32
    element_idx_t = np.int32


def exodusII_to_raw(input_mesh, output_folder):
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    mkdir(output_folder)
    mkdir(f"{output_folder}/blocks")

    nc = netCDF4.Dataset(input_mesh)

    if "coord" in nc.variables:
        coords = nc.variables["coord"]
    else:
        coords = []
        if "coordx" in nc.variables:
            coordx = nc.variables["coordx"]
            coords.append(coordx)
        if "coordy" in nc.variables:
            coordy = nc.variables["coordy"]
            coords.append(coordy)
        if "coordz" in nc.variables:
            coordz = nc.variables["coordz"]
            coords.append(coordz)

        coords = np.array(coords)

    dims, nnodes = coords.shape

    coordnames = ["x", "y", "z", "t"]

    for i in range(0, dims):
        x = np.array(coords[i, :]).astype(geom_t)
        x.tofile(f"{output_folder}/{coordnames[i]}.raw")

    n_time_steps = 1
    if "time_whole" in nc.variables:
        time_whole = nc.variables["time_whole"]
        n_time_steps = time_whole.shape[0]
        t = np.array(time_whole[:]).astype(np.float32)
        t.tofile(f"{output_folder}/time_whole.raw")

    print(f"n_time_steps = {n_time_steps}")

    if "name_nod_var" in nc.variables:
        name_nod_var = nc.variables["name_nod_var"]
        nvars, __ = name_nod_var.shape
        print(f"Point data, nvars = {nvars}")

        point_data_dir = f"{output_folder}/point_data"
        mkdir(point_data_dir)

        nodal_prefix = "vals_nod_var"
        for i in range(0, nvars):
            var_key = f"{nodal_prefix}{i+1}"
            var = nc.variables[var_key]

            var_name = netCDF4.chartostring(name_nod_var[i, :])
            print(f" - {var_name}, dtype {var.dtype}")

            var_path_prefix = f"{point_data_dir}/{var_name}"

            if n_time_steps <= 1:
                path = f"{var_path_prefix}.raw"

                data = np.array(var[:])
                data.tofile(path)
            else:
                size_padding = int(np.ceil(np.log10(n_time_steps)))

                format_string = f"%s.%0.{size_padding}d.raw"

                for t in range(0, n_time_steps):
                    data = np.array(var[t, :])

                    path = format_string % (var_path_prefix, t)
                    data.tofile(path)

    def s2n_quad4():
        return [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
        ]

    def s2n_hex8():
        return [
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]

    def s2n_tet4():
        return [
            [0, 1, 3],
            [1, 2, 3],
            [0, 3, 2],
            [0, 1, 2],
        ]

    def s2n_tri3():
        return [
            [0, 1],
            [1, 2],
            [2, 0],
        ]

    ss_to_nodelist = {}
    ss_to_nodelist["QUAD4"] = s2n_quad4()
    ss_to_nodelist["HEX8"] = s2n_hex8()
    ss_to_nodelist["HEX"] = s2n_hex8()
    ss_to_nodelist["TET4"] = s2n_tet4()
    ss_to_nodelist["TETRA"] = s2n_tet4()
    ss_to_nodelist["tetra"] = s2n_tet4()
    ss_to_nodelist["tetra4"] = s2n_tet4()
    ss_to_nodelist["TRI3"] = s2n_tri3()

    #########################################
    # Elements
    #########################################

    num_elem = nc.dimensions["num_elem"].size
    print(f"num_elem = {num_elem}")

    num_el_blk = nc.dimensions["num_el_blk"].size
    print(f"num_el_blk = {num_el_blk}")

    if num_el_blk == 1:
        connect = nc.variables["connect1"]
        elem_type = connect.elem_type
        print(f"elem_type = {elem_type}")

        nelements, nnodesxelem = connect.shape

        for i in range(0, nnodesxelem):
            ii = np.array(connect[:, i]).astype(idx_t) - 1
            ii.tofile(f"{output_folder}/i{i}.raw")

    else:
        num_nod_per_el_ref = 0
        for b in range(0, num_el_blk):
            var_name = f"num_nod_per_el{b+1}"
            num_nod_per_el = nc.dimensions[var_name].size
            print(f"{var_name} = {num_nod_per_el}")

            if num_nod_per_el_ref == 0:
                num_nod_per_el_ref = num_nod_per_el
            else:
                assert num_nod_per_el_ref == num_nod_per_el

        connect = np.zeros((num_elem, num_nod_per_el_ref), dtype=idx_t)

        offset = 0
        for b in range(0, num_el_blk):
            connect_b = nc.variables[f"connect{b+1}"]
            elem_type = connect_b.elem_type
            print(f"elem_type = {elem_type}")

            name = None

            try:
                eb_prop_b = nc.variables[f"eb_prop{b+2}"]
                if eb_prop_b != None:
                    name = eb_prop_b.__dict__["name"]
                    name = name.lower()
            except:
                eb_prop_b = None

            # if name != None:
            # if name in exclude:
            # 	print(f'Skipping block: {name}')
            # 	continue
            # print(f'name = {name}')

            nelements, nnodesxelem = connect_b.shape

            block_begin = offset
            block_end = offset + nelements

            if name != None:
                np.array([block_begin, block_end], dtype=np.int64).tofile(
                    f"{output_folder}/blocks/{name}.int64.raw"
                )

            connect[block_begin:block_end, :] = connect_b[:].astype(idx_t)
            offset += nelements

        for i in range(0, nnodesxelem):
            ii = np.array(connect[:, i]).astype(idx_t) - 1
            ii.tofile(f"{output_folder}/i{i}.raw")

    #########################################
    # Sidesets
    #########################################
    num_sidesets = 0

    if "num_side_sets" in nc.dimensions:
        num_sidesets = nc.dimensions["num_side_sets"].size
    else:
        return
    print(f"num_sidesets={num_sidesets}")

    s2n_map = ss_to_nodelist[elem_type]
    nnodesxside = len(s2n_map[0])

    ss_names = nc.variables["ss_names"]

    sideset_dir = f"{output_folder}/sidesets"
    if num_sidesets > 0:
        mkdir(sideset_dir)

    for i in range(0, num_sidesets):
        ssidx = i + 1

        name = netCDF4.chartostring(ss_names[i])

        if name == "":
            name = f"sideset{ssidx}"

        print(f"sideset = {name}")

        key = f"elem_ss{ssidx}"
        e_ss = nc.variables[key]

        key = f"side_ss{ssidx}"
        s_ss = nc.variables[key]

        this_sideset_dir = f"{sideset_dir}/{name}"
        mkdir(this_sideset_dir)

        idx = [None] * nnodesxside
        for d in range(0, nnodesxside):
            idx[d] = []

        parent = np.zeros(len(e_ss[:]), dtype=element_idx_t)
        local_face_idx = np.zeros(len(e_ss[:]), dtype=np.int16)

        for n in range(0, len(e_ss[:])):
            e = e_ss[n] - 1
            s = s_ss[n] - 1

            parent[n] = e
            local_face_idx[n] = s

            lnodes = s2n_map[s]

            for d in range(0, nnodesxside):
                ln = lnodes[d]
                node = connect[e, ln] - 1

                # if(node == 162):
                # 	pdb.set_trace()

                idx[d].append(node)

        local_face_idx.tofile(f"{this_sideset_dir}/lfi.int16.raw")
        # parent.tofile(f"{this_sideset_dir}/parent.{str(element_idx_t.__name__)}.raw")
        parent.tofile(f"{this_sideset_dir}/parent.raw")

        for d in range(0, nnodesxside):
            path = f"{this_sideset_dir}/{name}.{d}.raw"
            ii = np.array(idx[d]).astype(idx_t)
            ii.tofile(path)


if __name__ == "__main__":

    usage = f"usage: {sys.argv[0]} <input_mesh> <output_folder>"

    if len(sys.argv) < 3:
        print(usage)
        exit()

    input_mesh = sys.argv[1]
    output_folder = sys.argv[2]

    try:
        opts, args = getopt.getopt(sys.argv[3:], "h", ["help"])

    except getopt.GetoptError as err:
        print(err)
        print(usage)
        sys.exit(1)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(usage)
            sys.exit()
        # elif opt in ("--exclude"):
        # 	 temp = arg.split(',')
        # 	 for t in temp:
        # 	 	exclude.append(t.lower())
        # 	 print(f'Excluding {exclude}')

    exodusII_to_raw(input_mesh, output_folder)
