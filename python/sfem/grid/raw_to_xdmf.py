#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os


def raw_to_xdmf(path_to_meta_file):
    # Initial variable definition.
    ox = oy = oz = 0
    nx = ny = nz = 0
    dx = dy = dz = 1
    block_size = 0
    time_steps = 0
    endianess = tp = precision = number_type = ""
    attribute_type = "Vector"
    binary_path = ""
    rpath = True

    with open(path_to_meta_file, "r") as f:
        Lines = f.readlines()
        for i in Lines:
            if i[:4] == "nx: ":
                nx = int(i[4:])
            elif i[:4] == "ny: ":
                ny = int(i[4:])
            elif i[:4] == "nz: ":
                nz = int(i[4:])
            elif i[:4] == "ox: ":
                ox = float(i[4:])
            elif i[:4] == "oy: ":
                oy = float(i[4:])
            elif i[:4] == "oz: ":
                oz = float(i[4:])
            elif i[:4] == "dx: ":
                dx = float(i[4:])
            elif i[:4] == "dy: ":
                dy = float(i[4:])
            elif i[:4] == "dz: ":
                dz = float(i[4:])
            elif i[:11] == "endianess: ":
                endianess = i[11:]
                endianess = endianess.replace("\n", "")
            elif i[:12] == "block_size: ":
                block_size = int(i[12:])
            elif i[:12] == "time_steps: ":
                time_steps = int(i[12:])
            elif i[:6] == "path: ":
                binary_path = i[6:]
            elif i[:6] == "rpath: ":
                rpath = int(i[7:])
            elif i[:6] == "type: ":
                tp = i[6:]
                if tp == "long\n":
                    precision = "8"
                    number_type = "Int"
                elif tp == "double\n":
                    precision = "8"
                    number_type = "Float"
                elif tp == "float\n":
                    precision = "4"
                    number_type = "Float"
                elif tp == "int\n":
                    precision = "4"
                    number_type = "Int"
                elif tp == "char\n":
                    precision = "1"
                    number_type = "Int"

    path = os.path.abspath(path_to_meta_file)
    pdir = os.path.dirname(path)
    fname, fextension = os.path.splitext(binary_path)
    fname = os.path.basename(fname)

    if rpath:
        binary_path = pdir + "/" + binary_path

    print(binary_path)

    if pdir == "":
        pdir = "./"

    if not os.path.exists(pdir):
        os.mkdir(f"{pdir}")

    path = pdir
    xdmf_path = fname + ".xdmf"
    file_name = fname

    # Define attribute_type
    if block_size == 1:
        attribute_type = "Scalar"

    # #################################TIME-VARIANT#########################################
    time_string = ""
    if int(time_steps) > 1:
        assert False

    #         n_grids = int(time_steps)
    #         time_string_header = \
    #             """<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
    # <Xdmf
    #     xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">
    #     <Domain>
    #         <Topology name="topo" TopologyType="3DCoRectMesh"
    #             Dimensions="{dim}">
    #         </Topology>
    #         <Geometry name="geo" Type="ORIGIN_DXDYDZ">
    #             <!-- Origin -->
    #             <DataItem Format="XML" Dimensions="3">
    #                 0.0 0.0 0.0
    #             </DataItem>
    #             <!-- DxDyDz -->
    #             <DataItem Format="XML" Dimensions="3">
    #                 1 1 1
    #             </DataItem>
    #         </Geometry>""".format(dim=''
    #                  + str(nx) + ' ' + str(ny) + ' ' + str(nz) + '')
    #         time_string_global = \
    #             """
    #         <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
    #             <Time TimeType="HyperSlab">
    #                 <DataItem Format="XML" NumberType="{number_type}" Dimensions="3">
    #                     <!-- start stride count-->
    #                     0.0 1.0 {n_grids}
    #                 </DataItem>
    #             </Time>""".format(n_grids=n_grids,
    #                 number_type=number_type)
    #         time_string_local_final = ''
    #         for i in range(0, int(time_steps)):
    #             time_string_local_final = time_string_local_final \
    #                 + """
    #             <Grid Name="{grid_name}" GridType="Uniform">
    #                 <Topology Reference="/Xdmf/Domain/Topology[1]"/>
    #                 <Geometry Reference="/Xdmf/Domain/Geometry[1]"/>
    #                     <Attribute Name="{file_name}" Center="Node" AttributeType="{attribute_type}">
    #                         <DataItem Format="Binary" Precision="{precision}" Endian="{endianess}"
    #                             Dimensions="{dim} {block_size}" NumberType="{number_type}">
    #                                 {file_name_t}.raw
    #                         </DataItem>
    #                     </Attribute>
    #             </Grid>""".format(
    #                 dim='' + str(nx) + ' ' + str(ny) + ' ' + str(nz) + '',
    #                 endianess=endianess,
    #                 file_name=file_name,
    #                 file_name_t=file_name + '_t' + str(i),
    #                 precision=precision,
    #                 number_type=number_type,
    #                 block_size=block_size,
    #                 attribute_type=attribute_type,
    #                 grid_name='T' + str(i),
    #                 )
    #         time_footer = """
    #         </Grid>
    #     </Domain>
    # </Xdmf>"""
    #         time_string = time_string_header + time_string_global \
    #             + time_string_local_final + time_footer

    # #################################NORMAL-VARIANT#########################################

    xdmf_string = """<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf 
    xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">
    <Domain>
        <Topology name="topo" TopologyType="3DCoRectMesh"
            Dimensions="{dim}">
        </Topology>
        <Geometry name="geo" Type="ORIGIN_DXDYDZ">
            <!-- Origin -->
            <DataItem Format="XML" Dimensions="3">
                {oz} {oy} {ox}
            </DataItem>
            <!-- DxDyDz -->
            <DataItem Format="XML" Dimensions="3">
                {dz} {dy} {dx}
            </DataItem>
        </Geometry>
        <Grid Name="T1" GridType="Uniform">
            <Topology Reference="/Xdmf/Domain/Topology[1]"/>
            <Geometry Reference="/Xdmf/Domain/Geometry[1]"/>
            <Attribute Name="U" Center="Node" AttributeType="{attribute_type}">
                <DataItem Format="Binary" Dimensions="{dim} {block_size}" Endian="{endianess}" Precision="{precision}" NumberType="{number_type}">
                    {binary_path}
                </DataItem>
            </Attribute>
        </Grid>
    </Domain>
</Xdmf>""".format(
        dim="" + str(nz) + " " + str(ny) + " " + str(nx) + "",
        ox=ox,
        oy=oy,
        oz=oz,
        dx=dx,
        dy=dy,
        dz=dz,
        endianess=endianess,
        binary_path=binary_path,
        block_size=block_size,
        precision=precision,
        number_type=number_type,
        attribute_type=attribute_type,
    )

    # print(f'Writing {xdmf_path} ...')
    # print(xdmf_string)

    # if int(time_steps) == 1:
    textfile = open(xdmf_path, "w")
    textfile.write(xdmf_string)
    textfile.close()
    # else:
    #     textfile = open(xdmf_path, 'w')
    #     textfile.write(time_string)
    #     textfile.close()


if __name__ == "__main__":
    raw_to_xdmf(sys.argv[1])
