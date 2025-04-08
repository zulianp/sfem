#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, PathPatch
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
import mpl_toolkits.mplot3d.art3d as art3d

import taichi as ti
import taichi.math

ENTITY_VERTEX_0 = 0
ENTITY_VERTEX_1 = 1
ENTITY_VERTEX_2 = 2
ENTITY_EDGE_0 = 3
ENTITY_EDGE_1 = 4
ENTITY_EDGE_2 = 5
ENTITY_FACE = 6

# import pdb


@ti.func
def isedge(entity):
    return entity == ENTITY_EDGE_0 or entity == ENTITY_EDGE_1 or entity == ENTITY_EDGE_2


@ti.func
def isface(entity):
    return entity == ENTITY_FACE


@ti.func
def isvertex(entity):
    return (
        entity == ENTITY_VERTEX_0
        or entity == ENTITY_VERTEX_1
        or entity == ENTITY_VERTEX_2
    )


@ti.func
def count_common_nodes(f0, f1):
    ret = 0
    for i in range(0, 3):
        for j in range(0, 3):
            ret += f0[i] == f1[j]
    return ret


@ti.func
def are_adj(f0, f1):
    return count_common_nodes(f0, f1) == 2


@ti.func
def mmax(a, b):
    if a > b:
        return a
    return b


# def mdot(a, b):
# return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

mdot = taichi.math.dot


@ti.func
def point_to_triangle(point, triangle):
    diff = triangle[0] - point
    edge0 = triangle[1] - triangle[0]
    edge1 = triangle[2] - triangle[0]
    a00 = mdot(edge0, edge0)
    a01 = mdot(edge0, edge1)
    a11 = mdot(edge1, edge1)
    b0 = mdot(diff, edge0)
    b1 = mdot(diff, edge1)
    det = a00 * a11 - a01 * a01

    s = a01 * b1 - a11 * b0
    t = a01 * b0 - a00 * b1
    entity = -1

    if s + t <= det:
        if s < 0:
            if t < 0:  # region 4:
                if b0 < 0:
                    t = 0
                    if -b0 >= a00:
                        s = 1
                        entity = ENTITY_VERTEX_1
                    else:
                        s = -b0 / a00
                        entity = ENTITY_EDGE_0
                else:
                    s = 0
                    if b1 >= 0:
                        t = 0
                        entity = ENTITY_VERTEX_0
                    elif -b1 >= a11:
                        t = 1
                        entity = ENTITY_VERTEX_2
                    else:
                        t = -b1 / a11
                        entity = ENTITY_EDGE_2

            else:  # region 3:
                s = 0
                if b1 >= 0:
                    t = 0
                    entity = ENTITY_VERTEX_0
                elif -b1 >= a11:
                    t = 1
                    entity = ENTITY_VERTEX_2
                else:
                    t = -b1 / a11
                    entity = ENTITY_EDGE_2

        elif t < 0:  # region 5
            t = 0
            if b0 >= 0:
                s = 0
                entity = ENTITY_VERTEX_0
            elif -b0 >= a00:
                s = 1
                entity = ENTITY_VERTEX_1
            else:
                s = -b0 / a00
                entity = ENTITY_EDGE_0

        else:  # region 0:
            # minimum at interior point
            s /= det
            t /= det
            entity = ENTITY_FACE
    else:
        if s < 0:  # region 2:
            tmp0 = a01 + b0
            tmp1 = a11 + b1
            if tmp1 > tmp0:

                numer = tmp1 - tmp0
                denom = a00 - 2 * a01 + a11
                if numer >= denom:
                    s = 1
                    t = 0
                    entity = ENTITY_VERTEX_1
                else:
                    s = numer / denom
                    t = 1 - s
                    entity = ENTITY_EDGE_1
            else:
                s = 0
                if tmp1 <= 0:
                    t = 1
                    entity = ENTITY_VERTEX_2
                elif b1 >= 0:
                    t = 0
                    entity = ENTITY_VERTEX_0
                else:
                    t = -b1 / a11
                    entity = ENTITY_EDGE_2

        elif t < 0:  # region 6
            tmp0 = a01 + b1
            tmp1 = a00 + b0
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a00 - 2 * a01 + a11
                if numer >= denom:
                    t = 1
                    s = 0
                    entity = ENTITY_VERTEX_2
                else:
                    t = numer / denom
                    s = 1 - t
                    entity = ENTITY_EDGE_1
            else:
                t = 0
                if tmp1 <= 0:
                    s = 1
                    entity = ENTITY_VERTEX_1
                elif b0 >= 0:
                    s = 0
                    entity = ENTITY_VERTEX_0
                else:
                    s = -b0 / a00
                    entity = ENTITY_EDGE_0

        else:  # region 1:
            numer = a11 + b1 - a01 - b0
            if numer <= 0:
                s = 0
                t = 1
                entity = ENTITY_VERTEX_2
            else:
                denom = a00 - 2 * a01 + a11
                if numer >= denom:
                    s = 1
                    t = 0
                    entity = ENTITY_VERTEX_1
                else:
                    s = numer / denom
                    t = 1 - s
                    entity = ENTITY_EDGE_1

    result = triangle[0] + s * edge0 + t * edge1

    # pdb.set_trace()
    return result, s, t, entity


# p = np.array([0.5, 1, 0.2])
# t0 = np.array([0, 0, 0])
# t1 = np.array([1, 0, 2])
# t2 = np.array([0, 1, 0])

# t = [t0, t1, t2]

# q = point_to_triangle(p, t)

# print(q)

# ax = plt.figure().add_subplot(projection='3d')
# x = [t0[0], t1[0], t2[0]]
# y = [t0[1], t1[1], t2[1]]
# z = [t0[2], t1[2], t2[2]]
# d = p - q

# ax.scatter([p[0]], [p[1]], [p[2]])
# ax.scatter([q[0]], [q[1]], [q[2]], marker='o')
# ax.quiver(q[0], q[1], q[2], d[0], d[1], d[2], color='red')
# ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

# ax.axis('equal')
# # fig.tight_layout()
# plt.show()
