#!/usr/bin/env python3

from aahex8 import *


def simplify(val):
    if abs(val) < 1e-15:
        return 0
    return val


def sum(M):
    rows, cols = M.shape

    ret = 0
    for i in range(0, rows):
        for j in range(0, cols):
            ret += M[i, j]
    return ret


# Compute interpolation/projection
def mass_matrix(level, interpolation):
    fe = AAHex8()

    x = sp.Matrix(8, 1, [0, 1, 1, 0, 0, 1, 1, 0])
    y = sp.Matrix(8, 1, [0, 0, 1, 1, 0, 0, 1, 1])
    z = sp.Matrix(8, 1, [0, 0, 0, 0, 1, 1, 1, 1])
    q = sp.Matrix(3, 1, [qx, qy, qz])
    f = fe.fun(q)

    h = 1.0 / level
    h3 = h * h * h

    ndofs = (level + 1) * (level + 1) * (level + 1)
    ld = [1, (level + 1), (level + 1) * (level + 1)]

    B = sp.zeros(ndofs, 8)

    if not interpolation:
        D = sp.zeros(ndofs, ndofs)
        D_e = sp.zeros(8, 8)

        # sub-element matrix
        sum_D_e = 0
        for l in range(0, 8):
            for m in range(0, 8):
                D_e[l, m] = sp.integrate(
                    f[l] * f[m] * h3, (q[2], 0, 1), (q[1], 0, 1), (q[0], 0, 1)
                )
                sum_D_e += D_e[l, m]

        # print(sum_D_e)
        assert abs(sum_D_e - h3) < 1e-16

    def idx(xi, yi, zi):
        return xi * ld[0] + yi * ld[1] + zi * ld[2]

    for zi in range(0, level):
        for yi in range(0, level):
            for xi in range(0, level):
                print(f"cell ({xi}, {yi}, {zi})")

                tx = xi * h
                ty = yi * h
                tz = zi * h

                ii = [
                    # Bottom
                    idx(xi, yi, zi),
                    idx(xi + 1, yi, zi),
                    idx(xi + 1, yi + 1, zi),
                    idx(xi, yi + 1, zi),
                    # Top
                    idx(xi, yi, zi + 1),
                    idx(xi + 1, yi, zi + 1),
                    idx(xi + 1, yi + 1, zi + 1),
                    idx(xi, yi + 1, zi + 1),
                ]

                if interpolation:
                    B_e = sp.zeros(8, fe.n_nodes())
                    for l in range(0, 8):
                        q_m = sp.Matrix(
                            3, 1, [tx + x[l] * h, ty + y[l] * h, tz + z[l] * h]
                        )
                        f_m = fe.fun(q_m)
                        for m in range(0, len(f_m)):
                            B_e[l, m] = f_m[m]

                    for l in range(0, 8):
                        for m in range(0, 8):
                            B[ii[l], m] = B_e[l, m]
                else:
                    qx_m = tx + q[0] * h
                    qy_m = ty + q[1] * h
                    qz_m = tz + q[2] * h

                    q_m = sp.Matrix(3, 1, [qx_m, qy_m, qz_m])
                    f_m = fe.fun(q_m)

                    B_e = sp.zeros(8, len(f_m))
                    sum_Be = 0
                    for l in range(0, 8):
                        for m in range(0, len(f_m)):
                            B_e[l, m] = sp.integrate(
                                f[l] * f_m[m] * h3,
                                (q[2], 0, 1),
                                (q[1], 0, 1),
                                (q[0], 0, 1),
                            )
                            sum_Be += B_e[l, m]

                    # print(sum_Be)
                    assert abs(sum_Be - h3) < 1e-16

                    for l in range(0, 8):
                        for m in range(0, 8):
                            D[ii[l], ii[m]] += D_e[l, m]

                        for m in range(0, len(f_m)):
                            B[ii[l], m] += B_e[l, m]

    if not interpolation:
        print(D)
        print(sum(D - D.T))

        D_inv = D.inv()

        return D_inv * B
    else:
        return B


if __name__ == "__main__":
    level = 8
    T = mass_matrix(level, True)
    # T = mass_matrix(level, False) # Does not work yet

    ndofs = (level + 1) * (level + 1) * (level + 1)
    print(f"Op {ndofs}, {8}")
    measure = 0
    for i in range(0, ndofs):
        row_sum = 0
        for j in range(0, 8):
            val = simplify(T[i, j])
            measure += val
            row_sum += val

            print(f"{val}", end="\t")
        print(f"\t(sum={row_sum})\n")

    print(f"measure={measure}")
