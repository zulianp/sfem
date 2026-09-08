#!/usr/bin/env python3
"""
Reference TET4 CVFEM Navier-Stokes element kernel with branch-free upwind
convection.

The geometry contract follows the SFEM code-generation framework affine path:

    geometry = jacobian_adjugate + jacobian_determinant

where:

    jacobian_adjugate = det(J) J^{-1}

The hot residual kernel does not load physical shape gradients. For TET4 it
forms reference differences in registers and maps them with:

    grad_x u = grad_X u * jacobian_adjugate / jacobian_determinant

CVFEM subcontrol-surface area vectors are also not stored. They are formed
from reference SCS area vectors with:

    A_x = A_X * jacobian_adjugate

This is the same adjugate/determinant split used by generated affine kernels:
gradients divide by determinant, weak-form area/measure contractions use the
adjugate directly.

Residual sign convention:

  For oriented SCS (i,j), A_ij points from nodal control volume i to j.
  The flux F_ij is added to node i and subtracted from node j. This is an
  outward-flux/divergence residual convention. If your solver stores negative
  divergence, flip the final residual sign once.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse


Vector3 = list[float]
Matrix33 = list[list[float]]
Matrix43 = list[list[float]]


SCS_LEFT = (0, 0, 0, 1, 1, 2)
SCS_RIGHT = (1, 2, 3, 2, 3, 3)

# Oriented reference CVFEM SCS area vectors for edge/SCS pairs:
# (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
REFERENCE_SCS_AREA = (
    (1.0 / 12.0, 1.0 / 24.0, 1.0 / 24.0),
    (1.0 / 24.0, 1.0 / 12.0, 1.0 / 24.0),
    (1.0 / 24.0, 1.0 / 24.0, 1.0 / 12.0),
    (-1.0 / 24.0, 1.0 / 24.0, 0.0),
    (-1.0 / 24.0, 0.0, 1.0 / 24.0),
    (0.0, -1.0 / 24.0, 1.0 / 24.0),
)


@dataclass(frozen=True)
class Tet4AffineGeometry:
    """Affine TET4 geometry in the SFEM codegen adjugate/determinant form."""

    jacobian_adjugate: Matrix33
    jacobian_determinant: float
    volume: float
    scv_volume: list[float]


def _check_3x3(a: Matrix33, name: str) -> Matrix33:
    if len(a) != 3:
        raise ValueError(f"{name} must have 3 rows")
    out: Matrix33 = []
    for row in a:
        if len(row) != 3:
            raise ValueError(f"{name} must have 3 columns")
        out.append([float(row[0]), float(row[1]), float(row[2])])
    return out


def _check_4x3(a: Matrix43, name: str) -> Matrix43:
    if len(a) != 4:
        raise ValueError(f"{name} must have 4 rows")
    out: Matrix43 = []
    for row in a:
        if len(row) != 3:
            raise ValueError(f"{name} must have 3 columns")
        out.append([float(row[0]), float(row[1]), float(row[2])])
    return out


def _check_4(a: list[float], name: str) -> list[float]:
    if len(a) != 4:
        raise ValueError(f"{name} must have length 4")
    return [float(a[0]), float(a[1]), float(a[2]), float(a[3])]


def _determinant_3x3(a: Matrix33) -> float:
    a00, a01, a02 = a[0]
    a10, a11, a12 = a[1]
    a20, a21, a22 = a[2]
    return (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )


def _inverse_3x3(a: Matrix33) -> Matrix33:
    a00, a01, a02 = a[0]
    a10, a11, a12 = a[1]
    a20, a21, a22 = a[2]

    c00 = a11 * a22 - a12 * a21
    c01 = -(a10 * a22 - a12 * a20)
    c02 = a10 * a21 - a11 * a20

    c10 = -(a01 * a22 - a02 * a21)
    c11 = a00 * a22 - a02 * a20
    c12 = -(a00 * a21 - a01 * a20)

    c20 = a01 * a12 - a02 * a11
    c21 = -(a00 * a12 - a02 * a10)
    c22 = a00 * a11 - a01 * a10

    det = a00 * c00 + a01 * c01 + a02 * c02
    if det == 0.0:
        raise ValueError("singular 3x3 matrix")

    inv_det = 1.0 / det
    return [
        [c00 * inv_det, c10 * inv_det, c20 * inv_det],
        [c01 * inv_det, c11 * inv_det, c21 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det],
    ]


def tet4_jacobian_from_coordinates(x_in: Matrix43) -> Matrix33:
    """
    Convenience helper for tests and demos.

    Generated affine kernels do not need coordinates in the hot path; they
    route precomputed adjugate/determinant streams. This helper only builds
    those streams from coordinates for this standalone script.
    """

    x = _check_4x3(x_in, "x")
    return [
        [x[1][0] - x[0][0], x[2][0] - x[0][0], x[3][0] - x[0][0]],
        [x[1][1] - x[0][1], x[2][1] - x[0][1], x[3][1] - x[0][1]],
        [x[1][2] - x[0][2], x[2][2] - x[0][2], x[3][2] - x[0][2]],
    ]


def adjugate_and_determinant_from_jacobian(j_in: Matrix33) -> tuple[Matrix33, float]:
    """Return adjugate = det(J) J^{-1} and determinant = det(J)."""

    j = _check_3x3(j_in, "jacobian")
    determinant = _determinant_3x3(j)
    if determinant <= 0.0:
        raise ValueError("expected positive affine TET4 Jacobian determinant")

    inv_j = _inverse_3x3(j)
    adjugate = [[0.0, 0.0, 0.0] for _ in range(3)]
    for r in range(3):
        for c in range(3):
            adjugate[r][c] = determinant * inv_j[r][c]
    return adjugate, determinant


def affine_geometry_from_adjugate_determinant(
    jacobian_adjugate_in: Matrix33,
    jacobian_determinant: float,
) -> Tet4AffineGeometry:
    """Build the lightweight geometry object from codegen-style streams."""

    adj = _check_3x3(jacobian_adjugate_in, "jacobian_adjugate")
    det = float(jacobian_determinant)
    if det <= 0.0:
        raise ValueError("expected positive affine TET4 Jacobian determinant")
    volume = det / 6.0
    scv = [0.25 * volume, 0.25 * volume, 0.25 * volume, 0.25 * volume]
    return Tet4AffineGeometry(adj, det, volume, scv)


def affine_geometry_from_jacobian(jacobian: Matrix33) -> Tet4AffineGeometry:
    """Demo/test helper: J -> codegen-style adjugate/determinant geometry."""

    adj, det = adjugate_and_determinant_from_jacobian(jacobian)
    return affine_geometry_from_adjugate_determinant(adj, det)


def _coefficient_at_scs(value: float | list[float], s: int, i: int, j: int) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if len(value) == 4:
        return 0.5 * (float(value[i]) + float(value[j]))
    if len(value) == 6:
        return float(value[s])
    raise ValueError("coefficient must be scalar, length 4, or length 6")


def _scs_area_from_adjugate(adj: Matrix33, s: int) -> Vector3:
    """
    A_phys = A_ref * adjugate.

    Since adjugate = det(J) J^{-1}, this is the cofactor transform for oriented
    area vectors on positively oriented affine TET4 elements.
    """

    ar0, ar1, ar2 = REFERENCE_SCS_AREA[s]
    return [
        ar0 * adj[0][0] + ar1 * adj[1][0] + ar2 * adj[2][0],
        ar0 * adj[0][1] + ar1 * adj[1][1] + ar2 * adj[2][1],
        ar0 * adj[0][2] + ar1 * adj[1][2] + ar2 * adj[2][2],
    ]


def tet4_velocity_gradient_from_adjugate(
    geom: Tet4AffineGeometry,
    velocity: Matrix43,
) -> Matrix33:
    """
    Compute grad_x u without loading physical shape gradients.

    Reference TET4 derivative of a nodal field f is:

        grad_X f = [f1 - f0, f2 - f0, f3 - f0]

    Physical gradient:

        grad_x f = grad_X f * adjugate / determinant
    """

    u = _check_4x3(velocity, "velocity")
    adj = geom.jacobian_adjugate
    inv_det = 1.0 / geom.jacobian_determinant

    grad = [[0.0, 0.0, 0.0] for _ in range(3)]
    for comp in range(3):
        d0 = u[1][comp] - u[0][comp]
        d1 = u[2][comp] - u[0][comp]
        d2 = u[3][comp] - u[0][comp]
        grad[comp][0] = (d0 * adj[0][0] + d1 * adj[1][0] + d2 * adj[2][0]) * inv_det
        grad[comp][1] = (d0 * adj[0][1] + d1 * adj[1][1] + d2 * adj[2][1]) * inv_det
        grad[comp][2] = (d0 * adj[0][2] + d1 * adj[1][2] + d2 * adj[2][2]) * inv_det
    return grad


def tet4_cvfem_ns_upwind_residual(
    geom: Tet4AffineGeometry,
    velocity: Matrix43,
    pressure: list[float],
    rho: float | list[float] = 1.0,
    mu: float | list[float] = 0.0,
    divergence_correction: bool = False,
) -> tuple[Matrix43, list[float]]:
    """
    Compute one TET4 CVFEM Navier-Stokes residual.

    Geometry loads in the hot path:

      - jacobian_adjugate[3][3]
      - jacobian_determinant

    No physical gradN table and no physical SCS area table are loaded.
    """

    u = _check_4x3(velocity, "velocity")
    p = _check_4(pressure, "pressure")

    adj = geom.jacobian_adjugate
    inv_det = 1.0 / geom.jacobian_determinant

    u0x, u0y, u0z = u[0][0], u[0][1], u[0][2]
    u1x, u1y, u1z = u[1][0], u[1][1], u[1][2]
    u2x, u2y, u2z = u[2][0], u[2][1], u[2][2]
    u3x, u3y, u3z = u[3][0], u[3][1], u[3][2]

    # Reference differences, exactly what a generated TET4 affine kernel wants.
    dux0 = u1x - u0x
    dux1 = u2x - u0x
    dux2 = u3x - u0x

    duy0 = u1y - u0y
    duy1 = u2y - u0y
    duy2 = u3y - u0y

    duz0 = u1z - u0z
    duz1 = u2z - u0z
    duz2 = u3z - u0z

    # G_ab = partial_b u_a = grad_X u_a * adjugate / determinant.
    g00 = (dux0 * adj[0][0] + dux1 * adj[1][0] + dux2 * adj[2][0]) * inv_det
    g01 = (dux0 * adj[0][1] + dux1 * adj[1][1] + dux2 * adj[2][1]) * inv_det
    g02 = (dux0 * adj[0][2] + dux1 * adj[1][2] + dux2 * adj[2][2]) * inv_det

    g10 = (duy0 * adj[0][0] + duy1 * adj[1][0] + duy2 * adj[2][0]) * inv_det
    g11 = (duy0 * adj[0][1] + duy1 * adj[1][1] + duy2 * adj[2][1]) * inv_det
    g12 = (duy0 * adj[0][2] + duy1 * adj[1][2] + duy2 * adj[2][2]) * inv_det

    g20 = (duz0 * adj[0][0] + duz1 * adj[1][0] + duz2 * adj[2][0]) * inv_det
    g21 = (duz0 * adj[0][1] + duz1 * adj[1][1] + duz2 * adj[2][1]) * inv_det
    g22 = (duz0 * adj[0][2] + duz1 * adj[1][2] + duz2 * adj[2][2]) * inv_det

    div_corr = (2.0 / 3.0) * (g00 + g11 + g22) if divergence_correction else 0.0

    rm = [[0.0, 0.0, 0.0] for _ in range(4)]
    rc = [0.0, 0.0, 0.0, 0.0]

    ux = (u0x, u1x, u2x, u3x)
    uy = (u0y, u1y, u2y, u3y)
    uz = (u0z, u1z, u2z, u3z)

    for s in range(6):
        i = SCS_LEFT[s]
        j = SCS_RIGHT[s]

        ax, ay, az = _scs_area_from_adjugate(adj, s)

        rho_s = _coefficient_at_scs(rho, s, i, j)
        mu_s = _coefficient_at_scs(mu, s, i, j)

        adv_x = 0.5 * (ux[i] + ux[j])
        adv_y = 0.5 * (uy[i] + uy[j])
        adv_z = 0.5 * (uz[i] + uz[j])

        mdot = rho_s * (adv_x * ax + adv_y * ay + adv_z * az)

        # Branch-free first-order upwind split.
        mdot_abs = abs(mdot)
        mdot_pos = 0.5 * (mdot + mdot_abs)
        mdot_neg = 0.5 * (mdot - mdot_abs)

        conv_x = mdot_pos * ux[i] + mdot_neg * ux[j]
        conv_y = mdot_pos * uy[i] + mdot_neg * uy[j]
        conv_z = mdot_pos * uz[i] + mdot_neg * uz[j]

        p_mid = 0.5 * (p[i] + p[j])

        tau_x = mu_s * (
            (2.0 * g00 - div_corr) * ax
            + (g01 + g10) * ay
            + (g02 + g20) * az
        )
        tau_y = mu_s * (
            (g10 + g01) * ax
            + (2.0 * g11 - div_corr) * ay
            + (g12 + g21) * az
        )
        tau_z = mu_s * (
            (g20 + g02) * ax
            + (g21 + g12) * ay
            + (2.0 * g22 - div_corr) * az
        )

        flux_x = conv_x + p_mid * ax - tau_x
        flux_y = conv_y + p_mid * ay - tau_y
        flux_z = conv_z + p_mid * az - tau_z

        rm[i][0] += flux_x
        rm[i][1] += flux_y
        rm[i][2] += flux_z
        rc[i] += mdot

        rm[j][0] -= flux_x
        rm[j][1] -= flux_y
        rm[j][2] -= flux_z
        rc[j] -= mdot

    return rm, rc


N_NODE = 4
N_FIELD = 4
N_DOF = N_NODE * N_FIELD


def _dof(node: int, field: int) -> int:
    return node * N_FIELD + field


def _shape_grad_x(adj: Matrix33, inv_det: float) -> list[Vector3]:
    """Physical TET4 shape gradients from adjugate/determinant."""

    g0 = [0.0, 0.0, 0.0]
    g1 = [0.0, 0.0, 0.0]
    g2 = [0.0, 0.0, 0.0]
    g3 = [0.0, 0.0, 0.0]
    for b in range(3):
        a0 = adj[0][b] * inv_det
        a1 = adj[1][b] * inv_det
        a2 = adj[2][b] * inv_det
        g1[b] = a0
        g2[b] = a1
        g3[b] = a2
        g0[b] = -(a0 + a1 + a2)
    return [g0, g1, g2, g3]


def tet4_cvfem_ns_upwind_jacobian(
    geom: Tet4AffineGeometry,
    velocity: Matrix43,
    pressure: list[float],
    rho: float = 1.0,
    mu: float = 0.0,
) -> list[list[float]]:
    """
    Consistent 16x16 element Jacobian, node-major:

        Ke[(a * 4 + fi) * 16 + (b * 4 + fj)] = d R_{a,fi} / d u_{b,fj}

    Fields: 0=ux, 1=uy, 2=uz, 3=p. Residual: rx, ry, rz, rc.
    d|mdot|/d mdot = sign(mdot), 0 at mdot == 0.
    Scalar rho, mu (matches the C++ spike kernel). No divergence correction.
    """

    u = _check_4x3(velocity, "velocity")
    p = _check_4(pressure, "pressure")
    rho_s = float(rho)
    mu_s = float(mu)
    adj = geom.jacobian_adjugate
    inv_det = 1.0 / geom.jacobian_determinant
    gx = _shape_grad_x(adj, inv_det)

    ux = [u[a][0] for a in range(4)]
    uy = [u[a][1] for a in range(4)]
    uz = [u[a][2] for a in range(4)]

    ke = [[0.0] * N_DOF for _ in range(N_DOF)]

    def add_tau_derivs(ii: int, jj: int, ax: float, ay: float, az: float) -> None:
        for k in range(4):
            gk0, gk1, gk2 = gx[k]
            dtx_ux = mu_s * (2.0 * gk0 * ax + gk1 * ay + gk2 * az)
            dtx_uy = mu_s * (gk0 * ay)
            dtx_uz = mu_s * (gk0 * az)
            dty_ux = mu_s * (gk1 * ax)
            dty_uy = mu_s * (gk0 * ax + 2.0 * gk1 * ay + gk2 * az)
            dty_uz = mu_s * (gk1 * az)
            dtz_ux = mu_s * (gk2 * ax)
            dtz_uy = mu_s * (gk2 * ay)
            dtz_uz = mu_s * (gk0 * ax + gk1 * ay + 2.0 * gk2 * az)
            col_x = _dof(k, 0)
            col_y = _dof(k, 1)
            col_z = _dof(k, 2)
            for row_node, sgn_f in ((ii, 1.0), (jj, -1.0)):
                ke[_dof(row_node, 0)][col_x] -= sgn_f * dtx_ux
                ke[_dof(row_node, 0)][col_y] -= sgn_f * dtx_uy
                ke[_dof(row_node, 0)][col_z] -= sgn_f * dtx_uz
                ke[_dof(row_node, 1)][col_x] -= sgn_f * dty_ux
                ke[_dof(row_node, 1)][col_y] -= sgn_f * dty_uy
                ke[_dof(row_node, 1)][col_z] -= sgn_f * dty_uz
                ke[_dof(row_node, 2)][col_x] -= sgn_f * dtz_ux
                ke[_dof(row_node, 2)][col_y] -= sgn_f * dtz_uy
                ke[_dof(row_node, 2)][col_z] -= sgn_f * dtz_uz

    for s in range(6):
        i = SCS_LEFT[s]
        j = SCS_RIGHT[s]
        ax, ay, az = _scs_area_from_adjugate(adj, s)

        adv_x = 0.5 * (ux[i] + ux[j])
        adv_y = 0.5 * (uy[i] + uy[j])
        adv_z = 0.5 * (uz[i] + uz[j])
        mdot = rho_s * (adv_x * ax + adv_y * ay + adv_z * az)
        mdot_abs = abs(mdot)
        mdot_pos = 0.5 * (mdot + mdot_abs)
        mdot_neg = 0.5 * (mdot - mdot_abs)
        sgn = 0.0 if mdot == 0.0 else (1.0 if mdot > 0.0 else -1.0)
        d_pos = 0.5 * (1.0 + sgn)
        d_neg = 0.5 * (1.0 - sgn)

        dmdot_dux = rho_s * 0.5 * ax
        dmdot_duy = rho_s * 0.5 * ay
        dmdot_duz = rho_s * 0.5 * az

        def conv_col(dmdot_dq: float, duxi: float, duxj: float, duyi: float, duyj: float, duzi: float, duzj: float):
            dpos = d_pos * dmdot_dq
            dneg = d_neg * dmdot_dq
            dcx = dpos * ux[i] + mdot_pos * duxi + dneg * ux[j] + mdot_neg * duxj
            dcy = dpos * uy[i] + mdot_pos * duyi + dneg * uy[j] + mdot_neg * duyj
            dcz = dpos * uz[i] + mdot_pos * duzi + dneg * uz[j] + mdot_neg * duzj
            return dcx, dcy, dcz

        for k in (i, j):
            duxi = 1.0 if k == i else 0.0
            duxj = 1.0 if k == j else 0.0
            dcx, dcy, dcz = conv_col(dmdot_dux, duxi, duxj, 0.0, 0.0, 0.0, 0.0)
            col = _dof(k, 0)
            ke[_dof(i, 0)][col] += dcx
            ke[_dof(i, 1)][col] += dcy
            ke[_dof(i, 2)][col] += dcz
            ke[_dof(j, 0)][col] -= dcx
            ke[_dof(j, 1)][col] -= dcy
            ke[_dof(j, 2)][col] -= dcz
            ke[_dof(i, 3)][col] += dmdot_dux
            ke[_dof(j, 3)][col] -= dmdot_dux

            dcx, dcy, dcz = conv_col(dmdot_duy, 0.0, 0.0, duxi, duxj, 0.0, 0.0)
            col = _dof(k, 1)
            ke[_dof(i, 0)][col] += dcx
            ke[_dof(i, 1)][col] += dcy
            ke[_dof(i, 2)][col] += dcz
            ke[_dof(j, 0)][col] -= dcx
            ke[_dof(j, 1)][col] -= dcy
            ke[_dof(j, 2)][col] -= dcz
            ke[_dof(i, 3)][col] += dmdot_duy
            ke[_dof(j, 3)][col] -= dmdot_duy

            dcx, dcy, dcz = conv_col(dmdot_duz, 0.0, 0.0, 0.0, 0.0, duxi, duxj)
            col = _dof(k, 2)
            ke[_dof(i, 0)][col] += dcx
            ke[_dof(i, 1)][col] += dcy
            ke[_dof(i, 2)][col] += dcz
            ke[_dof(j, 0)][col] -= dcx
            ke[_dof(j, 1)][col] -= dcy
            ke[_dof(j, 2)][col] -= dcz
            ke[_dof(i, 3)][col] += dmdot_duz
            ke[_dof(j, 3)][col] -= dmdot_duz

        dp_mid = 0.5
        for k in (i, j):
            col = _dof(k, 3)
            ke[_dof(i, 0)][col] += dp_mid * ax
            ke[_dof(i, 1)][col] += dp_mid * ay
            ke[_dof(i, 2)][col] += dp_mid * az
            ke[_dof(j, 0)][col] -= dp_mid * ax
            ke[_dof(j, 1)][col] -= dp_mid * ay
            ke[_dof(j, 2)][col] -= dp_mid * az

        add_tau_derivs(i, j, ax, ay, az)

    return ke


def _residual_dof_vector(
    geom: Tet4AffineGeometry,
    velocity: Matrix43,
    pressure: list[float],
    rho: float,
    mu: float,
) -> list[float]:
    rm, rc = tet4_cvfem_ns_upwind_residual(geom, velocity, pressure, rho=rho, mu=mu)
    out = [0.0] * N_DOF
    for a in range(4):
        out[_dof(a, 0)] = rm[a][0]
        out[_dof(a, 1)] = rm[a][1]
        out[_dof(a, 2)] = rm[a][2]
        out[_dof(a, 3)] = rc[a]
    return out


def _fd_jacobian(
    geom: Tet4AffineGeometry,
    velocity: Matrix43,
    pressure: list[float],
    rho: float,
    mu: float,
    eps: float,
) -> list[list[float]]:
    u0 = [[velocity[a][c] for c in range(3)] for a in range(4)]
    p0 = list(pressure)
    ke = [[0.0] * N_DOF for _ in range(N_DOF)]
    inv_2eps = 0.5 / eps
    for col in range(N_DOF):
        node = col // N_FIELD
        field = col % N_FIELD

        u_plus = [[u0[a][c] for c in range(3)] for a in range(4)]
        p_plus = list(p0)
        u_minus = [[u0[a][c] for c in range(3)] for a in range(4)]
        p_minus = list(p0)
        if field < 3:
            u_plus[node][field] += eps
            u_minus[node][field] -= eps
        else:
            p_plus[node] += eps
            p_minus[node] -= eps
        r_plus = _residual_dof_vector(geom, u_plus, p_plus, rho, mu)
        r_minus = _residual_dof_vector(geom, u_minus, p_minus, rho, mu)
        for row in range(N_DOF):
            ke[row][col] = (r_plus[row] - r_minus[row]) * inv_2eps
    return ke


def tet4_kernel_cost_model() -> dict[str, float]:
    """Scalar add/mul/div model matching cvfem_tet4_ns_upwind_simd_microkernel.

    Counts source operations. Does not count abs/ternary neg, float→double
    casts, or residual zero-stores. Three SCS faces have a zero reference-area
    component and use 9 FLOPs instead of 15.
    """

    flops_inv_det = 1.0
    flops_ref_diff = 9.0
    flops_grad = 9.0 * 6.0
    flops_area = 3.0 * 15.0 + 3.0 * 9.0
    flops_scs_body = 6.0 + 6.0 + 4.0 + 2.0 + 27.0 + 18.0 + 8.0
    total = flops_inv_det + flops_ref_diff + flops_grad + flops_area + 6.0 * flops_scs_body
    return {
        "gradient_flops_from_adjugate": flops_inv_det + flops_ref_diff + flops_grad,
        "area_transform_flops": flops_area,
        "upwind_flux_flops_per_scs_excluding_area": flops_scs_body,
        "scs_per_element": 6.0,
        "total_flops": total,
        "connectivity_bytes": 4.0 * 4.0,
        "field_gather_bytes": 4.0 * 4.0 * 8.0,
        "adjugate_plus_determinant_bytes": 10.0 * 4.0,
        "residual_read_write_bytes": 2.0 * 4.0 * 4.0 * 8.0,
        "cold_total_bytes": 16.0 + 128.0 + 40.0 + 256.0,
    }


def _assert_close(a: float, b: float, tol: float, label: str) -> None:
    if abs(a - b) > tol:
        raise AssertionError(f"{label}: {a} != {b}, tol={tol}")


def _assert_vec_close(a: list[float], b: list[float], tol: float, label: str) -> None:
    if len(a) != len(b):
        raise AssertionError(f"{label}: length mismatch")
    for i in range(len(a)):
        _assert_close(a[i], b[i], tol, f"{label}[{i}]")


def _sum_cols_4x3(a: Matrix43) -> Vector3:
    return [
        a[0][0] + a[1][0] + a[2][0] + a[3][0],
        a[0][1] + a[1][1] + a[2][1] + a[3][1],
        a[0][2] + a[1][2] + a[2][2] + a[3][2],
    ]


def run_self_tests() -> None:
    x = [
        [0.0, 0.0, 0.0],
        [2.0, 0.1, 0.0],
        [0.2, 1.7, 0.3],
        [0.1, 0.4, 1.4],
    ]
    geom = affine_geometry_from_jacobian(tet4_jacobian_from_coordinates(x))

    _assert_close(sum(geom.scv_volume), geom.volume, 1e-14, "SCV volume sum")

    velocity = [[0.0, 0.0, 0.0] for _ in range(4)]
    pressure = [0.0, 0.0, 0.0, 0.0]
    for a in range(4):
        xx, yy, zz = x[a]
        velocity[a][0] = 1.0 + 2.0 * xx - 0.5 * yy + 0.25 * zz
        velocity[a][1] = -3.0 + 0.75 * xx + 1.5 * yy - 2.0 * zz
        velocity[a][2] = 0.5 - 1.25 * xx + 0.1 * yy + 0.6 * zz
        pressure[a] = 4.0 - xx + 0.2 * yy + 0.3 * zz

    expected_grad = [
        [2.0, -0.5, 0.25],
        [0.75, 1.5, -2.0],
        [-1.25, 0.1, 0.6],
    ]
    got_grad = tet4_velocity_gradient_from_adjugate(geom, velocity)
    for i in range(3):
        _assert_vec_close(got_grad[i], expected_grad[i], 1e-13, "linear gradient")

    identity_geom = affine_geometry_from_jacobian(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    for s in range(6):
        _assert_vec_close(
            _scs_area_from_adjugate(identity_geom.jacobian_adjugate, s),
            list(REFERENCE_SCS_AREA[s]),
            1e-14,
            "identity SCS area",
        )

    rm, rc = tet4_cvfem_ns_upwind_residual(
        geom,
        velocity,
        pressure,
        rho=[1.0, 1.2, 0.9, 1.1],
        mu=[0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
        divergence_correction=True,
    )

    _assert_vec_close(_sum_cols_4x3(rm), [0.0, 0.0, 0.0], 1e-13, "momentum conservation")
    _assert_close(sum(rc), 0.0, 1e-13, "continuity conservation")
    _assert_close(tet4_kernel_cost_model()["total_flops"], 562.0, 0.0, "folded kernel flop model")

    rho_j = 1.0
    mu_j = 0.01
    ke = tet4_cvfem_ns_upwind_jacobian(geom, velocity, pressure, rho=rho_j, mu=mu_j)
    ke_fd = _fd_jacobian(geom, velocity, pressure, rho_j, mu_j, 1.0e-6)
    max_abs = 0.0
    ke_scale = 0.0
    for row in range(N_DOF):
        for col in range(N_DOF):
            ke_scale = max(ke_scale, abs(ke[row][col]), abs(ke_fd[row][col]))
            max_abs = max(max_abs, abs(ke[row][col] - ke_fd[row][col]))
    rel_norm = max_abs / max(ke_scale, 1.0e-30)
    if rel_norm > 1.0e-8:
        raise AssertionError(f"analytical vs FD Jacobian rel {rel_norm}, abs {max_abs}")

    for col in range(N_DOF):
        for fi in range(3):
            row_sum = sum(ke[_dof(a, fi)][col] for a in range(4))
            _assert_close(row_sum, 0.0, 1e-12, f"momentum Jacobian column {col} field {fi}")
        cont_sum = sum(ke[_dof(a, 3)][col] for a in range(4))
        _assert_close(cont_sum, 0.0, 1e-12, f"continuity Jacobian column {col}")

    v0 = [[0.0, 0.0, 0.0] for _ in range(4)]
    ke0 = tet4_cvfem_ns_upwind_jacobian(geom, v0, pressure, rho=rho_j, mu=mu_j)
    ke0_fd = _fd_jacobian(geom, v0, pressure, rho_j, mu_j, 1.0e-6)
    max_abs0 = 0.0
    ke0_scale = 0.0
    for row in range(N_DOF):
        for col in range(N_DOF):
            ke0_scale = max(ke0_scale, abs(ke0[row][col]), abs(ke0_fd[row][col]))
            max_abs0 = max(max_abs0, abs(ke0[row][col] - ke0_fd[row][col]))
    rel_norm0 = max_abs0 / max(ke0_scale, 1.0e-30)
    if rel_norm0 > 1.0e-5:
        raise AssertionError(f"zero-velocity (kink) Jacobian rel {rel_norm0}, abs {max_abs0}")


def _print_matrix(name: str, a: list[list[float]]) -> None:
    print(name)
    for row in a:
        print("  " + " ".join(f"{v: .10e}" for v in row))


def _print_vector(name: str, a: list[float]) -> None:
    print(name)
    print("  " + " ".join(f"{v: .10e}" for v in a))


def demo() -> None:
    geom = affine_geometry_from_jacobian(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    velocity = [
        [1.0, 0.0, 0.0],
        [1.2, 0.1, 0.0],
        [0.9, 0.4, 0.1],
        [1.1, 0.2, 0.5],
    ]
    pressure = [1.0, 1.1, 0.9, 1.2]

    rm, rc = tet4_cvfem_ns_upwind_residual(
        geom,
        velocity,
        pressure,
        rho=1.0,
        mu=0.01,
        divergence_correction=False,
    )

    _print_matrix("jacobian_adjugate", geom.jacobian_adjugate)
    print("jacobian_determinant")
    print(f"  {geom.jacobian_determinant:.10e}")
    print("volume")
    print(f"  {geom.volume:.10e}")
    _print_vector("scv_volume", geom.scv_volume)
    _print_matrix("velocity gradient from adjugate/determinant", tet4_velocity_gradient_from_adjugate(geom, velocity))
    print("scs_area from adjugate")
    for s in range(6):
        print("  " + " ".join(f"{v: .10e}" for v in _scs_area_from_adjugate(geom.jacobian_adjugate, s)))
    _print_matrix("momentum residual", rm)
    _print_vector("continuity residual", rc)
    _print_vector("local conservative momentum sum", _sum_cols_4x3(rm))
    print("local conservative continuity sum")
    print(f"  {sum(rc): .10e}")
    print("cost model")
    for key, value in tet4_kernel_cost_model().items():
        print(f"  {key}: {value:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reference TET4 CVFEM Navier-Stokes upwind element kernel"
    )
    parser.add_argument("--demo", action="store_true", help="print one example residual")
    args = parser.parse_args()

    run_self_tests()
    if args.demo:
        demo()
    else:
        print("self-tests passed")


if __name__ == "__main__":
    main()
