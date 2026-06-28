import ctypes
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import sympy as sp

sys.path.insert(0, os.path.dirname(__file__))

from symbolic import (
    DeformationGradient,
    DimensionSpecialization,
    ExpressionRole,
    KernelExpressions,
    KernelTemplateParameter,
    LayoutKind,
    ReferenceShapeGradients,
    ScopeKind,
    data_layout,
    displacement_gradient_from_reference,
    execution_scope,
    generate_cpp_kernel,
    generate_openmp_cpp_kernel,
    generate_sfem_soa_cpp_files_for_element,
    hessian_action_from_energy,
    matrix_inner,
    residual_from_energy,
    sfem_element_quadrature_rule,
    sfem_supported_element_types,
    sfem_soa_element_specialization,
    sfem_soa_element_specializations,
    sfem_soa_kernel_form,
    sfem_soa_weak_form,
    vector_symbols,
)


def neohookean_ogden_energy(F, mu, lmbda):
    dim = F.shape[0]
    J = F.det()
    I1 = matrix_inner(F, F)
    logJ = sp.log(J)
    return mu * sp.Rational(1, 2) * (I1 - dim) - mu * logJ + (
        lmbda * sp.Rational(1, 2) * logJ * logJ
    )


def matrix_determinant(A):
    dim = len(A)
    if dim == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    if dim == 3:
        return (
            A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
            - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
            + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
        )
    raise ValueError("unsupported matrix dimension")


def matrix_inverse(A):
    det = matrix_determinant(A)
    if abs(det) == 0.0:
        raise ValueError("singular matrix")
    dim = len(A)
    if dim == 2:
        return (
            (A[1][1] / det, -A[0][1] / det),
            (-A[1][0] / det, A[0][0] / det),
        )
    if dim == 3:
        c00 = A[1][1] * A[2][2] - A[1][2] * A[2][1]
        c01 = -(A[1][0] * A[2][2] - A[1][2] * A[2][0])
        c02 = A[1][0] * A[2][1] - A[1][1] * A[2][0]
        c10 = -(A[0][1] * A[2][2] - A[0][2] * A[2][1])
        c11 = A[0][0] * A[2][2] - A[0][2] * A[2][0]
        c12 = -(A[0][0] * A[2][1] - A[0][1] * A[2][0])
        c20 = A[0][1] * A[1][2] - A[0][2] * A[1][1]
        c21 = -(A[0][0] * A[1][2] - A[0][2] * A[1][0])
        c22 = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        return (
            (c00 / det, c10 / det, c20 / det),
            (c01 / det, c11 / det, c21 / det),
            (c02 / det, c12 / det, c22 / det),
        )
    raise ValueError("unsupported matrix dimension")


def matrix_transpose(A):
    return tuple(tuple(A[row][col] for row in range(len(A))) for col in range(len(A[0])))


def matrix_multiply(A, B):
    rows = len(A)
    cols = len(B[0])
    inner = len(B)
    return tuple(
        tuple(sum(A[row][k] * B[k][col] for k in range(inner)) for col in range(cols))
        for row in range(rows)
    )


def matrix_add(A, B):
    return tuple(
        tuple(A[row][col] + B[row][col] for col in range(len(A[0])))
        for row in range(len(A))
    )


def matrix_scale(alpha, A):
    return tuple(tuple(alpha * value for value in row) for row in A)


def neohookean_first_piola(F, mu, lmbda):
    det_F = matrix_determinant(F)
    F_inv = matrix_inverse(F)
    F_inv_t = matrix_transpose(F_inv)
    pressure = lmbda * math.log(det_F) - mu
    return matrix_add(matrix_scale(mu, F), matrix_scale(pressure, F_inv_t))


def neohookean_linearized_first_piola(F, dF, mu, lmbda):
    det_F = matrix_determinant(F)
    F_inv = matrix_inverse(F)
    F_inv_t = matrix_transpose(F_inv)
    dF_t = matrix_transpose(dF)
    trace_F_inv_dF = sum(
        F_inv[row][col] * dF[col][row]
        for row in range(len(F))
        for col in range(len(F))
    )
    pressure = lmbda * math.log(det_F) - mu
    dF_inv_t = matrix_scale(-1.0, matrix_multiply(matrix_multiply(F_inv_t, dF_t), F_inv_t))
    return matrix_add(
        matrix_add(matrix_scale(mu, dF), matrix_scale(lmbda * trace_F_inv_dF, F_inv_t)),
        matrix_scale(pressure, dF_inv_t),
    )


def reference_gradients_at_q(quadrature_rule, q):
    dim = quadrature_rule.dim
    n_shape = quadrature_rule.n_shape
    return tuple(
        tuple(
            quadrature_rule.reference_gradients[(q * n_shape + shape) * dim + component]
            for component in range(dim)
        )
        for shape in range(n_shape)
    )


def physical_jacobian(coords, grad_ref):
    dim = len(coords[0])
    return tuple(
        tuple(
            sum(coords[shape][row] * grad_ref[shape][col] for shape in range(len(coords)))
            for col in range(dim)
        )
        for row in range(dim)
    )


def displacement_reference_gradient(values, grad_ref):
    dim = len(values[0])
    return tuple(
        tuple(
            sum(values[shape][row] * grad_ref[shape][col] for shape in range(len(values)))
            for col in range(dim)
        )
        for row in range(dim)
    )


def identity_plus(A):
    dim = len(A)
    return tuple(
        tuple((1.0 if row == col else 0.0) + A[row][col] for col in range(dim))
        for row in range(dim)
    )


def reference_neohookean_gradient(quadrature_rule, coords, displacement, mu, lmbda):
    dim = quadrature_rule.dim
    n_shape = quadrature_rule.n_shape
    result = [[0.0 for _ in range(dim)] for _ in range(n_shape)]
    for q, qw in enumerate(quadrature_rule.weights):
        grad_ref = reference_gradients_at_q(quadrature_rule, q)
        J = physical_jacobian(coords, grad_ref)
        det_J = matrix_determinant(J)
        adjugate = matrix_scale(det_J, matrix_inverse(J))
        grad_u_ref = displacement_reference_gradient(displacement, grad_ref)
        grad_u = matrix_multiply(grad_u_ref, matrix_scale(1.0 / det_J, adjugate))
        P = neohookean_first_piola(identity_plus(grad_u), mu, lmbda)
        loperand = matrix_scale(qw, matrix_multiply(P, matrix_transpose(adjugate)))
        for shape in range(n_shape):
            for row in range(dim):
                result[shape][row] += sum(
                    loperand[row][col] * grad_ref[shape][col] for col in range(dim)
                )
    return tuple(tuple(row) for row in result)


def reference_neohookean_apply(quadrature_rule, coords, displacement, direction, mu, lmbda):
    dim = quadrature_rule.dim
    n_shape = quadrature_rule.n_shape
    result = [[0.0 for _ in range(dim)] for _ in range(n_shape)]
    for q, qw in enumerate(quadrature_rule.weights):
        grad_ref = reference_gradients_at_q(quadrature_rule, q)
        J = physical_jacobian(coords, grad_ref)
        det_J = matrix_determinant(J)
        adjugate = matrix_scale(det_J, matrix_inverse(J))
        transform = matrix_scale(1.0 / det_J, adjugate)
        grad_u = matrix_multiply(displacement_reference_gradient(displacement, grad_ref), transform)
        trial_grad = matrix_multiply(displacement_reference_gradient(direction, grad_ref), transform)
        dP = neohookean_linearized_first_piola(identity_plus(grad_u), trial_grad, mu, lmbda)
        loperand = matrix_scale(qw, matrix_multiply(dP, matrix_transpose(adjugate)))
        for shape in range(n_shape):
            for row in range(dim):
                result[shape][row] += sum(
                    loperand[row][col] * grad_ref[shape][col] for col in range(dim)
                )
    return tuple(tuple(row) for row in result)


def element_geometry_stream_values(quadrature_rule, coords):
    dim = quadrature_rule.dim
    adjugate_streams = [[] for _ in range(dim * dim)]
    determinant_stream = []
    for q in range(quadrature_rule.n_qp):
        grad_ref = reference_gradients_at_q(quadrature_rule, q)
        J = physical_jacobian(coords, grad_ref)
        det_J = matrix_determinant(J)
        adjugate = matrix_scale(det_J, matrix_inverse(J))
        determinant_stream.append(det_J)
        for row in range(dim):
            for col in range(dim):
                adjugate_streams[row * dim + col].append(adjugate[row][col])
    return adjugate_streams + [determinant_stream]


def field_stream_values(values):
    dim = len(values[0])
    return [[values[shape][component]] for shape in range(len(values)) for component in range(dim)]


def read_field_stream_values(streams, dim, n_shape):
    return tuple(
        tuple(streams[shape * dim + component][0] for component in range(dim))
        for shape in range(n_shape)
    )


def affine_coords(reference_coords, A, b):
    dim = len(reference_coords[0])
    return tuple(
        tuple(b[row] + sum(A[row][col] * coord[col] for col in range(dim)) for row in range(dim))
        for coord in reference_coords
    )


def shear_displacement(coords):
    dim = len(coords[0])
    values = []
    for coord in coords:
        if dim == 2:
            values.append((0.17 * coord[1], -0.05 * coord[0]))
        else:
            values.append((0.17 * coord[1] + 0.03 * coord[2], -0.04 * coord[0], 0.02 * coord[1]))
    return tuple(values)


def max_abs_difference(left, right):
    return max(
        abs(left[shape][component] - right[shape][component])
        for shape in range(len(left))
        for component in range(len(left[0]))
    )


def max_abs_value(values):
    return max(
        abs(values[shape][component])
        for shape in range(len(values))
        for component in range(len(values[0]))
    )


def c_double_array(values):
    return (ctypes.c_double * len(values))(*values)


def reference_element_coords(element_type):
    if element_type == "TRI3":
        return ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    if element_type == "TET4":
        return (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    if element_type == "HEX8":
        return (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        )
    raise ValueError("unsupported element type")


def deformed_element_coords(element_type):
    coords = reference_element_coords(element_type)
    dim = len(coords[0])
    if dim == 2:
        return affine_coords(coords, ((1.23, 0.19), (0.11, 0.94)), (0.07, -0.03))
    return affine_coords(
        coords,
        (
            (1.21, 0.13, 0.08),
            (0.07, 0.97, 0.14),
            (0.04, 0.09, 1.16),
        ),
        (0.05, -0.02, 0.04),
    )


def nonaffine_hex8_coords():
    coords = [list(coord) for coord in reference_element_coords("HEX8")]
    coords[5][0] += 0.04
    coords[5][2] -= 0.03
    coords[6][0] += 0.06
    coords[6][1] -= 0.04
    coords[6][2] += 0.12
    coords[7][1] += 0.03
    coords[7][2] += 0.05
    return tuple(tuple(coord) for coord in coords)


def generated_neohookean_weak_form_files(element_type, prefix, vector_size=16, local_prefix=None):
    specialization = sfem_soa_element_specialization(element_type, vector_size=vector_size)
    dim = specialization.dim
    F = sp.Matrix(
        dim,
        dim,
        tuple(sp.symbols("F[%d]" % i) for i in range(dim * dim)),
    )
    weak_form = sfem_soa_weak_form(neohookean_ogden_energy(F, *sp.symbols("mu lmbda")), F)
    return specialization, generate_sfem_soa_cpp_files_for_element(
        (
            sfem_soa_kernel_form(
                "gradient",
                weak_form=weak_form,
                output_mode="accumulate",
            ),
            sfem_soa_kernel_form(
                "apply",
                weak_form=weak_form,
                has_direction=True,
                output_mode="accumulate",
            ),
        ),
        prefix=prefix,
        specialization=specialization,
        local_prefix=local_prefix,
    )


def compile_generated_shared_library(compiler, tmpdir, generated_files, operator_filename, library_name):
    for generated in generated_files:
        with open(os.path.join(tmpdir, generated.path), "w", encoding="utf-8") as output:
            output.write(generated.source)

    library_path = os.path.join(tmpdir, library_name)
    source_path = os.path.join(tmpdir, operator_filename)
    if sys.platform == "darwin":
        command = [
            compiler,
            "-std=c++11",
            "-O3",
            "-fPIC",
            "-dynamiclib",
            source_path,
            "-o",
            library_path,
        ]
    else:
        command = [
            compiler,
            "-std=c++11",
            "-O3",
            "-fPIC",
            "-shared",
            source_path,
            "-o",
            library_path,
        ]
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return ctypes.CDLL(library_path)


def call_generated_neohookean_kernel(
    library,
    prefix,
    element_type,
    form_name,
    quadrature_rule,
    coords,
    displacement,
    direction,
    mu,
    lmbda,
    isoparametric=False,
):
    dim = quadrature_rule.dim
    n_shape = quadrature_rule.n_shape
    pointer_type = ctypes.POINTER(ctypes.c_double)
    geometry = [c_double_array(stream) for stream in element_geometry_stream_values(quadrature_rule, coords)]
    coordinates = [c_double_array(stream) for stream in field_stream_values(coords)]
    u_streams = [c_double_array(stream) for stream in field_stream_values(displacement)]
    h_streams = [c_double_array(stream) for stream in field_stream_values(direction)] if direction is not None else []
    outputs = [c_double_array((0.0,)) for _ in range(n_shape * dim)]
    function = getattr(
        library,
        "%s_%s_%s_%ssoa"
        % (
            prefix,
            element_type.lower(),
            form_name,
            "isoparametric_" if isoparametric else "",
        ),
    )
    function.restype = ctypes.c_int
    geometry_or_coordinates = coordinates if isoparametric else geometry
    pointer_count = len(geometry_or_coordinates) + len(u_streams) + len(h_streams) + len(outputs)
    function.argtypes = (
        [ctypes.c_ssize_t]
        + [pointer_type] * len(geometry_or_coordinates)
        + [ctypes.c_double, ctypes.c_double]
        + [pointer_type] * (pointer_count - len(geometry_or_coordinates))
    )
    status = function(
        ctypes.c_ssize_t(1),
        *geometry_or_coordinates,
        ctypes.c_double(mu),
        ctypes.c_double(lmbda),
        *u_streams,
        *h_streams,
        *outputs,
    )
    if status != 0:
        raise RuntimeError("generated kernel returned status %d" % status)
    return read_field_stream_values(outputs, dim, n_shape)


def c_pointer_array(arrays, ctype):
    pointer_type = ctypes.POINTER(ctype)
    return (pointer_type * len(arrays))(*arrays)


def call_generated_neohookean_mesh_kernel(
    library,
    prefix,
    element_type,
    form_name,
    quadrature_rule,
    coords,
    displacement,
    direction,
    mu,
    lmbda,
    geometry_mode,
    scalar_ctype=ctypes.c_double,
):
    dim = quadrature_rule.dim
    n_shape = quadrature_rule.n_shape
    real_pointer = ctypes.POINTER(scalar_ctype)
    geometry_ctype = ctypes.c_double
    geometry_pointer = ctypes.POINTER(geometry_ctype)
    idx_pointer = ctypes.POINTER(ctypes.c_ssize_t)
    scalar_array = lambda values: (scalar_ctype * len(values))(*values)
    element_arrays = [(ctypes.c_ssize_t * 1)(shape) for shape in range(n_shape)]
    elements = (idx_pointer * n_shape)(*element_arrays)
    coordinate_arrays = [
        (geometry_ctype * len(component_values))(*component_values)
        for component_values in zip(*coords)
    ]
    points = c_pointer_array(coordinate_arrays, geometry_ctype)
    u_global = [scalar_array(component_values) for component_values in zip(*displacement)]
    h_global = [scalar_array(component_values) for component_values in zip(*direction)] if direction is not None else []
    outputs = [scalar_array((0.0,) * n_shape) for _ in range(dim)]
    suffix = "_float" if scalar_ctype is ctypes.c_float else ""
    function = getattr(
        library,
        "%s_%s_%s_%s_mesh_soa%s"
        % (prefix, element_type.lower(), form_name, geometry_mode, suffix),
    )
    function.restype = ctypes.c_int
    base_argtypes = [ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.POINTER(idx_pointer)]
    base_args = [ctypes.c_ssize_t(1), ctypes.c_ssize_t(n_shape), elements]
    if geometry_mode == "affine":
        geometry_streams = element_geometry_stream_values(quadrature_rule, coords)
        geometry_arrays = [
            scalar_array((stream[0],)) for stream in geometry_streams
        ]
        base_argtypes.extend([real_pointer] * len(geometry_arrays))
        base_args.extend(geometry_arrays)
    elif geometry_mode == "isoparametric":
        base_argtypes.append(ctypes.POINTER(geometry_pointer))
        base_args.append(points)
    else:
        raise ValueError("unsupported mesh geometry mode")

    function.argtypes = (
        base_argtypes
        + [scalar_ctype, scalar_ctype, ctypes.c_ssize_t]
        + [real_pointer] * dim
        + ([ctypes.c_ssize_t] + [real_pointer] * dim if direction is not None else [])
        + [ctypes.c_ssize_t]
        + [real_pointer] * dim
    )
    status = function(
        *base_args,
        scalar_ctype(mu),
        scalar_ctype(lmbda),
        ctypes.c_ssize_t(1),
        *u_global,
        *([ctypes.c_ssize_t(1)] + h_global if direction is not None else []),
        ctypes.c_ssize_t(1),
        *outputs,
    )
    if status != 0:
        raise RuntimeError("generated mesh kernel returned status %d" % status)
    return tuple(tuple(outputs[component][shape] for component in range(dim)) for shape in range(n_shape))


def compiler_vectorization_flags(compiler):
    version = subprocess.run(
        [compiler, "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.lower()
    if "clang" in version:
        return ["-O3", "-Rpass=loop-vectorize", "-Werror=pass-failed"], "clang"
    if "gcc" in version or "g++" in version:
        return ["-O3", "-fopt-info-vec-optimized"], "gcc"
    return None, "unknown"


def assert_generated_lane_loops_vectorized(
    test_case,
    compiler,
    source_path,
    object_path,
    local_header="generated_neohookean_ogden_local.hpp",
    minimum_matches=3,
):
    flags, compiler_kind = compiler_vectorization_flags(compiler)
    if flags is None:
        test_case.skipTest("compiler does not expose a supported vectorization report")

    completed = subprocess.run(
        [
            compiler,
            "-std=c++11",
            *flags,
            "-c",
            source_path,
            "-o",
            object_path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    report = completed.stdout + completed.stderr
    if compiler_kind == "clang":
        pattern = r"%s:\d+:\d+: remark: vectorized loop" % re.escape(local_header)
    else:
        pattern = r"%s:.*loop vectorized" % re.escape(local_header)

    matches = re.findall(pattern, report)
    test_case.assertGreaterEqual(
        len(matches),
        minimum_matches,
        "expected generated local lane loops to be vectorized; "
        "compiler report was:\n%s" % report,
    )


class NeoHookeanOgdenFrameworkTest(unittest.TestCase):
    def test_sfem_element_specialization_api_covers_relevant_elements(self):
        required = (
            "TET4",
            "HEX8",
            "HEX27",
            "PROTEUS_HEX8",
            "PROTEUS_HEX27",
            "PROTEUS_HEX64",
            "TET10",
            "QUAD4",
            "TRI3",
            "TRI6",
        )
        self.assertEqual(set(sfem_supported_element_types()), set(required))

        specializations = sfem_soa_element_specializations(required, vector_size=16)
        by_type = {specialization.element_type: specialization for specialization in specializations}

        expected_shape = {
            "TRI3": (2, 3, 1),
            "TRI6": (2, 6, 3),
            "QUAD4": (2, 4, 4),
            "TET4": (3, 4, 1),
            "TET10": (3, 10, 4),
            "HEX8": (3, 8, 8),
            "HEX27": (3, 27, 27),
            "PROTEUS_HEX8": (3, 8, 8),
            "PROTEUS_HEX27": (3, 27, 27),
            "PROTEUS_HEX64": (3, 64, 64),
        }
        for element_type, (dim, n_shape, n_qp) in expected_shape.items():
            specialization = by_type[element_type]
            self.assertEqual(specialization.dim, dim)
            self.assertEqual(specialization.n_shape, n_shape)
            self.assertEqual(specialization.n_qp, n_qp)
            self.assertEqual(specialization.vector_size, 16)
            self.assertEqual(
                len(specialization.quadrature_rule.reference_gradients),
                dim * n_shape * n_qp,
            )

    def test_tensor_product_elements_expose_1d_quadrature_and_shapes(self):
        quad4 = sfem_element_quadrature_rule("QUAD4")
        self.assertTrue(quad4.is_tensor_product)
        self.assertEqual(quad4.tensor_product_dim, 2)
        self.assertEqual(quad4.tensor_product_n_qp_1d, 2)
        self.assertEqual(quad4.tensor_product_n_shape_1d, 2)
        self.assertEqual(len(quad4.tensor_product_weights_1d), 2)
        self.assertEqual(len(quad4.tensor_product_shape_values_1d), 4)
        self.assertEqual(
            quad4.tensor_product_shape_gradients_1d,
            (-1.0, 1.0, -1.0, 1.0),
        )
        self.assertEqual(len(quad4.weights), 4)

        hex8 = sfem_element_quadrature_rule("HEX8")
        self.assertTrue(hex8.is_tensor_product)
        self.assertEqual(hex8.tensor_product_dim, 3)
        self.assertEqual(hex8.tensor_product_n_qp_1d, 2)
        self.assertEqual(hex8.tensor_product_n_shape_1d, 2)
        self.assertEqual(len(hex8.tensor_product_weights_1d), 2)
        self.assertEqual(len(hex8.tensor_product_shape_values_1d), 4)
        self.assertEqual(
            hex8.tensor_product_shape_gradients_1d,
            (-1.0, 1.0, -1.0, 1.0),
        )
        self.assertEqual(len(hex8.weights), 8)

        hex27 = sfem_element_quadrature_rule("HEX27")
        self.assertTrue(hex27.is_tensor_product)
        self.assertEqual(hex27.tensor_product_dim, 3)
        self.assertEqual(hex27.tensor_product_n_qp_1d, 3)
        self.assertEqual(hex27.tensor_product_n_shape_1d, 3)
        self.assertEqual(len(hex27.tensor_product_weights_1d), 3)
        self.assertEqual(len(hex27.tensor_product_shape_values_1d), 9)
        self.assertEqual(len(hex27.tensor_product_shape_gradients_1d), 9)
        self.assertEqual(hex27.tensor_product_shape_values_1d[3:6], (0.0, 1.0, 0.0))
        self.assertEqual(hex27.tensor_product_shape_gradients_1d[3:6], (-1.0, 0.0, 1.0))
        self.assertEqual(len(hex27.weights), 27)

        proteus_hex27 = sfem_element_quadrature_rule("PROTEUS_HEX27")
        self.assertTrue(proteus_hex27.is_tensor_product)
        self.assertEqual(proteus_hex27.tensor_product_dim, 3)
        self.assertEqual(proteus_hex27.tensor_product_n_qp_1d, 3)
        self.assertEqual(proteus_hex27.tensor_product_n_shape_1d, 3)
        self.assertEqual(len(proteus_hex27.weights), 27)

        proteus_hex64 = sfem_element_quadrature_rule("PROTEUS_HEX64")
        self.assertTrue(proteus_hex64.is_tensor_product)
        self.assertEqual(proteus_hex64.tensor_product_dim, 3)
        self.assertEqual(proteus_hex64.tensor_product_n_qp_1d, 4)
        self.assertEqual(proteus_hex64.tensor_product_n_shape_1d, 4)
        self.assertEqual(len(proteus_hex64.weights), 64)

    def test_generated_quad4_soa_kernel_uses_tensor_product_reference_data(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        specialization = sfem_soa_element_specialization("QUAD4", vector_size=8)
        dim = specialization.dim
        n_nodes = specialization.n_shape
        grad_ref = ReferenceShapeGradients("grad_ref", n_nodes=n_nodes, dim=dim)
        displacement = vector_symbols("u", n_nodes * dim)
        mu, lmbda, qw = sp.symbols("mu lmbda qw")

        energy = qw * sum(
            displacement[node * dim + d] * grad_ref.gradient(node, d)
            for node in range(n_nodes)
            for d in range(dim)
        )
        graph = (
            KernelExpressions()
            .add(ExpressionRole.OPERATOR_EVALUATION, energy)
            .build_graph(
                data_symbols=tuple(displacement) + (qw,),
                symbolic_objects=(grad_ref,),
                temporary_prefix="quad4_tp_tmp",
            )
        )

        generated_files = generate_sfem_soa_cpp_files_for_element(
            (
                sfem_soa_kernel_form(
                    "objective",
                    graph,
                ),
            ),
            prefix="generated_quad4_tensor_product",
            specialization=specialization,
        )

        source_by_path = {generated.path: generated.source for generated in generated_files}
        operator_source = source_by_path["generated_quad4_tensor_product_operator.cpp"]
        local_source = source_by_path["generated_quad4_tensor_product_local.hpp"]

        self.assertIn("struct generated_quad4_tensor_product_isoparametric_reference_data", operator_source)
        self.assertIn("generated_quad4_tensor_product_isoparametric_reference_data<real_t>::shape_1d()", operator_source)
        self.assertIn("generated_quad4_tensor_product_isoparametric_reference_data<real_t>::grad_1d()", operator_source)
        self.assertIn("generated_quad4_tensor_product_isoparametric_reference_data<real_t>::q_weight_1d()", operator_source)
        self.assertNotIn("generated_quad4_tensor_product_quad4_grad_ref", operator_source)
        self.assertNotIn("generated_quad4_tensor_product_quad4_q_weight[", operator_source)
        self.assertNotIn("GRAD_REF_NCOMPONENTS", operator_source)
        self.assertNotIn("GRAD_REF_NCOMPONENTS", local_source)
        self.assertIn("const scalar_t *const SFEM_RESTRICT shape_1d", operator_source)
        self.assertIn("const scalar_t *const SFEM_RESTRICT grad_1d", operator_source)
        self.assertIn("const scalar_t *const SFEM_RESTRICT q_weight_1d", operator_source)
        self.assertIn("const int qx = q % N_QP_1D;", operator_source)
        self.assertIn("const int qy = q / N_QP_1D;", operator_source)
        self.assertIn(
            "const scalar_t tensor_q_weight = q_weight_1d[qx] * q_weight_1d[qy];",
            operator_source,
        )
        self.assertIn(
            "return sfem::codegen::generated_quad4_tensor_product_quad4_objective_soa_impl<real_t, 4, 4, 8>",
            operator_source,
        )
        self.assertIn("const scalar_t *const SFEM_RESTRICT shape_1d", local_source)
        self.assertIn("const scalar_t *const SFEM_RESTRICT grad_1d", local_source)
        self.assertNotIn("grad_ref_data", local_source)
        self.assertIn("static constexpr int N_QP_1D = 2;", local_source)
        self.assertIn(
            "grad_ref[0] = grad_1d[qx * N_SHAPE_1D + 0] * shape_1d[qy * N_SHAPE_1D + 0];",
            local_source,
        )
        self.assertIn(
            "grad_ref[7] = shape_1d[qx * N_SHAPE_1D + 0] * grad_1d[qy * N_SHAPE_1D + 1];",
            local_source,
        )

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            for generated in generated_files:
                with open(os.path.join(tmpdir, generated.path), "w", encoding="utf-8") as output:
                    output.write(generated.source)

            subprocess.run(
                [
                    compiler,
                    "-std=c++11",
                    "-O3",
                    "-c",
                    os.path.join(tmpdir, "generated_quad4_tensor_product_operator.cpp"),
                    "-o",
                    os.path.join(tmpdir, "generated_quad4_tensor_product_operator.o"),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def test_generated_hex8_soa_kernel_uses_tensor_product_reference_data(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        specialization = sfem_soa_element_specialization("HEX8", vector_size=8)
        dim = specialization.dim
        n_nodes = specialization.n_shape
        grad_ref = ReferenceShapeGradients("grad_ref", n_nodes=n_nodes, dim=dim)
        displacement = vector_symbols("u", n_nodes * dim)
        qw = sp.symbols("qw")

        energy = qw * sum(
            displacement[node * dim + d] * grad_ref.gradient(node, d)
            for node in range(n_nodes)
            for d in range(dim)
        )
        graph = (
            KernelExpressions()
            .add(ExpressionRole.OPERATOR_EVALUATION, energy)
            .build_graph(
                data_symbols=tuple(displacement) + (qw,),
                symbolic_objects=(grad_ref,),
                temporary_prefix="hex8_tp_tmp",
            )
        )

        generated_files = generate_sfem_soa_cpp_files_for_element(
            (
                sfem_soa_kernel_form(
                    "objective",
                    graph,
                ),
            ),
            prefix="generated_hex8_tensor_product",
            specialization=specialization,
        )

        source_by_path = {generated.path: generated.source for generated in generated_files}
        operator_source = source_by_path["generated_hex8_tensor_product_operator.cpp"]
        local_source = source_by_path["generated_hex8_tensor_product_local.hpp"]

        self.assertIn("struct generated_hex8_tensor_product_isoparametric_reference_data", operator_source)
        self.assertIn("generated_hex8_tensor_product_isoparametric_reference_data<real_t>::shape_1d()", operator_source)
        self.assertIn("generated_hex8_tensor_product_isoparametric_reference_data<real_t>::grad_1d()", operator_source)
        self.assertIn("generated_hex8_tensor_product_isoparametric_reference_data<real_t>::q_weight_1d()", operator_source)
        self.assertNotIn("generated_hex8_tensor_product_hex8_grad_ref", operator_source)
        self.assertNotIn("GRAD_REF_NCOMPONENTS", operator_source)
        self.assertNotIn("GRAD_REF_NCOMPONENTS", local_source)
        self.assertIn("const int qx = q % N_QP_1D;", operator_source)
        self.assertIn("const int qy = (q / N_QP_1D) % N_QP_1D;", operator_source)
        self.assertIn("const int qz = q / (N_QP_1D * N_QP_1D);", operator_source)
        self.assertIn(
            "const scalar_t tensor_q_weight = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];",
            operator_source,
        )
        self.assertIn(
            "return sfem::codegen::generated_hex8_tensor_product_hex8_objective_soa_impl<real_t, 8, 8, 8>",
            operator_source,
        )
        self.assertNotIn("grad_ref_data", local_source)
        self.assertIn(
            "grad_ref[23] = shape_1d[qx * N_SHAPE_1D + 0] * shape_1d[qy * N_SHAPE_1D + 1] * grad_1d[qz * N_SHAPE_1D + 1];",
            local_source,
        )

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            for generated in generated_files:
                with open(os.path.join(tmpdir, generated.path), "w", encoding="utf-8") as output:
                    output.write(generated.source)

            subprocess.run(
                [
                    compiler,
                    "-std=c++11",
                    "-O3",
                    "-c",
                    os.path.join(tmpdir, "generated_hex8_tensor_product_operator.cpp"),
                    "-o",
                    os.path.join(tmpdir, "generated_hex8_tensor_product_operator.o"),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def test_generated_weak_form_kernel_uses_shape_loops_and_loperand(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        specialization = sfem_soa_element_specialization("TRI3", vector_size=8)
        dim = specialization.dim
        F = sp.Matrix(
            dim,
            dim,
            tuple(sp.symbols("F[%d]" % i) for i in range(dim * dim)),
        )
        weak_form = sfem_soa_weak_form(neohookean_ogden_energy(F, *sp.symbols("mu lmbda")), F)

        generated_files = generate_sfem_soa_cpp_files_for_element(
            (
                sfem_soa_kernel_form(
                    "objective",
                    weak_form=weak_form,
                    output_mode="accumulate",
                ),
                sfem_soa_kernel_form(
                    "gradient",
                    weak_form=weak_form,
                    output_mode="accumulate",
                ),
                sfem_soa_kernel_form(
                    "apply",
                    weak_form=weak_form,
                    has_direction=True,
                    output_mode="accumulate",
                ),
            ),
            prefix="generated_weak_neohookean",
            specialization=specialization,
        )

        source_by_path = {generated.path: generated.source for generated in generated_files}
        local_source = source_by_path["generated_weak_neohookean_local.hpp"]
        operator_source = source_by_path["generated_weak_neohookean_operator.cpp"]

        self.assertIn("for (int shape = 0; shape < N_SHAPE; ++shape)", local_source)
        self.assertNotIn("scalar_t grad_ref", local_source)
        self.assertNotIn("grad_ref[shape", local_source)
        self.assertNotIn("grad_ref_data", local_source)
        self.assertIn("struct generated_weak_neohookean_isoparametric_reference_data", operator_source)
        self.assertIn("generated_weak_neohookean_isoparametric_reference_data<real_t>::grad_ref_x()", operator_source)
        self.assertIn("generated_weak_neohookean_isoparametric_reference_data<real_t>::grad_ref_y()", operator_source)
        self.assertIn(
            "grad_u_ref0 += weak_u_streams[shape * 2 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];",
            local_source,
        )
        self.assertIn(
            "grad_h_ref0 += weak_h_streams[shape * 2 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];",
            local_source,
        )
        self.assertIn("const scalar_t trial_grad0", local_source)
        self.assertIn("const scalar_t material0", local_source)
        self.assertIn("const scalar_t loperand0", local_source)
        self.assertNotIn("scalar_t F[4];", local_source)
        self.assertNotIn("F[0] = 1.0 + grad_u[0];", local_source)
        self.assertNotIn("scalar_t u[N_SHAPE", local_source)
        self.assertNotIn("scalar_t du[N_SHAPE", local_source)
        self.assertNotIn("scalar_t element_vector[N_SHAPE", local_source)
        self.assertIn(
            "weak_out_streams[shape * 2 + 0][lane] += loperand0 * grad_ref_x[q * N_SHAPE + shape]",
            local_source,
        )
        self.assertIn("generated_weak_neohookean_tri3_apply_soa_impl<real_t, 1, 3, 8>", operator_source)

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            for generated in generated_files:
                with open(os.path.join(tmpdir, generated.path), "w", encoding="utf-8") as output:
                    output.write(generated.source)

            subprocess.run(
                [
                    compiler,
                    "-std=c++11",
                    "-O3",
                    "-c",
                    os.path.join(tmpdir, "generated_weak_neohookean_operator.cpp"),
                    "-o",
                    os.path.join(tmpdir, "generated_weak_neohookean_operator.o"),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def test_generated_tensor_product_weak_form_uses_1d_gradients_directly(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        specialization = sfem_soa_element_specialization("QUAD4", vector_size=8)
        dim = specialization.dim
        F = sp.Matrix(
            dim,
            dim,
            tuple(sp.symbols("F[%d]" % i) for i in range(dim * dim)),
        )
        weak_form = sfem_soa_weak_form(neohookean_ogden_energy(F, *sp.symbols("mu lmbda")), F)

        generated_files = generate_sfem_soa_cpp_files_for_element(
            (
                sfem_soa_kernel_form(
                    "gradient",
                    weak_form=weak_form,
                    output_mode="accumulate",
                ),
                sfem_soa_kernel_form(
                    "apply",
                    weak_form=weak_form,
                    has_direction=True,
                    output_mode="accumulate",
                ),
            ),
            prefix="generated_quad4_weak_neohookean",
            specialization=specialization,
        )

        source_by_path = {generated.path: generated.source for generated in generated_files}
        local_source = source_by_path["generated_quad4_weak_neohookean_local.hpp"]
        tensor_source = source_by_path["tensor_product_kernels.hpp"]

        self.assertIn("const scalar_t *const SFEM_RESTRICT shape_1d", local_source)
        self.assertIn("const scalar_t *const SFEM_RESTRICT grad_1d", local_source)
        self.assertIn("for (int q = 0; q < N_QP; ++q)", local_source)
        self.assertIn('#include "tensor_product_kernels.hpp"', local_source)
        self.assertIn("scalar_t value_x[Q * S * VECTOR_SIZE]", tensor_source)
        self.assertIn("scalar_t stage_x[Q * S * VECTOR_SIZE]", tensor_source)
        self.assertIn("for (ptrdiff_t lane = 0; lane < nelems; ++lane)", local_source)
        self.assertNotIn("grad_ref_data", local_source)
        self.assertIn(
            "tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 2>",
            local_source,
        )
        self.assertIn(
            "tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 2>",
            local_source,
        )
        self.assertNotIn("scalar_t element_vector[N_SHAPE", local_source)

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            for generated in generated_files:
                with open(os.path.join(tmpdir, generated.path), "w", encoding="utf-8") as output:
                    output.write(generated.source)

            assert_generated_lane_loops_vectorized(
                self,
                compiler,
                os.path.join(tmpdir, "generated_quad4_weak_neohookean_operator.cpp"),
                os.path.join(tmpdir, "generated_quad4_weak_neohookean_operator.o"),
                local_header="generated_quad4_weak_neohookean_local.hpp",
                minimum_matches=3,
            )

    def test_tensor_product_isoparametric_objective_geometry_uses_sum_factorization(self):
        specialization = sfem_soa_element_specialization("QUAD4", vector_size=8)
        dim = specialization.dim
        F = sp.Matrix(
            dim,
            dim,
            tuple(sp.symbols("F[%d]" % i) for i in range(dim * dim)),
        )
        weak_form = sfem_soa_weak_form(neohookean_ogden_energy(F, *sp.symbols("mu lmbda")), F)

        generated_files = generate_sfem_soa_cpp_files_for_element(
            (
                sfem_soa_kernel_form(
                    "objective",
                    weak_form=weak_form,
                    output_mode="accumulate",
                ),
            ),
            prefix="generated_quad4_iso_objective",
            specialization=specialization,
        )

        operator_source = {
            generated.path: generated.source for generated in generated_files
        }["generated_quad4_iso_objective_operator.cpp"]
        for marker, terminator in (
            (
                "static SFEM_INLINE int generated_quad4_iso_objective_quad4_objective_isoparametric_soa_impl",
                'extern "C" int generated_quad4_iso_objective_quad4_objective_isoparametric_soa',
            ),
            (
                "static SFEM_INLINE int generated_quad4_iso_objective_quad4_objective_isoparametric_mesh_soa_impl",
                'extern "C" int generated_quad4_iso_objective_quad4_objective_isoparametric_mesh_soa',
            ),
        ):
            section = operator_source.split(marker, 1)[1].split(terminator, 1)[0]
            self.assertIn(
                "coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE]",
                section,
            )
            self.assertIn(
                "tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 2>",
                section,
            )
            self.assertNotIn(
                "for (int shape = 0; shape < N_SHAPE; ++shape)",
                section,
            )

    def test_generated_hex27_weak_form_uses_q2_tensor_product_api(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        specialization = sfem_soa_element_specialization("HEX27", vector_size=8)
        dim = specialization.dim
        F = sp.Matrix(
            dim,
            dim,
            tuple(sp.symbols("F[%d]" % i) for i in range(dim * dim)),
        )
        weak_form = sfem_soa_weak_form(neohookean_ogden_energy(F, *sp.symbols("mu lmbda")), F)

        generated_files = generate_sfem_soa_cpp_files_for_element(
            (
                sfem_soa_kernel_form(
                    "gradient",
                    weak_form=weak_form,
                    output_mode="accumulate",
                ),
                sfem_soa_kernel_form(
                    "apply",
                    weak_form=weak_form,
                    has_direction=True,
                    output_mode="accumulate",
                ),
            ),
            prefix="generated_hex27_weak_neohookean",
            specialization=specialization,
        )

        source_by_path = {generated.path: generated.source for generated in generated_files}
        local_source = source_by_path["generated_hex27_weak_neohookean_local.hpp"]
        tensor_source = source_by_path["tensor_product_kernels.hpp"]
        operator_source = source_by_path["generated_hex27_weak_neohookean_operator.cpp"]

        self.assertIn("struct generated_hex27_weak_neohookean_isoparametric_reference_data", operator_source)
        self.assertIn("generated_hex27_weak_neohookean_isoparametric_reference_data<real_t>::shape_1d()", operator_source)
        self.assertIn("generated_hex27_weak_neohookean_isoparametric_reference_data<real_t>::grad_1d()", operator_source)
        self.assertIn("generated_hex27_weak_neohookean_isoparametric_reference_data<real_t>::q_weight_1d()", operator_source)
        self.assertIn("generated_hex27_weak_neohookean_hex27_apply_soa_impl<real_t, 27, 27, 8>", operator_source)
        self.assertIn("static constexpr int N_QP_1D = 3;", local_source)
        self.assertIn("static constexpr int N_SHAPE_1D = 3;", local_source)
        self.assertIn("for (int q = 0; q < N_QP; ++q)", local_source)
        self.assertIn('#include "tensor_product_kernels.hpp"', local_source)
        self.assertIn("scalar_t value_x[Q * S * S * VECTOR_SIZE]", tensor_source)
        self.assertIn("scalar_t value_xy[Q * Q * S * VECTOR_SIZE]", tensor_source)
        self.assertIn("scalar_t stage_xy_x[Q * S * S * VECTOR_SIZE]", tensor_source)
        self.assertIn("for (ptrdiff_t lane = 0; lane < nelems; ++lane)", local_source)
        self.assertNotIn("grad_ref_data", local_source)
        self.assertIn(
            "tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>",
            local_source,
        )
        self.assertIn(
            "tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>",
            local_source,
        )
        self.assertNotIn("scalar_t element_vector[N_SHAPE", local_source)

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            for generated in generated_files:
                with open(os.path.join(tmpdir, generated.path), "w", encoding="utf-8") as output:
                    output.write(generated.source)

            subprocess.run(
                [
                    compiler,
                    "-std=c++11",
                    "-O3",
                    "-c",
                    os.path.join(tmpdir, "generated_hex27_weak_neohookean_operator.cpp"),
                    "-o",
                    os.path.join(tmpdir, "generated_hex27_weak_neohookean_operator.o"),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def test_generated_tensor_product_shared_local_supports_hex8_and_hex27(self):
        local_prefix = "generated_neohookean_ogden_d3_tensor_product"
        source_by_element = {}
        tensor_source_by_element = {}
        operator_by_element = {}
        for element_type in ("HEX8", "HEX27"):
            specialization, generated_files = generated_neohookean_weak_form_files(
                element_type,
                "generated_neohookean_ogden_%s" % element_type.lower(),
                vector_size=8,
                local_prefix=local_prefix,
            )
            del specialization
            source_by_path = {generated.path: generated.source for generated in generated_files}
            operator_source = source_by_path[
                "generated_neohookean_ogden_%s_operator.cpp" % element_type.lower()
            ]
            self.assertIn('#include "%s_local.hpp"' % local_prefix, operator_source)
            self.assertIn(
                "%s_%s_apply_soa_impl<real_t, %d, %d, 8>"
                % (
                    "generated_neohookean_ogden_%s" % element_type.lower(),
                    element_type.lower(),
                    8 if element_type == "HEX8" else 27,
                    8 if element_type == "HEX8" else 27,
                ),
                operator_source,
            )
            operator_by_element[element_type] = operator_source
            source_by_element[element_type] = source_by_path["%s_local.hpp" % local_prefix]
            tensor_source_by_element[element_type] = source_by_path["tensor_product_kernels.hpp"]

        self.assertEqual(source_by_element["HEX8"], source_by_element["HEX27"])
        self.assertEqual(tensor_source_by_element["HEX8"], tensor_source_by_element["HEX27"])
        shared_local = source_by_element["HEX8"]
        tensor_source = tensor_source_by_element["HEX8"]
        self.assertIn("u_streams[N_SHAPE * 3]", shared_local)
        self.assertIn("out_streams[N_SHAPE * 3]", shared_local)
        self.assertIn("integer_root(N_QP, 3)", tensor_source)
        self.assertIn("integer_root(N_SHAPE, 3)", tensor_source)
        self.assertIn("for (int q = 0; q < N_QP; ++q)", shared_local)
        self.assertIn("tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>", shared_local)
        self.assertIn("tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>", shared_local)
        self.assertIn("const int shape = sx + S * (sy + S * sz);", tensor_source)
        self.assertNotIn("tensor_shape_index", shared_local)
        self.assertNotIn("scalar_t u[N_SHAPE", shared_local)
        self.assertNotIn("scalar_t du[N_SHAPE", shared_local)
        self.assertNotIn("scalar_t element_vector[N_SHAPE", shared_local)
        self.assertNotIn("static_assert(N_SHAPE == 8", shared_local)
        self.assertNotIn("static_assert(N_SHAPE == 27", shared_local)
        self.assertIn(
            "block_ux0, block_uy0, block_uz0, "
            "block_ux1, block_uy1, block_uz1, "
            "block_ux3, block_uy3, block_uz3, "
            "block_ux2, block_uy2, block_uz2",
            operator_by_element["HEX8"],
        )
        self.assertIn(
            "block_ux0, block_uy0, block_uz0, "
            "block_ux8, block_uy8, block_uz8, "
            "block_ux1, block_uy1, block_uz1, "
            "block_ux11, block_uy11, block_uz11",
            operator_by_element["HEX27"],
        )

    def test_generated_neohookean_action_matches_python_reference(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        mu = 1.7
        lmbda = 2.3
        for element_type in ("TRI3", "TET4", "HEX8"):
            with self.subTest(element_type=element_type):
                prefix = "generated_%s_neohookean_action" % element_type.lower()
                specialization, generated_files = generated_neohookean_weak_form_files(
                    element_type,
                    prefix,
                    vector_size=8,
                )
                quadrature_rule = specialization.quadrature_rule
                with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
                    library = compile_generated_shared_library(
                        compiler,
                        tmpdir,
                        generated_files,
                        "%s_operator.cpp" % prefix,
                        "lib%s.%s"
                        % (
                            prefix,
                            "dylib" if sys.platform == "darwin" else "so",
                        ),
                    )
                    for geometry_name, coords in (
                        ("reference", reference_element_coords(element_type)),
                        ("deformed", deformed_element_coords(element_type)),
                    ):
                        dim = quadrature_rule.dim
                        zero = tuple((0.0,) * dim for _ in range(quadrature_rule.n_shape))
                        shear = shear_displacement(coords)
                        for displacement_name, displacement in (
                            ("zero", zero),
                            ("shear", shear),
                        ):
                            with self.subTest(
                                element_type=element_type,
                                geometry=geometry_name,
                                displacement=displacement_name,
                            ):
                                expected_gradient = reference_neohookean_gradient(
                                    quadrature_rule,
                                    coords,
                                    displacement,
                                    mu,
                                    lmbda,
                                )
                                actual_gradient = call_generated_neohookean_kernel(
                                    library,
                                    prefix,
                                    element_type,
                                    "gradient",
                                    quadrature_rule,
                                    coords,
                                    displacement,
                                    None,
                                    mu,
                                    lmbda,
                                )
                                gradient_scale = max(1.0, max_abs_value(expected_gradient))
                                self.assertLessEqual(
                                    max_abs_difference(actual_gradient, expected_gradient),
                                    1.0e-10 * gradient_scale,
                                )

                                expected_apply = reference_neohookean_apply(
                                    quadrature_rule,
                                    coords,
                                    displacement,
                                    shear,
                                    mu,
                                    lmbda,
                                )
                                actual_apply = call_generated_neohookean_kernel(
                                    library,
                                    prefix,
                                    element_type,
                                    "apply",
                                    quadrature_rule,
                                    coords,
                                    displacement,
                                    shear,
                                    mu,
                                    lmbda,
                                )
                                apply_scale = max(1.0, max_abs_value(expected_apply))
                                self.assertLessEqual(
                                    max_abs_difference(actual_apply, expected_apply),
                                    1.0e-10 * apply_scale,
                                )

    def test_generated_isoparametric_neohookean_action_matches_python_reference(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        prefix = "generated_hex8_neohookean_isoparametric_action"
        specialization, generated_files = generated_neohookean_weak_form_files(
            "HEX8",
            prefix,
            vector_size=8,
        )
        quadrature_rule = specialization.quadrature_rule
        coords = nonaffine_hex8_coords()
        displacement = shear_displacement(coords)
        direction = tuple((0.03 * xyz[2], -0.02 * xyz[0], 0.04 * xyz[1]) for xyz in coords)
        mu = 1.7
        lmbda = 2.3

        source_by_path = {generated.path: generated.source for generated in generated_files}
        operator_source = source_by_path["%s_operator.cpp" % prefix]
        self.assertIn(
            'extern "C" int %s_hex8_gradient_isoparametric_soa' % prefix,
            operator_source,
        )
        self.assertIn("const real_t *const SFEM_RESTRICT x0", operator_source)
        self.assertIn("block_coordinate_streams[DIM * N_SHAPE]", operator_source)
        self.assertIn(
            "block_jacobian_determinant0[q * VECTOR_SIZE + lane] = J00 * (J11 * J22",
            operator_source,
        )
        self.assertIn(
            'extern "C" int %s_hex8_gradient_affine_mesh_soa' % prefix,
            operator_source,
        )
        self.assertIn(
            'extern "C" int %s_hex8_gradient_isoparametric_mesh_soa' % prefix,
            operator_source,
        )
        self.assertIn("idx_t **const SFEM_RESTRICT elements", operator_source)
        self.assertIn("const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate0", operator_source)
        self.assertIn("g_jacobian_adjugate0 + evbegin", operator_source)
        self.assertIn("g_jacobian_determinant0 + evbegin", operator_source)
        affine_mesh_source = operator_source.split(
            "static SFEM_INLINE int %s_hex8_gradient_affine_mesh_soa_impl" % prefix,
            1,
        )[1].split(
            "static SFEM_INLINE int %s_hex8_gradient_isoparametric_mesh_soa_impl"
            % prefix,
            1,
        )[0]
        self.assertNotIn("scalar_t block_jacobian_adjugate0[VECTOR_SIZE]", affine_mesh_source)
        self.assertNotIn("g_jacobian_adjugate[(evbegin + lane)", affine_mesh_source)
        self.assertIn(
            "const geometry_t *const *const SFEM_RESTRICT points",
            operator_source,
        )
        self.assertIn(
            "const geom_t *const *const SFEM_RESTRICT points",
            operator_source,
        )
        self.assertIn(
            "template <typename scalar_t>\nstatic SFEM_INLINE int %s_hex8_gradient_affine_mesh_soa_impl"
            % prefix,
            operator_source,
        )
        self.assertIn(
            "static constexpr int N_QP = 8;",
            affine_mesh_source,
        )
        self.assertIn(
            'extern "C" int %s_hex8_gradient_affine_mesh_soa_float' % prefix,
            operator_source,
        )
        self.assertIn(
            "%s_hex8_gradient_affine_mesh_soa_impl<double>" % prefix,
            operator_source,
        )
        self.assertIn(
            "%s_hex8_gradient_affine_mesh_soa_impl<float>" % prefix,
            operator_source,
        )
        self.assertIn("#pragma omp atomic update", operator_source)
        mesh_impl_signature = operator_source.split(
            "static SFEM_INLINE int %s_hex8_gradient_isoparametric_mesh_soa_impl" % prefix,
            1,
        )[1].split(") {", 1)[0]
        self.assertNotIn("shape_1d", mesh_impl_signature)
        self.assertNotIn("grad_1d", mesh_impl_signature)
        self.assertNotIn("q_weight_1d", mesh_impl_signature)
        self.assertIn(
            "struct generated_hex8_neohookean_isoparametric_action_isoparametric_reference_data",
            operator_source,
        )
        self.assertIn(
            "generated_hex8_neohookean_isoparametric_action_isoparametric_reference_data<scalar_t>::shape_1d()",
            operator_source,
        )
        self.assertIn(
            "static const scalar_t data[4] = {scalar_t(",
            operator_source,
        )
        isoparametric_mesh_source = operator_source.split(
            "static SFEM_INLINE int %s_hex8_gradient_isoparametric_mesh_soa_impl"
            % prefix,
            1,
        )[1].split(
            'extern "C" int %s_hex8_gradient_isoparametric_mesh_soa' % prefix,
            1,
        )[0]
        self.assertIn(
            "coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE]",
            isoparametric_mesh_source,
        )
        self.assertIn(
            "tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>",
            isoparametric_mesh_source,
        )
        self.assertNotIn(
            "for (int shape = 0; shape < N_SHAPE; ++shape)",
            isoparametric_mesh_source,
        )

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            library = compile_generated_shared_library(
                compiler,
                tmpdir,
                generated_files,
                "%s_operator.cpp" % prefix,
                "lib%s.%s" % (prefix, "dylib" if sys.platform == "darwin" else "so"),
            )
            expected_gradient = reference_neohookean_gradient(
                quadrature_rule,
                coords,
                displacement,
                mu,
                lmbda,
            )
            actual_gradient = call_generated_neohookean_kernel(
                library,
                prefix,
                "HEX8",
                "gradient",
                quadrature_rule,
                coords,
                displacement,
                None,
                mu,
                lmbda,
                isoparametric=True,
            )
            gradient_scale = max(1.0, max_abs_value(expected_gradient))
            self.assertLessEqual(
                max_abs_difference(actual_gradient, expected_gradient),
                1.0e-10 * gradient_scale,
            )

            expected_apply = reference_neohookean_apply(
                quadrature_rule,
                coords,
                displacement,
                direction,
                mu,
                lmbda,
            )
            actual_apply = call_generated_neohookean_kernel(
                library,
                prefix,
                "HEX8",
                "apply",
                quadrature_rule,
                coords,
                displacement,
                direction,
                mu,
                lmbda,
                isoparametric=True,
            )
            apply_scale = max(1.0, max_abs_value(expected_apply))
            self.assertLessEqual(
                max_abs_difference(actual_apply, expected_apply),
                1.0e-10 * apply_scale,
            )

            actual_mesh_gradient = call_generated_neohookean_mesh_kernel(
                library,
                prefix,
                "HEX8",
                "gradient",
                quadrature_rule,
                coords,
                displacement,
                None,
                mu,
                lmbda,
                "isoparametric",
            )
            self.assertLessEqual(
                max_abs_difference(actual_mesh_gradient, expected_gradient),
                1.0e-10 * gradient_scale,
            )

            actual_mesh_apply = call_generated_neohookean_mesh_kernel(
                library,
                prefix,
                "HEX8",
                "apply",
                quadrature_rule,
                coords,
                displacement,
                direction,
                mu,
                lmbda,
                "isoparametric",
            )
            self.assertLessEqual(
                max_abs_difference(actual_mesh_apply, expected_apply),
                1.0e-10 * apply_scale,
            )
            actual_mesh_apply_float = call_generated_neohookean_mesh_kernel(
                library,
                prefix,
                "HEX8",
                "apply",
                quadrature_rule,
                coords,
                displacement,
                direction,
                mu,
                lmbda,
                "isoparametric",
                scalar_ctype=ctypes.c_float,
            )
            self.assertLessEqual(
                max_abs_difference(actual_mesh_apply_float, expected_apply),
                2.0e-5 * apply_scale,
            )

            affine_coords = deformed_element_coords("HEX8")
            affine_displacement = shear_displacement(affine_coords)
            affine_direction = tuple(
                (0.03 * xyz[2], -0.02 * xyz[0], 0.04 * xyz[1]) for xyz in affine_coords
            )
            expected_affine_apply = reference_neohookean_apply(
                quadrature_rule,
                affine_coords,
                affine_displacement,
                affine_direction,
                mu,
                lmbda,
            )
            actual_affine_mesh_apply = call_generated_neohookean_mesh_kernel(
                library,
                prefix,
                "HEX8",
                "apply",
                quadrature_rule,
                affine_coords,
                affine_displacement,
                affine_direction,
                mu,
                lmbda,
                "affine",
            )
            affine_apply_scale = max(1.0, max_abs_value(expected_affine_apply))
            self.assertLessEqual(
                max_abs_difference(actual_affine_mesh_apply, expected_affine_apply),
                1.0e-10 * affine_apply_scale,
            )

    def test_passes_neohookean_ogden_strain_energy_to_framework(self):
        mu, lmbda = sp.symbols("mu lambda")
        q = sp.symbols("q")
        F_obj = DeformationGradient("F", 3)
        F = F_obj.as_matrix()
        variables = F_obj.entries
        directions = vector_symbols("dF", len(variables))

        energy = neohookean_ogden_energy(F, mu, lmbda)
        residual = residual_from_energy(energy, variables)
        hessian_action = hessian_action_from_energy(energy, variables, directions)

        graph = (
            KernelExpressions()
            .energy(energy)
            .residual(residual)
            .hessian_action(hessian_action)
            .build_graph(
                symbolic_objects=(F_obj,),
                scopes=(execution_scope(ScopeKind.QUADRATURE, (q,)),),
                temporary_prefix="nh_tmp",
            )
        )

        roles = [expr.role for expr in graph.outputs]
        self.assertEqual(roles.count(ExpressionRole.ENERGY), 1)
        self.assertEqual(roles.count(ExpressionRole.RESIDUAL), 9)
        self.assertEqual(roles.count(ExpressionRole.HESSIAN_ACTION), 9)
        self.assertGreater(graph.cost.flops, 0)
        self.assertGreaterEqual(len(graph.evaluation_plan.statements), 19)
        self.assertGreater(graph.cost.estimated_registers, 0)

    def test_neohookean_ogden_first_derivative_on_diagonal_F(self):
        mu, lmbda = sp.symbols("mu lambda")
        a, b, c = sp.symbols("a b c", positive=True)
        F_obj = DeformationGradient("F", 3)
        F = F_obj.as_matrix()
        energy = neohookean_ogden_energy(F, mu, lmbda)
        residual = residual_from_energy(energy, F_obj.entries)

        substitutions = {
            F[0, 0]: a,
            F[0, 1]: 0,
            F[0, 2]: 0,
            F[1, 0]: 0,
            F[1, 1]: b,
            F[1, 2]: 0,
            F[2, 0]: 0,
            F[2, 1]: 0,
            F[2, 2]: c,
        }
        actual = sp.simplify(residual[0].subs(substitutions))
        expected = mu * a + (lmbda * sp.log(a * b * c) - mu) / a

        self.assertEqual(sp.simplify(actual - expected), 0)

    def test_neohookean_ogden_from_reference_gradients_and_displacement_coeffs(self):
        mu, lmbda, qw = sp.symbols("mu lambda qw")
        q = sp.symbols("q")
        dim = 2
        n_nodes = 2

        grad_ref = ReferenceShapeGradients(
            "grad_ref",
            n_nodes=n_nodes,
            dim=dim,
            layout=data_layout(LayoutKind.AOS),
        )
        displacement = vector_symbols("u", n_nodes * dim)
        trial_direction = vector_symbols("du", n_nodes * dim)

        reference_gradients = []
        for node in range(n_nodes):
            for row in range(dim):
                reference_gradients.append(grad_ref.tensor_gradient(node, row))

        disp_grad = displacement_gradient_from_reference(
            displacement,
            reference_gradients,
            sp.eye(dim),
        )
        F = sp.eye(dim) + disp_grad
        energy = neohookean_ogden_energy(F, mu, lmbda) * qw
        residual = residual_from_energy(energy, displacement)
        hessian_action = hessian_action_from_energy(
            energy,
            displacement,
            trial_direction,
        )

        graph = (
            KernelExpressions()
            .energy(energy)
            .residual(residual)
            .hessian_action(hessian_action)
            .build_graph(
                data_symbols=tuple(displacement) + tuple(trial_direction) + (mu, lmbda, qw),
                symbolic_objects=(grad_ref,),
                scopes=(execution_scope(ScopeKind.QUADRATURE, (qw, q)),),
                temporary_prefix="nh_ref_tmp",
            )
        )

        roles = [expr.role for expr in graph.outputs]
        self.assertEqual(roles.count(ExpressionRole.ENERGY), 1)
        self.assertEqual(roles.count(ExpressionRole.RESIDUAL), n_nodes * dim)
        self.assertEqual(roles.count(ExpressionRole.HESSIAN_ACTION), n_nodes * dim)

        grad_symbol = grad_ref.gradient(node=1, component=0)
        grad_node = graph.graph.nodes[grad_symbol]
        self.assertEqual(grad_node["layout_kind"], LayoutKind.AOS)
        self.assertEqual(grad_node["node"], 1)
        self.assertEqual(grad_node["dim_component"], 0)
        self.assertEqual(grad_node["layout_offset"], 4 * grad_node["layout_index"] + 2)

        self.assertIn(ScopeKind.QUADRATURE, graph.evaluation_plan.outputs[0].scopes)
        self.assertEqual(graph.evaluation_plan.outputs[0].hoist_scope, ScopeKind.QUADRATURE)
        self.assertEqual(graph.specialization, DimensionSpecialization(dim, "grad_ref"))
        self.assertIn(
            KernelTemplateParameter("grad_ref_n_nodes", n_nodes, "grad_ref"),
            graph.template_parameters,
        )
        self.assertIn(
            KernelTemplateParameter("grad_ref_dim", dim, "grad_ref"),
            graph.template_parameters,
        )

    def test_compiles_generated_neohookean_ogden_cpp_kernel(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        mu, lmbda, qw = sp.symbols("mu lmbda qw")
        quadrature_rule = sfem_element_quadrature_rule("TRI3")
        dim = quadrature_rule.dim
        n_nodes = quadrature_rule.n_shape
        grad_ref = ReferenceShapeGradients("grad_ref", n_nodes=n_nodes, dim=dim)
        displacement = vector_symbols("u", n_nodes * dim)

        reference_gradients = []
        for node in range(n_nodes):
            for row in range(dim):
                reference_gradients.append(grad_ref.tensor_gradient(node, row))

        disp_grad = displacement_gradient_from_reference(
            displacement,
            reference_gradients,
            sp.eye(dim),
        )
        energy = neohookean_ogden_energy(sp.eye(dim) + disp_grad, mu, lmbda) * qw
        residual = residual_from_energy(energy, displacement)
        graph = (
            KernelExpressions()
            .energy(energy)
            .residual(residual)
            .build_graph(
                data_symbols=tuple(displacement) + (mu, lmbda, qw),
                symbolic_objects=(grad_ref,),
                temporary_prefix="nh_compile_tmp",
            )
        )
        generated = generate_cpp_kernel(
            graph,
            function_name="generic_expression_kernel",
        )

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            source_path = os.path.join(tmpdir, "generic_expression_kernel.cpp")
            object_path = os.path.join(tmpdir, "generic_expression_kernel.o")
            with open(source_path, "w", encoding="utf-8") as source_file:
                source_file.write(generated.source)

            subprocess.run(
                [
                    compiler,
                    "-std=c++11",
                    "-O2",
                    "-c",
                    source_path,
                    "-o",
                    object_path,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        self.assertIn('extern "C" void generic_expression_kernel', generated.source)

    def test_compiles_generated_neohookean_ogden_openmp_kernel_with_wrapper(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        mu, lmbda, qw = sp.symbols("mu lmbda qw")
        quadrature_rule = sfem_element_quadrature_rule("TRI3")
        dim = quadrature_rule.dim
        n_nodes = quadrature_rule.n_shape
        grad_ref = ReferenceShapeGradients("grad_ref", n_nodes=n_nodes, dim=dim)
        displacement = vector_symbols("u", n_nodes * dim)

        reference_gradients = []
        for node in range(n_nodes):
            for row in range(dim):
                reference_gradients.append(grad_ref.tensor_gradient(node, row))

        disp_grad = displacement_gradient_from_reference(
            displacement,
            reference_gradients,
            sp.eye(dim),
        )
        energy = neohookean_ogden_energy(sp.eye(dim) + disp_grad, mu, lmbda) * qw
        residual = residual_from_energy(energy, displacement)
        graph = (
            KernelExpressions()
            .energy(energy)
            .residual(residual)
            .build_graph(
                data_symbols=tuple(displacement) + (mu, lmbda, qw),
                symbolic_objects=(grad_ref,),
                temporary_prefix="nh_omp_tmp",
            )
        )
        generated = generate_openmp_cpp_kernel(
            graph,
            function_name="generic_expression_openmp_kernel",
            wrapper_name="GenericExpressionOpenMPOperator",
        )

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            source_path = os.path.join(tmpdir, "generic_expression_openmp_kernel.cpp")
            object_path = os.path.join(tmpdir, "generic_expression_openmp_kernel.o")
            with open(source_path, "w", encoding="utf-8") as source_file:
                source_file.write(generated.source)

            subprocess.run(
                [
                    compiler,
                    "-std=c++11",
                    "-O2",
                    "-c",
                    source_path,
                    "-o",
                    object_path,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        self.assertIn("#pragma omp parallel for", generated.source)
        self.assertIn("struct GenericExpressionOpenMPOperator", generated.source)
        self.assertIn('extern "C" void generic_expression_openmp_kernel', generated.source)

    def test_compiles_generated_neohookean_ogden_sfem_soa_kernel(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("c++ compiler is not available")

        mu, lmbda, qw = sp.symbols("mu lmbda qw")
        specialization = sfem_soa_element_specialization("TRI3", vector_size=8)
        quadrature_rule = specialization.quadrature_rule
        dim = quadrature_rule.dim
        n_nodes = quadrature_rule.n_shape
        grad_ref = ReferenceShapeGradients("grad_ref", n_nodes=n_nodes, dim=dim)
        jacobian_adjugate = sp.Matrix(
            dim,
            dim,
            tuple(
                sp.symbols("jacobian_adjugate[%d]" % i)
                for i in range(dim * dim)
            ),
        )
        jacobian_determinant = sp.symbols("jacobian_determinant[0]")
        displacement = vector_symbols("u", n_nodes * dim)
        trial_direction = vector_symbols("du", n_nodes * dim)

        reference_gradients = []
        for node in range(n_nodes):
            for row in range(dim):
                reference_gradients.append(grad_ref.tensor_gradient(node, row))

        disp_grad = displacement_gradient_from_reference(
            displacement,
            reference_gradients,
            jacobian_adjugate / jacobian_determinant,
        )
        energy = (
            neohookean_ogden_energy(sp.eye(dim) + disp_grad, mu, lmbda)
            * qw
            * jacobian_determinant
        )
        residual = residual_from_energy(energy, displacement)
        hessian_action = hessian_action_from_energy(
            energy,
            displacement,
            trial_direction,
        )

        geometry_data = tuple(jacobian_adjugate) + (jacobian_determinant,)
        common_data = tuple(displacement) + geometry_data + (mu, lmbda, qw)
        apply_data = tuple(displacement) + tuple(trial_direction) + geometry_data + (mu, lmbda, qw)

        def expression_graph(expression, data_symbols, prefix):
            return (
                KernelExpressions()
                .add(ExpressionRole.OPERATOR_EVALUATION, expression)
                .build_graph(
                    data_symbols=data_symbols,
                    symbolic_objects=(grad_ref,),
                    temporary_prefix=prefix,
                )
            )

        generated_files = generate_sfem_soa_cpp_files_for_element(
            (
                sfem_soa_kernel_form(
                    "objective",
                    expression_graph(energy, common_data, "nh_obj_tmp"),
                ),
                sfem_soa_kernel_form(
                    "gradient",
                    expression_graph(residual, common_data, "nh_grad_tmp"),
                ),
                sfem_soa_kernel_form(
                    "apply",
                    expression_graph(hessian_action, apply_data, "nh_apply_tmp"),
                    has_direction=True,
                ),
            ),
            prefix="generated_neohookean_ogden",
            specialization=specialization,
        )

        source_by_path = {generated.path: generated.source for generated in generated_files}
        operator_source = source_by_path["generated_neohookean_ogden_operator.cpp"]
        local_source = source_by_path["generated_neohookean_ogden_local.hpp"]
        math_source = source_by_path["kernel_math.hpp"]
        diagnostics_source = source_by_path["kernel_diagnostics.hpp"]

        self.assertIn("template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>", operator_source)
        self.assertIn('#include "kernel_math.hpp"', local_source)
        self.assertIn("static SFEM_INLINE T pow_2", math_source)
        self.assertIn("static SFEM_INLINE T pow_m2", math_source)
        self.assertIn("struct generated_neohookean_ogden_isoparametric_reference_data", operator_source)
        self.assertIn("generated_neohookean_ogden_isoparametric_reference_data<real_t>::grad_ref_x()", operator_source)
        self.assertIn("generated_neohookean_ogden_isoparametric_reference_data<real_t>::grad_ref_y()", operator_source)
        self.assertIn("generated_neohookean_ogden_isoparametric_reference_data<real_t>::q_weight()", operator_source)
        self.assertIn('#include "kernel_diagnostics.hpp"', operator_source)
        self.assertNotIn("struct SfemKernelDiagnostics", operator_source)
        self.assertIn("#ifndef SFEM_CODEGEN_KERNEL_DIAGNOSTICS_HPP", diagnostics_source)
        self.assertIn("namespace sfem", diagnostics_source)
        self.assertIn("namespace codegen", diagnostics_source)
        self.assertIn("struct KernelDiagnostics", diagnostics_source)
        self.assertIn("add_instructions_per_qp_lane", diagnostics_source)
        self.assertIn("mul_instructions_per_qp_lane", diagnostics_source)
        self.assertIn("div_instructions_per_qp_lane", diagnostics_source)
        self.assertIn("sqrt_instructions_per_qp_lane", diagnostics_source)
        self.assertIn("pow_instructions_per_qp_lane", diagnostics_source)
        self.assertIn("exp_instructions_per_qp_lane", diagnostics_source)
        self.assertIn("log_instructions_per_qp_lane", diagnostics_source)
        self.assertIn("trig_instructions_per_qp_lane", diagnostics_source)
        self.assertIn("load_instructions_per_qp_lane", diagnostics_source)
        self.assertIn("store_instructions_per_qp_lane", diagnostics_source)
        self.assertIn("double add_cpi", diagnostics_source)
        self.assertIn("double div_cpi", diagnostics_source)
        self.assertIn("double exp_cpi", diagnostics_source)
        self.assertIn("double log_cpi", diagnostics_source)
        self.assertIn("double trig_cpi", diagnostics_source)
        self.assertIn("int vector_size", diagnostics_source)
        self.assertIn("geometry_streams", diagnostics_source)
        self.assertIn("reference_scalars", diagnostics_source)
        self.assertIn("output_reads_per_element", diagnostics_source)
        self.assertIn(
            'extern "C" const sfem::codegen::KernelDiagnostics *generated_neohookean_ogden_tri3_apply_soa_diagnostics',
            operator_source,
        )
        self.assertIn(
            'extern "C" double generated_neohookean_ogden_tri3_apply_soa_arithmetic_intensity',
            operator_source,
        )
        self.assertIn("KernelDiagnostics_total_bytes", diagnostics_source)
        self.assertIn("KernelDiagnostics_print_rate", diagnostics_source)
        self.assertIn("#include <stdio.h>", diagnostics_source)
        self.assertIn(
            'extern "C" void generated_neohookean_ogden_tri3_apply_affine_mesh_soa_print_rate',
            operator_source,
        )
        self.assertIn(
            'extern "C" void generated_neohookean_ogden_tri3_apply_isoparametric_mesh_soa_float_print_rate',
            operator_source,
        )
        self.assertIn("static SFEM_INLINE int generated_neohookean_ogden_tri3_apply_soa_impl", operator_source)
        self.assertIn('extern "C" int generated_neohookean_ogden_tri3_apply_soa', operator_source)
        self.assertIn(
            "return sfem::codegen::generated_neohookean_ogden_tri3_apply_soa_impl<real_t, 1, 3, 8>",
            operator_source,
        )
        self.assertNotIn("accumulator_t", operator_source)
        self.assertNotIn("accumulator_t", local_source)
        self.assertNotIn("typedef double scalar_t;", local_source)
        self.assertIn("generated_neohookean_ogden_isoparametric_reference_data<real_t>::grad_ref_x()", operator_source)
        self.assertIn("generated_neohookean_ogden_isoparametric_reference_data<real_t>::grad_ref_y()", operator_source)
        self.assertIn("generated_neohookean_ogden_isoparametric_reference_data<real_t>::q_weight()", operator_source)
        self.assertIn("static_assert(N_QP == 1", operator_source)
        self.assertIn("static_assert(N_SHAPE == 3", operator_source)
        self.assertIn("for (int q = 0; q < N_QP; ++q)", operator_source)
        self.assertIn("block_ux0[VECTOR_SIZE]", operator_source)
        self.assertIn("block_jacobian_adjugate0[VECTOR_SIZE]", operator_source)
        self.assertIn("block_jacobian_determinant0[VECTOR_SIZE]", operator_source)
        self.assertIn(
            "block_jacobian_adjugate0[lane] = jacobian_adjugate0[(ptrdiff_t)q * nelements + evbegin + lane];",
            operator_source,
        )
        apply_wrapper_source = operator_source.split(
            'extern "C" int generated_neohookean_ogden_tri3_apply_soa',
            1,
        )[1]
        apply_wrapper_source = apply_wrapper_source.split(
            "return sfem::codegen::generated_neohookean_ogden_tri3_apply_soa_impl",
            1,
        )[0]
        self.assertNotIn("grad_ref", apply_wrapper_source)
        self.assertNotIn("qw", apply_wrapper_source)
        self.assertIn("const real_t *const SFEM_RESTRICT ux0", operator_source)
        self.assertIn("#pragma omp simd", local_source)
        self.assertIn("template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>", local_source)
        self.assertIn("generated_neohookean_ogden_apply_block", local_source)
        self.assertIn("const int q", local_source)
        self.assertIn("const scalar_t *const SFEM_RESTRICT grad_ref_x", local_source)
        self.assertIn("const scalar_t *const SFEM_RESTRICT grad_ref_y", local_source)
        self.assertNotIn("const scalar_t *const SFEM_RESTRICT grad_ref_data", local_source)
        self.assertNotIn("GRAD_REF_NCOMPONENTS", local_source)
        self.assertIn("scalar_t grad_ref[N_SHAPE * 2];", local_source)
        self.assertIn("scalar_t u[N_SHAPE * 2];", local_source)
        self.assertIn(
            "grad_ref[0] = grad_ref_x[q * N_SHAPE + 0];",
            local_source,
        )
        self.assertIn(
            "grad_ref[5] = grad_ref_y[q * N_SHAPE + 2];",
            local_source,
        )
        self.assertNotIn("grad_ref_data[(q * N_SHAPE", local_source)
        self.assertIn("jacobian_adjugate[0]", local_source)
        self.assertIn("jacobian_determinant[0]", local_source)

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            for generated in generated_files:
                with open(os.path.join(tmpdir, generated.path), "w", encoding="utf-8") as output:
                    output.write(generated.source)

            source_path = os.path.join(tmpdir, "generated_neohookean_ogden_operator.cpp")
            object_path = os.path.join(tmpdir, "generated_neohookean_ogden_operator.o")
            assert_generated_lane_loops_vectorized(
                self,
                compiler,
                source_path,
                object_path,
            )


if __name__ == "__main__":
    unittest.main()
