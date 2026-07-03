from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess

from codegen.framework.fem.basis import BasisFamily
from codegen.framework.fem.tensor_product import TensorProductOperation

from .common import CodeInspectionArtifacts
from .tools import _extract_single_top_level_operation, _serialize_spirv_module


@dataclass(frozen=True)
class MetalSmokeTestResult:
    harness_path: Path
    executable_path: Path
    compile_returncode: int
    compile_stdout: str
    compile_stderr: str
    run_returncode: int = -1
    run_stdout: str = ""
    run_stderr: str = ""

    @property
    def compiled(self):
        return self.compile_returncode == 0

    @property
    def ran(self):
        return self.run_returncode >= 0

    @property
    def success(self):
        return self.compiled and self.run_returncode == 0

    @property
    def no_default_device(self):
        return self.run_returncode == 77


@dataclass(frozen=True)
class TensorProductContractionStage:
    name: str
    operation: str
    derivative: int
    axis: int
    basis: str
    transpose_basis: bool
    lhs_rows: int
    lhs_cols: int
    rhs_cols: int

    def __post_init__(self):
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "operation", str(self.operation))
        object.__setattr__(self, "derivative", int(self.derivative))
        object.__setattr__(self, "axis", int(self.axis))
        object.__setattr__(self, "basis", str(self.basis))
        object.__setattr__(self, "transpose_basis", bool(self.transpose_basis))
        object.__setattr__(self, "lhs_rows", int(self.lhs_rows))
        object.__setattr__(self, "lhs_cols", int(self.lhs_cols))
        object.__setattr__(self, "rhs_cols", int(self.rhs_cols))
        if not self.name or not self.operation or not self.basis:
            raise ValueError("sum-factorization stages require names, operations, and basis tags")
        if self.derivative < 0 or self.axis < 0:
            raise ValueError("sum-factorization derivative and axis must be non-negative")
        if self.lhs_rows <= 0 or self.lhs_cols <= 0 or self.rhs_cols <= 0:
            raise ValueError("sum-factorization stage matrix sizes must be positive")

    @property
    def rhs_rows(self):
        return self.lhs_cols

    @property
    def result_rows(self):
        return self.lhs_rows

    @property
    def result_cols(self):
        return self.rhs_cols

    @property
    def lhs_tensor_type(self):
        return "tensor<%dx%dxf32>" % (self.lhs_rows, self.lhs_cols)

    @property
    def rhs_tensor_type(self):
        return "tensor<%dx%dxf32>" % (self.rhs_rows, self.rhs_cols)

    @property
    def result_tensor_type(self):
        return "tensor<%dx%dxf32>" % (self.result_rows, self.result_cols)

    @property
    def lhs_vector_type(self):
        return "vector<%dxf32>" % (self.lhs_rows * self.lhs_cols)

    @property
    def rhs_vector_type(self):
        return "vector<%dxf32>" % (self.rhs_rows * self.rhs_cols)

    @property
    def result_vector_type(self):
        return "vector<%dxf32>" % (self.result_rows * self.result_cols)

    @property
    def lhs_memref_type(self):
        return "memref<%dx%dxf32>" % (self.lhs_rows, self.lhs_cols)

    @property
    def rhs_memref_type(self):
        return "memref<%dx%dxf32>" % (self.rhs_rows, self.rhs_cols)

    @property
    def result_memref_type(self):
        return "memref<%dx%dxf32>" % (self.result_rows, self.result_cols)

    def to_dict(self):
        return {
            "name": self.name,
            "operation": self.operation,
            "derivative": self.derivative,
            "axis": self.axis,
            "basis": self.basis,
            "transpose_basis": self.transpose_basis,
            "lhs_rows": self.lhs_rows,
            "lhs_cols": self.lhs_cols,
            "rhs_cols": self.rhs_cols,
        }


@dataclass(frozen=True)
class TensorProductSumFactorIR:
    material_name: str
    element_type: str
    element_label: str
    dim: int
    n_shape: int
    n_qp: int
    n_shape_1d: int
    n_qp_1d: int
    quadrature_order: int
    vector_size: int
    shape_values_1d: tuple
    shape_gradients_1d: tuple
    weights_1d: tuple
    field_gradient_stages: tuple
    test_gradient_stages: tuple

    def __post_init__(self):
        dim = int(self.dim)
        n_shape = int(self.n_shape)
        n_qp = int(self.n_qp)
        n_shape_1d = int(self.n_shape_1d)
        n_qp_1d = int(self.n_qp_1d)
        field_gradient_stages = tuple(self.field_gradient_stages)
        test_gradient_stages = tuple(self.test_gradient_stages)
        object.__setattr__(self, "material_name", str(self.material_name))
        object.__setattr__(self, "element_type", str(self.element_type).upper())
        object.__setattr__(self, "element_label", str(self.element_label).lower())
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "n_shape", n_shape)
        object.__setattr__(self, "n_qp", n_qp)
        object.__setattr__(self, "n_shape_1d", n_shape_1d)
        object.__setattr__(self, "n_qp_1d", n_qp_1d)
        object.__setattr__(self, "quadrature_order", int(self.quadrature_order))
        object.__setattr__(self, "vector_size", int(self.vector_size))
        object.__setattr__(self, "shape_values_1d", tuple(float(v) for v in self.shape_values_1d))
        object.__setattr__(self, "shape_gradients_1d", tuple(float(v) for v in self.shape_gradients_1d))
        object.__setattr__(self, "weights_1d", tuple(float(v) for v in self.weights_1d))
        object.__setattr__(self, "field_gradient_stages", field_gradient_stages)
        object.__setattr__(self, "test_gradient_stages", test_gradient_stages)
        if dim not in (2, 3):
            raise ValueError("tensor-product sum-factorization IR supports dimensions 2 and 3")
        if n_shape != n_shape_1d ** dim:
            raise ValueError("tensor-product IR n_shape must equal n_shape_1d ** dim")
        if n_qp != n_qp_1d ** dim:
            raise ValueError("tensor-product IR n_qp must equal n_qp_1d ** dim")
        expected_1d = n_qp_1d * n_shape_1d
        if len(self.shape_values_1d) != expected_1d:
            raise ValueError("shape_values_1d must have n_qp_1d * n_shape_1d entries")
        if len(self.shape_gradients_1d) != expected_1d:
            raise ValueError("shape_gradients_1d must have n_qp_1d * n_shape_1d entries")
        if len(self.weights_1d) != n_qp_1d:
            raise ValueError("weights_1d must have n_qp_1d entries")
        for stage in field_gradient_stages + test_gradient_stages:
            if not isinstance(stage, TensorProductContractionStage):
                raise TypeError("sum-factorization stages must be TensorProductContractionStage objects")

    @property
    def stages(self):
        return self.field_gradient_stages + self.test_gradient_stages

    @property
    def function_prefix(self):
        return "sfem_%s_%s_sum_factor" % (self.material_name, self.element_label)

    def to_dict(self):
        return {
            "material_name": self.material_name,
            "element_type": self.element_type,
            "element_label": self.element_label,
            "dim": self.dim,
            "n_shape": self.n_shape,
            "n_qp": self.n_qp,
            "n_shape_1d": self.n_shape_1d,
            "n_qp_1d": self.n_qp_1d,
            "quadrature_order": self.quadrature_order,
            "vector_size": self.vector_size,
            "shape_values_1d": list(self.shape_values_1d),
            "shape_gradients_1d": list(self.shape_gradients_1d),
            "weights_1d": list(self.weights_1d),
            "field_gradient_stages": [stage.to_dict() for stage in self.field_gradient_stages],
            "test_gradient_stages": [stage.to_dict() for stage in self.test_gradient_stages],
        }


@dataclass(frozen=True)
class TensorProductLaplaceFormIR:
    sum_factor: TensorProductSumFactorIR
    parameter_name: str = "kappa"
    parameter_default: float = 1.0

    def __post_init__(self):
        if not isinstance(self.sum_factor, TensorProductSumFactorIR):
            raise TypeError("sum_factor must be a TensorProductSumFactorIR")
        object.__setattr__(self, "parameter_name", str(self.parameter_name))
        object.__setattr__(self, "parameter_default", float(self.parameter_default))
        if self.sum_factor.material_name != "laplace":
            raise ValueError("TensorProductLaplaceFormIR requires the laplace material")
        if self.parameter_name != "kappa":
            raise ValueError("laplace tensor-product form currently supports kappa only")

    @property
    def function_prefix(self):
        return "%s_laplace_apply" % self.sum_factor.function_prefix

    def to_dict(self):
        values = self.sum_factor.to_dict()
        values.update(
            {
                "form": "laplace",
                "parameter_name": self.parameter_name,
                "parameter_default": self.parameter_default,
            }
        )
        return values


class TensorProductLaplaceFormLinalgLowering:
    def __init__(self, ir):
        if not isinstance(ir, TensorProductLaplaceFormIR):
            raise TypeError("ir must be a TensorProductLaplaceFormIR")
        self.ir = ir
        self.sum_factor_lowering = TensorProductSumFactorMLIRLowering(ir.sum_factor)

    def render_linalg_pipeline_module(self):
        sf = self.ir.sum_factor
        lines = [
            'module attributes {sfem.material = "%s", sfem.element = "%s", '
            'sfem.lowering = "tensor_product_laplace_form_linalg_pipeline", sfem.form = "laplace"} {'
            % (sf.material_name, sf.element_type)
        ]
        lines.extend(self.sum_factor_lowering._render_linalg_pipeline_body())
        lines.extend(self._render_apply_function())
        lines.append("}")
        return "\n".join(lines) + "\n"

    def write_inspection_artifacts(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / ("%s.linalg_pipeline.mlir" % self.ir.function_prefix)
        path.write_text(self.render_linalg_pipeline_module())
        return CodeInspectionArtifacts(
            output_dir=str(output_dir),
            files=(str(path),),
        )

    def iree_metal_compile_command(self, input_mlir, output_vmfb):
        return [
            "iree-compile",
            str(input_mlir),
            "--iree-hal-target-backends=metal-spirv",
            "--iree-metal-compile-to-metallib=false",
            "-o",
            str(output_vmfb),
        ]

    def compile_with_iree_metal(self, output_dir, *, iree_compile=None):
        iree_compile = iree_compile or shutil.which("iree-compile")
        if iree_compile is None:
            raise FileNotFoundError("iree-compile is not available")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_mlir = output_dir / ("%s.linalg_pipeline.mlir" % self.ir.function_prefix)
        input_mlir.write_text(self.render_linalg_pipeline_module())
        output_vmfb = output_dir / ("%s.linalg_pipeline.metal.vmfb" % self.ir.function_prefix)
        command = self.iree_metal_compile_command(input_mlir, output_vmfb)
        command[0] = iree_compile
        result = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return output_vmfb, result

    def _render_apply_function(self):
        sf = self.ir.sum_factor
        field_stages = tuple(stage for stage in sf.field_gradient_stages if stage.derivative == 0)
        test_stages = tuple(stage for stage in sf.test_gradient_stages if stage.derivative == 0)
        input_type = field_stages[0].rhs_tensor_type
        output_type = test_stages[-1].result_tensor_type
        q_by_s = "tensor<%dx%dxf32>" % (sf.n_qp_1d, sf.n_shape_1d)
        s_by_q = "tensor<%dx%dxf32>" % (sf.n_shape_1d, sf.n_qp_1d)
        weight_type = "tensor<%dxf32>" % sf.n_qp_1d
        kappa_type = "tensor<1xf32>"
        signature = (
            "%%shape_1d: %s, %%grad_1d: %s, %%shape_1d_t: %s, %%grad_1d_t: %s, "
            "%%weights_1d: %s, %%kappa: %s, %%u: %s"
            % (q_by_s, q_by_s, s_by_q, s_by_q, weight_type, kappa_type, input_type)
        )
        lines = [
            "",
            "  func.func @%s_linalg_pipeline(%s) -> %s attributes "
            '{sfem.form = "laplace", sfem.parameter = "kappa", sfem.linalg.pipeline = "tensor"} {'
            % (self.ir.function_prefix, signature, output_type)
        ]
        residuals = []
        for derivative in range(sf.dim):
            field_result = "%%field_d%d" % derivative
            lines.extend(self._render_field_pipeline_call(derivative, field_result, input_type))
            weighted_result = "%%weighted_d%d" % derivative
            lines.extend(self._render_weighted_gradient(derivative, field_result, weighted_result))
            test_result = "%%test_d%d" % derivative
            lines.extend(self._render_test_pipeline_call(derivative, weighted_result, test_result))
            residuals.append(test_result)
        lines.extend(self._render_sum_residuals(residuals, output_type))
        lines.extend(["    return %%result : %s" % output_type, "  }"])
        return lines

    def _render_field_pipeline_call(self, derivative, result, input_type):
        sf = self.ir.sum_factor
        stages = tuple(stage for stage in sf.field_gradient_stages if stage.derivative == derivative)
        basis_args = ["%grad_1d" if stage.basis == "grad_1d" else "%shape_1d" for stage in stages]
        function_name = "%s_field_gradient_d%d_linalg_pipeline" % (sf.function_prefix, derivative)
        arg_values = basis_args + ["%u"]
        arg_types = [stage.lhs_tensor_type for stage in stages] + [input_type]
        return [
            "    %s = func.call @%s(%s) : (%s) -> %s"
            % (
                result,
                function_name,
                ", ".join(arg_values),
                ", ".join(arg_types),
                stages[-1].result_tensor_type,
            )
        ]

    def _render_test_pipeline_call(self, derivative, weighted, result):
        sf = self.ir.sum_factor
        stages = tuple(stage for stage in sf.test_gradient_stages if stage.derivative == derivative)
        basis_args = ["%grad_1d_t" if stage.basis == "grad_1d_t" else "%shape_1d_t" for stage in stages]
        function_name = "%s_test_gradient_d%d_linalg_pipeline" % (sf.function_prefix, derivative)
        arg_values = basis_args + [weighted]
        arg_types = [stage.lhs_tensor_type for stage in stages] + [stages[0].rhs_tensor_type]
        return [
            "    %s = func.call @%s(%s) : (%s) -> %s"
            % (
                result,
                function_name,
                ", ".join(arg_values),
                ", ".join(arg_types),
                stages[-1].result_tensor_type,
            )
        ]

    def _render_weighted_gradient(self, derivative, source, result):
        sf = self.ir.sum_factor
        stages = tuple(stage for stage in sf.field_gradient_stages if stage.derivative == derivative)
        tensor_type = stages[-1].result_tensor_type
        extents = _stage_output_extents(sf, stages[-1])
        axes = _stage_view_axes(sf.dim, stages[-1].axis)
        weight_maps = [
            "affine_map<(d0, d1) -> (%s)>" % _axis_coordinate_affine_expr(axis, axes, extents)
            for axis in range(sf.dim)
        ]
        indexing_maps = ["affine_map<(d0, d1) -> (d0, d1)>"] + weight_maps
        indexing_maps.append("affine_map<(d0, d1) -> (0)>")
        indexing_maps.append("affine_map<(d0, d1) -> (d0, d1)>")
        ins_values = [source] + ["%weights_1d" for _ in range(sf.dim)] + ["%kappa"]
        ins_types = [tensor_type] + ["tensor<%dxf32>" % sf.n_qp_1d for _ in range(sf.dim)] + ["tensor<1xf32>"]
        block_args = ["%grad: f32"] + ["%%w%d: f32" % axis for axis in range(sf.dim)] + ["%kappa_value: f32", "%out: f32"]
        lines = [
            "    %s_empty = tensor.empty() : %s" % (result, tensor_type),
            "    %s = linalg.generic {" % result,
            "      indexing_maps = [%s]," % ", ".join(indexing_maps),
            '      iterator_types = ["parallel", "parallel"]} ins(%s : %s) outs(%s_empty : %s) {'
            % (", ".join(ins_values), ", ".join(ins_types), result, tensor_type),
            "    ^bb0(%s):" % ", ".join(block_args),
            "      %weighted_kappa = arith.mulf %grad, %kappa_value : f32",
        ]
        previous = "%weighted_kappa"
        for axis in range(sf.dim):
            current = "%%weighted_axis%d" % axis
            lines.append("      %s = arith.mulf %s, %%w%d : f32" % (current, previous, axis))
            previous = current
        lines.extend(
            [
                "      linalg.yield %s : f32" % previous,
                "    } -> %s" % tensor_type,
            ]
        )
        return lines

    def _render_sum_residuals(self, residuals, output_type):
        if len(residuals) < 1:
            raise ValueError("laplace form requires at least one derivative residual")
        if len(residuals) == 1:
            return ["    %%result = tensor.cast %s : %s to %s" % (residuals[0], output_type, output_type)]
        indexing_maps = ["affine_map<(d0, d1) -> (d0, d1)>"] * (len(residuals) + 1)
        ins_types = [output_type for _ in residuals]
        block_args = ["%%r%d: f32" % index for index in range(len(residuals))] + ["%out: f32"]
        lines = [
            "    %%result_empty = tensor.empty() : %s" % output_type,
            "    %result = linalg.generic {",
            "      indexing_maps = [%s]," % ", ".join(indexing_maps),
            '      iterator_types = ["parallel", "parallel"]} ins(%s : %s) outs(%%result_empty : %s) {'
            % (", ".join(residuals), ", ".join(ins_types), output_type),
            "    ^bb0(%s):" % ", ".join(block_args),
            "      %sum0 = arith.addf %r0, %r1 : f32",
        ]
        previous = "%sum0"
        for index in range(2, len(residuals)):
            current = "%%sum%d" % (index - 1)
            lines.append("      %s = arith.addf %s, %%r%d : f32" % (current, previous, index))
            previous = current
        lines.extend(
            [
                "      linalg.yield %s : f32" % previous,
                "    } -> %s" % output_type,
            ]
        )
        return lines


class TensorProductLaplaceReferenceEvaluator:
    def __init__(self, ir):
        if not isinstance(ir, TensorProductLaplaceFormIR):
            raise TypeError("ir must be a TensorProductLaplaceFormIR")
        self.ir = ir

    def apply_local(self, u, kappa=None):
        sf = self.ir.sum_factor
        if len(u) != sf.n_shape:
            raise ValueError("local vector length must match tensor-product shape count")
        kappa = self.ir.parameter_default if kappa is None else float(kappa)
        result = []
        for row in range(sf.n_shape):
            row_idx = _tensor_product_multi_index(row, sf.n_shape_1d, sf.dim)
            value = 0.0
            for q in range(sf.n_qp):
                q_idx = _tensor_product_multi_index(q, sf.n_qp_1d, sf.dim)
                weight = _tensor_product_weight(sf, q_idx)
                dot = 0.0
                for derivative in range(sf.dim):
                    test = _tensor_product_basis(sf, q_idx, row_idx, derivative)
                    grad_u = 0.0
                    for trial in range(sf.n_shape):
                        trial_idx = _tensor_product_multi_index(trial, sf.n_shape_1d, sf.dim)
                        grad_u += float(u[trial]) * _tensor_product_basis(sf, q_idx, trial_idx, derivative)
                    dot += test * grad_u
                value += kappa * weight * dot
            result.append(float(value))
        return tuple(result)

    def apply_ebe(self, connectivity, u, node_degree, node_to_element_map, node_to_local_idx, kappa=None):
        sf = self.ir.sum_factor
        num_elements = len(connectivity)
        num_nodes = len(node_degree)
        if num_elements <= 0 or num_nodes <= 0:
            raise ValueError("EBE reference requires positive element and node counts")
        element_out = []
        for elem in range(num_elements):
            local_u = []
            for local in range(sf.n_shape):
                node = int(_rank2_value(connectivity, elem, local, sf.n_shape))
                local_u.append(float(u[node]))
            element_out.append(self.apply_local(local_u, kappa=kappa))
        out = []
        max_node_degree = _rank2_width(node_to_element_map)
        for node in range(num_nodes):
            degree = int(node_degree[node])
            acc = 0.0
            for i in range(degree):
                elem = int(_rank2_value(node_to_element_map, node, i, max_node_degree))
                local = int(_rank2_value(node_to_local_idx, node, i, max_node_degree))
                acc += element_out[elem][local]
            out.append(float(acc))
        return tuple(tuple(row) for row in element_out), tuple(out)


class TensorProductSumFactorReferenceEvaluator:
    def __init__(self, ir):
        if not isinstance(ir, TensorProductSumFactorIR):
            raise TypeError("ir must be a TensorProductSumFactorIR")
        self.ir = ir

    def apply_stage(self, stage, operand):
        if not isinstance(stage, TensorProductContractionStage):
            raise TypeError("stage must be a TensorProductContractionStage")
        if len(operand) != stage.rhs_rows * stage.rhs_cols:
            raise ValueError("operand length does not match stage RHS shape")
        basis = self._stage_basis_matrix(stage)
        result = []
        for row in range(stage.result_rows):
            for col in range(stage.result_cols):
                value = 0.0
                for k in range(stage.lhs_cols):
                    value += basis[row * stage.lhs_cols + k] * float(operand[k * stage.rhs_cols + col])
                result.append(float(value))
        return tuple(result)

    def apply_pipeline(self, stages, operand):
        stages = tuple(stages)
        if not stages:
            raise ValueError("sum-factorization pipeline requires stages")
        values = tuple(float(value) for value in operand)
        previous_stage = None
        for stage in stages:
            if previous_stage is not None:
                values = self._reorder_between_stages(values, previous_stage, stage)
            if len(values) != stage.rhs_rows * stage.rhs_cols:
                raise ValueError("stage operand element count does not match pipeline state")
            values = self.apply_stage(stage, values)
            previous_stage = stage
        return values

    def field_gradient(self, u, derivative):
        stages = tuple(stage for stage in self.ir.field_gradient_stages if stage.derivative == derivative)
        values = self.apply_pipeline(stages, u)
        return self._reorder_stage_output_to_canonical(values, stages[-1])

    def direct_field_gradient(self, u, derivative):
        sf = self.ir
        if len(u) != sf.n_shape:
            raise ValueError("field vector length must match tensor-product shape count")
        values = []
        for q in range(sf.n_qp):
            q_idx = _tensor_product_multi_index_row_major(q, sf.n_qp_1d, sf.dim)
            value = 0.0
            for trial in range(sf.n_shape):
                trial_idx = _tensor_product_multi_index_row_major(trial, sf.n_shape_1d, sf.dim)
                value += float(u[trial]) * _tensor_product_basis(sf, q_idx, trial_idx, derivative)
            values.append(float(value))
        return tuple(values)

    def test_contraction(self, q_values, derivative):
        stages = tuple(stage for stage in self.ir.test_gradient_stages if stage.derivative == derivative)
        values = self._reorder_canonical_to_stage_input(q_values, stages[0])
        values = self.apply_pipeline(stages, values)
        return self._reorder_stage_output_to_canonical(values, stages[-1])

    def direct_test_contraction(self, q_values, derivative):
        sf = self.ir
        if len(q_values) != sf.n_qp:
            raise ValueError("quadrature vector length must match tensor-product quadrature count")
        values = []
        for row in range(sf.n_shape):
            row_idx = _tensor_product_multi_index_row_major(row, sf.n_shape_1d, sf.dim)
            value = 0.0
            for q in range(sf.n_qp):
                q_idx = _tensor_product_multi_index_row_major(q, sf.n_qp_1d, sf.dim)
                value += float(q_values[q]) * _tensor_product_basis(sf, q_idx, row_idx, derivative)
            values.append(float(value))
        return tuple(values)

    def apply_laplace_local(self, u, kappa=1.0):
        sf = self.ir
        if len(u) != sf.n_shape:
            raise ValueError("local vector length must match tensor-product shape count")
        result = [0.0 for _ in range(sf.n_shape)]
        for derivative in range(sf.dim):
            gradients = self.field_gradient(u, derivative)
            weighted = []
            for q, value in enumerate(gradients):
                q_idx = _tensor_product_multi_index(q, sf.n_qp_1d, sf.dim)
                weighted.append(float(kappa) * _tensor_product_weight(sf, q_idx) * value)
            contracted = self.test_contraction(weighted, derivative)
            for i, value in enumerate(contracted):
                result[i] += value
        return tuple(float(value) for value in result)

    def direct_laplace_local(self, u, kappa=1.0):
        sf = self.ir
        if len(u) != sf.n_shape:
            raise ValueError("local vector length must match tensor-product shape count")
        result = []
        for row in range(sf.n_shape):
            row_idx = _tensor_product_multi_index_row_major(row, sf.n_shape_1d, sf.dim)
            value = 0.0
            for q in range(sf.n_qp):
                q_idx = _tensor_product_multi_index_row_major(q, sf.n_qp_1d, sf.dim)
                weight = _tensor_product_weight(sf, q_idx)
                dot = 0.0
                for derivative in range(sf.dim):
                    test = _tensor_product_basis(sf, q_idx, row_idx, derivative)
                    grad_u = 0.0
                    for trial in range(sf.n_shape):
                        trial_idx = _tensor_product_multi_index_row_major(trial, sf.n_shape_1d, sf.dim)
                        grad_u += float(u[trial]) * _tensor_product_basis(sf, q_idx, trial_idx, derivative)
                    dot += test * grad_u
                value += float(kappa) * weight * dot
            result.append(float(value))
        return tuple(result)

    def _stage_basis_matrix(self, stage):
        sf = self.ir
        values = sf.shape_gradients_1d if "grad" in stage.basis else sf.shape_values_1d
        basis = []
        if stage.transpose_basis:
            for row in range(stage.lhs_rows):
                for col in range(stage.lhs_cols):
                    basis.append(values[col * sf.n_shape_1d + row])
        else:
            for row in range(stage.lhs_rows):
                for col in range(stage.lhs_cols):
                    basis.append(values[row * sf.n_shape_1d + col])
        return tuple(float(value) for value in basis)

    def _reorder_between_stages(self, values, previous_stage, next_stage):
        sf = self.ir
        source_extents = _stage_output_extents(sf, previous_stage)
        target_extents = _stage_input_extents(sf, next_stage)
        if source_extents != target_extents:
            raise ValueError("adjacent sum-factor stages do not have matching canonical extents")
        source_axes = _stage_view_axes(sf.dim, previous_stage.axis)
        target_axes = _stage_view_axes(sf.dim, next_stage.axis)
        if source_axes == target_axes:
            return values
        result = [0.0 for _ in range(len(values))]
        for linear in range(len(values)):
            multi = _linear_to_multi_index(linear, source_extents)
            source = _view_offset(multi, source_axes, source_extents)
            target = _view_offset(multi, target_axes, target_extents)
            result[target] = values[source]
        return tuple(result)

    def _reorder_canonical_to_stage_input(self, values, stage):
        extents = _stage_input_extents(self.ir, stage)
        return _reorder_tensor_values(values, tuple(range(self.ir.dim)), _stage_view_axes(self.ir.dim, stage.axis), extents)

    def _reorder_stage_output_to_canonical(self, values, stage):
        extents = _stage_output_extents(self.ir, stage)
        return _reorder_tensor_values(values, _stage_view_axes(self.ir.dim, stage.axis), tuple(range(self.ir.dim)), extents)


class TensorProductSumFactorMLIRLowering:
    def __init__(self, ir):
        if not isinstance(ir, TensorProductSumFactorIR):
            raise TypeError("ir must be a TensorProductSumFactorIR")
        self.ir = ir

    def render_linalg_module(self):
        lines = [
            'module attributes {sfem.material = "%s", sfem.element = "%s", '
            'sfem.lowering = "tensor_product_sum_factor_generic_gpu"} {'
            % (self.ir.material_name, self.ir.element_type)
        ]
        for stage in self.ir.stages:
            lines.extend(self._render_stage_function(stage))
        lines.append("}")
        return "\n".join(lines) + "\n"

    def render_linalg_pipeline_module(self):
        self._check_pipeline_stage_types()
        lines = [
            'module attributes {sfem.material = "%s", sfem.element = "%s", '
            'sfem.lowering = "tensor_product_sum_factor_linalg_pipeline"} {'
            % (self.ir.material_name, self.ir.element_type)
        ]
        lines.extend(self._render_linalg_pipeline_body())
        lines.append("}")
        return "\n".join(lines) + "\n"

    def _render_linalg_pipeline_body(self):
        self._check_pipeline_stage_types()
        lines = []
        for stage in self.ir.stages:
            lines.extend(self._render_stage_function(stage))
        for operation, stages in (
            ("field_gradient", self.ir.field_gradient_stages),
            ("test_gradient_contraction", self.ir.test_gradient_stages),
        ):
            for derivative in range(self.ir.dim):
                derivative_stages = tuple(stage for stage in stages if stage.derivative == derivative)
                lines.extend(self._render_linalg_pipeline_function(operation, derivative, derivative_stages))
        return lines

    def render_vector_module(self):
        lines = [
            'module attributes {sfem.material = "%s", sfem.element = "%s", '
            'sfem.lowering = "tensor_product_sum_factor_vector"} {'
            % (self.ir.material_name, self.ir.element_type)
        ]
        for stage in self.ir.stages:
            lines.extend(self._render_vector_stage_function(stage))
        lines.append("}")
        return "\n".join(lines) + "\n"

    def render_matrix_unit_module(self, tile_size=None):
        tile_size = self.ir.vector_size if tile_size is None else int(tile_size)
        if tile_size <= 0:
            raise ValueError("matrix unit tile size must be positive")
        lines = [
            'module attributes {sfem.material = "%s", sfem.element = "%s", '
            'sfem.lowering = "tensor_product_sum_factor_matrix_unit", '
            "sfem.matrix_unit.tile_size = %d : i64} {"
            % (self.ir.material_name, self.ir.element_type, tile_size)
        ]
        for stage in self.ir.stages:
            lines.extend(self._render_matrix_unit_stage_function(stage, tile_size))
        lines.append("}")
        return "\n".join(lines) + "\n"

    def render_matrix_unit_memref_module(self, tile_size=None):
        tile_size = self.ir.vector_size if tile_size is None else int(tile_size)
        if tile_size <= 0:
            raise ValueError("matrix unit tile size must be positive")
        lines = [
            'module attributes {sfem.material = "%s", sfem.element = "%s", '
            'sfem.lowering = "tensor_product_sum_factor_matrix_unit_memref", '
            "sfem.matrix_unit.tile_size = %d : i64} {"
            % (self.ir.material_name, self.ir.element_type, tile_size)
        ]
        for stage in self.ir.stages:
            lines.extend(self._render_matrix_unit_memref_stage_function(stage, tile_size))
        lines.append("}")
        return "\n".join(lines) + "\n"

    def render_matrix_unit_pipeline_module(self, tile_size=None):
        tile_size = self.ir.vector_size if tile_size is None else int(tile_size)
        if tile_size <= 0:
            raise ValueError("matrix unit tile size must be positive")
        self._check_pipeline_stage_types()
        lines = [
            'module attributes {sfem.material = "%s", sfem.element = "%s", '
            'sfem.lowering = "tensor_product_sum_factor_matrix_unit_pipeline", '
            "sfem.matrix_unit.tile_size = %d : i64} {"
            % (self.ir.material_name, self.ir.element_type, tile_size)
        ]
        for stage in self.ir.stages:
            lines.extend(self._render_matrix_unit_memref_stage_function(stage, tile_size))
        for operation, stages in (
            ("field_gradient", self.ir.field_gradient_stages),
            ("test_gradient_contraction", self.ir.test_gradient_stages),
        ):
            for derivative in range(self.ir.dim):
                derivative_stages = tuple(stage for stage in stages if stage.derivative == derivative)
                lines.extend(self._render_matrix_unit_pipeline_function(operation, derivative, derivative_stages))
        lines.append("}")
        return "\n".join(lines) + "\n"

    def render_gpu_module(self):
        lines = [
            'module attributes {gpu.container_module, sfem.material = "%s", sfem.element = "%s", '
            'sfem.lowering = "tensor_product_sum_factor_gpu"} {'
            % (self.ir.material_name, self.ir.element_type)
        ]
        for stage in self.ir.stages:
            lines.extend(self._render_gpu_launch_function(stage))
        lines.append("  gpu.module @%s_gpu_kernels {" % self.ir.function_prefix)
        for stage in self.ir.stages:
            lines.extend(self._render_gpu_kernel(stage))
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def render_spirv_opencl_module(self):
        lines = [
            "module {",
            "  spirv.module Logical OpenCL requires #spirv.vce<v1.0, [Kernel, Addresses], []> attributes "
            '{sfem.material = "%s", sfem.element = "%s", '
            'sfem.lowering = "tensor_product_sum_factor_spirv_opencl", '
            'sfem.opencl.execution_model = "Kernel", sfem.opencl.memory_model = "OpenCL"} {'
            % (self.ir.material_name, self.ir.element_type),
        ]
        for stage in self.ir.stages:
            for kind, size in (
                ("basis", stage.lhs_rows * stage.lhs_cols),
                ("operand", stage.lhs_cols * stage.rhs_cols),
                ("result", stage.result_rows * stage.result_cols),
            ):
                lines.append(
                    "    spirv.GlobalVariable @%s : !spirv.ptr<!spirv.array<%d x f32>, CrossWorkgroup>"
                    % (self._spirv_opencl_global_name(stage, kind), size)
                )
        lines.append("    spirv.GlobalVariable @gid : !spirv.ptr<vector<3xi64>, Input>")
        for stage in self.ir.stages:
            lines.extend(self._render_spirv_opencl_stage_function(stage))
        for stage in self.ir.stages:
            lines.append(
                '    spirv.EntryPoint "Kernel" @%s, @%s, @%s, @%s, @gid'
                % (
                    self._spirv_opencl_kernel_name(stage),
                    self._spirv_opencl_global_name(stage, "basis"),
                    self._spirv_opencl_global_name(stage, "operand"),
                    self._spirv_opencl_global_name(stage, "result"),
                )
            )
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def render_spirv_opencl_module_op(self):
        return _extract_single_top_level_operation(self.render_spirv_opencl_module(), "spirv.module")

    def spirv_opencl_dispatch_manifest(self):
        stages = []
        for index, stage in enumerate(self.ir.stages):
            stages.append(
                {
                    "index": index,
                    "stage": stage.name,
                    "operation": stage.operation,
                    "derivative": stage.derivative,
                    "axis": stage.axis,
                    "basis": stage.basis,
                    "kernel": self._spirv_opencl_kernel_name(stage),
                    "global_work_items": stage.result_rows * stage.result_cols,
                    "result_rows": stage.result_rows,
                    "result_cols": stage.result_cols,
                    "lhs_rows": stage.lhs_rows,
                    "lhs_cols": stage.lhs_cols,
                    "rhs_cols": stage.rhs_cols,
                    "basis_elements": stage.lhs_rows * stage.lhs_cols,
                    "operand_elements": stage.lhs_cols * stage.rhs_cols,
                    "result_elements": stage.result_rows * stage.result_cols,
                }
            )
        return {
            "lowering": "tensor_product_sum_factor_spirv_opencl",
            "material": self.ir.material_name,
            "element": self.ir.element_type,
            "function_prefix": self.ir.function_prefix,
            "dim": self.ir.dim,
            "n_stages": len(stages),
            "stages": stages,
        }

    def render_metal_source(self):
        lines = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
        ]
        for stage in self.ir.stages:
            lines.extend(self._render_metal_kernel(stage))
        return "\n".join(lines) + "\n"

    def write_inspection_artifacts(
        self,
        output_dir,
        *,
        include_metal_source=True,
        include_metal_smoke_harness=True,
        include_spirv_binary=True,
        mlir_translate=None,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = output_dir / self.ir.function_prefix

        linalg_path = prefix.with_suffix(".linalg.mlir")
        linalg_pipeline_path = prefix.with_suffix(".linalg_pipeline.mlir")
        vector_path = prefix.with_suffix(".vector.mlir")
        matrix_unit_path = prefix.with_suffix(".matrix_unit.mlir")
        matrix_unit_memref_path = prefix.with_suffix(".matrix_unit_memref.mlir")
        matrix_unit_pipeline_path = prefix.with_suffix(".matrix_unit_pipeline.mlir")
        gpu_path = prefix.with_suffix(".gpu.mlir")
        spirv_opencl_path = prefix.with_suffix(".spirv.opencl.mlir")
        spirv_opencl_op_path = prefix.with_suffix(".spirv.opencl.op.mlir")
        spirv_opencl_binary_path = prefix.with_suffix(".spirv.opencl.spv")
        spirv_opencl_dispatch_path = prefix.with_suffix(".spirv.opencl.dispatch.json")
        metal_path = prefix.with_suffix(".metal")
        harness_path = prefix.with_suffix(".metal_smoke.mm")

        linalg_path.write_text(self.render_linalg_module())
        linalg_pipeline_path.write_text(self.render_linalg_pipeline_module())
        vector_path.write_text(self.render_vector_module())
        matrix_unit_path.write_text(self.render_matrix_unit_module())
        matrix_unit_memref_path.write_text(self.render_matrix_unit_memref_module())
        matrix_unit_pipeline_path.write_text(self.render_matrix_unit_pipeline_module())
        gpu_path.write_text(self.render_gpu_module())
        spirv_opencl_path.write_text(self.render_spirv_opencl_module())
        spirv_opencl_op_path.write_text(self.render_spirv_opencl_module_op())
        spirv_opencl_dispatch_path.write_text(
            json.dumps(self.spirv_opencl_dispatch_manifest(), indent=2, sort_keys=True) + "\n"
        )

        files = [
            linalg_path,
            linalg_pipeline_path,
            vector_path,
            matrix_unit_path,
            matrix_unit_memref_path,
            matrix_unit_pipeline_path,
            gpu_path,
            spirv_opencl_path,
            spirv_opencl_op_path,
            spirv_opencl_dispatch_path,
        ]
        if include_spirv_binary:
            _serialize_spirv_module(
                spirv_opencl_op_path,
                spirv_opencl_binary_path,
                mlir_translate=mlir_translate,
            )
            files.append(spirv_opencl_binary_path)
        if include_metal_source:
            metal_path.write_text(self.render_metal_source())
            files.append(metal_path)
        if include_metal_smoke_harness:
            harness_path.write_text(self.render_metal_smoke_test_harness())
            files.append(harness_path)

        return CodeInspectionArtifacts(
            output_dir=str(output_dir),
            files=tuple(str(path) for path in files),
        )

    def render_metal_smoke_test_harness(self, stage=None):
        stages = self.ir.stages if stage is None else (stage,)
        source = _objc_string_literal(self.render_metal_source())
        return _METAL_SMOKE_TEST_TEMPLATE % {
            "source": source,
            "stage_count": len(stages),
            "stage_calls": "\n".join(
                self._render_metal_smoke_stage_call(index, stage) for index, stage in enumerate(stages)
            ),
        }

    def _render_metal_smoke_stage_call(self, index, stage):
        basis = _float_initializer(
            _deterministic_values(stage.lhs_rows * stage.lhs_cols, scale=0.25, offset=1.0 + index)
        )
        operand = _float_initializer(
            _deterministic_values(stage.rhs_rows * stage.rhs_cols, scale=0.125, offset=2.0 + index)
        )
        return (
            "        static const float basis_%(index)d[%(lhs_size)d] = {%(basis)s};\n"
            "        static const float operand_%(index)d[%(rhs_size)d] = {%(operand)s};\n"
            "        status = run_stage(device, library, queue, @\"%(kernel_name)s\", "
            "basis_%(index)d, sizeof(basis_%(index)d), operand_%(index)d, sizeof(operand_%(index)d), "
            "%(result_size)d, %(lhs_cols)d, %(rhs_cols)d, %(result_rows)d, %(result_cols)d);\n"
            "        if (status != 0) {\n"
            "            return status;\n"
            "        }\n"
        ) % {
            "index": index,
            "kernel_name": self._metal_kernel_name(stage),
            "lhs_size": stage.lhs_rows * stage.lhs_cols,
            "rhs_size": stage.rhs_rows * stage.rhs_cols,
            "result_size": stage.result_rows * stage.result_cols,
            "lhs_cols": stage.lhs_cols,
            "rhs_cols": stage.rhs_cols,
            "result_rows": stage.result_rows,
            "result_cols": stage.result_cols,
            "basis": basis,
            "operand": operand,
        }

    def iree_metal_compile_command(self, input_mlir, output_vmfb):
        return [
            "iree-compile",
            str(input_mlir),
            "--iree-hal-target-backends=metal-spirv",
            "--iree-metal-compile-to-metallib=false",
            "-o",
            str(output_vmfb),
        ]

    def compile_with_iree_metal(self, output_dir, *, iree_compile=None, input_kind="linalg_pipeline"):
        iree_compile = iree_compile or shutil.which("iree-compile")
        if iree_compile is None:
            raise FileNotFoundError("iree-compile is not available")
        if input_kind not in ("linalg", "linalg_pipeline", "matrix_unit", "matrix_unit_memref", "matrix_unit_pipeline"):
            raise ValueError(
                "IREE Metal input_kind must be 'linalg', 'linalg_pipeline', 'matrix_unit', "
                "'matrix_unit_memref', or 'matrix_unit_pipeline'"
            )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if input_kind == "matrix_unit_pipeline":
            input_mlir = output_dir / ("%s.matrix_unit_pipeline.mlir" % self.ir.function_prefix)
            input_mlir.write_text(self.render_matrix_unit_pipeline_module())
            output_vmfb = output_dir / ("%s.matrix_unit_pipeline.metal.vmfb" % self.ir.function_prefix)
        elif input_kind == "linalg_pipeline":
            input_mlir = output_dir / ("%s.linalg_pipeline.mlir" % self.ir.function_prefix)
            input_mlir.write_text(self.render_linalg_pipeline_module())
            output_vmfb = output_dir / ("%s.linalg_pipeline.metal.vmfb" % self.ir.function_prefix)
        elif input_kind == "matrix_unit_memref":
            input_mlir = output_dir / ("%s.matrix_unit_memref.mlir" % self.ir.function_prefix)
            input_mlir.write_text(self.render_matrix_unit_memref_module())
            output_vmfb = output_dir / ("%s.matrix_unit_memref.metal.vmfb" % self.ir.function_prefix)
        elif input_kind == "matrix_unit":
            input_mlir = output_dir / ("%s.matrix_unit.mlir" % self.ir.function_prefix)
            input_mlir.write_text(self.render_matrix_unit_module())
            output_vmfb = output_dir / ("%s.matrix_unit.metal.vmfb" % self.ir.function_prefix)
        else:
            input_mlir = output_dir / ("%s.linalg.mlir" % self.ir.function_prefix)
            input_mlir.write_text(self.render_linalg_module())
            output_vmfb = output_dir / ("%s.linalg.metal.vmfb" % self.ir.function_prefix)
        command = self.iree_metal_compile_command(input_mlir, output_vmfb)
        command[0] = iree_compile
        result = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return output_vmfb, result

    def run_metal_smoke_test(self, output_dir, stage=None, *, xcrun=None):
        xcrun = xcrun or shutil.which("xcrun")
        if xcrun is None:
            raise FileNotFoundError("xcrun is not available")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        harness = output_dir / ("%s_metal_smoke.mm" % self.ir.function_prefix)
        executable = output_dir / ("%s_metal_smoke" % self.ir.function_prefix)
        harness.write_text(self.render_metal_smoke_test_harness(stage=stage))
        compile_result = subprocess.run(
            [
                xcrun,
                "clang++",
                str(harness),
                "-fobjc-arc",
                "-framework",
                "Foundation",
                "-framework",
                "Metal",
                "-o",
                str(executable),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if compile_result.returncode != 0:
            return MetalSmokeTestResult(
                harness,
                executable,
                compile_result.returncode,
                compile_result.stdout,
                compile_result.stderr,
            )
        run_result = subprocess.run(
            [str(executable)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return MetalSmokeTestResult(
            harness,
            executable,
            compile_result.returncode,
            compile_result.stdout,
            compile_result.stderr,
            run_result.returncode,
            run_result.stdout,
            run_result.stderr,
        )

    def _spirv_opencl_kernel_name(self, stage):
        return "%s_%s_spirv_opencl" % (self.ir.function_prefix, stage.name)

    def _spirv_opencl_global_name(self, stage, kind):
        return "%s_%s_%s" % (self.ir.function_prefix, stage.name, kind)

    def _render_spirv_opencl_stage_function(self, stage):
        names = _SSANamer()
        basis_name = self._spirv_opencl_global_name(stage, "basis")
        operand_name = self._spirv_opencl_global_name(stage, "operand")
        result_name = self._spirv_opencl_global_name(stage, "result")
        basis_array_type = "!spirv.ptr<!spirv.array<%d x f32>, CrossWorkgroup>" % (stage.lhs_rows * stage.lhs_cols)
        operand_array_type = "!spirv.ptr<!spirv.array<%d x f32>, CrossWorkgroup>" % (stage.lhs_cols * stage.rhs_cols)
        result_array_type = "!spirv.ptr<!spirv.array<%d x f32>, CrossWorkgroup>" % (stage.result_rows * stage.result_cols)
        scalar_ptr_type = "!spirv.ptr<f32, CrossWorkgroup>"
        lines = [
            "    spirv.func @%s() \"None\" {" % self._spirv_opencl_kernel_name(stage),
            "      %gid_addr = spirv.mlir.addressof @gid : !spirv.ptr<vector<3xi64>, Input>",
            "      %gid_vec = spirv.Load \"Input\" %gid_addr : vector<3xi64>",
            "      %gid_x64 = spirv.CompositeExtract %gid_vec[0 : i32] : vector<3xi64>",
            "      %idx = spirv.UConvert %gid_x64 : i64 to i32",
            "      %%basis_addr = spirv.mlir.addressof @%s : %s" % (basis_name, basis_array_type),
            "      %%operand_addr = spirv.mlir.addressof @%s : %s" % (operand_name, operand_array_type),
            "      %%result_addr = spirv.mlir.addressof @%s : %s" % (result_name, result_array_type),
        ]

        result_cols = names.value("result_cols")
        lhs_cols = names.value("lhs_cols")
        rhs_cols = names.value("rhs_cols")
        row = names.value("row")
        col = names.value("col")
        acc = names.value("acc")
        lines.extend(
            [
                "      %s = spirv.Constant %d : i32" % (result_cols, stage.result_cols),
                "      %s = spirv.Constant %d : i32" % (lhs_cols, stage.lhs_cols),
                "      %s = spirv.Constant %d : i32" % (rhs_cols, stage.rhs_cols),
                "      %s = spirv.UDiv %%idx, %s : i32" % (row, result_cols),
                "      %s = spirv.UMod %%idx, %s : i32" % (col, result_cols),
                "      %s = spirv.Constant 0.0 : f32" % acc,
            ]
        )
        acc_value = acc
        for k in range(stage.lhs_cols):
            k_value = names.value("k")
            lhs_row_offset = names.value("lhs_row_offset")
            lhs_index = names.value("lhs_index")
            rhs_row_offset = names.value("rhs_row_offset")
            rhs_index = names.value("rhs_index")
            lhs_ptr = names.value("lhs_ptr")
            rhs_ptr = names.value("rhs_ptr")
            lhs = names.value("lhs")
            rhs = names.value("rhs")
            product = names.value("product")
            next_acc = names.value("acc")
            lines.extend(
                [
                    "      %s = spirv.Constant %d : i32" % (k_value, k),
                    "      %s = spirv.IMul %s, %s : i32" % (lhs_row_offset, row, lhs_cols),
                    "      %s = spirv.IAdd %s, %s : i32" % (lhs_index, lhs_row_offset, k_value),
                    "      %s = spirv.IMul %s, %s : i32" % (rhs_row_offset, k_value, rhs_cols),
                    "      %s = spirv.IAdd %s, %s : i32" % (rhs_index, rhs_row_offset, col),
                    "      %s = spirv.AccessChain %%basis_addr[%s] : %s, i32 -> %s"
                    % (lhs_ptr, lhs_index, basis_array_type, scalar_ptr_type),
                    "      %s = spirv.AccessChain %%operand_addr[%s] : %s, i32 -> %s"
                    % (rhs_ptr, rhs_index, operand_array_type, scalar_ptr_type),
                    "      %s = spirv.Load \"CrossWorkgroup\" %s : f32" % (lhs, lhs_ptr),
                    "      %s = spirv.Load \"CrossWorkgroup\" %s : f32" % (rhs, rhs_ptr),
                    "      %s = spirv.FMul %s, %s : f32" % (product, lhs, rhs),
                    "      %s = spirv.FAdd %s, %s : f32" % (next_acc, acc_value, product),
                ]
            )
            acc_value = next_acc

        result_ptr = names.value("result_ptr")
        lines.extend(
            [
                "      %s = spirv.AccessChain %%result_addr[%%idx] : %s, i32 -> %s"
                % (result_ptr, result_array_type, scalar_ptr_type),
                "      spirv.Store \"CrossWorkgroup\" %s, %s : f32" % (result_ptr, acc_value),
                "      spirv.Return",
                "    }",
            ]
        )
        return lines

    def _render_stage_function(self, stage):
        function_name = "%s_%s" % (self.ir.function_prefix, stage.name)
        return [
            "",
            "  func.func @%s(%%basis: %s, %%operand: %s) -> %s attributes "
            '{sfem.sum_factor.operation = "%s", sfem.sum_factor.basis = "%s", '
            "sfem.sum_factor.axis = %d : i64, sfem.sum_factor.derivative = %d : i64} {"
            % (
                function_name,
                stage.lhs_tensor_type,
                stage.rhs_tensor_type,
                stage.result_tensor_type,
                stage.operation,
                stage.basis,
                stage.axis,
                stage.derivative,
            ),
            f"    %empty = tensor.empty() : {stage.result_tensor_type}",
            "    %zero = arith.constant 0.0 : f32",
            "    %init = linalg.fill ins(%zero : f32) "
            f"outs(%empty : {stage.result_tensor_type}) -> {stage.result_tensor_type}",
            "    %result = linalg.matmul ins(%basis, %operand : "
            f"{stage.lhs_tensor_type}, {stage.rhs_tensor_type}) "
            f"outs(%init : {stage.result_tensor_type}) -> {stage.result_tensor_type}",
            f"    return %result : {stage.result_tensor_type}",
            "  }",
        ]

    def _render_linalg_pipeline_function(self, operation, derivative, stages):
        if not stages:
            raise ValueError("sum-factorization pipeline requires at least one stage")
        label = "field_gradient" if operation == "field_gradient" else "test_gradient"
        function_name = "%s_%s_d%d_linalg_pipeline" % (self.ir.function_prefix, label, derivative)
        signature = []
        for index, stage in enumerate(stages):
            signature.append("%%basis%d: %s" % (index, stage.lhs_tensor_type))
        signature.append("%%input: %s" % stages[0].rhs_tensor_type)
        return_type = stages[-1].result_tensor_type

        lines = [
            "",
            "  func.func @%s(%s) -> %s attributes "
            '{sfem.sum_factor.operation = "%s", sfem.sum_factor.derivative = %d : i64, '
            'sfem.linalg.pipeline = "tensor"} {'
            % (function_name, ", ".join(signature), return_type, operation, derivative),
        ]
        operand = "%input"
        operand_type = stages[0].rhs_tensor_type
        previous_stage = None
        for index, stage in enumerate(stages):
            call_operand = operand
            if previous_stage is not None:
                call_operand = "%%bridge%d" % index
                lines.extend(self._render_linalg_pipeline_bridge(index, previous_stage, stage, operand, operand_type))
            result = "%%stage%d" % index
            stage_function_name = "%s_%s" % (self.ir.function_prefix, stage.name)
            lines.append(
                "    %s = func.call @%s(%%basis%d, %s) : (%s, %s) -> %s"
                % (
                    result,
                    stage_function_name,
                    index,
                    call_operand,
                    stage.lhs_tensor_type,
                    stage.rhs_tensor_type,
                    stage.result_tensor_type,
                )
            )
            operand = result
            operand_type = stage.result_tensor_type
            previous_stage = stage
        lines.extend(["    return %s : %s" % (operand, return_type), "  }"])
        return lines

    def _render_linalg_pipeline_bridge(self, bridge_index, previous_stage, next_stage, source, source_type):
        sf = self.ir
        source_extents = _stage_output_extents(sf, previous_stage)
        target_extents = _stage_input_extents(sf, next_stage)
        if source_extents != target_extents:
            raise ValueError("adjacent sum-factor stages do not have matching canonical extents")
        source_axes = _stage_view_axes(sf.dim, previous_stage.axis)
        target_axes = _stage_view_axes(sf.dim, next_stage.axis)
        if source_axes == target_axes and source_type == next_stage.rhs_tensor_type:
            return ["    %%bridge%d = tensor.cast %s : %s to %s" % (bridge_index, source, source_type, source_type)]

        source_map = _stage_view_affine_map(source_axes, target_axes, target_extents)
        return [
            "    %%bridge%d_empty = tensor.empty() : %s" % (bridge_index, next_stage.rhs_tensor_type),
            "    %%bridge%d = linalg.generic {" % bridge_index,
            "      indexing_maps = [%s, affine_map<(d0, d1) -> (d0, d1)>]," % source_map,
            '      iterator_types = ["parallel", "parallel"]} ins(%s : %s) outs(%%bridge%d_empty : %s) {'
            % (source, source_type, bridge_index, next_stage.rhs_tensor_type),
            "    ^bb0(%in: f32, %out: f32):",
            "      linalg.yield %in : f32",
            "    } -> %s" % next_stage.rhs_tensor_type,
        ]

    def _render_vector_stage_function(self, stage):
        function_name = "%s_%s_vector" % (self.ir.function_prefix, stage.name)
        return [
            "",
            "  func.func @%s(%%basis: %s, %%operand: %s) -> %s attributes "
            '{sfem.sum_factor.operation = "%s", sfem.sum_factor.basis = "%s", '
            "sfem.sum_factor.axis = %d : i64, sfem.sum_factor.derivative = %d : i64, "
            'sfem.vector.matrix_multiply = "flattened"} {'
            % (
                function_name,
                stage.lhs_vector_type,
                stage.rhs_vector_type,
                stage.result_vector_type,
                stage.operation,
                stage.basis,
                stage.axis,
                stage.derivative,
            ),
            "    %%result = vector.matrix_multiply %%basis, %%operand "
            "{lhs_rows = %d : i32, lhs_columns = %d : i32, rhs_columns = %d : i32} "
            ": (%s, %s) -> %s"
            % (
                stage.lhs_rows,
                stage.lhs_cols,
                stage.rhs_cols,
                stage.lhs_vector_type,
                stage.rhs_vector_type,
                stage.result_vector_type,
            ),
            f"    return %result : {stage.result_vector_type}",
            "  }",
        ]

    def _render_matrix_unit_stage_function(self, stage, tile_size):
        function_name = "%s_%s_matrix_unit" % (self.ir.function_prefix, stage.name)
        lhs_rows = _align_up(stage.lhs_rows, tile_size)
        lhs_cols = _align_up(stage.lhs_cols, tile_size)
        rhs_cols = _align_up(stage.rhs_cols, tile_size)
        lhs_vector_type = "vector<%dxf32>" % (lhs_rows * lhs_cols)
        rhs_vector_type = "vector<%dxf32>" % (lhs_cols * rhs_cols)
        result_vector_type = "vector<%dxf32>" % (lhs_rows * rhs_cols)
        return [
            "",
            "  func.func @%s(%%basis: %s, %%operand: %s) -> %s attributes "
            '{sfem.sum_factor.operation = "%s", sfem.sum_factor.basis = "%s", '
            "sfem.sum_factor.axis = %d : i64, sfem.sum_factor.derivative = %d : i64, "
            'sfem.vector.matrix_multiply = "padded", sfem.matrix_unit.tile_size = %d : i64, '
            "sfem.matrix_unit.raw_lhs_rows = %d : i64, sfem.matrix_unit.raw_lhs_columns = %d : i64, "
            "sfem.matrix_unit.raw_rhs_columns = %d : i64} {"
            % (
                function_name,
                lhs_vector_type,
                rhs_vector_type,
                result_vector_type,
                stage.operation,
                stage.basis,
                stage.axis,
                stage.derivative,
                tile_size,
                stage.lhs_rows,
                stage.lhs_cols,
                stage.rhs_cols,
            ),
            "    %%result = vector.matrix_multiply %%basis, %%operand "
            "{lhs_rows = %d : i32, lhs_columns = %d : i32, rhs_columns = %d : i32} "
            ": (%s, %s) -> %s"
            % (
                lhs_rows,
                lhs_cols,
                rhs_cols,
                lhs_vector_type,
                rhs_vector_type,
                result_vector_type,
            ),
            f"    return %result : {result_vector_type}",
            "  }",
        ]

    def _check_pipeline_stage_types(self):
        for stages in (self.ir.field_gradient_stages, self.ir.test_gradient_stages):
            for derivative in range(self.ir.dim):
                derivative_stages = tuple(stage for stage in stages if stage.derivative == derivative)
                if not derivative_stages:
                    raise ValueError("sum-factorization pipeline is missing derivative stages")
                for previous, current in zip(derivative_stages, derivative_stages[1:]):
                    previous_size = previous.result_rows * previous.result_cols
                    current_size = current.rhs_rows * current.rhs_cols
                    if previous_size != current_size:
                        raise ValueError(
                            "sum-factorization pipeline requires adjacent stage buffer element counts to match"
                        )

    def _render_matrix_unit_pipeline_function(self, operation, derivative, stages):
        if not stages:
            raise ValueError("sum-factorization pipeline requires at least one stage")
        label = "field_gradient" if operation == "field_gradient" else "test_gradient"
        function_name = "%s_%s_d%d_matrix_unit_pipeline" % (self.ir.function_prefix, label, derivative)
        signature = []
        for index, stage in enumerate(stages):
            signature.append("%%basis%d: %s" % (index, stage.lhs_memref_type))
        signature.append("%%input: %s" % stages[0].rhs_memref_type)
        signature.append("%%output: %s" % stages[-1].result_memref_type)

        lines = [
            "",
            "  func.func @%s(%s) attributes "
            '{sfem.sum_factor.operation = "%s", sfem.sum_factor.derivative = %d : i64, '
            'sfem.vector.matrix_multiply = "padded_memref_pipeline"} {'
            % (function_name, ", ".join(signature), operation, derivative),
        ]
        max_index = max(
            max(stage.rhs_rows, stage.rhs_cols, stage.result_rows, stage.result_cols)
            for stage in stages
        )
        for value in range(max_index):
            lines.append("    %%c%d = arith.constant %d : index" % (value, value))
        scratches = []
        operand = "%input"
        operand_type = stages[0].rhs_memref_type
        previous_stage = None
        for index, stage in enumerate(stages):
            call_operand = operand
            if previous_stage is not None:
                call_operand = "%%bridge%d" % index
                scratches.append((call_operand, stage.rhs_memref_type))
                lines.append("    %s = memref.alloc() : %s" % (call_operand, stage.rhs_memref_type))
                lines.extend(self._render_pipeline_bridge_copy(index, previous_stage, stage, operand, operand_type, call_operand))
            result = "%output"
            if index != len(stages) - 1:
                result = "%%scratch%d" % index
                scratches.append((result, stage.result_memref_type))
                lines.append("    %s = memref.alloc() : %s" % (result, stage.result_memref_type))
            stage_function_name = "%s_%s_matrix_unit_memref" % (self.ir.function_prefix, stage.name)
            lines.append(
                "    func.call @%s(%%basis%d, %s, %s) : (%s, %s, %s) -> ()"
                % (
                    stage_function_name,
                    index,
                    call_operand,
                    result,
                    stage.lhs_memref_type,
                    stage.rhs_memref_type,
                    stage.result_memref_type,
                )
            )
            operand = result
            operand_type = stage.result_memref_type
            previous_stage = stage
        for scratch_name, scratch_type in reversed(tuple(scratches)):
            lines.append("    memref.dealloc %s : %s" % (scratch_name, scratch_type))
        lines.extend(["    return", "  }"])
        return lines

    def _render_pipeline_bridge_copy(self, bridge_index, previous_stage, next_stage, source, source_type, target):
        sf = self.ir
        source_extents = _stage_output_extents(sf, previous_stage)
        target_extents = _stage_input_extents(sf, next_stage)
        if source_extents != target_extents:
            raise ValueError("adjacent sum-factor stages do not have matching canonical extents")
        source_axes = _stage_view_axes(sf.dim, previous_stage.axis)
        target_axes = _stage_view_axes(sf.dim, next_stage.axis)
        lines = []
        for linear in range(_product(source_extents)):
            multi = _linear_to_multi_index(linear, source_extents)
            source_offset = _view_offset(multi, source_axes, source_extents)
            target_offset = _view_offset(multi, target_axes, target_extents)
            source_row = source_offset // previous_stage.result_cols
            source_col = source_offset % previous_stage.result_cols
            target_row = target_offset // next_stage.rhs_cols
            target_col = target_offset % next_stage.rhs_cols
            lines.extend(
                [
                    "    %%bridge%d_value_%d = memref.load %s[%%c%d, %%c%d] : %s"
                    % (bridge_index, linear, source, source_row, source_col, source_type),
                    "    memref.store %%bridge%d_value_%d, %s[%%c%d, %%c%d] : %s"
                    % (bridge_index, linear, target, target_row, target_col, next_stage.rhs_memref_type),
                ]
            )
        return lines

    def _render_matrix_unit_memref_stage_function(self, stage, tile_size):
        function_name = "%s_%s_matrix_unit_memref" % (self.ir.function_prefix, stage.name)
        lhs_rows = _align_up(stage.lhs_rows, tile_size)
        lhs_cols = _align_up(stage.lhs_cols, tile_size)
        rhs_cols = _align_up(stage.rhs_cols, tile_size)
        lhs_matrix_vector_type = "vector<%dx%dxf32>" % (lhs_rows, lhs_cols)
        rhs_matrix_vector_type = "vector<%dx%dxf32>" % (lhs_cols, rhs_cols)
        result_matrix_vector_type = "vector<%dx%dxf32>" % (lhs_rows, rhs_cols)
        lhs_flat_vector_type = "vector<%dxf32>" % (lhs_rows * lhs_cols)
        rhs_flat_vector_type = "vector<%dxf32>" % (lhs_cols * rhs_cols)
        result_flat_vector_type = "vector<%dxf32>" % (lhs_rows * rhs_cols)
        return [
            "",
            "  func.func @%s(%%basis: %s, %%operand: %s, %%result: %s) attributes "
            '{sfem.sum_factor.operation = "%s", sfem.sum_factor.basis = "%s", '
            "sfem.sum_factor.axis = %d : i64, sfem.sum_factor.derivative = %d : i64, "
            'sfem.vector.matrix_multiply = "padded_memref", sfem.matrix_unit.tile_size = %d : i64, '
            "sfem.matrix_unit.raw_lhs_rows = %d : i64, sfem.matrix_unit.raw_lhs_columns = %d : i64, "
            "sfem.matrix_unit.raw_rhs_columns = %d : i64} {"
            % (
                function_name,
                stage.lhs_memref_type,
                stage.rhs_memref_type,
                stage.result_memref_type,
                stage.operation,
                stage.basis,
                stage.axis,
                stage.derivative,
                tile_size,
                stage.lhs_rows,
                stage.lhs_cols,
                stage.rhs_cols,
            ),
            "    %c0 = arith.constant 0 : index",
            "    %zero = arith.constant 0.0 : f32",
            "    %%basis_tile = vector.transfer_read %%basis[%%c0, %%c0], %%zero : %s, %s"
            % (stage.lhs_memref_type, lhs_matrix_vector_type),
            "    %%operand_tile = vector.transfer_read %%operand[%%c0, %%c0], %%zero : %s, %s"
            % (stage.rhs_memref_type, rhs_matrix_vector_type),
            "    %%basis_flat = vector.shape_cast %%basis_tile : %s to %s"
            % (lhs_matrix_vector_type, lhs_flat_vector_type),
            "    %%operand_flat = vector.shape_cast %%operand_tile : %s to %s"
            % (rhs_matrix_vector_type, rhs_flat_vector_type),
            "    %%result_flat = vector.matrix_multiply %%basis_flat, %%operand_flat "
            "{lhs_rows = %d : i32, lhs_columns = %d : i32, rhs_columns = %d : i32} "
            ": (%s, %s) -> %s"
            % (
                lhs_rows,
                lhs_cols,
                rhs_cols,
                lhs_flat_vector_type,
                rhs_flat_vector_type,
                result_flat_vector_type,
            ),
            "    %%result_tile = vector.shape_cast %%result_flat : %s to %s"
            % (result_flat_vector_type, result_matrix_vector_type),
            "    vector.transfer_write %%result_tile, %%result[%%c0, %%c0] : %s, %s"
            % (result_matrix_vector_type, stage.result_memref_type),
            "    return",
            "  }",
        ]

    def _gpu_kernel_name(self, stage):
        return "%s_%s_kernel" % (self.ir.function_prefix, stage.name)

    def _gpu_launch_name(self, stage):
        return "%s_%s_gpu" % (self.ir.function_prefix, stage.name)

    def _render_gpu_launch_function(self, stage):
        launch_name = self._gpu_launch_name(stage)
        kernel_name = self._gpu_kernel_name(stage)
        kernel_module_name = "%s_gpu_kernels" % self.ir.function_prefix
        return [
            "",
            "  func.func @%s(%%basis: %s, %%operand: %s, %%result: %s) attributes "
            '{sfem.sum_factor.operation = "%s", sfem.sum_factor.basis = "%s", '
            "sfem.sum_factor.axis = %d : i64, sfem.sum_factor.derivative = %d : i64} {"
            % (
                launch_name,
                stage.lhs_memref_type,
                stage.rhs_memref_type,
                stage.result_memref_type,
                stage.operation,
                stage.basis,
                stage.axis,
                stage.derivative,
            ),
            "    %c1 = arith.constant 1 : index",
            f"    %threads_x = arith.constant {stage.result_cols} : index",
            f"    %threads_y = arith.constant {stage.result_rows} : index",
            "    gpu.launch_func @%s::@%s blocks in (%%c1, %%c1, %%c1) "
            "threads in (%%threads_x, %%threads_y, %%c1) args(%%basis : %s, %%operand : %s, %%result : %s)"
            % (
                kernel_module_name,
                kernel_name,
                stage.lhs_memref_type,
                stage.rhs_memref_type,
                stage.result_memref_type,
            ),
            "    return",
            "  }",
        ]

    def _render_gpu_kernel(self, stage):
        kernel_name = self._gpu_kernel_name(stage)
        basis = _gpu_kernel_argument("basis", stage.lhs_memref_type, 0)
        operand = _gpu_kernel_argument("operand", stage.rhs_memref_type, 1)
        result = _gpu_kernel_argument("result", stage.result_memref_type, 2)
        return [
            "",
            "    gpu.func @%s(%s, %s, %s) kernel attributes "
            '{sfem.sum_factor.operation = "%s", sfem.sum_factor.basis = "%s", '
            "sfem.sum_factor.axis = %d : i64, sfem.sum_factor.derivative = %d : i64, %s} {"
            % (
                kernel_name,
                basis,
                operand,
                result,
                stage.operation,
                stage.basis,
                stage.axis,
                stage.derivative,
                _spirv_entry_point_abi(stage.result_rows, stage.result_cols, 1),
            ),
            "      %tx = gpu.thread_id x",
            "      %ty = gpu.thread_id y",
            "      %c0 = arith.constant 0 : index",
            "      %c1 = arith.constant 1 : index",
            f"      %contract_extent = arith.constant {stage.lhs_cols} : index",
            "      %zero = arith.constant 0.0 : f32",
            "      %sum = scf.for %k = %c0 to %contract_extent step %c1 iter_args(%acc = %zero) -> (f32) {",
            f"        %a = memref.load %basis[%ty, %k] : {stage.lhs_memref_type}",
            f"        %b = memref.load %operand[%k, %tx] : {stage.rhs_memref_type}",
            "        %prod = arith.mulf %a, %b : f32",
            "        %next = arith.addf %acc, %prod : f32",
            "        scf.yield %next : f32",
            "      }",
            f"      memref.store %sum, %result[%ty, %tx] : {stage.result_memref_type}",
            "      gpu.return",
            "    }",
        ]

    def _metal_kernel_name(self, stage):
        return "%s_%s_metal" % (self.ir.function_prefix, stage.name)

    def _render_metal_kernel(self, stage):
        return [
            "kernel void %s(" % self._metal_kernel_name(stage),
            "        device const float *basis [[buffer(0)]],",
            "        device const float *operand [[buffer(1)]],",
            "        device float *result [[buffer(2)]],",
            "        uint2 tid [[thread_position_in_grid]]) {",
            "    const uint col = tid.x;",
            "    const uint row = tid.y;",
            "    float acc = 0.0f;",
            "    #pragma unroll",
            "    for (uint k = 0; k < %d; ++k) {" % stage.lhs_cols,
            "        acc += basis[row * %d + k] * operand[k * %d + col];"
            % (stage.lhs_cols, stage.rhs_cols),
            "    }",
            "    result[row * %d + col] = acc;" % stage.result_cols,
            "}",
            "",
        ]


class TensorProductLaplaceFormMetalLowering:
    def __init__(self, ir):
        if not isinstance(ir, TensorProductLaplaceFormIR):
            raise TypeError("ir must be a TensorProductLaplaceFormIR")
        self.ir = ir

    def render_metal_source(self):
        sf = self.ir.sum_factor
        lines = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
            "constant float sfem_shape_1d[%d] = {%s};"
            % (len(sf.shape_values_1d), _float_initializer(sf.shape_values_1d)),
            "constant float sfem_grad_1d[%d] = {%s};"
            % (len(sf.shape_gradients_1d), _float_initializer(sf.shape_gradients_1d)),
            "constant float sfem_weight_1d[%d] = {%s};"
            % (len(sf.weights_1d), _float_initializer(sf.weights_1d)),
            "",
        ]
        if sf.dim == 2:
            lines.extend(self._render_quad_kernel())
        elif sf.dim == 3:
            lines.extend(self._render_hex_kernel())
        else:
            raise ValueError("unsupported laplace tensor-product dimension")
        return "\n".join(lines) + "\n"

    def render_metal_smoke_test_harness(self):
        sf = self.ir.sum_factor
        source = _objc_string_literal(self.render_metal_source())
        u = _deterministic_values(sf.n_shape, scale=0.03125, offset=0.5)
        return _LAPLACE_METAL_SMOKE_TEST_TEMPLATE % {
            "source": source,
            "kernel_name": self._metal_kernel_name(),
            "n_shape": sf.n_shape,
            "u": _float_initializer(u),
            "kappa": _c_float_literal(self.ir.parameter_default),
            "host_reference": self._render_host_reference_function(),
        }

    def write_inspection_artifacts(self, output_dir, *, include_metal_smoke_harness=True):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = output_dir / self.ir.function_prefix

        metal_path = prefix.with_suffix(".metal")
        harness_path = prefix.with_suffix(".metal_smoke.mm")

        metal_path.write_text(self.render_metal_source())

        files = [metal_path]
        if include_metal_smoke_harness:
            harness_path.write_text(self.render_metal_smoke_test_harness())
            files.append(harness_path)

        return CodeInspectionArtifacts(
            output_dir=str(output_dir),
            files=tuple(str(path) for path in files),
        )

    def run_metal_smoke_test(self, output_dir, *, xcrun=None):
        xcrun = xcrun or shutil.which("xcrun")
        if xcrun is None:
            raise FileNotFoundError("xcrun is not available")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        harness = output_dir / ("%s_metal_smoke.mm" % self.ir.function_prefix)
        executable = output_dir / ("%s_metal_smoke" % self.ir.function_prefix)
        harness.write_text(self.render_metal_smoke_test_harness())
        compile_result = subprocess.run(
            [
                xcrun,
                "clang++",
                str(harness),
                "-fobjc-arc",
                "-framework",
                "Foundation",
                "-framework",
                "Metal",
                "-o",
                str(executable),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if compile_result.returncode != 0:
            return MetalSmokeTestResult(
                harness,
                executable,
                compile_result.returncode,
                compile_result.stdout,
                compile_result.stderr,
            )
        run_result = subprocess.run(
            [str(executable)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return MetalSmokeTestResult(
            harness,
            executable,
            compile_result.returncode,
            compile_result.stdout,
            compile_result.stderr,
            run_result.returncode,
            run_result.stdout,
            run_result.stderr,
        )

    def _metal_kernel_name(self):
        return "%s_metal" % self.ir.function_prefix

    def _render_quad_kernel(self):
        sf = self.ir.sum_factor
        q = sf.n_qp_1d
        s = sf.n_shape_1d
        return [
            "kernel void %s(" % self._metal_kernel_name(),
            "        device const float *u [[buffer(0)]],",
            "        device float *out [[buffer(1)]],",
            "        constant float &kappa [[buffer(2)]],",
            "        uint row [[thread_position_in_grid]]) {",
            "    const uint rx = row %% %d;" % s,
            "    const uint ry = row / %d;" % s,
            "    float value = 0.0f;",
            "    #pragma unroll",
            "    for (uint qy = 0; qy < %d; ++qy) {" % q,
            "        #pragma unroll",
            "        for (uint qx = 0; qx < %d; ++qx) {" % q,
            "            const float wt = sfem_weight_1d[qx] * sfem_weight_1d[qy];",
            "            const float test_gx = sfem_grad_1d[qx * %d + rx] * sfem_shape_1d[qy * %d + ry];"
            % (s, s),
            "            const float test_gy = sfem_shape_1d[qx * %d + rx] * sfem_grad_1d[qy * %d + ry];"
            % (s, s),
            "            float grad_u_x = 0.0f;",
            "            float grad_u_y = 0.0f;",
            "            #pragma unroll",
            "            for (uint sy = 0; sy < %d; ++sy) {" % s,
            "                #pragma unroll",
            "                for (uint sx = 0; sx < %d; ++sx) {" % s,
            "                    const uint trial = sx + %d * sy;" % s,
            "                    const float coeff = u[trial];",
            "                    grad_u_x += coeff * sfem_grad_1d[qx * %d + sx] * sfem_shape_1d[qy * %d + sy];"
            % (s, s),
            "                    grad_u_y += coeff * sfem_shape_1d[qx * %d + sx] * sfem_grad_1d[qy * %d + sy];"
            % (s, s),
            "                }",
            "            }",
            "            value += kappa * wt * (test_gx * grad_u_x + test_gy * grad_u_y);",
            "        }",
            "    }",
            "    out[row] = value;",
            "}",
            "",
        ]

    def _render_hex_kernel(self):
        sf = self.ir.sum_factor
        q = sf.n_qp_1d
        s = sf.n_shape_1d
        return [
            "kernel void %s(" % self._metal_kernel_name(),
            "        device const float *u [[buffer(0)]],",
            "        device float *out [[buffer(1)]],",
            "        constant float &kappa [[buffer(2)]],",
            "        uint row [[thread_position_in_grid]]) {",
            "    const uint rx = row %% %d;" % s,
            "    const uint ry = (row / %d) %% %d;" % (s, s),
            "    const uint rz = row / %d;" % (s * s),
            "    float value = 0.0f;",
            "    #pragma unroll",
            "    for (uint qz = 0; qz < %d; ++qz) {" % q,
            "        #pragma unroll",
            "        for (uint qy = 0; qy < %d; ++qy) {" % q,
            "            #pragma unroll",
            "            for (uint qx = 0; qx < %d; ++qx) {" % q,
            "                const float wt = sfem_weight_1d[qx] * sfem_weight_1d[qy] * sfem_weight_1d[qz];",
            "                const float test_gx = sfem_grad_1d[qx * %d + rx] * sfem_shape_1d[qy * %d + ry] * sfem_shape_1d[qz * %d + rz];"
            % (s, s, s),
            "                const float test_gy = sfem_shape_1d[qx * %d + rx] * sfem_grad_1d[qy * %d + ry] * sfem_shape_1d[qz * %d + rz];"
            % (s, s, s),
            "                const float test_gz = sfem_shape_1d[qx * %d + rx] * sfem_shape_1d[qy * %d + ry] * sfem_grad_1d[qz * %d + rz];"
            % (s, s, s),
            "                float grad_u_x = 0.0f;",
            "                float grad_u_y = 0.0f;",
            "                float grad_u_z = 0.0f;",
            "                #pragma unroll",
            "                for (uint sz = 0; sz < %d; ++sz) {" % s,
            "                    #pragma unroll",
            "                    for (uint sy = 0; sy < %d; ++sy) {" % s,
            "                        #pragma unroll",
            "                        for (uint sx = 0; sx < %d; ++sx) {" % s,
            "                            const uint trial = sx + %d * (sy + %d * sz);" % (s, s),
            "                            const float coeff = u[trial];",
            "                            grad_u_x += coeff * sfem_grad_1d[qx * %d + sx] * sfem_shape_1d[qy * %d + sy] * sfem_shape_1d[qz * %d + sz];"
            % (s, s, s),
            "                            grad_u_y += coeff * sfem_shape_1d[qx * %d + sx] * sfem_grad_1d[qy * %d + sy] * sfem_shape_1d[qz * %d + sz];"
            % (s, s, s),
            "                            grad_u_z += coeff * sfem_shape_1d[qx * %d + sx] * sfem_shape_1d[qy * %d + sy] * sfem_grad_1d[qz * %d + sz];"
            % (s, s, s),
            "                        }",
            "                    }",
            "                }",
            "                value += kappa * wt * (test_gx * grad_u_x + test_gy * grad_u_y + test_gz * grad_u_z);",
            "            }",
            "        }",
            "    }",
            "    out[row] = value;",
            "}",
            "",
        ]

    def _render_host_reference_function(self):
        sf = self.ir.sum_factor
        arrays = [
            "static const float shape_1d[%d] = {%s};"
            % (len(sf.shape_values_1d), _float_initializer(sf.shape_values_1d)),
            "static const float grad_1d[%d] = {%s};"
            % (len(sf.shape_gradients_1d), _float_initializer(sf.shape_gradients_1d)),
            "static const float weight_1d[%d] = {%s};"
            % (len(sf.weights_1d), _float_initializer(sf.weights_1d)),
        ]
        if sf.dim == 2:
            body = self._render_quad_host_reference_body()
        elif sf.dim == 3:
            body = self._render_hex_host_reference_body()
        else:
            raise ValueError("unsupported laplace tensor-product dimension")
        return "\n".join(
            [
                "static void reference_apply(const float *u, float *out, const float kappa) {",
                *("    %s" % line for line in arrays),
                *("    %s" % line for line in body),
                "}",
            ]
        )

    def _render_quad_host_reference_body(self):
        sf = self.ir.sum_factor
        q = sf.n_qp_1d
        s = sf.n_shape_1d
        return [
            "for (unsigned row = 0; row < %d; ++row) {" % sf.n_shape,
            "    const unsigned rx = row %% %d;" % s,
            "    const unsigned ry = row / %d;" % s,
            "    float value = 0.0f;",
            "    for (unsigned qy = 0; qy < %d; ++qy) {" % q,
            "        for (unsigned qx = 0; qx < %d; ++qx) {" % q,
            "            const float wt = weight_1d[qx] * weight_1d[qy];",
            "            const float test_gx = grad_1d[qx * %d + rx] * shape_1d[qy * %d + ry];"
            % (s, s),
            "            const float test_gy = shape_1d[qx * %d + rx] * grad_1d[qy * %d + ry];"
            % (s, s),
            "            float grad_u_x = 0.0f;",
            "            float grad_u_y = 0.0f;",
            "            for (unsigned sy = 0; sy < %d; ++sy) {" % s,
            "                for (unsigned sx = 0; sx < %d; ++sx) {" % s,
            "                    const unsigned trial = sx + %d * sy;" % s,
            "                    const float coeff = u[trial];",
            "                    grad_u_x += coeff * grad_1d[qx * %d + sx] * shape_1d[qy * %d + sy];"
            % (s, s),
            "                    grad_u_y += coeff * shape_1d[qx * %d + sx] * grad_1d[qy * %d + sy];"
            % (s, s),
            "                }",
            "            }",
            "            value += kappa * wt * (test_gx * grad_u_x + test_gy * grad_u_y);",
            "        }",
            "    }",
            "    out[row] = value;",
            "}",
        ]

    def _render_hex_host_reference_body(self):
        sf = self.ir.sum_factor
        q = sf.n_qp_1d
        s = sf.n_shape_1d
        return [
            "for (unsigned row = 0; row < %d; ++row) {" % sf.n_shape,
            "    const unsigned rx = row %% %d;" % s,
            "    const unsigned ry = (row / %d) %% %d;" % (s, s),
            "    const unsigned rz = row / %d;" % (s * s),
            "    float value = 0.0f;",
            "    for (unsigned qz = 0; qz < %d; ++qz) {" % q,
            "        for (unsigned qy = 0; qy < %d; ++qy) {" % q,
            "            for (unsigned qx = 0; qx < %d; ++qx) {" % q,
            "                const float wt = weight_1d[qx] * weight_1d[qy] * weight_1d[qz];",
            "                const float test_gx = grad_1d[qx * %d + rx] * shape_1d[qy * %d + ry] * shape_1d[qz * %d + rz];"
            % (s, s, s),
            "                const float test_gy = shape_1d[qx * %d + rx] * grad_1d[qy * %d + ry] * shape_1d[qz * %d + rz];"
            % (s, s, s),
            "                const float test_gz = shape_1d[qx * %d + rx] * shape_1d[qy * %d + ry] * grad_1d[qz * %d + rz];"
            % (s, s, s),
            "                float grad_u_x = 0.0f;",
            "                float grad_u_y = 0.0f;",
            "                float grad_u_z = 0.0f;",
            "                for (unsigned sz = 0; sz < %d; ++sz) {" % s,
            "                    for (unsigned sy = 0; sy < %d; ++sy) {" % s,
            "                        for (unsigned sx = 0; sx < %d; ++sx) {" % s,
            "                            const unsigned trial = sx + %d * (sy + %d * sz);" % (s, s),
            "                            const float coeff = u[trial];",
            "                            grad_u_x += coeff * grad_1d[qx * %d + sx] * shape_1d[qy * %d + sy] * shape_1d[qz * %d + sz];"
            % (s, s, s),
            "                            grad_u_y += coeff * shape_1d[qx * %d + sx] * grad_1d[qy * %d + sy] * shape_1d[qz * %d + sz];"
            % (s, s, s),
            "                            grad_u_z += coeff * shape_1d[qx * %d + sx] * shape_1d[qy * %d + sy] * grad_1d[qz * %d + sz];"
            % (s, s, s),
            "                        }",
            "                    }",
            "                }",
            "                value += kappa * wt * (test_gx * grad_u_x + test_gy * grad_u_y + test_gz * grad_u_z);",
            "            }",
            "        }",
            "    }",
            "    out[row] = value;",
            "}",
        ]


class TensorProductLaplaceFormGPULowering:
    def __init__(self, ir):
        if not isinstance(ir, TensorProductLaplaceFormIR):
            raise TypeError("ir must be a TensorProductLaplaceFormIR")
        self.ir = ir

    def render_gpu_module(self):
        sf = self.ir.sum_factor
        lines = [
            'module attributes {gpu.container_module, sfem.material = "%s", sfem.element = "%s", '
            'sfem.lowering = "tensor_product_laplace_form_gpu"} {'
            % (sf.material_name, sf.element_type),
            "",
            "  func.func @%s_gpu(%%u: memref<%dxf32>, %%out: memref<%dxf32>, "
            "%%kappa: memref<1xf32>, %%shape_1d: memref<%dxf32>, %%grad_1d: memref<%dxf32>, "
            "%%weight_1d: memref<%dxf32>) attributes {sfem.form = \"laplace\", sfem.parameter = \"kappa\"} {"
            % (
                self.ir.function_prefix,
                sf.n_shape,
                sf.n_shape,
                len(sf.shape_values_1d),
                len(sf.shape_gradients_1d),
                len(sf.weights_1d),
            ),
            "    %c1 = arith.constant 1 : index",
            f"    %threads = arith.constant {sf.n_shape} : index",
            "    gpu.launch_func @%s_gpu_kernels::@%s_kernel blocks in (%%c1, %%c1, %%c1) "
            "threads in (%%threads, %%c1, %%c1) args(%%u : memref<%dxf32>, %%out : memref<%dxf32>, "
            "%%kappa : memref<1xf32>, %%shape_1d : memref<%dxf32>, %%grad_1d : memref<%dxf32>, "
            "%%weight_1d : memref<%dxf32>)"
            % (
                self.ir.function_prefix,
                self.ir.function_prefix,
                sf.n_shape,
                sf.n_shape,
                len(sf.shape_values_1d),
                len(sf.shape_gradients_1d),
                len(sf.weights_1d),
            ),
            "    return",
            "  }",
            "",
            "  gpu.module @%s_gpu_kernels {" % self.ir.function_prefix,
        ]
        if sf.dim == 2:
            lines.extend(self._render_quad_kernel())
        elif sf.dim == 3:
            lines.extend(self._render_hex_kernel())
        else:
            raise ValueError("unsupported laplace tensor-product dimension")
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def write_inspection_artifacts(
        self,
        output_dir,
        *,
        include_metal_source=True,
        include_metal_smoke_harness=True,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = output_dir / self.ir.function_prefix

        gpu_path = prefix.with_suffix(".gpu.mlir")
        metal_path = prefix.with_suffix(".metal")
        harness_path = prefix.with_suffix(".metal_smoke.mm")

        gpu_path.write_text(self.render_gpu_module())

        files = [gpu_path]
        if include_metal_source or include_metal_smoke_harness:
            metal = TensorProductLaplaceFormMetalLowering(self.ir)
            if include_metal_source:
                metal_path.write_text(metal.render_metal_source())
                files.append(metal_path)
            if include_metal_smoke_harness:
                harness_path.write_text(metal.render_metal_smoke_test_harness())
                files.append(harness_path)

        return CodeInspectionArtifacts(
            output_dir=str(output_dir),
            files=tuple(str(path) for path in files),
        )

    def _kernel_signature(self):
        sf = self.ir.sum_factor
        return (
            "    gpu.func @%s_kernel(%s, %s, %s, %s, %s, %s) kernel "
            "attributes {sfem.form = \"laplace\", sfem.parameter = \"kappa\", %s} {"
            % (
                self.ir.function_prefix,
                _gpu_kernel_argument("u", "memref<%dxf32>" % sf.n_shape, 0),
                _gpu_kernel_argument("out", "memref<%dxf32>" % sf.n_shape, 1),
                _gpu_kernel_argument("kappa_ref", "memref<1xf32>", 2),
                _gpu_kernel_argument("shape_1d", "memref<%dxf32>" % len(sf.shape_values_1d), 3),
                _gpu_kernel_argument("grad_1d", "memref<%dxf32>" % len(sf.shape_gradients_1d), 4),
                _gpu_kernel_argument("weight_1d", "memref<%dxf32>" % len(sf.weights_1d), 5),
                _spirv_entry_point_abi(sf.n_shape, 1, 1),
            )
        )

    def _render_quad_kernel(self):
        sf = self.ir.sum_factor
        s = sf.n_shape_1d
        q = sf.n_qp_1d
        n_shape = sf.n_shape
        ref_size = len(sf.shape_values_1d)
        return [
            self._kernel_signature(),
            "      %row = gpu.thread_id x",
            "      %c0 = arith.constant 0 : index",
            "      %c1 = arith.constant 1 : index",
            f"      %S = arith.constant {s} : index",
            f"      %Q = arith.constant {q} : index",
            "      %zero = arith.constant 0.0 : f32",
            "      %rx = arith.remui %row, %S : index",
            "      %ry = arith.divui %row, %S : index",
            "      %kappa = memref.load %kappa_ref[%c0] : memref<1xf32>",
            "      %sum_y = scf.for %qy = %c0 to %Q step %c1 iter_args(%acc_y = %zero) -> (f32) {",
            "        %sum_x = scf.for %qx = %c0 to %Q step %c1 iter_args(%acc_x = %acc_y) -> (f32) {",
            "          %qx_s = arith.muli %qx, %S : index",
            "          %qy_s = arith.muli %qy, %S : index",
            "          %test_gx_i = arith.addi %qx_s, %rx : index",
            "          %test_gy_i = arith.addi %qy_s, %ry : index",
            f"          %test_gx_a = memref.load %grad_1d[%test_gx_i] : memref<{ref_size}xf32>",
            f"          %test_gx_b = memref.load %shape_1d[%test_gy_i] : memref<{ref_size}xf32>",
            "          %test_gx = arith.mulf %test_gx_a, %test_gx_b : f32",
            f"          %test_gy_a = memref.load %shape_1d[%test_gx_i] : memref<{ref_size}xf32>",
            f"          %test_gy_b = memref.load %grad_1d[%test_gy_i] : memref<{ref_size}xf32>",
            "          %test_gy = arith.mulf %test_gy_a, %test_gy_b : f32",
            "          %grad_y_x, %grad_y_y = scf.for %sy = %c0 to %S step %c1 iter_args(%acc_gx_y = %zero, %acc_gy_y = %zero) -> (f32, f32) {",
            "            %grad_x_x, %grad_x_y = scf.for %sx = %c0 to %S step %c1 iter_args(%acc_gx_x = %acc_gx_y, %acc_gy_x = %acc_gy_y) -> (f32, f32) {",
            "              %trial_y = arith.muli %sy, %S : index",
            "              %trial = arith.addi %sx, %trial_y : index",
            f"              %coeff = memref.load %u[%trial] : memref<{n_shape}xf32>",
            "              %gx_i = arith.addi %qx_s, %sx : index",
            "              %gy_i = arith.addi %qy_s, %sy : index",
            f"              %gx_a = memref.load %grad_1d[%gx_i] : memref<{ref_size}xf32>",
            f"              %gx_b = memref.load %shape_1d[%gy_i] : memref<{ref_size}xf32>",
            f"              %gy_a = memref.load %shape_1d[%gx_i] : memref<{ref_size}xf32>",
            f"              %gy_b = memref.load %grad_1d[%gy_i] : memref<{ref_size}xf32>",
            "              %gx_basis = arith.mulf %gx_a, %gx_b : f32",
            "              %gy_basis = arith.mulf %gy_a, %gy_b : f32",
            "              %gx_term = arith.mulf %coeff, %gx_basis : f32",
            "              %gy_term = arith.mulf %coeff, %gy_basis : f32",
            "              %next_gx = arith.addf %acc_gx_x, %gx_term : f32",
            "              %next_gy = arith.addf %acc_gy_x, %gy_term : f32",
            "              scf.yield %next_gx, %next_gy : f32, f32",
            "            }",
            "            scf.yield %grad_x_x, %grad_x_y : f32, f32",
            "          }",
            "          %dot_x = arith.mulf %test_gx, %grad_y_x : f32",
            "          %dot_y = arith.mulf %test_gy, %grad_y_y : f32",
            "          %dot = arith.addf %dot_x, %dot_y : f32",
            f"          %wx = memref.load %weight_1d[%qx] : memref<{q}xf32>",
            f"          %wy = memref.load %weight_1d[%qy] : memref<{q}xf32>",
            "          %w = arith.mulf %wx, %wy : f32",
            "          %scaled0 = arith.mulf %kappa, %w : f32",
            "          %scaled = arith.mulf %scaled0, %dot : f32",
            "          %next = arith.addf %acc_x, %scaled : f32",
            "          scf.yield %next : f32",
            "        }",
            "        scf.yield %sum_x : f32",
            "      }",
            f"      memref.store %sum_y, %out[%row] : memref<{n_shape}xf32>",
            "      gpu.return",
            "    }",
        ]

    def _render_hex_kernel(self):
        sf = self.ir.sum_factor
        s = sf.n_shape_1d
        q = sf.n_qp_1d
        n_shape = sf.n_shape
        ref_size = len(sf.shape_values_1d)
        return [
            self._kernel_signature(),
            "      %row = gpu.thread_id x",
            "      %c0 = arith.constant 0 : index",
            "      %c1 = arith.constant 1 : index",
            f"      %S = arith.constant {s} : index",
            f"      %SS = arith.constant {s * s} : index",
            f"      %Q = arith.constant {q} : index",
            "      %zero = arith.constant 0.0 : f32",
            "      %rx = arith.remui %row, %S : index",
            "      %row_div_s = arith.divui %row, %S : index",
            "      %ry = arith.remui %row_div_s, %S : index",
            "      %rz = arith.divui %row, %SS : index",
            "      %kappa = memref.load %kappa_ref[%c0] : memref<1xf32>",
            "      %sum_z = scf.for %qz = %c0 to %Q step %c1 iter_args(%acc_z = %zero) -> (f32) {",
            "        %sum_y = scf.for %qy = %c0 to %Q step %c1 iter_args(%acc_y = %acc_z) -> (f32) {",
            "          %sum_x = scf.for %qx = %c0 to %Q step %c1 iter_args(%acc_x = %acc_y) -> (f32) {",
            "            %qx_s = arith.muli %qx, %S : index",
            "            %qy_s = arith.muli %qy, %S : index",
            "            %qz_s = arith.muli %qz, %S : index",
            "            %ix = arith.addi %qx_s, %rx : index",
            "            %iy = arith.addi %qy_s, %ry : index",
            "            %iz = arith.addi %qz_s, %rz : index",
            f"            %test_gx_a = memref.load %grad_1d[%ix] : memref<{ref_size}xf32>",
            f"            %test_gx_b = memref.load %shape_1d[%iy] : memref<{ref_size}xf32>",
            f"            %test_gx_c = memref.load %shape_1d[%iz] : memref<{ref_size}xf32>",
            "            %test_gx_ab = arith.mulf %test_gx_a, %test_gx_b : f32",
            "            %test_gx = arith.mulf %test_gx_ab, %test_gx_c : f32",
            f"            %test_gy_a = memref.load %shape_1d[%ix] : memref<{ref_size}xf32>",
            f"            %test_gy_b = memref.load %grad_1d[%iy] : memref<{ref_size}xf32>",
            f"            %test_gy_c = memref.load %shape_1d[%iz] : memref<{ref_size}xf32>",
            "            %test_gy_ab = arith.mulf %test_gy_a, %test_gy_b : f32",
            "            %test_gy = arith.mulf %test_gy_ab, %test_gy_c : f32",
            f"            %test_gz_a = memref.load %shape_1d[%ix] : memref<{ref_size}xf32>",
            f"            %test_gz_b = memref.load %shape_1d[%iy] : memref<{ref_size}xf32>",
            f"            %test_gz_c = memref.load %grad_1d[%iz] : memref<{ref_size}xf32>",
            "            %test_gz_ab = arith.mulf %test_gz_a, %test_gz_b : f32",
            "            %test_gz = arith.mulf %test_gz_ab, %test_gz_c : f32",
            "            %grad_z_x, %grad_z_y, %grad_z_z = scf.for %sz = %c0 to %S step %c1 iter_args(%acc_gx_z = %zero, %acc_gy_z = %zero, %acc_gz_z = %zero) -> (f32, f32, f32) {",
            "              %grad_y_x, %grad_y_y, %grad_y_z = scf.for %sy = %c0 to %S step %c1 iter_args(%acc_gx_y = %acc_gx_z, %acc_gy_y = %acc_gy_z, %acc_gz_y = %acc_gz_z) -> (f32, f32, f32) {",
            "                %grad_x_x, %grad_x_y, %grad_x_z = scf.for %sx = %c0 to %S step %c1 iter_args(%acc_gx_x = %acc_gx_y, %acc_gy_x = %acc_gy_y, %acc_gz_x = %acc_gz_y) -> (f32, f32, f32) {",
            "                  %sy_s = arith.muli %sy, %S : index",
            "                  %sz_ss = arith.muli %sz, %SS : index",
            "                  %trial_y = arith.addi %sx, %sy_s : index",
            "                  %trial = arith.addi %trial_y, %sz_ss : index",
            f"                  %coeff = memref.load %u[%trial] : memref<{n_shape}xf32>",
            "                  %jx = arith.addi %qx_s, %sx : index",
            "                  %jy = arith.addi %qy_s, %sy : index",
            "                  %jz = arith.addi %qz_s, %sz : index",
            f"                  %gx_a = memref.load %grad_1d[%jx] : memref<{ref_size}xf32>",
            f"                  %gx_b = memref.load %shape_1d[%jy] : memref<{ref_size}xf32>",
            f"                  %gx_c = memref.load %shape_1d[%jz] : memref<{ref_size}xf32>",
            "                  %gx_ab = arith.mulf %gx_a, %gx_b : f32",
            "                  %gx_basis = arith.mulf %gx_ab, %gx_c : f32",
            f"                  %gy_a = memref.load %shape_1d[%jx] : memref<{ref_size}xf32>",
            f"                  %gy_b = memref.load %grad_1d[%jy] : memref<{ref_size}xf32>",
            f"                  %gy_c = memref.load %shape_1d[%jz] : memref<{ref_size}xf32>",
            "                  %gy_ab = arith.mulf %gy_a, %gy_b : f32",
            "                  %gy_basis = arith.mulf %gy_ab, %gy_c : f32",
            f"                  %gz_a = memref.load %shape_1d[%jx] : memref<{ref_size}xf32>",
            f"                  %gz_b = memref.load %shape_1d[%jy] : memref<{ref_size}xf32>",
            f"                  %gz_c = memref.load %grad_1d[%jz] : memref<{ref_size}xf32>",
            "                  %gz_ab = arith.mulf %gz_a, %gz_b : f32",
            "                  %gz_basis = arith.mulf %gz_ab, %gz_c : f32",
            "                  %gx_term = arith.mulf %coeff, %gx_basis : f32",
            "                  %gy_term = arith.mulf %coeff, %gy_basis : f32",
            "                  %gz_term = arith.mulf %coeff, %gz_basis : f32",
            "                  %next_gx = arith.addf %acc_gx_x, %gx_term : f32",
            "                  %next_gy = arith.addf %acc_gy_x, %gy_term : f32",
            "                  %next_gz = arith.addf %acc_gz_x, %gz_term : f32",
            "                  scf.yield %next_gx, %next_gy, %next_gz : f32, f32, f32",
            "                }",
            "                scf.yield %grad_x_x, %grad_x_y, %grad_x_z : f32, f32, f32",
            "              }",
            "              scf.yield %grad_y_x, %grad_y_y, %grad_y_z : f32, f32, f32",
            "            }",
            "            %dot_x = arith.mulf %test_gx, %grad_z_x : f32",
            "            %dot_y = arith.mulf %test_gy, %grad_z_y : f32",
            "            %dot_xy = arith.addf %dot_x, %dot_y : f32",
            "            %dot_z = arith.mulf %test_gz, %grad_z_z : f32",
            "            %dot = arith.addf %dot_xy, %dot_z : f32",
            f"            %wx = memref.load %weight_1d[%qx] : memref<{q}xf32>",
            f"            %wy = memref.load %weight_1d[%qy] : memref<{q}xf32>",
            f"            %wz = memref.load %weight_1d[%qz] : memref<{q}xf32>",
            "            %wxy = arith.mulf %wx, %wy : f32",
            "            %w = arith.mulf %wxy, %wz : f32",
            "            %scaled0 = arith.mulf %kappa, %w : f32",
            "            %scaled = arith.mulf %scaled0, %dot : f32",
            "            %next = arith.addf %acc_x, %scaled : f32",
            "            scf.yield %next : f32",
            "          }",
            "          scf.yield %sum_x : f32",
            "        }",
            "        scf.yield %sum_y : f32",
            "      }",
            f"      memref.store %sum_z, %out[%row] : memref<{n_shape}xf32>",
            "      gpu.return",
            "    }",
        ]


class TensorProductLaplaceFormBatchedGPULowering(TensorProductLaplaceFormGPULowering):
    """Batched EBE map lowering for tensor-product Laplace forms.

    The emitted GPU kernel launches one block per element and one thread per
    local test function.  It gathers the local trial vector through static
    connectivity and stores an element-local residual scratch buffer, so the
    map phase stays branch-free and does not require global atomics.
    """

    def __init__(self, ir, *, max_elements, max_nodes):
        super().__init__(ir)
        self.max_elements = int(max_elements)
        self.max_nodes = int(max_nodes)
        if self.max_elements <= 0 or self.max_nodes <= 0:
            raise ValueError("batched tensor-product GPU bounds must be positive")

    @property
    def connectivity_memref_type(self):
        return "memref<%dx%dxindex>" % (self.max_elements, self.ir.sum_factor.n_shape)

    @property
    def u_memref_type(self):
        return "memref<%dxf32>" % self.max_nodes

    @property
    def element_out_memref_type(self):
        return "memref<%dx%dxf32>" % (self.max_elements, self.ir.sum_factor.n_shape)

    def render_gpu_module(self):
        sf = self.ir.sum_factor
        lines = [
            'module attributes {gpu.container_module, sfem.material = "%s", sfem.element = "%s", '
            'sfem.lowering = "tensor_product_laplace_ebe_gpu_map"} {'
            % (sf.material_name, sf.element_type),
            "",
            "  func.func @%s_ebe_gpu(%%connectivity: %s, %%u: %s, %%element_out: %s, "
            "%%kappa: memref<1xf32>, %%shape_1d: memref<%dxf32>, %%grad_1d: memref<%dxf32>, "
            "%%weight_1d: memref<%dxf32>) attributes {sfem.form = \"laplace\", "
            'sfem.mesh_phase = "ebe_map", sfem.parameter = "kappa"} {'
            % (
                self.ir.function_prefix,
                self.connectivity_memref_type,
                self.u_memref_type,
                self.element_out_memref_type,
                len(sf.shape_values_1d),
                len(sf.shape_gradients_1d),
                len(sf.weights_1d),
            ),
            "    %c1 = arith.constant 1 : index",
            f"    %blocks = arith.constant {self.max_elements} : index",
            f"    %threads = arith.constant {sf.n_shape} : index",
            "    gpu.launch_func @%s_ebe_gpu_kernels::@%s_ebe_kernel blocks in (%%blocks, %%c1, %%c1) "
            "threads in (%%threads, %%c1, %%c1) args(%%connectivity : %s, %%u : %s, %%element_out : %s, "
            "%%kappa : memref<1xf32>, %%shape_1d : memref<%dxf32>, %%grad_1d : memref<%dxf32>, "
            "%%weight_1d : memref<%dxf32>)"
            % (
                self.ir.function_prefix,
                self.ir.function_prefix,
                self.connectivity_memref_type,
                self.u_memref_type,
                self.element_out_memref_type,
                len(sf.shape_values_1d),
                len(sf.shape_gradients_1d),
                len(sf.weights_1d),
            ),
            "    return",
            "  }",
            "",
            "  gpu.module @%s_ebe_gpu_kernels {" % self.ir.function_prefix,
        ]
        if sf.dim == 2:
            lines.extend(self._render_quad_kernel())
        elif sf.dim == 3:
            lines.extend(self._render_hex_kernel())
        else:
            raise ValueError("unsupported laplace tensor-product dimension")
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def write_inspection_artifacts(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        gpu_path = output_dir / ("%s.ebe.gpu.mlir" % self.ir.function_prefix)
        gpu_path.write_text(self.render_gpu_module())
        return CodeInspectionArtifacts(
            output_dir=str(output_dir),
            files=(str(gpu_path),),
        )

    def _kernel_signature(self):
        sf = self.ir.sum_factor
        return (
            "    gpu.func @%s_ebe_kernel(%s, %s, %s, %s, %s, %s, %s) kernel "
            "attributes {sfem.form = \"laplace\", sfem.mesh_phase = \"ebe_map\", "
            'sfem.parameter = "kappa", %s} {'
            % (
                self.ir.function_prefix,
                _gpu_kernel_argument("connectivity", self.connectivity_memref_type, 0),
                _gpu_kernel_argument("u", self.u_memref_type, 1),
                _gpu_kernel_argument("element_out", self.element_out_memref_type, 2),
                _gpu_kernel_argument("kappa_ref", "memref<1xf32>", 3),
                _gpu_kernel_argument("shape_1d", "memref<%dxf32>" % len(sf.shape_values_1d), 4),
                _gpu_kernel_argument("grad_1d", "memref<%dxf32>" % len(sf.shape_gradients_1d), 5),
                _gpu_kernel_argument("weight_1d", "memref<%dxf32>" % len(sf.weights_1d), 6),
                _spirv_entry_point_abi(sf.n_shape, 1, 1),
            )
        )

    def _render_quad_kernel(self):
        return self._batched_kernel_lines(super()._render_quad_kernel())

    def _render_hex_kernel(self):
        return self._batched_kernel_lines(super()._render_hex_kernel())

    def _batched_kernel_lines(self, lines):
        sf = self.ir.sum_factor
        rewritten = []
        for line in lines:
            rewritten.append(line)
            if line == "      %row = gpu.thread_id x":
                rewritten.append("      %elem = gpu.block_id x")
                continue
            if "%coeff = memref.load %u[%trial]" in line:
                rewritten.pop()
                rewritten.append(
                    f"              %node = memref.load %connectivity[%elem, %trial] : {self.connectivity_memref_type}"
                )
                rewritten.append(
                    f"              %coeff = memref.load %u[%node] : {self.u_memref_type}"
                )
                continue
            if line.strip().startswith("memref.store %sum_") and "%out[%row]" in line:
                rewritten.pop()
                sum_name = line.strip().split(",")[0].split()[1]
                rewritten.append(
                    "      memref.store %s, %%element_out[%%elem, %%row] : %s"
                    % (sum_name, self.element_out_memref_type)
                )
                continue
        expected_coeff_loads = 1
        actual_coeff_loads = sum(1 for line in rewritten if "memref.load %u[%node]" in line)
        if actual_coeff_loads != expected_coeff_loads:
            raise ValueError(
                "expected to rewrite one local coefficient load for %s, found %d"
                % (sf.element_type, actual_coeff_loads)
            )
        return rewritten


class TensorProductLaplaceFormEBEGPULowering(TensorProductLaplaceFormBatchedGPULowering):
    """Atomics-free EBE map/reduce lowering for scalar tensor-product Laplace."""

    def __init__(self, ir, *, max_elements, max_nodes, max_node_degree):
        super().__init__(ir, max_elements=max_elements, max_nodes=max_nodes)
        self.max_node_degree = int(max_node_degree)
        if self.max_node_degree <= 0:
            raise ValueError("tensor-product EBE GPU max_node_degree must be positive")

    @property
    def node_degree_memref_type(self):
        return "memref<%dxindex>" % self.max_nodes

    @property
    def inverse_topology_memref_type(self):
        return "memref<%dx%dxindex>" % (self.max_nodes, self.max_node_degree)

    @property
    def output_memref_type(self):
        return "memref<%dxf32>" % self.max_nodes

    def render_gpu_module(self):
        sf = self.ir.sum_factor
        lines = [
            'module attributes {gpu.container_module, sfem.material = "%s", sfem.element = "%s", '
            'sfem.lowering = "tensor_product_laplace_ebe_gpu"} {'
            % (sf.material_name, sf.element_type),
            "",
            "  func.func @%s_ebe_gpu(%%connectivity: %s, %%u: %s, %%element_out: %s, "
            "%%node_degree: %s, %%node_to_element_map: %s, %%node_to_local_idx: %s, %%out: %s, "
            "%%kappa: memref<1xf32>, %%shape_1d: memref<%dxf32>, %%grad_1d: memref<%dxf32>, "
            "%%weight_1d: memref<%dxf32>) attributes {sfem.form = \"laplace\", "
            'sfem.mesh_phases = "ebe_map,ebe_reduce", sfem.parameter = "kappa"} {'
            % (
                self.ir.function_prefix,
                self.connectivity_memref_type,
                self.u_memref_type,
                self.element_out_memref_type,
                self.node_degree_memref_type,
                self.inverse_topology_memref_type,
                self.inverse_topology_memref_type,
                self.output_memref_type,
                len(sf.shape_values_1d),
                len(sf.shape_gradients_1d),
                len(sf.weights_1d),
            ),
            "    %c1 = arith.constant 1 : index",
            f"    %map_blocks = arith.constant {self.max_elements} : index",
            f"    %map_threads = arith.constant {sf.n_shape} : index",
            f"    %reduce_threads = arith.constant {self.max_nodes} : index",
            "    gpu.launch_func @%s_ebe_gpu_kernels::@%s_ebe_map_kernel blocks in (%%map_blocks, %%c1, %%c1) "
            "threads in (%%map_threads, %%c1, %%c1) args(%%connectivity : %s, %%u : %s, %%element_out : %s, "
            "%%kappa : memref<1xf32>, %%shape_1d : memref<%dxf32>, %%grad_1d : memref<%dxf32>, "
            "%%weight_1d : memref<%dxf32>)"
            % (
                self.ir.function_prefix,
                self.ir.function_prefix,
                self.connectivity_memref_type,
                self.u_memref_type,
                self.element_out_memref_type,
                len(sf.shape_values_1d),
                len(sf.shape_gradients_1d),
                len(sf.weights_1d),
            ),
            "    gpu.launch_func @%s_ebe_gpu_kernels::@%s_ebe_reduce_kernel blocks in (%%c1, %%c1, %%c1) "
            "threads in (%%reduce_threads, %%c1, %%c1) args(%%element_out : %s, %%node_degree : %s, "
            "%%node_to_element_map : %s, %%node_to_local_idx : %s, %%out : %s)"
            % (
                self.ir.function_prefix,
                self.ir.function_prefix,
                self.element_out_memref_type,
                self.node_degree_memref_type,
                self.inverse_topology_memref_type,
                self.inverse_topology_memref_type,
                self.output_memref_type,
            ),
            "    return",
            "  }",
            "",
            "  gpu.module @%s_ebe_gpu_kernels {" % self.ir.function_prefix,
        ]
        if sf.dim == 2:
            lines.extend(self._render_quad_kernel())
        elif sf.dim == 3:
            lines.extend(self._render_hex_kernel())
        else:
            raise ValueError("unsupported laplace tensor-product dimension")
        lines.extend(self._render_reduce_kernel())
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def write_inspection_artifacts(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        gpu_path = output_dir / ("%s.ebe.full.gpu.mlir" % self.ir.function_prefix)
        gpu_path.write_text(self.render_gpu_module())
        return CodeInspectionArtifacts(
            output_dir=str(output_dir),
            files=(str(gpu_path),),
        )

    def _kernel_signature(self):
        sf = self.ir.sum_factor
        return (
            "    gpu.func @%s_ebe_map_kernel(%s, %s, %s, %s, %s, %s, %s) kernel "
            "attributes {sfem.form = \"laplace\", sfem.mesh_phase = \"ebe_map\", "
            'sfem.parameter = "kappa", %s} {'
            % (
                self.ir.function_prefix,
                _gpu_kernel_argument("connectivity", self.connectivity_memref_type, 0),
                _gpu_kernel_argument("u", self.u_memref_type, 1),
                _gpu_kernel_argument("element_out", self.element_out_memref_type, 2),
                _gpu_kernel_argument("kappa_ref", "memref<1xf32>", 3),
                _gpu_kernel_argument("shape_1d", "memref<%dxf32>" % len(sf.shape_values_1d), 4),
                _gpu_kernel_argument("grad_1d", "memref<%dxf32>" % len(sf.shape_gradients_1d), 5),
                _gpu_kernel_argument("weight_1d", "memref<%dxf32>" % len(sf.weights_1d), 6),
                _spirv_entry_point_abi(sf.n_shape, 1, 1),
            )
        )

    def _render_reduce_kernel(self):
        return [
            "",
            "    gpu.func @%s_ebe_reduce_kernel(%s, %s, %s, %s, %s) kernel "
            'attributes {sfem.form = "laplace", sfem.mesh_phase = "ebe_reduce", %s} {'
            % (
                self.ir.function_prefix,
                _gpu_kernel_argument("element_out", self.element_out_memref_type, 0),
                _gpu_kernel_argument("node_degree", self.node_degree_memref_type, 1),
                _gpu_kernel_argument("node_to_element_map", self.inverse_topology_memref_type, 2),
                _gpu_kernel_argument("node_to_local_idx", self.inverse_topology_memref_type, 3),
                _gpu_kernel_argument("out", self.output_memref_type, 4),
                _spirv_entry_point_abi(self.max_nodes, 1, 1),
            ),
            "      %node = gpu.thread_id x",
            "      %c0 = arith.constant 0 : index",
            "      %c1 = arith.constant 1 : index",
            "      %zero = arith.constant 0.0 : f32",
            f"      %degree = memref.load %node_degree[%node] : {self.node_degree_memref_type}",
            "      %sum = scf.for %i = %c0 to %degree step %c1 iter_args(%acc = %zero) -> (f32) {",
            f"        %elem = memref.load %node_to_element_map[%node, %i] : {self.inverse_topology_memref_type}",
            f"        %local = memref.load %node_to_local_idx[%node, %i] : {self.inverse_topology_memref_type}",
            f"        %value = memref.load %element_out[%elem, %local] : {self.element_out_memref_type}",
            "        %next = arith.addf %acc, %value : f32",
            "        scf.yield %next : f32",
            "      }",
            f"      memref.store %sum, %out[%node] : {self.output_memref_type}",
            "      gpu.return",
            "    }",
        ]


class TensorProductLaplaceFormEBEMetalLowering(TensorProductLaplaceFormMetalLowering):
    """Metal EBE map/reduce lowering for scalar tensor-product Laplace."""

    def __init__(self, ir, *, max_elements, max_nodes, max_node_degree):
        super().__init__(ir)
        self.max_elements = int(max_elements)
        self.max_nodes = int(max_nodes)
        self.max_node_degree = int(max_node_degree)
        if self.max_elements <= 0 or self.max_nodes <= 0 or self.max_node_degree <= 0:
            raise ValueError("Metal EBE bounds must be positive")

    def render_metal_source(self):
        sf = self.ir.sum_factor
        lines = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
            "constant float sfem_shape_1d[%d] = {%s};"
            % (len(sf.shape_values_1d), _float_initializer(sf.shape_values_1d)),
            "constant float sfem_grad_1d[%d] = {%s};"
            % (len(sf.shape_gradients_1d), _float_initializer(sf.shape_gradients_1d)),
            "constant float sfem_weight_1d[%d] = {%s};"
            % (len(sf.weights_1d), _float_initializer(sf.weights_1d)),
            "",
        ]
        if sf.dim == 2:
            lines.extend(self._render_map_kernel_from_local(self._render_quad_kernel()))
        elif sf.dim == 3:
            lines.extend(self._render_map_kernel_from_local(self._render_hex_kernel()))
        else:
            raise ValueError("unsupported laplace tensor-product dimension")
        lines.extend(self._render_reduce_kernel())
        return "\n".join(lines) + "\n"

    def render_metal_smoke_test_harness(self):
        sf = self.ir.sum_factor
        fixture_elements = 1
        fixture_nodes = sf.n_shape
        if self.max_elements >= 2 and self.max_nodes >= 2 * sf.n_shape - 1 and self.max_node_degree >= 2:
            fixture_elements = 2
            fixture_nodes = 2 * sf.n_shape - 1

        connectivity_rows = [tuple(range(sf.n_shape))]
        if fixture_elements == 2:
            connectivity_rows.append(tuple(range(sf.n_shape - 1, 2 * sf.n_shape - 1)))
        connectivity = tuple(node for row in connectivity_rows for node in row)

        node_degree = [0 for _ in range(fixture_nodes)]
        node_to_element_map = [0 for _ in range(fixture_nodes * self.max_node_degree)]
        node_to_local_idx = [0 for _ in range(fixture_nodes * self.max_node_degree)]
        for elem, row in enumerate(connectivity_rows):
            for local, node in enumerate(row):
                degree = node_degree[node]
                map_index = node * self.max_node_degree + degree
                node_to_element_map[map_index] = elem
                node_to_local_idx[map_index] = local
                node_degree[node] = degree + 1

        u = _deterministic_values(fixture_nodes, scale=0.03125, offset=0.5)
        return _LAPLACE_EBE_METAL_SMOKE_TEST_TEMPLATE % {
            "source": _objc_string_literal(self.render_metal_source()),
            "map_kernel_name": self._map_kernel_name(),
            "reduce_kernel_name": self._reduce_kernel_name(),
            "host_reference": self._render_host_reference_function(),
            "n_shape": sf.n_shape,
            "max_elements": fixture_elements,
            "max_nodes": fixture_nodes,
            "max_node_degree": self.max_node_degree,
            "connectivity": _uint_initializer(connectivity),
            "node_degree": _uint_initializer(node_degree),
            "node_to_element_map": _uint_initializer(node_to_element_map),
            "node_to_local_idx": _uint_initializer(node_to_local_idx),
            "u": _float_initializer(u),
            "kappa": _c_float_literal(self.ir.parameter_default),
        }

    def write_inspection_artifacts(self, output_dir, *, include_metal_smoke_harness=True):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = output_dir / self.ir.function_prefix
        metal_path = prefix.with_suffix(".ebe.metal")
        harness_path = prefix.with_suffix(".ebe_metal_smoke.mm")
        metal_path.write_text(self.render_metal_source())
        files = [metal_path]
        if include_metal_smoke_harness:
            harness_path.write_text(self.render_metal_smoke_test_harness())
            files.append(harness_path)
        return CodeInspectionArtifacts(
            output_dir=str(output_dir),
            files=tuple(str(path) for path in files),
        )

    def run_metal_smoke_test(self, output_dir, *, xcrun=None):
        xcrun = xcrun or shutil.which("xcrun")
        if xcrun is None:
            raise FileNotFoundError("xcrun is not available")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        harness = output_dir / ("%s_ebe_metal_smoke.mm" % self.ir.function_prefix)
        executable = output_dir / ("%s_ebe_metal_smoke" % self.ir.function_prefix)
        harness.write_text(self.render_metal_smoke_test_harness())
        compile_result = subprocess.run(
            [
                xcrun,
                "clang++",
                str(harness),
                "-fobjc-arc",
                "-framework",
                "Foundation",
                "-framework",
                "Metal",
                "-o",
                str(executable),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if compile_result.returncode != 0:
            return MetalSmokeTestResult(
                harness,
                executable,
                compile_result.returncode,
                compile_result.stdout,
                compile_result.stderr,
            )
        run_result = subprocess.run(
            [str(executable)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return MetalSmokeTestResult(
            harness,
            executable,
            compile_result.returncode,
            compile_result.stdout,
            compile_result.stderr,
            run_result.returncode,
            run_result.stdout,
            run_result.stderr,
        )

    def _map_kernel_name(self):
        return "%s_ebe_map_metal" % self.ir.function_prefix

    def _reduce_kernel_name(self):
        return "%s_ebe_reduce_metal" % self.ir.function_prefix

    def _render_map_kernel_from_local(self, local_lines):
        sf = self.ir.sum_factor
        lines = [
            "kernel void %s(" % self._map_kernel_name(),
            "        device const uint *connectivity [[buffer(0)]],",
            "        device const float *u [[buffer(1)]],",
            "        device float *element_out [[buffer(2)]],",
            "        constant float &kappa [[buffer(3)]],",
            "        uint2 tid [[thread_position_in_grid]]) {",
            "    const uint row = tid.x;",
            "    const uint elem = tid.y;",
        ]
        for line in local_lines[5:]:
            if "const float coeff = u[trial];" in line:
                lines.append("                    const uint node = connectivity[elem * %d + trial];" % sf.n_shape)
                lines.append("                    const float coeff = u[node];")
                continue
            if line == "    out[row] = value;":
                lines.append("    element_out[elem * %d + row] = value;" % sf.n_shape)
                continue
            lines.append(line)
        return lines

    def _render_reduce_kernel(self):
        sf = self.ir.sum_factor
        return [
            "kernel void %s(" % self._reduce_kernel_name(),
            "        device const float *element_out [[buffer(0)]],",
            "        device const uint *node_degree [[buffer(1)]],",
            "        device const uint *node_to_element_map [[buffer(2)]],",
            "        device const uint *node_to_local_idx [[buffer(3)]],",
            "        device float *out [[buffer(4)]],",
            "        uint node [[thread_position_in_grid]]) {",
            "    const uint degree = node_degree[node];",
            "    float acc = 0.0f;",
            "    for (uint i = 0; i < degree; ++i) {",
            "        const uint map_index = node * %d + i;" % self.max_node_degree,
            "        const uint elem = node_to_element_map[map_index];",
            "        const uint local = node_to_local_idx[map_index];",
            "        acc += element_out[elem * %d + local];" % sf.n_shape,
            "    }",
            "    out[node] = acc;",
            "}",
            "",
        ]
def tensor_product_sum_factor_ir_from_material(
    material,
    *,
    element,
    vector_size=8,
    quadrature_order=None,
):
    from sfem import gen

    user_input = gen.UserInputStage.create(
        material,
        (element,),
        int(vector_size),
        quadrature_order,
    )
    return tensor_product_sum_factor_ir_from_user_input_stage(user_input)


def tensor_product_sum_factor_ir_from_user_input_stage(user_input):
    material = user_input.material
    contexts = tuple(user_input.element_contexts)
    if len(contexts) != 1:
        raise ValueError("tensor-product SFEM IR lowering requires a single element context")
    context = contexts[0]
    basis = context.basis_plan("cell")
    if basis.family is not BasisFamily.TENSOR_PRODUCT:
        raise ValueError("element '%s' does not use tensor-product basis evaluation" % context.element_type)
    field_plan = basis.field_evaluation_sum_factorization
    test_plan = basis.test_contraction_sum_factorization
    if TensorProductOperation.FIELD_GRADIENT not in field_plan.operations:
        raise ValueError("tensor-product field gradient plan is required")
    if TensorProductOperation.TEST_GRADIENT_CONTRACTION not in test_plan.operations:
        raise ValueError("tensor-product test gradient contraction plan is required")
    rule = context.specialization.quadrature_rule
    _validate_laplace_form_source(material, basis.dim)
    return TensorProductSumFactorIR(
        material_name=material.name,
        element_type=context.element_type,
        element_label=context.label,
        dim=basis.dim,
        n_shape=basis.n_shape,
        n_qp=basis.n_qp,
        n_shape_1d=basis.n_shape_1d,
        n_qp_1d=basis.n_qp_1d,
        quadrature_order=rule.order,
        vector_size=context.specialization.vector_size,
        shape_values_1d=rule.tensor_product_shape_values_1d,
        shape_gradients_1d=rule.tensor_product_shape_gradients_1d,
        weights_1d=rule.tensor_product_weights_1d,
        field_gradient_stages=_field_gradient_stages(basis.dim, basis.n_shape_1d, basis.n_qp_1d),
        test_gradient_stages=_test_gradient_stages(basis.dim, basis.n_shape_1d, basis.n_qp_1d),
    )


def _validate_laplace_form_source(material, dim):
    from codegen.framework.symbolic.forms import FormKind, StandardFormName

    if getattr(material, "name", None) != "laplace":
        raise ValueError("tensor-product sum-factorization IR currently supports the laplace form only")
    systems = getattr(material, "systems", None)
    if systems is None:
        raise ValueError("laplace tensor-product IR requires material equation systems")
    system = systems.for_dim(int(dim))
    collections = tuple(system.form_collections())
    residual_collections = tuple(
        collection
        for collection in collections
        if collection.kind is FormKind.RESIDUAL
        and len(collection.fields) == 1
        and collection.fields[0].name == "u"
    )
    if len(residual_collections) != 1:
        raise ValueError("laplace tensor-product IR requires one residual form for field 'u'")
    residual = residual_collections[0].standard_form(StandardFormName.ONE)
    expression = str(residual.expression)
    required_tokens = ["kappa"]
    for axis in range(int(dim)):
        required_tokens.append("u_grad_%d" % axis)
        required_tokens.append("u_test_grad_%d" % axis)
    missing = tuple(token for token in required_tokens if token not in expression)
    if missing:
        raise ValueError(
            "laplace residual form is missing expected gradient tokens: %s"
            % ", ".join(missing)
        )


def tensor_product_laplace_form_ir_from_material(
    material,
    *,
    element,
    vector_size=8,
    quadrature_order=None,
):
    return TensorProductLaplaceFormIR(
        tensor_product_sum_factor_ir_from_material(
            material,
            element=element,
            vector_size=vector_size,
            quadrature_order=quadrature_order,
        ),
        "kappa",
        dict(getattr(material, "parameter_defaults", ())).get("kappa", 1.0),
    )


def tensor_product_laplace_form_ir_from_user_input_stage(user_input):
    material = user_input.material
    return TensorProductLaplaceFormIR(
        tensor_product_sum_factor_ir_from_user_input_stage(user_input),
        "kappa",
        dict(getattr(material, "parameter_defaults", ())).get("kappa", 1.0),
    )


def _field_gradient_stages(dim, n_shape_1d, n_qp_1d):
    stages = []
    for derivative in range(dim):
        for axis in range(dim):
            before = n_qp_1d ** axis
            after = n_shape_1d ** (dim - axis - 1)
            basis = "grad_1d" if axis == derivative else "shape_1d"
            stages.append(
                TensorProductContractionStage(
                    "field_gradient_d%d_axis%d" % (derivative, axis),
                    "field_gradient",
                    derivative,
                    axis,
                    basis,
                    False,
                    n_qp_1d,
                    n_shape_1d,
                    before * after,
                )
            )
    return tuple(stages)


def _test_gradient_stages(dim, n_shape_1d, n_qp_1d):
    stages = []
    for derivative in range(dim):
        for axis in reversed(range(dim)):
            before = n_qp_1d ** axis
            after = n_shape_1d ** (dim - axis - 1)
            basis = "grad_1d_t" if axis == derivative else "shape_1d_t"
            stages.append(
                TensorProductContractionStage(
                    "test_gradient_d%d_axis%d" % (derivative, axis),
                    "test_gradient_contraction",
                    derivative,
                    axis,
                    basis,
                    True,
                    n_shape_1d,
                    n_qp_1d,
                    before * after,
                )
            )
    return tuple(stages)


def _deterministic_values(count, scale, offset):
    return tuple(float(offset + scale * (i + 1)) for i in range(int(count)))


def _tensor_product_multi_index(index, base, dim):
    values = []
    index = int(index)
    base = int(base)
    for _ in range(int(dim)):
        values.append(index % base)
        index //= base
    return tuple(values)


def _tensor_product_multi_index_row_major(index, base, dim):
    values = []
    index = int(index)
    base = int(base)
    dim = int(dim)
    for axis in range(dim):
        divisor = base ** (dim - axis - 1)
        values.append((index // divisor) % base)
    return tuple(values)


def _stage_view_axes(dim, axis):
    axis = int(axis)
    return (axis,) + tuple(range(axis)) + tuple(range(axis + 1, int(dim)))


def _stage_input_extents(sf, stage):
    if stage.operation == "field_gradient":
        return tuple(sf.n_qp_1d if axis < stage.axis else sf.n_shape_1d for axis in range(sf.dim))
    if stage.operation == "test_gradient_contraction":
        return tuple(sf.n_qp_1d if axis <= stage.axis else sf.n_shape_1d for axis in range(sf.dim))
    raise ValueError("unsupported sum-factorization stage operation")


def _stage_output_extents(sf, stage):
    if stage.operation == "field_gradient":
        return tuple(sf.n_qp_1d if axis <= stage.axis else sf.n_shape_1d for axis in range(sf.dim))
    if stage.operation == "test_gradient_contraction":
        return tuple(sf.n_qp_1d if axis < stage.axis else sf.n_shape_1d for axis in range(sf.dim))
    raise ValueError("unsupported sum-factorization stage operation")


def _stage_view_affine_map(source_axes, target_axes, extents):
    source_row = _axis_coordinate_affine_expr(source_axes[0], target_axes, extents)
    source_col = _stage_view_col_affine_expr(source_axes[1:], target_axes, extents)
    return "affine_map<(d0, d1) -> (%s, %s)>" % (source_row, source_col)


def _stage_view_col_affine_expr(col_axes, target_axes, extents):
    terms = []
    for index, axis in enumerate(col_axes):
        stride = _product(extents[col_axis] for col_axis in col_axes[index + 1 :])
        coordinate = _axis_coordinate_affine_expr(axis, target_axes, extents)
        if coordinate == "0":
            continue
        if stride == 1:
            terms.append(coordinate)
        else:
            terms.append("%s * %d" % (_affine_expr_term(coordinate), stride))
    return " + ".join(terms) if terms else "0"


def _axis_coordinate_affine_expr(axis, target_axes, extents):
    axis = int(axis)
    if axis == target_axes[0]:
        return "d0"
    col_axes = target_axes[1:]
    col_index = col_axes.index(axis)
    extent = int(extents[axis])
    if extent == 1:
        return "0"
    stride = _product(extents[col_axis] for col_axis in col_axes[col_index + 1 :])
    if stride == 1:
        return "d1 mod %d" % extent
    return "(d1 floordiv %d) mod %d" % (stride, extent)


def _affine_expr_term(expr):
    if " " in expr or "+" in expr:
        return "(%s)" % expr
    return expr


def _spirv_entry_point_abi(x, y, z):
    return "spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [%d, %d, %d]>" % (
        int(x),
        int(y),
        int(z),
    )


def _spirv_interface_var_abi(binding, descriptor_set=0):
    return "spirv.interface_var_abi = #spirv.interface_var_abi<(%d, %d)>" % (
        int(descriptor_set),
        int(binding),
    )


def _gpu_kernel_argument(name, memref_type, binding, descriptor_set=0):
    return "%%%s: %s {%s}" % (
        name,
        memref_type,
        _spirv_interface_var_abi(binding, descriptor_set),
    )


class _SSANamer:
    def __init__(self):
        self._next_id = 0

    def value(self, hint):
        name = "%%%s%d" % (hint, self._next_id)
        self._next_id += 1
        return name


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def _linear_to_multi_index(index, extents):
    values = []
    remaining = int(index)
    suffix = _product(extents)
    for extent in extents:
        suffix //= int(extent)
        values.append((remaining // suffix) % int(extent))
    return tuple(values)


def _view_offset(multi_index, axes, extents):
    offset = 0
    for axis in axes:
        offset = offset * int(extents[axis]) + int(multi_index[axis])
    return offset


def _reorder_tensor_values(values, source_axes, target_axes, extents):
    values = tuple(float(value) for value in values)
    if source_axes == target_axes:
        return values
    result = [0.0 for _ in range(len(values))]
    for linear in range(len(values)):
        multi = _linear_to_multi_index(linear, extents)
        source = _view_offset(multi, source_axes, extents)
        target = _view_offset(multi, target_axes, extents)
        result[target] = values[source]
    return tuple(result)


def _align_up(value, alignment):
    value = int(value)
    alignment = int(alignment)
    return ((value + alignment - 1) // alignment) * alignment


def _tensor_product_weight(sf, q_idx):
    value = 1.0
    for axis in range(sf.dim):
        value *= sf.weights_1d[q_idx[axis]]
    return value


def _tensor_product_basis(sf, q_idx, shape_idx, derivative):
    value = 1.0
    s = sf.n_shape_1d
    for axis in range(sf.dim):
        offset = q_idx[axis] * s + shape_idx[axis]
        if axis == derivative:
            value *= sf.shape_gradients_1d[offset]
        else:
            value *= sf.shape_values_1d[offset]
    return value


def _rank2_width(values):
    if not values:
        raise ValueError("rank-2 value container must be non-empty")
    first = values[0]
    if isinstance(first, (list, tuple)):
        return len(first)
    raise ValueError("flat rank-2 value containers require an explicit width")


def _rank2_value(values, row, col, width):
    row_values = values[row]
    if isinstance(row_values, (list, tuple)):
        return row_values[col]
    return values[row * width + col]


def _float_initializer(values):
    return ", ".join("%sf" % _c_float_literal(value) for value in values)


def _uint_initializer(values):
    return ", ".join("%du" % int(value) for value in values)


def _c_float_literal(value):
    literal = "%.9g" % float(value)
    if "." not in literal and "e" not in literal and "E" not in literal:
        literal += ".0"
    return literal


def _objc_string_literal(text):
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )
    return '@"%s"' % escaped


_METAL_SMOKE_TEST_TEMPLATE = r'''#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static int run_stage(id<MTLDevice> device,
                     id<MTLLibrary> library,
                     id<MTLCommandQueue> queue,
                     NSString *kernel_name,
                     const float *basis,
                     size_t basis_bytes,
                     const float *operand,
                     size_t operand_bytes,
                     unsigned result_size,
                     unsigned lhs_cols,
                     unsigned rhs_cols,
                     unsigned result_rows,
                     unsigned result_cols) {
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:kernel_name];
    if (!function) {
        std::fprintf(stderr, "Metal function lookup failed: %%s\n", [kernel_name UTF8String]);
        return 79;
    }

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        std::fprintf(stderr, "Metal pipeline creation failed for %%s: %%s\n",
                     [kernel_name UTF8String], [[error localizedDescription] UTF8String]);
        return 80;
    }

    float *expected = static_cast<float *>(std::calloc(result_size, sizeof(float)));
    if (!expected) {
        return 83;
    }
    for (unsigned row = 0; row < result_rows; ++row) {
        for (unsigned col = 0; col < result_cols; ++col) {
            float acc = 0.0f;
            for (unsigned k = 0; k < lhs_cols; ++k) {
                acc += basis[row * lhs_cols + k] * operand[k * rhs_cols + col];
            }
            expected[row * result_cols + col] = acc;
        }
    }

    id<MTLBuffer> basis_buffer = [device newBufferWithBytes:basis length:basis_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> operand_buffer = [device newBufferWithBytes:operand length:operand_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> result_buffer = [device newBufferWithLength:result_size * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:basis_buffer offset:0 atIndex:0];
    [encoder setBuffer:operand_buffer offset:0 atIndex:1];
    [encoder setBuffer:result_buffer offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(result_cols, result_rows, 1)
        threadsPerThreadgroup:MTLSizeMake(result_cols, result_rows, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] != MTLCommandBufferStatusCompleted) {
        std::fprintf(stderr, "Metal command failed for %%s\n", [kernel_name UTF8String]);
        std::free(expected);
        return 81;
    }

    const float *result = static_cast<const float *>([result_buffer contents]);
    for (unsigned i = 0; i < result_size; ++i) {
        if (std::fabs(result[i] - expected[i]) > 1.0e-5f) {
            std::fprintf(stderr, "Mismatch in %%s at %%u: got %%g expected %%g\n",
                         [kernel_name UTF8String], i, result[i], expected[i]);
            std::free(expected);
            return 82;
        }
    }
    std::free(expected);
    return 0;
}

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return 77;
        }

        NSString *source = %(source)s;
        NSError *error = nil;
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            std::fprintf(stderr, "Metal library compilation failed: %%s\n", [[error localizedDescription] UTF8String]);
            return 78;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        int status = 0;
%(stage_calls)s
        std::printf("sum_factor_stage_count=%%d\n", %(stage_count)d);
        return 0;
    }
}
'''


_LAPLACE_METAL_SMOKE_TEST_TEMPLATE = r'''#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>

%(host_reference)s

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return 77;
        }

        NSString *source = %(source)s;
        NSError *error = nil;
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            std::fprintf(stderr, "Metal library compilation failed: %%s\n", [[error localizedDescription] UTF8String]);
            return 78;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"%(kernel_name)s"];
        if (!function) {
            std::fprintf(stderr, "Metal function lookup failed\n");
            return 79;
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            std::fprintf(stderr, "Metal pipeline creation failed: %%s\n", [[error localizedDescription] UTF8String]);
            return 80;
        }

        const float kappa = %(kappa)sf;
        const float u[%(n_shape)d] = {%(u)s};
        float expected[%(n_shape)d];
        reference_apply(u, expected, kappa);

        id<MTLBuffer> u_buffer = [device newBufferWithBytes:u length:sizeof(u) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buffer = [device newBufferWithLength:sizeof(expected) options:MTLResourceStorageModeShared];
        id<MTLBuffer> kappa_buffer = [device newBufferWithBytes:&kappa length:sizeof(kappa) options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:u_buffer offset:0 atIndex:0];
        [encoder setBuffer:out_buffer offset:0 atIndex:1];
        [encoder setBuffer:kappa_buffer offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(%(n_shape)d, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(%(n_shape)d, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if ([command_buffer status] != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "Metal command failed\n");
            return 81;
        }

        const float *result = static_cast<const float *>([out_buffer contents]);
        for (unsigned i = 0; i < %(n_shape)d; ++i) {
            if (std::fabs(result[i] - expected[i]) > 1.0e-4f) {
                std::fprintf(stderr, "Mismatch at %%u: got %%g expected %%g\n", i, result[i], expected[i]);
                return 82;
            }
        }
        return 0;
    }
}
'''


_LAPLACE_EBE_METAL_SMOKE_TEST_TEMPLATE = r'''#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>

%(host_reference)s

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return 77;
        }

        NSString *source = %(source)s;
        NSError *error = nil;
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            std::fprintf(stderr, "Metal library compilation failed: %%s\n", [[error localizedDescription] UTF8String]);
            return 78;
        }

        id<MTLFunction> map_function = [library newFunctionWithName:@"%(map_kernel_name)s"];
        id<MTLFunction> reduce_function = [library newFunctionWithName:@"%(reduce_kernel_name)s"];
        if (!map_function || !reduce_function) {
            std::fprintf(stderr, "Metal EBE function lookup failed\n");
            return 79;
        }

        id<MTLComputePipelineState> map_pipeline = [device newComputePipelineStateWithFunction:map_function error:&error];
        if (!map_pipeline) {
            std::fprintf(stderr, "Metal map pipeline creation failed: %%s\n", [[error localizedDescription] UTF8String]);
            return 80;
        }
        id<MTLComputePipelineState> reduce_pipeline = [device newComputePipelineStateWithFunction:reduce_function error:&error];
        if (!reduce_pipeline) {
            std::fprintf(stderr, "Metal reduce pipeline creation failed: %%s\n", [[error localizedDescription] UTF8String]);
            return 81;
        }

        const float kappa = %(kappa)sf;
        const unsigned connectivity[%(max_elements)d * %(n_shape)d] = {%(connectivity)s};
        const unsigned node_degree[%(max_nodes)d] = {%(node_degree)s};
        const unsigned node_to_element_map[%(max_nodes)d * %(max_node_degree)d] = {%(node_to_element_map)s};
        const unsigned node_to_local_idx[%(max_nodes)d * %(max_node_degree)d] = {%(node_to_local_idx)s};
        const float u[%(max_nodes)d] = {%(u)s};
        float expected_element[%(max_elements)d * %(n_shape)d] = {0.0f};
        float expected[%(max_nodes)d] = {0.0f};

        for (unsigned elem = 0; elem < %(max_elements)d; ++elem) {
            float local_u[%(n_shape)d];
            float local_out[%(n_shape)d];
            for (unsigned local = 0; local < %(n_shape)d; ++local) {
                local_u[local] = u[connectivity[elem * %(n_shape)d + local]];
            }
            reference_apply(local_u, local_out, kappa);
            for (unsigned local = 0; local < %(n_shape)d; ++local) {
                expected_element[elem * %(n_shape)d + local] = local_out[local];
            }
        }
        for (unsigned node = 0; node < %(max_nodes)d; ++node) {
            float acc = 0.0f;
            for (unsigned i = 0; i < node_degree[node]; ++i) {
                const unsigned map_index = node * %(max_node_degree)d + i;
                const unsigned elem = node_to_element_map[map_index];
                const unsigned local = node_to_local_idx[map_index];
                acc += expected_element[elem * %(n_shape)d + local];
            }
            expected[node] = acc;
        }

        id<MTLBuffer> connectivity_buffer = [device newBufferWithBytes:connectivity length:sizeof(connectivity) options:MTLResourceStorageModeShared];
        id<MTLBuffer> u_buffer = [device newBufferWithBytes:u length:sizeof(u) options:MTLResourceStorageModeShared];
        id<MTLBuffer> element_out_buffer = [device newBufferWithLength:sizeof(expected_element) options:MTLResourceStorageModeShared];
        id<MTLBuffer> kappa_buffer = [device newBufferWithBytes:&kappa length:sizeof(kappa) options:MTLResourceStorageModeShared];
        id<MTLBuffer> node_degree_buffer = [device newBufferWithBytes:node_degree length:sizeof(node_degree) options:MTLResourceStorageModeShared];
        id<MTLBuffer> node_to_element_buffer = [device newBufferWithBytes:node_to_element_map length:sizeof(node_to_element_map) options:MTLResourceStorageModeShared];
        id<MTLBuffer> node_to_local_buffer = [device newBufferWithBytes:node_to_local_idx length:sizeof(node_to_local_idx) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buffer = [device newBufferWithLength:sizeof(expected) options:MTLResourceStorageModeShared];

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];

        id<MTLComputeCommandEncoder> map_encoder = [command_buffer computeCommandEncoder];
        [map_encoder setComputePipelineState:map_pipeline];
        [map_encoder setBuffer:connectivity_buffer offset:0 atIndex:0];
        [map_encoder setBuffer:u_buffer offset:0 atIndex:1];
        [map_encoder setBuffer:element_out_buffer offset:0 atIndex:2];
        [map_encoder setBuffer:kappa_buffer offset:0 atIndex:3];
        [map_encoder dispatchThreads:MTLSizeMake(%(n_shape)d, %(max_elements)d, 1)
            threadsPerThreadgroup:MTLSizeMake(%(n_shape)d, 1, 1)];
        [map_encoder endEncoding];

        id<MTLComputeCommandEncoder> reduce_encoder = [command_buffer computeCommandEncoder];
        [reduce_encoder setComputePipelineState:reduce_pipeline];
        [reduce_encoder setBuffer:element_out_buffer offset:0 atIndex:0];
        [reduce_encoder setBuffer:node_degree_buffer offset:0 atIndex:1];
        [reduce_encoder setBuffer:node_to_element_buffer offset:0 atIndex:2];
        [reduce_encoder setBuffer:node_to_local_buffer offset:0 atIndex:3];
        [reduce_encoder setBuffer:out_buffer offset:0 atIndex:4];
        [reduce_encoder dispatchThreads:MTLSizeMake(%(max_nodes)d, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(%(max_nodes)d, 1, 1)];
        [reduce_encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if ([command_buffer status] != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "Metal EBE command failed\n");
            return 82;
        }

        const float *result = static_cast<const float *>([out_buffer contents]);
        for (unsigned i = 0; i < %(max_nodes)d; ++i) {
            if (std::fabs(result[i] - expected[i]) > 1.0e-4f) {
                std::fprintf(stderr, "Mismatch at %%u: got %%g expected %%g\n", i, result[i], expected[i]);
                return 83;
            }
        }
        return 0;
    }
}
'''
