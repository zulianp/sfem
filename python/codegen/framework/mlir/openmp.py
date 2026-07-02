import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import sympy as sp
from dataclasses import dataclass

from .common import CodeInspectionArtifacts, MLIR_OPTIMIZATION_PLANS, mlir_optimization_strategy
from .model import linear_elasticity_mlir_model
from .runtime import build_inverted_topology
from .tools import (
    _find_mlir_opt,
    _find_mlir_runner,
    _find_runner_library,
    _parse_mlir_runner_i32_result,
    _translate_emitc_file_to_cpp,
    _translate_emitc_to_cpp,
    _translate_mlir_to_llvm_ir,
)


@dataclass(frozen=True)
class MLIRRunnerResult:
    success: bool
    stdout: str
    stderr: str
    entry_result: int
    openmp_module: str
    llvm_module: str


class MatrixFreeOpenMPMLIRLowering:
    SCF_TO_OPENMP_PASS = "--convert-scf-to-openmp"
    EMITC_PIPELINE = (
        "--convert-scf-to-emitc",
        "--convert-arith-to-emitc",
        "--convert-memref-to-emitc",
        "--convert-func-to-emitc",
        "--reconcile-unrealized-casts",
    )
    LLVM_PIPELINE = (
        "--convert-scf-to-openmp",
        "--convert-openmp-to-llvm",
        "--convert-scf-to-cf",
        "--convert-cf-to-llvm",
        "--convert-arith-to-llvm",
        "--finalize-memref-to-llvm",
        "--convert-func-to-llvm",
        "--reconcile-unrealized-casts",
    )

    def __init__(
        self,
        model=None,
        *,
        max_elements=1024,
        max_nodes=4096,
        max_node_degree=32,
        optimization_strategy=None,
    ):
        self.model = linear_elasticity_mlir_model() if model is None else model
        self.max_elements = int(max_elements)
        self.max_nodes = int(max_nodes)
        self.max_node_degree = int(max_node_degree)
        self.optimization_strategy = mlir_optimization_strategy(optimization_strategy)
        if self.max_elements <= 0 or self.max_nodes <= 0 or self.max_node_degree <= 0:
            raise ValueError("OpenMP MLIR bounds must be positive")

    @property
    def optimization_plan(self):
        return MLIR_OPTIMIZATION_PLANS[self.optimization_strategy]

    @property
    def function_name(self):
        return f"{self.model.mesh_kernel_name}_mlir_apply_openmp"

    @property
    def scratch_size(self):
        return self.max_elements * self.model.scratch_components

    @property
    def node_field_size(self):
        return self.max_nodes * self.model.n_field_components

    @property
    def connectivity_size(self):
        return self.max_elements * self.model.n_shape

    @property
    def geometry_adjugate_size(self):
        return self.max_elements * self.model.dim * self.model.dim

    @property
    def geometry_determinant_size(self):
        return self.max_elements

    @property
    def node_degree_size(self):
        return self.max_nodes

    @property
    def node_element_map_size(self):
        return self.max_nodes * self.max_node_degree

    def build_scf_module(self):
        return _SCFOpenMPModuleBuilder(self).build()

    def render_scf_module(self):
        return str(self.build_scf_module())

    def build_kernel_scf_module(self):
        return _SCFOpenMPModuleBuilder(self, include_main=False).build()

    def render_kernel_scf_module(self):
        return str(self.build_kernel_scf_module())

    def build_c_scf_module(self):
        return _SCFOpenMPModuleBuilder(
            self,
            include_main=False,
            parallel_loops=False,
        ).build()

    def render_c_scf_module(self):
        return str(self.build_c_scf_module())

    def optimize_scf_module(self, mlir_opt=None, module_text=None, optimization_strategy=None):
        return self._run_mlir_opt(
            (),
            mlir_opt=mlir_opt,
            module_text=module_text,
            optimization_strategy=optimization_strategy,
        )

    def optimize_kernel_scf_module(self, mlir_opt=None, optimization_strategy=None):
        return self.optimize_scf_module(
            mlir_opt=mlir_opt,
            module_text=self.render_kernel_scf_module(),
            optimization_strategy=optimization_strategy,
        )

    def lower_to_openmp_module(self, mlir_opt=None, optimization_strategy=None):
        return self._run_mlir_opt(
            (self.SCF_TO_OPENMP_PASS,),
            mlir_opt=mlir_opt,
            optimization_strategy=optimization_strategy,
        )

    def lower_to_llvm_module(self, mlir_opt=None, optimization_strategy=None):
        return self._run_mlir_opt(
            self.LLVM_PIPELINE,
            mlir_opt=mlir_opt,
            optimization_strategy=optimization_strategy,
        )

    def lower_kernel_to_openmp_module(self, mlir_opt=None, optimization_strategy=None):
        return self._run_mlir_opt(
            (self.SCF_TO_OPENMP_PASS,),
            mlir_opt=mlir_opt,
            module_text=self.render_kernel_scf_module(),
            optimization_strategy=optimization_strategy,
        )

    def lower_kernel_to_llvm_module(self, mlir_opt=None, optimization_strategy=None):
        return self._run_mlir_opt(
            self.LLVM_PIPELINE,
            mlir_opt=mlir_opt,
            module_text=self.render_kernel_scf_module(),
            optimization_strategy=optimization_strategy,
        )

    def lower_to_emitc_module(self, mlir_opt=None):
        return str(_EmitCKernelModuleBuilder(self).build())

    def lower_to_c_source(self, mlir_opt=None, mlir_translate=None):
        emitc_module = self.lower_to_emitc_module(mlir_opt=mlir_opt)
        return _translate_emitc_to_cpp(emitc_module, mlir_translate=mlir_translate)

    def write_inspection_artifacts(
        self,
        output_dir,
        *,
        mlir_opt=None,
        mlir_translate=None,
        include_llvm_ir=True,
        include_c_source=True,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = output_dir / self.function_name

        strategy = self.optimization_strategy
        strategy_suffix = strategy.value

        scf_path = prefix.with_suffix(".scf.mlir")
        optimized_scf_path = prefix.with_suffix(f".{strategy_suffix}.optimized.scf.mlir")
        kernel_scf_path = prefix.with_suffix(".kernel.scf.mlir")
        optimized_kernel_scf_path = prefix.with_suffix(f".kernel.{strategy_suffix}.optimized.scf.mlir")
        c_scf_path = prefix.with_suffix(".c.scf.mlir")
        openmp_path = prefix.with_suffix(".openmp.mlir")
        llvm_mlir_path = prefix.with_suffix(".llvm.mlir")
        llvm_ir_path = prefix.with_suffix(".ll")
        emitc_path = prefix.with_suffix(".emitc.mlir")
        c_path = prefix.with_suffix(".c")

        scf_path.write_text(self.render_scf_module())
        optimized_scf_path.write_text(self.optimize_scf_module(mlir_opt=mlir_opt))
        kernel_scf_path.write_text(self.render_kernel_scf_module())
        optimized_kernel_scf_path.write_text(self.optimize_kernel_scf_module(mlir_opt=mlir_opt))
        c_scf_path.write_text(self.render_c_scf_module())
        openmp_path.write_text(self.lower_to_openmp_module(mlir_opt=mlir_opt))
        llvm_mlir_path.write_text(self.lower_to_llvm_module(mlir_opt=mlir_opt))

        files = [
            scf_path,
            optimized_scf_path,
            kernel_scf_path,
            optimized_kernel_scf_path,
            c_scf_path,
            openmp_path,
            llvm_mlir_path,
        ]
        if include_llvm_ir:
            _translate_mlir_to_llvm_ir(
                llvm_mlir_path,
                llvm_ir_path,
                mlir_translate=mlir_translate,
            )
            files.append(llvm_ir_path)
        if include_c_source:
            emitc_path.write_text(self.lower_to_emitc_module(mlir_opt=mlir_opt))
            _translate_emitc_file_to_cpp(
                emitc_path,
                c_path,
                mlir_translate=mlir_translate,
            )
            files.extend((emitc_path, c_path))

        return CodeInspectionArtifacts(
            output_dir=str(output_dir),
            files=tuple(str(path) for path in files),
        )

    def run_with_mlir_runner(self, mlir_opt=None, mlir_runner=None, extra_env=None):
        llvm_module = self.lower_to_llvm_module(mlir_opt=mlir_opt)
        openmp_module = self.lower_to_openmp_module(mlir_opt=mlir_opt)
        mlir_runner = mlir_runner or _find_mlir_runner()
        runner_lib = _find_runner_library("libmlir_runner_utils.dylib")
        omp_lib = _find_runner_library("libomp.dylib")
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "2")
        if extra_env:
            env.update(extra_env)
        with tempfile.TemporaryDirectory(prefix="sfem_mlir_openmp_") as tmp:
            module_path = Path(tmp) / "linear_elasticity_openmp_llvm.mlir"
            module_path.write_text(llvm_module)
            result = subprocess.run(
                [
                    mlir_runner,
                    str(module_path),
                    "-e",
                    "main",
                    "--entry-point-result=i32",
                    f"-shared-libs={runner_lib},{omp_lib}",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
        entry_result = _parse_mlir_runner_i32_result(result.stdout)
        return MLIRRunnerResult(
            success=entry_result == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            entry_result=entry_result,
            openmp_module=openmp_module,
            llvm_module=llvm_module,
        )

    def _run_mlir_opt(self, passes, mlir_opt=None, module_text=None, optimization_strategy=None):
        mlir_opt = mlir_opt or _find_mlir_opt()
        strategy = mlir_optimization_strategy(
            self.optimization_strategy if optimization_strategy is None else optimization_strategy
        )
        optimization_passes = MLIR_OPTIMIZATION_PLANS[strategy].pre_lowering_passes
        result = subprocess.run(
            [mlir_opt, "-", *optimization_passes, *passes],
            input=self.render_scf_module() if module_text is None else module_text,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout


class _SCFOpenMPModuleBuilder:
    def __init__(self, lowering, include_main=True, parallel_loops=True):
        self.lowering = lowering
        self.include_main = bool(include_main)
        self.parallel_loops = bool(parallel_loops)
        self.ir = None
        self.func = None
        self.scf = None
        self.arith = None
        self.memref = None
        self.arith_enum = None

    def build(self):
        self._load_bindings()
        with self.ir.Context() as context, self.ir.Location.unknown():
            self.context = context
            module = self.ir.Module.create()
            with self.ir.InsertionPoint(module.body):
                self._build_kernel_function()
                if self.include_main:
                    self._build_main_function()
            return module

    def _load_bindings(self):
        from mlir import ir
        import mlir.dialects._func_ops_gen as func
        import mlir.dialects._scf_ops_gen as scf
        import mlir.dialects._arith_ops_gen as arith
        import mlir.dialects._memref_ops_gen as memref
        import mlir.dialects._arith_enum_gen as arith_enum

        self.ir = ir
        self.func = func
        self.scf = scf
        self.arith = arith
        self.memref = memref
        self.arith_enum = arith_enum

    def _build_kernel_function(self):
        self._validate_affine_tet4_apply()
        function = self.func.FuncOp(
            self.lowering.function_name,
            self.ir.TypeAttr.get(
                self.ir.FunctionType.get(
                    [
                        self._connectivity_type(),
                        self._field_type(),
                        self._geometry_adjugate_type(),
                        self._geometry_determinant_type(),
                        self.ir.F32Type.get(),
                        self.ir.F32Type.get(),
                        self._scratch_type(),
                        self._node_degree_type(),
                        self._node_element_map_type(),
                        self._node_element_map_type(),
                        self._field_type(),
                    ],
                    [],
                )
            ),
        )
        body = self.ir.Block.create_at_start(
            function.regions[0],
            [
                self._connectivity_type(),
                self._field_type(),
                self._geometry_adjugate_type(),
                self._geometry_determinant_type(),
                self.ir.F32Type.get(),
                self.ir.F32Type.get(),
                self._scratch_type(),
                self._node_degree_type(),
                self._node_element_map_type(),
                self._node_element_map_type(),
                self._field_type(),
            ],
        )
        with self.ir.InsertionPoint(body):
            (
                connectivity,
                direction,
                adjugate,
                determinant,
                lmbda,
                mu,
                scratch,
                node_degree,
                node_to_element_map,
                node_to_local_idx,
                output,
            ) = body.arguments
            zero = self._index_constant(0)
            one = self._index_constant(1)
            element_count = self._index_constant(self.lowering.max_elements)
            node_count = self._index_constant(self.lowering.max_nodes)
            self._parallel_element_apply(
                zero,
                element_count,
                one,
                connectivity,
                direction,
                adjugate,
                determinant,
                lmbda,
                mu,
                scratch,
            )
            self._parallel_inverse_topology_reduce(
                zero,
                node_count,
                one,
                scratch,
                node_degree,
                node_to_element_map,
                node_to_local_idx,
                output,
            )
            self.func.ReturnOp([])

    def _build_main_function(self):
        function = self.func.FuncOp(
            "main",
            self.ir.TypeAttr.get(
                self.ir.FunctionType.get([], [self.ir.IntegerType.get_signless(32)])
            ),
        )
        body = self.ir.Block.create_at_start(function.regions[0])
        with self.ir.InsertionPoint(body):
            zero = self._index_constant(0)
            field_count = self._index_constant(self.lowering.node_field_size)
            direction = self.memref.AllocOp(self._field_type(), [], []).result
            scratch = self.memref.AllocOp(self._scratch_type(), [], []).result
            output = self.memref.AllocOp(self._field_type(), [], []).result
            connectivity = self.memref.AllocOp(self._connectivity_type(), [], []).result
            adjugate = self.memref.AllocOp(self._geometry_adjugate_type(), [], []).result
            determinant = self.memref.AllocOp(self._geometry_determinant_type(), [], []).result
            node_degree = self.memref.AllocOp(self._node_degree_type(), [], []).result
            node_to_element_map = self.memref.AllocOp(self._node_element_map_type(), [], []).result
            node_to_local_idx = self.memref.AllocOp(self._node_element_map_type(), [], []).result
            self._fill_f32(direction, field_count, np.float32(0.0))
            self._initialize_fixture_direction(direction)
            self._fill_f32(output, field_count, np.float32(0.0))
            self._fill_f32(scratch, self._index_constant(self.lowering.scratch_size), np.float32(0.0))
            self._initialize_fixture_connectivity(connectivity)
            self._initialize_affine_identity_geometry(adjugate, determinant)
            self._initialize_fixture_inverse_topology(
                node_degree,
                node_to_element_map,
                node_to_local_idx,
            )
            self.func.CallOp(
                [],
                self.lowering.function_name,
                [
                    connectivity,
                    direction,
                    adjugate,
                    determinant,
                    self._f32_constant(2.0),
                    self._f32_constant(3.0),
                    scratch,
                    node_degree,
                    node_to_element_map,
                    node_to_local_idx,
                    output,
                ],
            )
            error_count = self.memref.AllocOp(
                self.ir.MemRefType.get([1], self.ir.IntegerType.get_signless(32)),
                [],
                [],
            ).result
            self.memref.StoreOp(self._i32_constant(0), error_count, [zero])
            self._verify_against_expected(output, error_count)
            result = self.memref.LoadOp(error_count, [zero]).result
            self.memref.DeallocOp(error_count)
            self.memref.DeallocOp(node_to_local_idx)
            self.memref.DeallocOp(node_to_element_map)
            self.memref.DeallocOp(node_degree)
            self.memref.DeallocOp(determinant)
            self.memref.DeallocOp(adjugate)
            self.memref.DeallocOp(connectivity)
            self.memref.DeallocOp(output)
            self.memref.DeallocOp(scratch)
            self.memref.DeallocOp(direction)
            self.func.ReturnOp([result])

    def _parallel_element_apply(
        self,
        lower,
        upper,
        step,
        connectivity,
        direction,
        adjugate,
        determinant,
        lmbda,
        mu,
        scratch,
    ):
        if self.parallel_loops:
            loop = self.scf.ParallelOp([], [lower], [upper], [step], [])
            body = self.ir.Block.create_at_start(loop.regions[0], [self.ir.IndexType.get()])
            with self.ir.InsertionPoint(body):
                self._element_apply_body(
                    body.arguments[0],
                    connectivity,
                    direction,
                    adjugate,
                    determinant,
                    lmbda,
                    mu,
                    scratch,
                )
                self.scf.ReduceOp([], 0)
            return

        loop = self.scf.ForOp([], lower, upper, step, [])
        body = self.ir.Block.create_at_start(loop.regions[0], [self.ir.IndexType.get()])
        with self.ir.InsertionPoint(body):
            self._element_apply_body(
                body.arguments[0],
                connectivity,
                direction,
                adjugate,
                determinant,
                lmbda,
                mu,
                scratch,
            )
            self.scf.YieldOp([])

    def _element_apply_body(
        self,
        elem,
        connectivity,
        direction,
        adjugate,
        determinant,
        lmbda,
        mu,
        scratch,
    ):
        trial_grad_ref = self._trial_reference_gradient(elem, connectivity, direction)
        trial_grad = self._transform_gradient(elem, trial_grad_ref, adjugate, determinant)
        symbols = {
            sp.Symbol("lmbda"): lmbda,
            sp.Symbol("mu"): mu,
        }
        for idx, value in enumerate(trial_grad):
            symbols[sp.Symbol("trial_grad%d" % idx)] = value
        material = tuple(
            self._lower_sympy_f32(expr, symbols)
            for expr in self.lowering.model.apply_material_expressions
        )
        loperand = self._transformed_loperand(elem, material, adjugate)
        self._store_element_apply(elem, loperand, scratch)

    def _parallel_inverse_topology_reduce(
        self,
        lower,
        upper,
        step,
        scratch,
        node_degree,
        node_to_element_map,
        node_to_local_idx,
        output,
    ):
        if self.parallel_loops:
            loop = self.scf.ParallelOp([], [lower], [upper], [step], [])
            body = self.ir.Block.create_at_start(loop.regions[0], [self.ir.IndexType.get()])
            with self.ir.InsertionPoint(body):
                self._inverse_topology_reduce_body(
                    body.arguments[0],
                    scratch,
                    node_degree,
                    node_to_element_map,
                    node_to_local_idx,
                    output,
                )
                self.scf.ReduceOp([], 0)
            return

        loop = self.scf.ForOp([], lower, upper, step, [])
        body = self.ir.Block.create_at_start(loop.regions[0], [self.ir.IndexType.get()])
        with self.ir.InsertionPoint(body):
            self._inverse_topology_reduce_body(
                body.arguments[0],
                scratch,
                node_degree,
                node_to_element_map,
                node_to_local_idx,
                output,
            )
            self.scf.YieldOp([])

    def _inverse_topology_reduce_body(
        self,
        node,
        scratch,
        node_degree,
        node_to_element_map,
        node_to_local_idx,
        output,
    ):
        degree = self.memref.LoadOp(node_degree, [node]).result
        for component in range(self.lowering.model.n_field_components):
            reduction = self.scf.ForOp(
                [self.ir.F32Type.get()],
                self._index_constant(0),
                degree,
                self._index_constant(1),
                [self._f32_constant(0.0)],
            )
            reduce_body = self.ir.Block.create_at_start(
                reduction.regions[0],
                [self.ir.IndexType.get(), self.ir.F32Type.get()],
            )
            with self.ir.InsertionPoint(reduce_body):
                i = reduce_body.arguments[0]
                acc = reduce_body.arguments[1]
                map_index = self._inverse_topology_index(node, i)
                elem = self.memref.LoadOp(node_to_element_map, [map_index]).result
                local = self.memref.LoadOp(node_to_local_idx, [map_index]).result
                scratch_index = self._scratch_index(elem, local, component)
                value = self.memref.LoadOp(scratch, [scratch_index]).result
                self.scf.YieldOp([self.arith.AddFOp(acc, value).result])
            self.memref.StoreOp(
                reduction.results[0],
                output,
                [self._node_component_index(node, component)],
            )

    def _trial_reference_gradient(self, elem, connectivity, direction):
        model = self.lowering.model
        values = []
        for row in range(model.dim):
            for col in range(model.dim):
                acc = self._f32_constant(0.0)
                for shape in range(model.n_shape):
                    conn_index = self._add_index(
                        self._mul_index(elem, self._index_constant(model.n_shape)),
                        self._index_constant(shape),
                    )
                    node = self.memref.LoadOp(connectivity, [conn_index]).result
                    field_index = self._node_component_index(node, row)
                    field_value = self.memref.LoadOp(direction, [field_index]).result
                    reference_gradient = self._f32_constant(
                        model.reference_gradients[shape * model.dim + col]
                    )
                    acc = self.arith.AddFOp(
                        acc,
                        self.arith.MulFOp(field_value, reference_gradient).result,
                    ).result
                values.append(acc)
        return tuple(values)

    def _transform_gradient(self, elem, reference_gradient, adjugate, determinant):
        model = self.lowering.model
        det = self.memref.LoadOp(determinant, [elem]).result
        inv_det = self.arith.DivFOp(self._f32_constant(1.0), det).result
        values = []
        for row in range(model.dim):
            for col in range(model.dim):
                acc = self._f32_constant(0.0)
                for k in range(model.dim):
                    adj = self.memref.LoadOp(
                        adjugate,
                        [self._adjugate_index(elem, k, col)],
                    ).result
                    term = self.arith.MulFOp(reference_gradient[row * model.dim + k], adj).result
                    acc = self.arith.AddFOp(acc, term).result
                values.append(self.arith.MulFOp(acc, inv_det).result)
        return tuple(values)

    def _transformed_loperand(self, elem, material, adjugate):
        model = self.lowering.model
        qw = self._f32_constant(model.quadrature_weights[0])
        values = []
        for row in range(model.dim):
            for col in range(model.dim):
                acc = self._f32_constant(0.0)
                for k in range(model.dim):
                    adj = self.memref.LoadOp(
                        adjugate,
                        [self._adjugate_index(elem, col, k)],
                    ).result
                    term = self.arith.MulFOp(material[row * model.dim + k], adj).result
                    acc = self.arith.AddFOp(acc, term).result
                values.append(self.arith.MulFOp(qw, acc).result)
        return tuple(values)

    def _store_element_apply(self, elem, loperand, scratch):
        model = self.lowering.model
        element_offset = self._mul_index(
            elem,
            self._index_constant(model.scratch_components),
        )
        for shape in range(model.n_shape):
            for row in range(model.dim):
                acc = self._f32_constant(0.0)
                for col in range(model.dim):
                    reference_gradient = self._f32_constant(
                        model.reference_gradients[shape * model.dim + col]
                    )
                    term = self.arith.MulFOp(
                        loperand[row * model.dim + col],
                        reference_gradient,
                    ).result
                    acc = self.arith.AddFOp(acc, term).result
                local_index = self._index_constant(shape * model.dim + row)
                self.memref.StoreOp(
                    acc,
                    scratch,
                    [self._add_index(element_offset, local_index)],
                )

    def _initialize_fixture_connectivity(self, connectivity):
        for elem, nodes in enumerate(self._runner_connectivity()):
            for local, node in enumerate(nodes):
                self.memref.StoreOp(
                    self._index_constant(node),
                    connectivity,
                    [self._index_constant(elem * self.lowering.model.n_shape + local)],
                )

    def _initialize_affine_identity_geometry(self, adjugate, determinant):
        model = self.lowering.model
        for elem in range(self.lowering.max_elements):
            for row in range(model.dim):
                for col in range(model.dim):
                    self.memref.StoreOp(
                        self._f32_constant(1.0 if row == col else 0.0),
                        adjugate,
                        [self._index_constant(elem * model.dim * model.dim + row * model.dim + col)],
                    )
            self.memref.StoreOp(
                self._f32_constant(1.0),
                determinant,
                [self._index_constant(elem)],
            )

    def _initialize_fixture_direction(self, direction):
        for idx, value in enumerate(self._runner_direction().reshape(-1)):
            self.memref.StoreOp(
                self._f32_constant(value),
                direction,
                [self._index_constant(idx)],
            )

    def _initialize_fixture_inverse_topology(self, node_degree, node_to_element_map, node_to_local_idx):
        inverted = build_inverted_topology(self._runner_connectivity(), self.lowering.max_nodes)
        for node, degree in enumerate(inverted.node_degree):
            self.memref.StoreOp(
                self._index_constant(int(degree)),
                node_degree,
                [self._index_constant(node)],
            )
        for node in range(self.lowering.max_nodes):
            for i in range(self.lowering.max_node_degree):
                value_index = node * self.lowering.max_node_degree + i
                elem = 0
                local = 0
                if i < inverted.node_to_element_map.shape[1]:
                    elem = int(inverted.node_to_element_map[node, i])
                    local = int(inverted.node_to_local_idx[node, i])
                self.memref.StoreOp(
                    self._index_constant(elem),
                    node_to_element_map,
                    [self._index_constant(value_index)],
                )
                self.memref.StoreOp(
                    self._index_constant(local),
                    node_to_local_idx,
                    [self._index_constant(value_index)],
                )

    def _verify_against_expected(self, output, error_count):
        expected = self._expected_fixture_output(lmbda=2.0, mu=3.0)
        zero = self._index_constant(0)
        for idx, value in enumerate(expected):
            actual = self.memref.LoadOp(output, [self._index_constant(idx)]).result
            ok = self.arith.CmpFOp(
                self.arith_enum.CmpFPredicate.OEQ,
                actual,
                self._f32_constant(value),
            ).result
            old_count = self.memref.LoadOp(error_count, [zero]).result
            incremented = self.arith.AddIOp(old_count, self._i32_constant(1)).result
            next_count = self.arith.SelectOp(ok, old_count, incremented).result
            self.memref.StoreOp(next_count, error_count, [zero])

    def _expected_fixture_output(self, lmbda, mu):
        model = self.lowering.model
        output = np.zeros((self.lowering.max_nodes, model.n_field_components), dtype=np.float32)
        direction = self._runner_direction()
        connectivity = self._runner_connectivity()
        qw = model.quadrature_weights[0]
        for elem in range(connectivity.shape[0]):
            trial_gradient = []
            for row in range(model.dim):
                for col in range(model.dim):
                    value = np.float32(0.0)
                    for shape in range(model.n_shape):
                        node = int(connectivity[elem, shape])
                        value = np.float32(
                            value
                            + np.float32(direction[node, row])
                            * np.float32(model.reference_gradients[shape * model.dim + col])
                        )
                    trial_gradient.append(float(value))
            substitutions = {
                sp.Symbol("lmbda"): float(lmbda),
                sp.Symbol("mu"): float(mu),
            }
            for idx, value in enumerate(trial_gradient):
                substitutions[sp.Symbol("trial_grad%d" % idx)] = value
            material = [
                np.float32(float(expr.subs(substitutions)))
                for expr in model.apply_material_expressions
            ]
            for shape in range(model.n_shape):
                node = int(connectivity[elem, shape])
                for row in range(model.dim):
                    value = np.float32(0.0)
                    for col in range(model.dim):
                        value = np.float32(
                            value
                            + np.float32(qw)
                            * material[row * model.dim + col]
                            * np.float32(model.reference_gradients[shape * model.dim + col])
                        )
                    output[node, row] = np.float32(output[node, row] + value)
        return tuple(float(v) for v in output.reshape(-1))

    def _runner_connectivity(self):
        if self.lowering.max_elements == 1 and self.lowering.max_nodes == 4:
            return np.array([[0, 1, 2, 3]], dtype=np.intp)
        if self.lowering.max_elements == 8 and self.lowering.max_nodes == 9:
            return np.array(
                [
                    [0, 1, 2, 4],
                    [1, 3, 2, 4],
                    [1, 5, 3, 4],
                    [3, 5, 6, 4],
                    [0, 2, 7, 4],
                    [2, 8, 7, 4],
                    [2, 3, 8, 4],
                    [3, 6, 8, 4],
                ],
                dtype=np.intp,
            )
        raise ValueError("OpenMP MLIR runner fixture supports either 1xTET4/4 nodes or 8xTET4/9 nodes")

    def _runner_direction(self):
        if self.lowering.max_nodes == 4:
            return np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        if self.lowering.max_nodes == 9:
            return np.arange(
                self.lowering.max_nodes * self.lowering.model.n_field_components,
                dtype=np.float32,
            ).reshape(self.lowering.max_nodes, self.lowering.model.n_field_components)
        raise ValueError("OpenMP MLIR runner fixture has no direction field for this node count")

    def _fill_f32(self, target, upper, value):
        loop = self.scf.ForOp([], self._index_constant(0), upper, self._index_constant(1), [])
        body = self.ir.Block.create_at_start(loop.regions[0], [self.ir.IndexType.get()])
        with self.ir.InsertionPoint(body):
            self.memref.StoreOp(self._f32_constant(value), target, [body.arguments[0]])
            self.scf.YieldOp([])

    def _field_type(self):
        return self.ir.MemRefType.get([self.lowering.node_field_size], self.ir.F32Type.get())

    def _scratch_type(self):
        return self.ir.MemRefType.get([self.lowering.scratch_size], self.ir.F32Type.get())

    def _connectivity_type(self):
        return self.ir.MemRefType.get(
            [self.lowering.connectivity_size],
            self.ir.IndexType.get(),
        )

    def _geometry_adjugate_type(self):
        return self.ir.MemRefType.get(
            [self.lowering.geometry_adjugate_size],
            self.ir.F32Type.get(),
        )

    def _geometry_determinant_type(self):
        return self.ir.MemRefType.get(
            [self.lowering.geometry_determinant_size],
            self.ir.F32Type.get(),
        )

    def _node_degree_type(self):
        return self.ir.MemRefType.get(
            [self.lowering.node_degree_size],
            self.ir.IndexType.get(),
        )

    def _node_element_map_type(self):
        return self.ir.MemRefType.get(
            [self.lowering.node_element_map_size],
            self.ir.IndexType.get(),
        )

    def _node_component_index(self, node, component):
        return self._add_index(
            self._mul_index(node, self._index_constant(self.lowering.model.n_field_components)),
            self._index_constant(component),
        )

    def _inverse_topology_index(self, node, local_degree_index):
        return self._add_index(
            self._mul_index(node, self._index_constant(self.lowering.max_node_degree)),
            local_degree_index,
        )

    def _scratch_index(self, elem, local_node, component):
        model = self.lowering.model
        return self._add_index(
            self._mul_index(elem, self._index_constant(model.scratch_components)),
            self._add_index(
                self._mul_index(local_node, self._index_constant(model.n_field_components)),
                self._index_constant(component),
            ),
        )

    def _adjugate_index(self, elem, row, col):
        model = self.lowering.model
        return self._add_index(
            self._mul_index(elem, self._index_constant(model.dim * model.dim)),
            self._index_constant(row * model.dim + col),
        )

    def _add_index(self, lhs, rhs):
        return self.arith.AddIOp(lhs, rhs).result

    def _mul_index(self, lhs, rhs):
        return self.arith.MulIOp(lhs, rhs).result

    def _lower_sympy_f32(self, expression, symbols):
        expression = sp.sympify(expression)
        if expression in symbols:
            return symbols[expression]
        if expression.is_Number:
            return self._f32_constant(float(expression))
        if expression.is_Add:
            args = tuple(expression.args)
            result = self._lower_sympy_f32(args[0], symbols)
            for arg in args[1:]:
                result = self.arith.AddFOp(result, self._lower_sympy_f32(arg, symbols)).result
            return result
        if expression.is_Mul:
            args = tuple(expression.args)
            result = self._lower_sympy_f32(args[0], symbols)
            for arg in args[1:]:
                result = self.arith.MulFOp(result, self._lower_sympy_f32(arg, symbols)).result
            return result
        if expression.is_Pow and expression.exp.is_Integer and int(expression.exp) == 2:
            value = self._lower_sympy_f32(expression.base, symbols)
            return self.arith.MulFOp(value, value).result
        raise ValueError(f"unsupported SymPy expression in MLIR lowering: {expression}")

    def _validate_affine_tet4_apply(self):
        model = self.lowering.model
        if model.element_type != "TET4" or model.dim != 3 or model.n_shape != 4 or model.n_qp != 1:
            raise ValueError("initial OpenMP MLIR local apply supports affine TET4 only")
        if self.lowering.max_node_degree < 1:
            raise ValueError("OpenMP MLIR inverse topology requires positive max node degree")
        if not self.include_main:
            return
        connectivity = self._runner_connectivity()
        inverted = build_inverted_topology(connectivity, self.lowering.max_nodes)
        required_degree = int(np.max(inverted.node_degree)) if inverted.node_degree.size else 0
        if required_degree > self.lowering.max_node_degree:
            raise ValueError(
                "OpenMP MLIR inverse topology max_node_degree=%d is smaller than required degree=%d"
                % (self.lowering.max_node_degree, required_degree)
            )

    def _index_constant(self, value):
        return self.arith.ConstantOp(
            self.ir.IntegerAttr.get(self.ir.IndexType.get(), int(value))
        ).result

    def _i32_constant(self, value):
        return self.arith.ConstantOp(
            self.ir.IntegerAttr.get(self.ir.IntegerType.get_signless(32), int(value))
        ).result

    def _f32_constant(self, value):
        return self.arith.ConstantOp(
            self.ir.FloatAttr.get(self.ir.F32Type.get(), float(value))
        ).result


class _EmitCKernelModuleBuilder:
    def __init__(self, lowering):
        self.lowering = lowering
        self.ir = None
        self.emitc = None

    def build(self):
        self._load_bindings()
        self._validate_affine_tet4_apply()
        with self.ir.Context() as context, self.ir.Location.unknown():
            self.context = context
            self.i64 = self.ir.IntegerType.get_signless(64)
            self.f32 = self.ir.F32Type.get()
            self.ptr_i64 = self.ir.Type.parse("!emitc.ptr<i64>")
            self.ptr_f32 = self.ir.Type.parse("!emitc.ptr<f32>")
            self.lv_i64 = self.ir.Type.parse("!emitc.lvalue<i64>")
            self.lv_f32 = self.ir.Type.parse("!emitc.lvalue<f32>")

            module = self.ir.Module.create()
            with self.ir.InsertionPoint(module.body):
                self.emitc.IncludeOp("stdint.h", is_standard_include=True)
                self._build_kernel_function()
            return module

    def _load_bindings(self):
        from mlir import ir
        import mlir.dialects.emitc as emitc

        self.ir = ir
        self.emitc = emitc

    @property
    def function_name(self):
        return f"{self.lowering.function_name}_c"

    def _build_kernel_function(self):
        function_type = self.ir.TypeAttr.get(
            self.ir.FunctionType.get(
                [
                    self.ptr_i64,
                    self.ptr_f32,
                    self.ptr_f32,
                    self.ptr_f32,
                    self.f32,
                    self.f32,
                    self.ptr_f32,
                    self.ptr_i64,
                    self.ptr_i64,
                    self.ptr_i64,
                    self.ptr_f32,
                ],
                [],
            )
        )
        function = self.emitc.FuncOp(self.function_name, function_type)
        body = self.ir.Block.create_at_start(
            function.regions[0],
            [
                self.ptr_i64,
                self.ptr_f32,
                self.ptr_f32,
                self.ptr_f32,
                self.f32,
                self.f32,
                self.ptr_f32,
                self.ptr_i64,
                self.ptr_i64,
                self.ptr_i64,
                self.ptr_f32,
            ],
        )
        with self.ir.InsertionPoint(body):
            (
                connectivity,
                direction,
                adjugate,
                determinant,
                lmbda,
                mu,
                scratch,
                node_degree,
                node_to_element_map,
                node_to_local_idx,
                output,
            ) = body.arguments
            self._for_i64(
                self._i64_constant(0),
                self._i64_constant(self.lowering.max_elements),
                self._i64_constant(1),
                lambda elem: self._element_apply_body(
                    elem,
                    connectivity,
                    direction,
                    adjugate,
                    determinant,
                    lmbda,
                    mu,
                    scratch,
                ),
            )
            self._for_i64(
                self._i64_constant(0),
                self._i64_constant(self.lowering.max_nodes),
                self._i64_constant(1),
                lambda node: self._inverse_topology_reduce_body(
                    node,
                    scratch,
                    node_degree,
                    node_to_element_map,
                    node_to_local_idx,
                    output,
                ),
            )
            self.emitc.ReturnOp()

    def _for_i64(self, lower, upper, step, body_builder):
        loop = self.emitc.ForOp(lower, upper, step)
        body = self.ir.Block.create_at_start(loop.regions[0], [self.i64])
        with self.ir.InsertionPoint(body):
            body_builder(body.arguments[0])
            self.emitc.YieldOp()

    def _element_apply_body(
        self,
        elem,
        connectivity,
        direction,
        adjugate,
        determinant,
        lmbda,
        mu,
        scratch,
    ):
        trial_grad_ref = self._trial_reference_gradient(elem, connectivity, direction)
        trial_grad = self._transform_gradient(elem, trial_grad_ref, adjugate, determinant)
        symbols = {
            sp.Symbol("lmbda"): lmbda,
            sp.Symbol("mu"): mu,
        }
        for idx, value in enumerate(trial_grad):
            symbols[sp.Symbol("trial_grad%d" % idx)] = value
        material = tuple(
            self._lower_sympy_f32(expr, symbols)
            for expr in self.lowering.model.apply_material_expressions
        )
        loperand = self._transformed_loperand(elem, material, adjugate)
        self._store_element_apply(elem, loperand, scratch)

    def _inverse_topology_reduce_body(
        self,
        node,
        scratch,
        node_degree,
        node_to_element_map,
        node_to_local_idx,
        output,
    ):
        degree = self._load_i64(node_degree, node)
        for component in range(self.lowering.model.n_field_components):
            acc = self.emitc.VariableOp(
                self.lv_f32,
                self.ir.FloatAttr.get(self.f32, 0.0),
            ).result
            self._for_i64(
                self._i64_constant(0),
                degree,
                self._i64_constant(1),
                lambda i, component=component, acc=acc: self._reduce_one_incidence(
                    node,
                    i,
                    component,
                    acc,
                    scratch,
                    node_to_element_map,
                    node_to_local_idx,
                ),
            )
            self._store_f32(
                output,
                self._node_component_index(node, component),
                self.emitc.LoadOp(self.f32, acc).result,
            )

    def _reduce_one_incidence(
        self,
        node,
        i,
        component,
        acc,
        scratch,
        node_to_element_map,
        node_to_local_idx,
    ):
        map_index = self._inverse_topology_index(node, i)
        elem = self._load_i64(node_to_element_map, map_index)
        local = self._load_i64(node_to_local_idx, map_index)
        value = self._load_f32(scratch, self._scratch_index(elem, local, component))
        old = self.emitc.LoadOp(self.f32, acc).result
        self.emitc.AssignOp(acc, self.emitc.AddOp(self.f32, old, value).result)

    def _trial_reference_gradient(self, elem, connectivity, direction):
        model = self.lowering.model
        values = []
        for row in range(model.dim):
            for col in range(model.dim):
                acc = self._f32_constant(0.0)
                for shape in range(model.n_shape):
                    node = self._load_i64(
                        connectivity,
                        self._add_i64(
                            self._mul_i64(elem, self._i64_constant(model.n_shape)),
                            self._i64_constant(shape),
                        ),
                    )
                    field_value = self._load_f32(direction, self._node_component_index(node, row))
                    reference_gradient = self._f32_constant(
                        model.reference_gradients[shape * model.dim + col]
                    )
                    acc = self.emitc.AddOp(
                        self.f32,
                        acc,
                        self.emitc.MulOp(self.f32, field_value, reference_gradient).result,
                    ).result
                values.append(acc)
        return tuple(values)

    def _transform_gradient(self, elem, reference_gradient, adjugate, determinant):
        model = self.lowering.model
        det = self._load_f32(determinant, elem)
        inv_det = self.emitc.DivOp(self.f32, self._f32_constant(1.0), det).result
        values = []
        for row in range(model.dim):
            for col in range(model.dim):
                acc = self._f32_constant(0.0)
                for k in range(model.dim):
                    adj = self._load_f32(adjugate, self._adjugate_index(elem, k, col))
                    term = self.emitc.MulOp(
                        self.f32,
                        reference_gradient[row * model.dim + k],
                        adj,
                    ).result
                    acc = self.emitc.AddOp(self.f32, acc, term).result
                values.append(self.emitc.MulOp(self.f32, acc, inv_det).result)
        return tuple(values)

    def _transformed_loperand(self, elem, material, adjugate):
        model = self.lowering.model
        qw = self._f32_constant(model.quadrature_weights[0])
        values = []
        for row in range(model.dim):
            for col in range(model.dim):
                acc = self._f32_constant(0.0)
                for k in range(model.dim):
                    adj = self._load_f32(adjugate, self._adjugate_index(elem, col, k))
                    term = self.emitc.MulOp(self.f32, material[row * model.dim + k], adj).result
                    acc = self.emitc.AddOp(self.f32, acc, term).result
                values.append(self.emitc.MulOp(self.f32, qw, acc).result)
        return tuple(values)

    def _store_element_apply(self, elem, loperand, scratch):
        model = self.lowering.model
        element_offset = self._mul_i64(elem, self._i64_constant(model.scratch_components))
        for shape in range(model.n_shape):
            for row in range(model.dim):
                acc = self._f32_constant(0.0)
                for col in range(model.dim):
                    reference_gradient = self._f32_constant(
                        model.reference_gradients[shape * model.dim + col]
                    )
                    term = self.emitc.MulOp(
                        self.f32,
                        loperand[row * model.dim + col],
                        reference_gradient,
                    ).result
                    acc = self.emitc.AddOp(self.f32, acc, term).result
                self._store_f32(
                    scratch,
                    self._add_i64(
                        element_offset,
                        self._i64_constant(shape * model.dim + row),
                    ),
                    acc,
                )

    def _node_component_index(self, node, component):
        return self._add_i64(
            self._mul_i64(node, self._i64_constant(self.lowering.model.n_field_components)),
            self._i64_constant(component),
        )

    def _inverse_topology_index(self, node, local_degree_index):
        return self._add_i64(
            self._mul_i64(node, self._i64_constant(self.lowering.max_node_degree)),
            local_degree_index,
        )

    def _scratch_index(self, elem, local_node, component):
        model = self.lowering.model
        return self._add_i64(
            self._mul_i64(elem, self._i64_constant(model.scratch_components)),
            self._add_i64(
                self._mul_i64(local_node, self._i64_constant(model.n_field_components)),
                self._i64_constant(component),
            ),
        )

    def _adjugate_index(self, elem, row, col):
        model = self.lowering.model
        return self._add_i64(
            self._mul_i64(elem, self._i64_constant(model.dim * model.dim)),
            self._i64_constant(row * model.dim + col),
        )

    def _load_i64(self, pointer, index):
        ref = self.emitc.SubscriptOp(self.lv_i64, pointer, [index]).result
        return self.emitc.LoadOp(self.i64, ref).result

    def _load_f32(self, pointer, index):
        ref = self.emitc.SubscriptOp(self.lv_f32, pointer, [index]).result
        return self.emitc.LoadOp(self.f32, ref).result

    def _store_f32(self, pointer, index, value):
        ref = self.emitc.SubscriptOp(self.lv_f32, pointer, [index]).result
        self.emitc.AssignOp(ref, value)

    def _i64_constant(self, value):
        return self.emitc.ConstantOp(
            self.i64,
            self.ir.IntegerAttr.get(self.i64, int(value)),
        ).result

    def _f32_constant(self, value):
        return self.emitc.ConstantOp(
            self.f32,
            self.ir.FloatAttr.get(self.f32, float(value)),
        ).result

    def _add_i64(self, lhs, rhs):
        return self.emitc.AddOp(self.i64, lhs, rhs).result

    def _mul_i64(self, lhs, rhs):
        return self.emitc.MulOp(self.i64, lhs, rhs).result

    def _lower_sympy_f32(self, expression, symbols):
        expression = sp.sympify(expression)
        if expression in symbols:
            return symbols[expression]
        if expression.is_Number:
            return self._f32_constant(float(expression))
        if expression.is_Add:
            args = tuple(expression.args)
            result = self._lower_sympy_f32(args[0], symbols)
            for arg in args[1:]:
                result = self.emitc.AddOp(
                    self.f32,
                    result,
                    self._lower_sympy_f32(arg, symbols),
                ).result
            return result
        if expression.is_Mul:
            args = tuple(expression.args)
            result = self._lower_sympy_f32(args[0], symbols)
            for arg in args[1:]:
                result = self.emitc.MulOp(
                    self.f32,
                    result,
                    self._lower_sympy_f32(arg, symbols),
                ).result
            return result
        if expression.is_Pow and expression.exp.is_Integer and int(expression.exp) == 2:
            value = self._lower_sympy_f32(expression.base, symbols)
            return self.emitc.MulOp(self.f32, value, value).result
        raise ValueError(f"unsupported SymPy expression in EmitC lowering: {expression}")

    def _validate_affine_tet4_apply(self):
        model = self.lowering.model
        if model.element_type != "TET4" or model.dim != 3 or model.n_shape != 4 or model.n_qp != 1:
            raise ValueError("initial EmitC MLIR local apply supports affine TET4 only")
        if self.lowering.max_elements <= 0 or self.lowering.max_nodes <= 0 or self.lowering.max_node_degree <= 0:
            raise ValueError("EmitC MLIR TET4 apply requires positive mesh bounds")
