from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from enum import Enum

from codegen.framework.ir.kernel_ast import (
    BlockNode,
    BufferDeclNode,
    LoopHeaderNode,
    LoopKind,
    LoopNode,
    add_assign_increment,
    expr_ref,
    iteration_range,
    iterator,
    pre_increment,
)


class TargetLanguage(Enum):
    CPP = "c++"
    CUDA = "cuda"
    HIP = "hip"


class ExecutionModel(Enum):
    VECTOR_LANES = "vector_lanes"
    VECTOR_LENGTH_AGNOSTIC = "vector_length_agnostic"
    SIMT_THREADS = "simt_threads"


class MatrixUnitKind(Enum):
    NONE = "none"
    ARM_SME = "arm_sme"
    CUDA_TENSOR_CORES = "cuda_tensor_cores"
    HIP_MATRIX_CORES = "hip_matrix_cores"


@dataclass(frozen=True)
class KernelVariantPolicy:
    mesh_layouts: tuple = ("standard",)
    thread_variants: tuple = ()
    warp_variants: tuple = ()
    packed_mesh_passes: tuple = ()

    @property
    def supports_packed_mesh(self):
        return "packed" in self.mesh_layouts


@dataclass(frozen=True)
class LoopLoweringPolicy:
    execution_model: ExecutionModel = ExecutionModel.VECTOR_LANES
    emits_lane_loop: bool = True
    maps_lane_to_thread: bool = False
    vectorize_lane_loop: bool = False
    parallel_element_loop: bool = False
    supports_shared_memory: bool = False
    lane_index: str = "lane"
    lane_index_type: str = "int"
    vector_size_symbol: str = "VS"
    thread_index: str = "threadIdx.x"
    block_index: str = "blockIdx.x"
    block_dim: str = "blockDim.x"
    grid_dim: str = "gridDim.x"
    vector_isa: str = ""
    vector_length_aware: bool = False
    preferred_vector_bits: int = 0
    matrix_unit: MatrixUnitKind = MatrixUnitKind.NONE


@dataclass(frozen=True)
class TargetPlatform:
    name: str
    language: TargetLanguage
    default_alignment: int = 64

    @property
    def generated_language(self):
        return self.language.value

    def includes(self):
        return ()

    def function_qualifier(self):
        return "static inline"

    def inline_qualifier(self):
        return "inline"

    def inline_definition_lines(self, inline_definition="inline"):
        return ()

    def restrict_qualifier(self):
        return ""

    def restrict_definition(self):
        """What `SFEM_RESTRICT` expands to in a generated header.

        Empty on a host target, which is the CPU tree's own choice; a device
        target spells the real qualifier.  Distinct from
        `restrict_qualifier`, which is the *name* a generated declaration uses.
        """
        return ""

    def parallel_for_pragma(self, schedule=None, reduction=None):
        return None

    def parallel_region_pragma(self):
        """The pragma opening a parallel region, or `None`.

        Distinct from `parallel_for_pragma` because a kernel that stages through
        thread-private scratch has to open the region *before* the loop, allocate
        into it, and only then share the iterations out -- which is one pragma too
        few for the combined form.  A target without one emits neither and runs
        the loop serially, which is correct if slow.
        """
        return None

    def worksharing_for_pragma(self, schedule=None):
        """The pragma sharing a loop's iterations inside an open region."""
        return None

    def vectorize_pragma(self):
        return None

    def atomic_update_pragma(self):
        return None

    def alignment_assumption(self, pointer, alignment=None):
        return str(pointer)

    def math_header(self):
        return "kernel_math.hpp"

    def math_helper_name(self, function, exponent=None):
        function = str(function)
        if function == "pow" and exponent is not None:
            return _pow_helper_name(exponent)
        return function

    def diagnostics_header(self):
        return "kernel_diagnostics.hpp"

    def diagnostic_print_function(self):
        return "sfem::codegen::KernelDiagnostics_print_rate"

    def kernel_launch_style(self):
        return "host_function"

    def wrapper_style(self):
        return "c_abi"

    def loop_lowering_policy(self):
        return LoopLoweringPolicy()

    def work_item_index(self):
        policy = self.loop_lowering_policy()
        return policy.lane_index if policy.emits_lane_loop else "0"

    def work_item_name(self, name, component):
        return "%s_%s%d" % (str(name), self.work_item_index(), int(component))

    def diagnostic_work_item(self):
        return self.work_item_index()

    def work_item_scope_node(self, body=()):
        """One work-item scope: a lane loop, or a bare block where it is a thread.

        The decision -- loop or block, which index, which type, vectorized or
        not -- belongs to the target and is made once, here.  It comes back as a
        node rather than as text because `emitters/ast_printer.py` owns the
        spelling; `targets` may import `ir` (index 4 over index 3) and may not
        import `emitters`, which is exactly the split this returns.

        Lifted out of `residual_codegen._work_item_loop_node`, which was the
        only one of six re-spellings that was both policy-driven and IR-shaped.
        The other five differed from it by return type, not by decision.
        """
        policy = self.loop_lowering_policy()
        if not policy.emits_lane_loop:
            return BlockNode(body=tuple(body))
        index = iterator(policy.lane_index, policy.lane_index_type)
        pragma = self.vectorize_pragma() if policy.vectorize_lane_loop else None
        return LoopNode(
            LoopKind.SIMD,
            index,
            iteration_range(0, expr_ref("ne", "tile_extent")),
            pre_increment(index),
            body=tuple(body),
            vectorized=bool(pragma),
        )

    def serial_work_item_scope_node(self, body=()):
        """The same scope, never vectorized.

        A scatter that two work items can land on the same node in is serial on
        purpose -- the packed scatters say so in their own docstrings.  Routing
        one through `work_item_scope_node` on a target whose policy vectorizes
        would add a `#pragma omp simd` that is not there today: a data race, and
        a quiet one, since the answer only changes when two lanes collide.
        """
        node = self.work_item_scope_node(body)
        if isinstance(node, LoopNode):
            return dataclass_replace(node, vectorized=False)
        return node

    def work_item_prologue_lines(self, indent=""):
        """Bind the work-item name where there is no loop to bind it.

        The one-element-in-hand kernels -- matrix assembly, which scatters into
        a sparse row and has nowhere to put a block -- run the blocked
        arithmetic with a single work item and declare the name themselves.
        That declaration is not the work-item *index* and must not be routed
        through `work_item_index()`: on a CPU target that returns the name, so
        the site would emit `const int lane = lane;`, and on a SIMT target it
        returns `0` and the site would emit `const int 0 = 0;`.
        """
        policy = self.loop_lowering_policy()
        if not policy.emits_lane_loop:
            return ()
        return (
            "%sconst %s %s = 0;" % (indent, policy.lane_index_type, policy.lane_index),
        )

    def work_item_subscript(self):
        """How a staged buffer is indexed at this work item: `[lane]`, or `[0]`."""
        return "[%s]" % self.work_item_index()

    def work_item_offset(self, outer, stride):
        """A flat offset into a per-point, per-work-item stream.

        `q * VS + lane` where the work items are lanes of one block, and
        `q * VS` where the index is zero.  The `+ lane` term is what vanishes --
        not the stride, which is often `geometry_stride`, a runtime mesh
        parameter rather than the block width.
        """
        policy = self.loop_lowering_policy()
        base = "%s * %s" % (outer, stride)
        if not policy.emits_lane_loop:
            return base
        return "%s + %s" % (base, policy.lane_index)

    def element_index(self, block="evb"):
        """The element this work item holds: `evb + lane`, or `evb` itself."""
        policy = self.loop_lowering_policy()
        if not policy.emits_lane_loop:
            return block
        return "%s + %s" % (block, policy.lane_index)

    def work_item_loop_lines(self, indent):
        policy = self.loop_lowering_policy()
        if not policy.emits_lane_loop:
            return ("%s{" % indent,)
        lines = []
        pragma = self.vectorize_pragma() if policy.vectorize_lane_loop else None
        if pragma:
            lines.append("%s%s" % (indent, pragma))
        index = policy.lane_index
        lines.append(
            "%sfor (%s %s = 0; %s < ne; ++%s) {"
            % (indent, policy.lane_index_type, index, index, index)
        )
        return tuple(lines)

    def parallel_element_loop_lines(self, schedule=None, reduction=None):
        pragma = self.parallel_for_pragma(schedule, reduction)
        return () if pragma is None else (pragma,)

    def mesh_loop_nodes(self):
        """The blocked pass over the mesh: `VS` elements per iteration.

        Beside `work_item_scope_node`, which opens the scope *inside* one
        block.  This opens the block loop itself and declares how many elements
        it actually holds, which is `VS` except at the tail.

        It belongs to the target for the same reason the work item does:
        whether a mesh is walked by a host loop or by a grid of threads is a
        target fact, and an emitter that writes the loop itself produces, on a
        device target, a `__global__` kernel in which every thread walks every
        element.  Nodes rather than text, because `targets` is index 4 and `ir`
        is index 3: a target may build a node and may not spell one.
        `emitters/ast_printer.mesh_loop_lines` spells it.
        """
        tile_iterator = iterator("evb", "ptrdiff_t")
        return (
            LoopHeaderNode(
                LoopNode(
                    LoopKind.TILE,
                    tile_iterator,
                    iteration_range(
                        expr_ref("0", "first_element"),
                        expr_ref("nelements", "element_count"),
                    ),
                    add_assign_increment(tile_iterator, expr_ref("VS", "vector_width")),
                )
            ),
            BufferDeclNode(
                "const int", "ne", (), "(int)MIN((ptrdiff_t)VS, nelements - evb)"
            ),
        )

    def element_loop_nodes(self, index="element", extent="nelements"):
        """The scalar pass over the mesh: one element per iteration.

        The other shape is `mesh_loop_nodes`.  A constant-P1 simplex takes this
        one -- its staging is the whole of its cost, so it works an element at a
        time rather than blocking.

        `index` and `extent` are parameters because the boundary emitter walks
        sides rather than elements and calls its counter `s`; what the target
        decides is how the pass is opened, not what it counts.
        """
        element_iterator = iterator(index, "ptrdiff_t")
        return (
            LoopHeaderNode(
                LoopNode(
                    LoopKind.KERNEL,
                    element_iterator,
                    iteration_range(
                        expr_ref("0", "first_element"),
                        expr_ref(extent, "element_count"),
                    ),
                    pre_increment(element_iterator),
                )
            ),
        )

    def op_class_prefix(self):
        """What a generated `Op` class is called here.

        SFEM names its device Ops for the device -- `GPULaplacian`,
        `GPULinearElasticity`, `GPUKelvinVoigtNewmark` -- so a host and a device
        Op for one material are two classes rather than one name fought over.
        """
        return ""

    def op_registration_prefix(self):
        """The key the factory answers to.  `gpu:Laplacian`, `gpu:em:Laplacian`."""
        return ""

    def op_file_suffix(self):
        """What keeps the two wrappers' translation units apart."""
        return ""

    def source_subdirectory(self):
        """Where this target's sources sit inside the directory they mirror.

        SFEM keeps device sources in a local `cuda/` folder beside the host
        sources of the same module: `operators/tet4/tet4_laplacian.cpp` is
        mirrored by `operators/tet4/cuda/cu_tet4_laplacian.cu`, and the same
        shape holds in `operators/hex8`, `operators/tet10`, `algebra`,
        `resampling`, `frontend` and inside `external/smesh`.  The generated
        tree follows the repository rather than inventing its own placement.

        Nothing for the host, whose sources are the ones being mirrored.
        """
        return ""

    def element_connectivity_accessor(self):
        """Where a generated `Op` reads its connectivity from.

        The mesh's own SoA on the host.  A device target reads the block's
        device copy -- `device_elements_SoA()` -- which is what every `gpu:` Op
        in SFEM hands its kernels, and what a `__global__` body can dereference.
        """
        return "domain.block->elements()->data()"

    def geometry_memory_space(self):
        """Where a generated `Op` asks smesh to leave its cached geometry.

        `JacobianAdjugateAndDeterminant::create_SoA` and `FFF::create_SoA` both
        end in `to_memory_space(buffer, space)`, so a device Op needs no copy of
        its own: it asks for the space it wants and gets a device array of
        device pointers, which is the shape the kernels already read.
        """
        return "smesh::MEMORY_SPACE_HOST"

    def op_stream_arguments(self):
        """What a generated `Op` passes after a kernel's own arguments."""
        return ()

    def entry_point_name(self, public_name):
        """What this target calls a public `extern "C"` symbol.

        Unchanged on the host.  A device target prefixes, because the host and
        the device implementation of one operator are two symbols in one
        library and SFEM names its own device kernels that way throughout.
        """
        return public_name

    def entry_point_suffix_parameters(self):
        """Parameters every public entry point takes here, after its own.

        Empty on the host.  Every `cu_` entry point in SFEM ends with a
        `void *stream`, and an `Op` holds one and hands it to each call.
        """
        return ()

    def header_name(self, stem):
        """What a generated header is called here.

        Beside `mesh_function_line`: the name of a translation unit is a target
        fact.  A device generation that writes `kernel_math.hpp` has not just
        chosen an inconvenient name -- it has written a file that collides by
        path with the CPU tree's file of the same name and different contents.
        """
        return "%s.hpp" % stem

    def header_guard_suffix(self):
        return "HPP"

    def mesh_function_line(self, implementation_name):
        """How the mesh kernel itself is declared."""
        return "%s int %s(" % (self.function_qualifier(), implementation_name)

    def success_return_lines(self, indent="  "):
        """The mesh kernel's status return, where it has one to give."""
        return ("%sreturn SFEM_SUCCESS;" % indent,)

    def mesh_launch_lines(
        self, implementation_name, template_args, arguments, indent="  ", extent="nelements"
    ):
        """How the `extern \"C\"` entry point reaches the mesh kernel.

        `extent` is how many work items there are.  A host target does not use
        it -- the kernel's own loop reads it -- but a device target sizes its
        grid from it, and the boundary family's sideset entry points count
        `nsides` rather than `nelements`.
        """
        return (
            "%sreturn sfem::codegen::%s<%s>(%s);"
            % (indent, implementation_name, template_args, ", ".join(arguments)),
        )

    def scatter_add_lines(self, lhs, rhs, indent, pragma_indent=None):
        """One accumulation into a node several elements may share.

        The pragma and the `+=` are one decision: a target without an atomic
        pragma does not want the pragma dropped and the `+=` kept.

        `pragma_indent` exists only because the tracked tree spells this pragma
        at column 0 in the matrix-scatter sites and at the statement's own
        indent everywhere else.  It reproduces an inconsistency faithfully
        rather than deciding it.
        """
        pragma = self.atomic_update_pragma()
        lines = []
        if pragma:
            lines.append(
                "%s%s" % (indent if pragma_indent is None else pragma_indent, pragma)
            )
        lines.append("%s%s += %s;" % (indent, lhs, rhs))
        return tuple(lines)

    def host_function_qualifier(self):
        """How to spell a function that runs on the host and nowhere else.

        Most of what a kernel touches has to compile for the device too, and
        `inline_qualifier()` answers for those.  A few things must not: the
        dispatch reporter is called from the `extern "C"` launcher and never
        from a kernel, and its body prints to `stderr`, which device code has no
        notion of.  Marking it `__host__ __device__` because that is what
        everything else gets is how it came to be compiled for a device that
        cannot run it.
        """
        return self.inline_qualifier()

    def kernel_callable_qualifier(self):
        """What a shared helper needs in order to be callable from a kernel here.

        Empty on a CPU target, and that is not an oversight: a host function
        called from host code needs nothing said about it.  A device kernel can
        only call a function the compiler was told to compile for the device, so
        a target that has device kernels answers with its inline qualifier.

        The reference tables are what forced this.  Their accessors carried no
        qualifier at all -- correct for the only target that had ever read them
        -- and `__global__` bodies call them, so a CUDA generation produced 36
        of its 37 nvcc errors from that one omission.
        """
        return self.inline_qualifier() if self.supports_device_kernels else ""

    @property
    def supports_device_kernels(self):
        return False

    def vectorization_diagnostic_flags(self, compiler="c++"):
        compiler = str(compiler)
        if "clang" in compiler or compiler in ("c++", "cc"):
            return ("-Rpass=loop-vectorize", "-Werror=pass-failed")
        if "gcc" in compiler or "g++" in compiler:
            return ("-fopt-info-vec-optimized",)
        return ()

    def target_compile_flags(self):
        return ()

    def variant_policy(self):
        return KernelVariantPolicy()

    @property
    def supports_matrix_units(self):
        return self.loop_lowering_policy().matrix_unit is not MatrixUnitKind.NONE


@dataclass(frozen=True)
class OpenMPTarget(TargetPlatform):
    name: str = "openmp"
    language: TargetLanguage = TargetLanguage.CPP
    default_alignment: int = 64

    def includes(self):
        return (
            "#ifdef _OPENMP",
            "#include <omp.h>",
            "#endif",
        )

    def function_qualifier(self):
        return "static SFEM_INLINE"

    def inline_qualifier(self):
        return "SFEM_INLINE"

    def inline_definition_lines(self, inline_definition="inline"):
        return (
            "#ifndef SFEM_INLINE",
            "#define SFEM_INLINE %s" % str(inline_definition),
            "#endif",
        )

    def restrict_qualifier(self):
        return "RSTR"

    def parallel_for_pragma(self, schedule=None, reduction=None):
        pragma = "#pragma omp parallel for"
        if schedule:
            pragma += " schedule(%s)" % str(schedule)
        if reduction:
            pragma += " reduction(%s)" % str(reduction)
        return pragma

    def parallel_region_pragma(self):
        return "#pragma omp parallel"

    def worksharing_for_pragma(self, schedule=None):
        pragma = "#pragma omp for"
        if schedule:
            pragma += " schedule(%s)" % str(schedule)
        return pragma

    def vectorize_pragma(self):
        return "#pragma omp simd"

    def atomic_update_pragma(self):
        return "#pragma omp atomic update"

    def alignment_assumption(self, pointer, alignment=None):
        alignment = self.default_alignment if alignment is None else int(alignment)
        return "__builtin_assume_aligned(%s, %d)" % (str(pointer), alignment)

    def loop_lowering_policy(self):
        return LoopLoweringPolicy(
            execution_model=ExecutionModel.VECTOR_LANES,
            emits_lane_loop=True,
            maps_lane_to_thread=False,
            vectorize_lane_loop=True,
            parallel_element_loop=True,
            supports_shared_memory=False,
        )


@dataclass(frozen=True)
class AVX512Target(OpenMPTarget):
    name: str = "avx512"
    default_alignment: int = 64

    def loop_lowering_policy(self):
        return LoopLoweringPolicy(
            execution_model=ExecutionModel.VECTOR_LANES,
            emits_lane_loop=True,
            maps_lane_to_thread=False,
            vectorize_lane_loop=True,
            parallel_element_loop=True,
            supports_shared_memory=False,
            vector_isa="avx512",
            preferred_vector_bits=512,
        )

    def target_compile_flags(self):
        return ("-mavx512f",)

    def variant_policy(self):
        return KernelVariantPolicy(
            mesh_layouts=("standard", "packed"),
            packed_mesh_passes=("one_pass", "two_pass"),
        )


@dataclass(frozen=True)
class ARMSVETarget(OpenMPTarget):
    name: str = "arm_sve"
    default_alignment: int = 64

    def loop_lowering_policy(self):
        return LoopLoweringPolicy(
            execution_model=ExecutionModel.VECTOR_LENGTH_AGNOSTIC,
            emits_lane_loop=True,
            maps_lane_to_thread=False,
            vectorize_lane_loop=True,
            parallel_element_loop=True,
            supports_shared_memory=False,
            lane_index_type="ptrdiff_t",
            vector_isa="arm_sve",
            vector_length_aware=True,
        )

    def target_compile_flags(self):
        return ("-march=armv8-a+sve",)

    def variant_policy(self):
        return KernelVariantPolicy(
            mesh_layouts=("standard", "packed"),
            packed_mesh_passes=("one_pass", "two_pass"),
        )


@dataclass(frozen=True)
class ARMSMETarget(ARMSVETarget):
    name: str = "arm_sme"

    def loop_lowering_policy(self):
        return LoopLoweringPolicy(
            execution_model=ExecutionModel.VECTOR_LENGTH_AGNOSTIC,
            emits_lane_loop=True,
            maps_lane_to_thread=False,
            vectorize_lane_loop=True,
            parallel_element_loop=True,
            supports_shared_memory=False,
            lane_index_type="ptrdiff_t",
            vector_isa="arm_sme",
            vector_length_aware=True,
            matrix_unit=MatrixUnitKind.ARM_SME,
        )

    def target_compile_flags(self):
        return ("-march=armv9-a+sme",)


@dataclass(frozen=True)
class CUDATarget(TargetPlatform):
    name: str = "cuda"
    language: TargetLanguage = TargetLanguage.CUDA
    default_alignment: int = 16

    def includes(self):
        return ("#include <cuda_runtime.h>",)

    def function_qualifier(self):
        return "__host__ __device__ __forceinline__"

    def inline_qualifier(self):
        return "__host__ __device__ __forceinline__"

    def restrict_qualifier(self):
        return "__restrict__"

    def restrict_definition(self):
        return "__restrict__"

    def parallel_for_pragma(self, schedule=None, reduction=None):
        return None

    def vectorize_pragma(self):
        return None

    def alignment_assumption(self, pointer, alignment=None):
        alignment = self.default_alignment if alignment is None else int(alignment)
        return "__builtin_assume_aligned(%s, %d)" % (str(pointer), alignment)

    def kernel_launch_style(self):
        return "cuda_grid_stride"

    def wrapper_style(self):
        return "cuda_launcher"

    def loop_lowering_policy(self):
        return LoopLoweringPolicy(
            execution_model=ExecutionModel.SIMT_THREADS,
            emits_lane_loop=False,
            maps_lane_to_thread=True,
            vectorize_lane_loop=False,
            parallel_element_loop=False,
            supports_shared_memory=True,
            vector_isa="cuda",
            matrix_unit=MatrixUnitKind.CUDA_TENSOR_CORES,
        )

    def host_function_qualifier(self):
        return "__host__ __forceinline__"

    def work_item_name(self, name, component):
        return "%s_value%d" % (str(name), int(component))

    def diagnostic_work_item(self):
        return "scalar"

    def scatter_add_lines(self, lhs, rhs, indent, pragma_indent=None):
        """`pragma_indent` is accepted and ignored: there is no pragma to place."""
        return ("%satomicAdd(&(%s), %s);" % (indent, lhs, rhs),)

    def mesh_loop_nodes(self):
        """A grid-stride pass over the mesh, one element per thread.

        `ne` is 1 rather than the block's tail count: here the block is one
        thread, and the work-item scope below it opens no loop at all.
        """
        return self._grid_stride_nodes(
            "evb", (BufferDeclNode("const int", "ne", (), "1"),)
        )

    def element_loop_nodes(self, index="element", extent="nelements"):
        return self._grid_stride_nodes(index, (), extent)

    def _grid_stride_nodes(self, index, trailing, extent="nelements"):
        kernel_iterator = iterator(index, "ptrdiff_t")
        return (
            LoopHeaderNode(
                LoopNode(
                    LoopKind.KERNEL,
                    kernel_iterator,
                    iteration_range(
                        expr_ref(
                            "(ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x",
                            "cuda_thread_start",
                        ),
                        expr_ref(extent, "element_count"),
                    ),
                    add_assign_increment(
                        kernel_iterator,
                        expr_ref("(ptrdiff_t)blockDim.x * gridDim.x", "cuda_grid_stride"),
                    ),
                )
            ),
            *trailing,
        )

    def op_class_prefix(self):
        return "GPU"

    def op_registration_prefix(self):
        return "gpu:"

    def op_file_suffix(self):
        return "_cuda"

    def source_subdirectory(self):
        return "cuda"

    def element_connectivity_accessor(self):
        #: `device_elements_SoA()` hands back `idx_t *const *` where the C ABI
        #: takes `idx_t **`, and the const is the buffer's rather than the
        #: kernel's -- `GPULaplacian` passes its own device elements to
        #: `cu_laplacian_apply` the same way.
        return "const_cast<idx_t **>(domain.block->device_elements_SoA()->data())"

    def geometry_memory_space(self):
        return "smesh::MEMORY_SPACE_DEVICE"

    def op_stream_arguments(self):
        return ("stream",)

    def entry_point_name(self, public_name):
        return "cu_%s" % public_name

    def entry_point_suffix_parameters(self):
        return ("void *const stream",)

    def header_name(self, stem):
        return "%s.cuh" % stem

    def header_guard_suffix(self):
        return "CUH"

    def mesh_function_line(self, implementation_name):
        return "__global__ void %s(" % implementation_name

    def success_return_lines(self, indent="  "):
        """Nothing: a `__global__` kernel returns void."""
        return ()

    def mesh_launch_lines(
        self, implementation_name, template_args, arguments, indent="  ", extent="nelements"
    ):
        return (
            "%sconst int block_size = 256;" % indent,
            "%sconst int grid_size = (int)((%s + block_size - 1) / block_size);"
            % (indent, extent),
            #: The stream the caller handed the entry point.  SFEM's own device
            #: kernels all take one and an `Op` holds one; a launch that ignored
            #: it would serialise onto the default stream and silently undo the
            #: caller's ordering.
            "%ssfem::codegen::%s<%s><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(%s);"
            % (indent, implementation_name, template_args, ", ".join(arguments)),
            "%sreturn SFEM_SUCCESS;" % indent,
        )

    @property
    def supports_device_kernels(self):
        return True

    def vectorization_diagnostic_flags(self, compiler="nvcc"):
        return ()

    def variant_policy(self):
        return KernelVariantPolicy(
            mesh_layouts=("standard", "packed"),
            thread_variants=("per_thread",),
            warp_variants=("per_warp",),
            packed_mesh_passes=("one_pass", "two_pass"),
        )


@dataclass(frozen=True)
class HIPTarget(CUDATarget):
    name: str = "hip"
    language: TargetLanguage = TargetLanguage.HIP

    def includes(self):
        return ("#include <hip/hip_runtime.h>",)

    def source_subdirectory(self):
        #: The repository has no `hip/` folder to follow, so this names the
        #: target rather than copying `cuda/`: two device trees in one
        #: directory would collide the way the host and device trees did.
        return "hip"

    def kernel_launch_style(self):
        return "hip_grid_stride"

    def wrapper_style(self):
        return "hip_launcher"

    def loop_lowering_policy(self):
        return LoopLoweringPolicy(
            execution_model=ExecutionModel.SIMT_THREADS,
            emits_lane_loop=False,
            maps_lane_to_thread=True,
            vectorize_lane_loop=False,
            parallel_element_loop=False,
            supports_shared_memory=True,
            vector_isa="hip",
            matrix_unit=MatrixUnitKind.HIP_MATRIX_CORES,
        )


def _pow_helper_name(exponent):
    if isinstance(exponent, int):
        value = exponent
    else:
        try:
            value = int(exponent)
        except (TypeError, ValueError):
            return "pow"
        if value != exponent:
            return "pow"
    if value < 0:
        return "pow_m%d" % abs(value)
    return "pow_%d" % value
