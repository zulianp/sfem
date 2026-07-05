from .model import linear_elasticity_mlir_model, MLIRLoweringSpec
from .opencl import MatrixFreeOpenCLMLIRLowering
from .openmp import MatrixFreeOpenMPMLIRLowering


class MatrixFreeEBEMLIRLowering:
    """MLIR lowering for the generated SFEM linear-elasticity kernel model.

    The module keeps local FEM algebra behind the generated local-kernel
    boundary and lowers the mesh-level EBE schedule: geometry availability,
    local apply call, collision-free element scratchpad, and deterministic
    node-wise reduction.  This is intentionally plan-driven: names, dimensions,
    shape counts, quadrature counts, parameters, and phase ordering come from
    the same KernelPlan used by the OpenMP backend.
    """

    def __init__(self, model=None, spec=None):
        self.model = linear_elasticity_mlir_model() if model is None else model
        self.spec = MLIRLoweringSpec() if spec is None else spec

    @classmethod
    def from_linear_elasticity(cls, element="TET4", vector_size=8, quadrature_order=None, spec=None):
        return cls(
            linear_elasticity_mlir_model(
                element=element,
                vector_size=vector_size,
                quadrature_order=quadrature_order,
            ),
            spec=spec,
        )

    def opencl(self, max_elements=1024, max_nodes=4096, max_node_degree=32, optimization_strategy=None):
        return MatrixFreeOpenCLMLIRLowering(
            self.model,
            max_elements=max_elements,
            max_nodes=max_nodes,
            max_node_degree=max_node_degree,
            optimization_strategy=optimization_strategy,
        )

    def openmp(self, max_elements=1024, max_nodes=4096, max_node_degree=32, optimization_strategy=None):
        return MatrixFreeOpenMPMLIRLowering(
            self.model,
            max_elements=max_elements,
            max_nodes=max_nodes,
            max_node_degree=max_node_degree,
            optimization_strategy=optimization_strategy,
        )

    def has_no_atomics(self, module_text):
        lowered = module_text.lower()
        lowered = lowered.replace("no_atomics", "")
        lowered = lowered.replace("no atomics", "")
        return "atomic" not in lowered

    def _function_name(self):
        return f"{self.model.mesh_kernel_name}_{self.spec.function_suffix}"
