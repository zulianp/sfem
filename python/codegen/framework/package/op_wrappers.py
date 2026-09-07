import collections
import json
import os
import re

from codegen.framework.emitters.cprinter import parameter_list_lines
from codegen.framework.plans.form_transformations import (
    symmetric_metric_component_count,
)


def generate_op_registration_files(manifests, function_name="register_generated_ops"):
    entries = _registration_entries_from_manifests(manifests)
    header_name = "sfem_generated_ops_registration.hpp"
    source_name = "sfem_generated_ops_registration.cpp"
    return {
        header_name: _registration_aggregate_header(function_name),
        source_name: _registration_aggregate_source(header_name, function_name, entries),
    }


def generate_op_files(material, elements, kernel_sources=None):
    c_abi_header = "sfem_%s_c_abi.hpp" % material.op_name if kernel_sources else None
    kernel_sources = dict(kernel_sources or {})
    dispatch_sources, declared_signatures = (
        _dispatch_sources(material, elements, c_abi_header, kernel_sources)
        if c_abi_header
        else ({}, {})
    )
    element_api_sources = _element_api_sources(material, elements, kernel_sources)
    abi_sources = dict(kernel_sources)
    abi_sources.update(dispatch_sources)
    systems_by_dim = _systems_by_dim(material, elements)
    equations = _representative_equations(systems_by_dim)
    if len(equations) > 1:
        header, source = _coupled_energy_residual_op(
            material, elements, c_abi_header, systems_by_dim, abi_sources
        )
    elif not equations:
        raise ValueError("generated Op wrappers require at least one equation")
    elif equations[0].name:
        raise ValueError("single-equation generated Op wrappers require an unnamed equation")
    elif equations[0].is_energy:
        form_collections = _single_equation_form_collections(systems_by_dim, equations[0])
        header, source = _hyperelastic_op(
            material, elements, c_abi_header, form_collections, abi_sources
        )
    elif equations[0].is_residual:
        form_collections = _single_equation_form_collections(systems_by_dim, equations[0])
        measures = {collection.measure for collection in form_collections.values()}
        if measures == {"ds"}:
            header, source = _boundary_residual_op(
                material, elements, c_abi_header, form_collections
            )
        elif "ds" in measures:
            raise ValueError("generated residual Op wrappers cannot mix dx and ds forms yet")
        else:
            header, source = _residual_op(
                material, elements, c_abi_header, form_collections, abi_sources
            )
    else:
        raise ValueError("unsupported generated Op equation form")
    wrapper_header = "op/sfem_%s.hpp" % material.op_name
    wrapper_source = "op/sfem_%s.cpp" % material.op_name
    registration_source = "op/sfem_%s_registration.cpp" % material.op_name
    files = {
        wrapper_header: header,
        wrapper_source: source,
        registration_source: _registration_source(material, wrapper_header),
    }
    files.update(dispatch_sources)
    files.update(element_api_sources)
    if c_abi_header:
        c_abi_path = "op/%s" % c_abi_header
        files[c_abi_path] = _c_abi_header(material, abi_sources)
        files["op/sfem_%s_manifest.json" % material.op_name] = _op_manifest(
            material,
            abi_sources,
            wrapper_header,
            wrapper_source,
            registration_source,
            c_abi_path,
            element_api_sources,
        )
    _verify_generated_calls(files, abi_sources, declared_signatures)
    return files


def _call_argument_count(source, start):
    """How many top-level arguments the call beginning at ``start`` passes.

    ``start`` indexes the opening parenthesis.  Returns ``None`` if the call is
    unterminated, which means this was not a call at all.
    """
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return len(_split_c_parameters(source[start + 1 : index]))
    return None


def _verify_declaration_matches_recovery(declared_signatures, recovered):
    """The extraction, checked against signatures whose shape is known.

    Not the whole parser: see the note at the call site for what this does and
    does not establish.
    """
    disagreements = []
    for name, declared in sorted((declared_signatures or {}).items()):
        parsed = recovered.get(name)
        if parsed is None or parsed.declaration is None:
            continue
        declared_names = [parameter.name for parameter in declared.parameters]
        parsed_names = [parameter.name for parameter in parsed.parameters]
        if declared_names != parsed_names:
            disagreements.append(
                "%s: declared %s, recovered %s"
                % (name, declared_names, parsed_names)
            )
            continue
        declared_extents = [parameter.extent for parameter in declared.parameters]
        parsed_extents = [parameter.extent for parameter in parsed.parameters]
        if declared_extents != parsed_extents:
            disagreements.append(
                "%s: array extents declared %s, recovered %s"
                % (name, declared_extents, parsed_extents)
            )
    if disagreements:
        raise ValueError(
            "the C ABI recovery disagrees with what was declared:\n    %s"
            % "\n    ".join(disagreements)
        )


def _verify_generated_calls(files, abi_sources, declared_signatures=None):
    """Every call the wrapper makes must match the kernel it calls.

    The wrapper and the kernels are written by different layers that derive the
    ABI separately, so a wrapper can be emitted that cannot possibly compile --
    and for Stokes and poro-hyperelasticity, one was, on every generation, for
    as long as those materials have existed.  Nothing noticed because nothing
    in the framework compiles a generated wrapper: the byte-identity snapshot
    and the reproducibility gate both operate on kernels, and the wrapper
    syntax check in run_m9_regression.sh skips itself whenever ryml.hpp is
    absent, which in a bare worktree is always.

    Arity is a weaker statement than the compiler's, and it is available here,
    now, with no build tree and no dependencies -- which is what makes it worth
    having.  It is checked at generation rather than in a test because a
    wrapper that cannot compile should not reach disk: the failure belongs
    where the mistake is, not three steps downstream in somebody's build.

    See ARCHITECTURE.html OP 16 for why the two layers disagree in the first
    place and what would make the disagreement unrepresentable.
    """
    # Both views, because they cover different files.  The private view skips
    # everything under `op/`, and the public entry points a wrapper actually
    # calls are defined in the generated `_dispatch.cpp` sources, which live
    # there -- so checking only the private view silently checks nothing for
    # the mixed-order materials, which is how this check first passed while the
    # defect it exists to catch was still present.
    signatures = dict(_c_abi_signatures(abi_sources))
    signatures.update(_c_abi_signatures(abi_sources, public_only=True))
    # Where both exist they must agree.  The dispatch entry points are authored
    # here from `group["params"]` and also recovered from the text they were
    # printed as, so this compares the recovery against a known answer, on
    # fifty to a hundred signatures per material, every generation.
    #
    # What it covers is the extraction: locating a declaration in the emitted
    # source and splitting its parameter list back out.  Verified by injection
    # -- dropping a parameter during extraction is caught.  What it does not
    # cover is `_parse_c_parameter`, because both sides go through it, so a
    # defect there moves declared and recovered together and cancels; dropping
    # every array extent is not caught.  That half is only covered by the
    # arity check downstream and by the compile spike.
    _verify_declaration_matches_recovery(declared_signatures, signatures)
    # What was declared then outranks what was parsed back.
    signatures.update(declared_signatures or {})
    if not signatures:
        return
    mismatches = []
    for path, source in sorted(files.items()):
        if not path.endswith(".cpp"):
            continue
        for name, signature in signatures.items():
            expected = len(signature.parameters)
            marker = "%s(" % name
            index = source.find(marker)
            while index >= 0:
                before = source[index - 1] if index else " "
                # Skip a longer name that merely ends with this one.
                if before.isalnum() or before == "_":
                    index = source.find(marker, index + len(marker))
                    continue
                actual = _call_argument_count(source, index + len(name))
                if actual is not None and actual != expected:
                    mismatches.append(
                        "%s calls %s with %d argument(s); it takes %d"
                        % (path, name, actual, expected)
                    )
                index = source.find(marker, index + len(marker))
    if mismatches:
        raise ValueError(
            "generated wrapper does not match the kernels it calls:\n    %s"
            % "\n    ".join(sorted(set(mismatches)))
        )


def _registration_entries_from_manifests(manifests):
    entries = []
    seen_operators = set()
    for manifest in manifests:
        if isinstance(manifest, str):
            manifest = json.loads(manifest)
        _validate_op_manifest(manifest)
        registration = manifest["registration"]
        operator_name = registration["operator_name"]
        if operator_name in seen_operators:
            raise ValueError(
                "generated Op manifest registration operator '%s' is duplicated"
                % operator_name
            )
        seen_operators.add(operator_name)
        entries.append(
            (
                operator_name,
                registration["function"].replace("sfem::", ""),
            )
        )
    return tuple(sorted(entries))


def _validate_op_manifest(manifest):
    if not isinstance(manifest, dict):
        raise TypeError("generated Op manifest must be a JSON object")
    if manifest.get("schema") != "sfem.generated_op_manifest.v1":
        raise ValueError("generated Op manifest has unsupported schema")

    material = manifest.get("material")
    op_name = manifest.get("op_name")
    if not _nonempty_string(material):
        raise ValueError("generated Op manifest requires a material name")
    if not _nonempty_string(op_name):
        raise ValueError("generated Op manifest requires an op_name")

    wrapper = _required_mapping(manifest, "wrapper")
    _required_string(wrapper, "header", "generated Op manifest wrapper")
    _required_string(wrapper, "source", "generated Op manifest wrapper")
    _required_string(wrapper, "c_abi_header", "generated Op manifest wrapper")

    registration = _required_mapping(manifest, "registration")
    _required_string(registration, "source", "generated Op manifest registration")
    function = _required_string(registration, "function", "generated Op manifest registration")
    operator_name = _required_string(
        registration,
        "operator_name",
        "generated Op manifest registration",
    )
    if operator_name != op_name:
        raise ValueError("generated Op manifest registration operator_name must match op_name")
    if not function.startswith("sfem::"):
        raise ValueError("generated Op manifest registration function must be namespace-qualified")

    factory = _required_mapping(manifest, "factory")
    _required_string(factory, "class", "generated Op manifest factory")
    _required_string(factory, "create", "generated Op manifest factory")
    _required_string(factory, "create_from_yaml", "generated Op manifest factory")

    include_paths = manifest.get("generated_include_paths")
    if not isinstance(include_paths, (list, tuple)) or not include_paths:
        raise ValueError("generated Op manifest requires generated_include_paths")
    if not all(_nonempty_string(path) for path in include_paths):
        raise ValueError("generated Op manifest include paths must be strings")

    c_abi = manifest.get("c_abi")
    if not isinstance(c_abi, (list, tuple)) or not c_abi:
        raise ValueError("generated Op manifest requires c_abi declarations")
    c_abi_names = _validate_manifest_c_abi(c_abi)
    _validate_manifest_runtime_operations(manifest.get("runtime_operations"), c_abi_names)


def _validate_manifest_c_abi(c_abi):
    names = set()
    for index, entry in enumerate(c_abi):
        if not isinstance(entry, dict):
            raise ValueError("generated Op manifest c_abi entry %d must be an object" % index)
        name = _required_string(entry, "name", "generated Op manifest c_abi entry")
        declaration = _required_string(
            entry,
            "declaration",
            "generated Op manifest c_abi entry",
        )
        if name in names:
            raise ValueError("generated Op manifest c_abi function '%s' is duplicated" % name)
        if 'extern "C"' not in declaration or not declaration.rstrip().endswith(";"):
            raise ValueError(
                "generated Op manifest c_abi function '%s' must be an extern C declaration"
                % name
            )
        if _c_abi_function_name(declaration) != name:
            raise ValueError(
                "generated Op manifest c_abi function '%s' does not match its declaration"
                % name
            )
        names.add(name)
    return names


def _validate_manifest_runtime_operations(runtime_operations, c_abi_names):
    if not isinstance(runtime_operations, (list, tuple)) or not runtime_operations:
        raise ValueError("generated Op manifest requires runtime_operations")
    for operation in runtime_operations:
        if not isinstance(operation, dict):
            raise ValueError("generated Op manifest runtime operation must be an object")
        _required_string(operation, "name", "generated Op manifest runtime operation")
        variants = operation.get("variants")
        if not isinstance(variants, (list, tuple)) or not variants:
            raise ValueError("generated Op manifest runtime operation requires variants")
        for variant in variants:
            if not isinstance(variant, dict):
                raise ValueError("generated Op manifest runtime variant must be an object")
            _required_string(variant, "variant", "generated Op manifest runtime variant")
            _required_string(variant, "scalar_type", "generated Op manifest runtime variant")
            function = _required_string(
                variant,
                "function",
                "generated Op manifest runtime variant",
            )
            _required_string(variant, "target", "generated Op manifest runtime variant")
            if function not in c_abi_names:
                raise ValueError(
                    "generated Op manifest runtime function '%s' is not declared in c_abi"
                    % function
                )


def _required_mapping(mapping, key):
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError("generated Op manifest requires %s metadata" % key)
    return value


def _required_string(mapping, key, context):
    value = mapping.get(key)
    if not _nonempty_string(value):
        raise ValueError("%s requires %s" % (context, key))
    return value


def _nonempty_string(value):
    return isinstance(value, str) and bool(value)


def _registration_aggregate_header(function_name):
    return """#pragma once

namespace sfem {
    void %(function)s();
}  // namespace sfem
""" % {
        "function": function_name,
    }


def _registration_aggregate_source(header_name, function_name, entries):
    declarations = "\n".join("    void %s();" % function for _, function in entries)
    calls = "\n".join("        %s();" % function for _, function in entries)
    if declarations:
        declarations += "\n"
    if calls:
        calls += "\n"
    return """#include "%(header)s"

namespace sfem {
%(declarations)s
    void %(function)s() {
%(calls)s    }
}  // namespace sfem
""" % {
        "header": header_name,
        "declarations": declarations,
        "function": function_name,
        "calls": calls,
    }


def _systems_by_dim(material, elements):
    systems = getattr(material, "systems", None)
    if systems is None:
        raise TypeError("generated Op wrappers require a CodeGenerator with equation systems")
    return {
        dim: systems.for_dim(dim)
        for dim in sorted({_element_dim(element) for element in elements})
    }


def _representative_equations(systems_by_dim):
    first_dim = next(iter(sorted(systems_by_dim)))
    return tuple(systems_by_dim[first_dim].equations)


def _single_equation_form_collections(systems_by_dim, representative_equation):
    orders = _equation_form_orders(representative_equation)
    collections = {}
    for dim, system in systems_by_dim.items():
        equations = tuple(system.equations)
        if len(equations) != 1:
            raise ValueError("single-equation generated Op wrappers require one equation per dimension")
        collections[dim] = system.form_collection(equations[0], orders=orders)
    return collections


def _equation_form_orders(equation):
    if equation.is_energy:
        orders = [_form_order_zero()]
        for kernel in equation.kernels:
            if kernel == "objective":
                orders.append(_form_order_zero())
            elif kernel == "gradient":
                orders.append(_form_order_one())
            elif kernel == "apply":
                orders.append(_form_order_two())
        return tuple(dict.fromkeys(orders))
    if equation.is_residual:
        return (_form_order_zero(), _form_order_one(), _form_order_two())
    raise ValueError("unsupported equation form")


def _inexact_declarations(material):
    """The inexact methods, where the material generates the kernels for them.

    `Op` declares them with defaults that refuse, so an Op that does not
    generate the split simply does not override them and callers see
    `inexact_supported() == false`.  Nothing else changes shape.
    """
    if not getattr(material, "inexact_apply", False):
        return ""
    return """
        bool inexact_supported() const override { return true; }
        int inexact_update(const real_t *const x) override;
        int inexact_apply(const real_t *const h, real_t *const out) override;"""


def _header(material, residual, publishes_value_steps=None):
    """The Op's declared interface.

    ``publishes_value_steps`` says whether this Op offers the Newton
    line-search 0-form.  It used to be spelled ``not residual``, which is true
    for the two single-equation paths -- an energy has an objective, a residual
    was never given one -- and false for a coupled Op, which has an energy
    equation and therefore a perfectly good objective, generates the
    ``objective_steps`` kernels for it, and then did not declare the method
    that would call them.  Adding a residual block to an energy material
    silently removed its line search.

    Whether the method is published is a question about the forms the material
    lowered to, not about which ``add_*`` call produced them, so it is asked
    separately here.  The default keeps the two single-equation paths spelling
    it the way they always have.
    """
    if publishes_value_steps is None:
        publishes_value_steps = not residual
    extra = """
        int update(const real_t *const x) override;
        int update(const real_t *const previous, const real_t *const current) override;
        void set_field(const char *name,
                       const std::shared_ptr<Buffer<real_t>> &values,
                       int component) override;""" if residual else ""
    value_steps = """
        int value_steps(const real_t *x,
                        const real_t *h,
                        const int nsteps,
                        const real_t *const steps,
                        real_t *const out) override;""" if publishes_value_steps else ""
    matrix_methods = """
        int hessian_bsr(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override;
        int hessian_dia(const real_t *const x,
                        const int *const diag_offsets,
                        const ptrdiff_t ndiag,
                        real_t *const values) override;""" if residual else """
        int hessian_bsr(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override;
        int hessian_dia(const real_t *const x,
                        const int *const diag_offsets,
                        const ptrdiff_t ndiag,
                        real_t *const values) override;
        int hessian_coo(const real_t *const x,
                        const ptrdiff_t nnz,
                        const idx_t *const rows,
                        const idx_t *const cols,
                        real_t *const values);
        int hessian_patch(const real_t *const x,
                          const count_t *const rowptr,
                          const idx_t *const colidx,
                          real_t *const values);
        int hessian_block_diag_sym(const real_t *const x,
                                   real_t *const values) override;"""
    return """#pragma once

#include "sfem_Op.hpp"
#include "sfem_NeumannConditions.hpp"

namespace sfem {
    class %(op)s final : public Op {
    public:
        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);

        explicit %(op)s(const std::shared_ptr<FunctionSpace> &space);
        ~%(op)s() override;

        const char *name() const override { return "%(op)s"; }
        bool is_linear() const override { return false; }
        ptrdiff_t n_dofs_domain() const override;
        ptrdiff_t n_dofs_image() const override;
        double flops_value() const override;
        double flops_gradient() const override;
        double flops_apply() const override;
        size_t memory_traffic_bytes_value() const override;
        size_t memory_traffic_bytes_gradient() const override;
        size_t memory_traffic_bytes_apply() const override;

        int initialize(const std::vector<std::string> &block_names = {}) override;%(extra)s
        int gradient(const real_t *const x, real_t *const out) override;
        int apply(const real_t *const x,
                  const real_t *const h,
                  real_t *const out) override;
        int value(const real_t *x, real_t *const out) override;%(value_steps)s%(inexact_methods)s
        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override;%(matrix_methods)s
        void set_option(const std::string &name, bool val) override;
        void set_value_in_block(const std::string &block_name,
                                const std::string &var_name,
                                real_t value) override;
#ifdef SFEM_ENABLE_RYAML
        std::shared_ptr<Op> create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                             const ryml::ConstNodeRef             &node) override;
#endif  // SFEM_ENABLE_RYAML

        //! The scalar type the kernels are asked for at run time.
        //!
        //! Mirrors GPULaplacian, which declares the same member with the same
        //! default and hands it to every kernel call.  SMESH_DEFAULT resolves
        //! to the build's real_t, so the default costs a caller nothing and is
        //! the common path rather than a fallback.  The Op interface itself is
        //! unchanged: its methods still take real_t*, which converts to void*
        //! at the call, exactly as gpu_laplacian_block_vector relies on.
        enum smesh::PrimitiveType real_type{smesh::SMESH_DEFAULT};

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem
""" % {
        "op": material.op_name,
        "extra": extra,
        "value_steps": value_steps,
        "matrix_methods": matrix_methods,
        "inexact_methods": _inexact_declarations(material),
    }


def _inexact_cache_field(material):
    """The stored tangent, declared only where the material generates one.

    It sits with the cached geometry because it has the same lifetime and the
    same owner: assembled from a state, valid until the caller says otherwise,
    and read by every apply in between.  Empty until `inexact_update` is called,
    which is what makes calling it a precondition of the inexact apply rather
    than a hint.
    """
    if not getattr(material, "inexact_apply", False):
        return ""
    return """            SharedBuffer<metric_tensor_t> inexact_tangent;
"""


def _inexact_tangent_components(material, form_collections, elements):
    """How many numbers the stored tangent takes per element.

    A property of the dimension and of whether the tangent is symmetric, both
    fixed for a material, so it is resolved here at generation time rather than
    queried at run time.  An energy's tangent is a Hessian and folds to
    `d^2 (d^2 + 1) / 2`; a residual's is a Jacobian and generally does not fold
    at all.
    """
    from codegen.framework.plans.inexact_apply import (
        flux_form_for_collection,
        flux_tangent_is_symmetric,
    )

    for collection in (form_collections or {}).values():
        for dim in sorted({_element_dim(element) for element in elements}):
            built = flux_form_for_collection(collection, dim)
            if built is None:
                continue
            flux_form, _is_deformation_gradient = built
            order = dim * dim
            if flux_tangent_is_symmetric(flux_form):
                return order * (order + 1) // 2
            return order * order
    return 0


def _inexact_definitions(
    material, form_collections, elements, kernel_sources,
    apply_dependencies_by_dim, n_field_components_by_dim,
):
    """`inexact_update` and `inexact_apply`, where the material has them.

    Both dispatch on the spatial dimension the way every other method here
    does, and both go through the same public C ABI the exact kernels use.
    Only the affine SoA variant exists for this path, so the body is a good
    deal smaller than `apply`'s: there is no packed or isoparametric case to
    choose between.

    `inexact_update` owns the store.  It sizes it on first use and refills it in
    place afterwards, because a Newton solve calls it once per step for the life
    of the operator, and reallocating each time would dominate a kernel whose
    whole purpose is to be cheap.
    """
    if not getattr(material, "inexact_apply", False):
        return ""
    components = _inexact_tangent_components(material, form_collections, elements)
    if not components:
        return ""

    update_lines = []
    apply_lines = []
    reachable = False
    for dim in (2, 3):
        tangent_abi = "%s_inexact_apply_tangent_%dd_affine_mesh_soa" % (material.name, dim)
        stored_abi = "%s_inexact_apply_stored_%dd_affine_mesh_soa" % (material.name, dim)
        if not _c_abi_function_exists(kernel_sources, tangent_abi, public_only=True):
            continue
        if not _c_abi_function_exists(kernel_sources, stored_abi, public_only=True):
            continue
        reachable = True
        prefix = "if" if not update_lines else "else if"
        n_components = (n_field_components_by_dim or {}).get(dim, dim)
        parameter_args = list(
            _dependency_domain_parameter_args(apply_dependencies_by_dim.get(dim))
        )
        adjugate = ", ".join("adjugate[%d]" % index for index in range(dim * dim))
        state = ", ".join(["%d" % n_components] + ["x + %d" % d for d in range(n_components)])
        increment = ", ".join(["%d" % n_components] + ["h + %d" % d for d in range(n_components)])
        output = ", ".join(["%d" % n_components] + ["out + %d" % d for d in range(n_components)])
        arguments = ["domain.element_type", "real_type", "nelements",
                     "domain.block->elements()->data()", adjugate, "determinant"]
        arguments.extend(parameter_args)
        arguments.append(state)
        arguments.append("1, nelements")
        arguments.append("cache->inexact_tangent->data()")
        update_lines.extend([
            "            %s (dim == %d) {" % (prefix, dim),
            "                return %s(" % tangent_abi,
            "                        %s);" % ",\n                        ".join(arguments),
            "            }",
        ])
        apply_arguments = ["domain.element_type", "real_type", "nelements",
                           "domain.block->elements()->data()",
                           "1, nelements",
                           "cache->inexact_tangent->data()",
                           increment, output]
        apply_lines.extend([
            "            %s (dim == %d) {" % (prefix, dim),
            "                return %s(" % stored_abi,
            "                        %s);" % ",\n                        ".join(apply_arguments),
            "            }",
        ])
    if not reachable:
        return ""

    return """

    int %%(op)s::inexact_update(const real_t *const x) {
        SFEM_TRACE_SCOPE("%%(op)s::inexact_update");
        auto mesh = impl_->space->mesh_ptr();
        const int dim = mesh->spatial_dimension();
        return impl_->domains->iterate([&](const OpDomain &domain) {
            auto cache = std::static_pointer_cast<AffineGeometryCache>(domain.user_data);
            if (!cache || !cache->jacobian_soa) {
                SFEM_ERROR("%%(op)s::inexact_update requires cached affine geometry\\n");
                return SFEM_FAILURE;
            }
            const ptrdiff_t nelements = domain.block->n_elements();
            if (!cache->inexact_tangent) {
                cache->inexact_tangent = sfem::create_host_buffer<metric_tensor_t>(
                        nelements * %(components)d);
            }
            auto adjugate = reinterpret_cast<const geom_t *const *>(
                    cache->jacobian_soa->jacobian_adjugate_SoA()->data());
            auto determinant = reinterpret_cast<const geom_t *>(
                    cache->jacobian_soa->jacobian_determinant()->data());
%(update_body)s
            SFEM_ERROR("%%(op)s::inexact_update has no kernel for dimension %%%%d\\n", dim);
            return SFEM_FAILURE;
        });
    }

    int %%(op)s::inexact_apply(const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("%%(op)s::inexact_apply");
        auto mesh = impl_->space->mesh_ptr();
        const int dim = mesh->spatial_dimension();
        return impl_->domains->iterate([&](const OpDomain &domain) {
            auto cache = std::static_pointer_cast<AffineGeometryCache>(domain.user_data);
            if (!cache || !cache->inexact_tangent) {
                SFEM_ERROR("%%(op)s::inexact_apply requires inexact_update first\\n");
                return SFEM_FAILURE;
            }
            const ptrdiff_t nelements = domain.block->n_elements();
%(apply_body)s
            SFEM_ERROR("%%(op)s::inexact_apply has no kernel for dimension %%%%d\\n", dim);
            return SFEM_FAILURE;
        });
    }""" % {
        "components": components,
        "update_body": "\n".join(update_lines),
        "apply_body": "\n".join(apply_lines),
    }


def _hyperelastic_op(
    material, elements, c_abi_header=None, form_collections=None, kernel_sources=None
):
    if form_collections is None:
        raise ValueError("energy generated Op requires form collections")
    n_field_components_by_dim = {}
    if kernel_sources is None:
        kernel_sources = {}
    parameters = tuple(str(name) for name, _ in material.parameter_defaults)
    defaults = _seed_lines(material.parameter_defaults)
    declarations = []
    gradient_cases = []
    apply_cases = []
    objective_cases = []
    objective_steps_cases = []
    hessian_crs_cases = []
    hessian_bsr_cases = []
    hessian_dia_cases = []
    hessian_coo_cases = []
    hessian_patch_cases = []
    generated_packed_apply = any("_packed_" in source for source in kernel_sources.values())
    packed_scratch_include = '#include "packed_thread_scratch.hpp"\n#include "smesh_env.hpp"' if generated_packed_apply else ""
    packed_scratch_prealloc = (
        """        impl_->use_packed_two_pass = smesh::Env::read("SFEM_PACKED_TWO_PASS", false);
        if (impl_->space->has_packed_mesh()) {
            auto packed = impl_->space->packed_mesh();
            const ptrdiff_t max_nodes_per_pack = packed->max_nodes_per_pack();
            const int dim = impl_->space->mesh_ptr()->spatial_dimension();
            const size_t scratch_size = (size_t)dim * (size_t)max_nodes_per_pack;
            sfem::codegen::prealloc_thread_scratch<real_t>(0, scratch_size);
            sfem::codegen::prealloc_thread_scratch<real_t>(1, scratch_size);
            sfem::codegen::prealloc_thread_scratch<real_t>(2, scratch_size);
            sfem::codegen::prealloc_thread_scratch<real_t>(3, scratch_size);
            impl_->packed_ghost_buf.resize((size_t)packed->n_blocks());
            for (int b = 0; b < packed->n_blocks(); ++b) {
                const ptrdiff_t n_ghost = packed->n_ghost_entries(b);
                const ptrdiff_t n_slots = (n_ghost > 0 ? n_ghost : 1) * (ptrdiff_t)dim;
                impl_->packed_ghost_buf[b] = create_host_buffer<real_t>(n_slots);
            }
        }"""
        if generated_packed_apply
        else ""
    )
    performance_cases = {"value": [], "gradient": [], "apply": []}
    dependencies_by_dim = {}
    gradient_affine_aos_flags = []
    apply_affine_aos_flags = []
    # Whether any affine entry point this wrapper calls takes the gradient
    # metric rather than the Jacobian adjugate.  The kernels decide; the
    # wrapper reads their declarations and caches what they ask for.
    affine_metric_flags = []
    for element in elements:
        dim = _element_dim(element)
        dependencies = dependencies_by_dim.get(dim)
        if dependencies is None:
            collection = form_collections[dim]
            dependencies = (
                collection.form_metadata(_form_order_zero()).dependencies,
                collection.form_metadata(_form_order_one()).dependencies,
                collection.form_metadata(_form_order_two()).dependencies,
            )
            dependencies_by_dim[dim] = dependencies
        objective_dependencies, gradient_dependencies, apply_dependencies = dependencies
        element_label = _element_name(element).lower()
        stem = "%s_%s" % (material.name, element_label)
        performance_cases["value"].append(
            _performance_case(
                element,
                ("%s_objective_soa_diagnostics" % stem,),
                affine_flags=("objective_uses_affine",),
            )
        )
        performance_cases["gradient"].append(
            _performance_case(
                element,
                ("%s_gradient_soa_diagnostics" % stem,),
                affine_flags=("gradient_uses_affine",),
            )
        )
        performance_cases["apply"].append(
            _performance_case(
                element,
                ("%s_apply_soa_diagnostics" % stem,),
                affine_flags=("apply_uses_affine",),
            )
        )
        n_field_components = _energy_field_component_count(collection, dim)
        n_field_components_by_dim[dim] = n_field_components
        components = _components(n_field_components)
        declarations.extend(
            _hyperelastic_declarations(stem, dim, parameters, dependencies, n_field_components_by_dim=n_field_components_by_dim)
        )
        objective_args = "".join(
            ", %s" % arg for arg in _dependency_domain_parameter_args(objective_dependencies)
        )
        gradient_args = "".join(
            ", %s" % arg for arg in _dependency_domain_parameter_args(gradient_dependencies)
        )
        apply_args = "".join(
            ", %s" % arg for arg in _dependency_domain_parameter_args(apply_dependencies)
        )
        gradient_common_isoparametric_args = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), points%s" % gradient_args
        )
        gradient_common_affine_args = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), %s%s"
            % (
                _affine_geometry_offsets_for(
                    kernel_sources, "%s_gradient_affine_mesh_soa" % stem, dim
                ),
                gradient_args,
            )
        )
        gradient_common_affine_aos_args = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), adjugate_aos, determinant%s"
            % gradient_args
        )
        apply_common_isoparametric_args = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), points%s" % apply_args
        )
        apply_common_affine_args = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), %s%s"
            % (
                _affine_geometry_offsets_for(
                    kernel_sources, "%s_apply_affine_mesh_soa" % stem, dim
                ),
                apply_args,
            )
        )
        apply_common_affine_aos_args = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), adjugate_aos, determinant%s"
            % apply_args
        )
        gradient_affine_uses_aos = _c_abi_function_exists(
            kernel_sources, "%s_gradient_affine_mesh_soa_aos_unit" % stem
        )
        apply_affine_uses_aos = _c_abi_function_exists(
            kernel_sources, "%s_apply_affine_mesh_soa_aos_unit" % stem
        )
        gradient_affine_aos_flags.append(gradient_affine_uses_aos)
        apply_affine_aos_flags.append(apply_affine_uses_aos)
        affine_metric_flags.extend(
            _affine_dispatch_uses_metric(kernel_sources, name)
            for name in (
                "%s_gradient_affine_mesh_soa" % stem,
                "%s_apply_affine_mesh_soa" % stem,
                "%s_objective_affine_mesh_soa" % stem,
                "%s_objective_steps_affine_mesh_soa" % stem,
                "%s_gradient_%dd_affine_mesh_soa" % (material.name, dim),
                "%s_apply_%dd_affine_mesh_soa" % (material.name, dim),
                "%s_objective_%dd_affine_mesh_soa" % (material.name, dim),
                "%s_objective_steps_%dd_affine_mesh_soa" % (material.name, dim),
            )
        )
        gradient_affine_args = ", ".join(
            _nonempty(
                gradient_common_affine_args,
                *_energy_field_args(
                    gradient_dependencies, dim, components, current="x"
                ),
                *_energy_output_args(dim, components),
            )
        )
        gradient_isoparametric_args = ", ".join(
            _nonempty(
                gradient_common_isoparametric_args,
                *_energy_field_args(
                    gradient_dependencies, dim, components, current="x"
                ),
                *_energy_output_args(dim, components),
            )
        )
        if gradient_affine_uses_aos:
            gradient_cases.append(
                _dual_aos_unit_case(
                    element,
                    "gradient_uses_affine",
                    "%s_gradient_affine_mesh_soa_aos_unit" % stem,
                    ", ".join(
                        _nonempty(
                            gradient_common_affine_aos_args,
                            *_energy_field_args(
                                gradient_dependencies, dim, components, current="x"
                            ),
                            *_energy_output_args(dim, components),
                        )
                    ),
                    "%s_gradient_affine_mesh_soa" % stem,
                    gradient_affine_args,
                    "%s_gradient_isoparametric_mesh_soa" % stem,
                    gradient_isoparametric_args,
                )
            )
        else:
            gradient_cases.append(
                _dual_case(
                    element,
                    "gradient_uses_affine",
                    "%s_gradient_affine_mesh_soa" % stem,
                    gradient_affine_args,
                    "%s_gradient_isoparametric_mesh_soa" % stem,
                    gradient_isoparametric_args,
                )
            )
        apply_affine_args = ", ".join(
            _nonempty(
                apply_common_affine_args,
                *_energy_field_args(
                    apply_dependencies,
                    dim,
                    components,
                    current="x",
                    direction="h",
                ),
                *_energy_output_args(dim, components),
            )
        )
        apply_isoparametric_args = ", ".join(
            _nonempty(
                apply_common_isoparametric_args,
                *_energy_field_args(
                    apply_dependencies,
                    dim,
                    components,
                    current="x",
                    direction="h",
                ),
                *_energy_output_args(dim, components),
            )
        )
        if apply_affine_uses_aos:
            apply_cases.append(
                _dual_aos_unit_case(
                    element,
                    "apply_uses_affine",
                    "%s_apply_affine_mesh_soa_aos_unit" % stem,
                    ", ".join(
                        _nonempty(
                            apply_common_affine_aos_args,
                            *_energy_field_args(
                                apply_dependencies,
                                dim,
                                components,
                                current="x",
                                direction="h",
                            ),
                            *_energy_output_args(dim, components),
                        )
                    ),
                    "%s_apply_affine_mesh_soa" % stem,
                    apply_affine_args,
                    "%s_apply_isoparametric_mesh_soa" % stem,
                    apply_isoparametric_args,
                )
            )
        else:
            apply_cases.append(
                _dual_case(
                    element,
                    "apply_uses_affine",
                    "%s_apply_affine_mesh_soa" % stem,
                    apply_affine_args,
                    "%s_apply_isoparametric_mesh_soa" % stem,
                    apply_isoparametric_args,
                )
            )
        objective_cases.append(
            _dual_status_case(
                element,
                "%s_objective_affine_mesh_soa" % stem,
                ", ".join(_nonempty(
                    "nelements, mesh->n_nodes(), domain.block->elements()->data(), %s, determinant%s"
                    % (
                        _affine_geometry_offsets_for(
                            kernel_sources,
                            "%s_objective_affine_mesh_soa" % stem,
                            dim,
                        ),
                        objective_args,
                    ),
                    *_energy_field_args(objective_dependencies, dim, components, current="x"),
                    "impl_->element_values.get()",
                )),
                "%s_objective_isoparametric_mesh_soa" % stem,
                ", ".join(_nonempty(
                    "nelements, mesh->n_nodes(), domain.block->elements()->data(), points%s"
                    % objective_args,
                    *_energy_field_args(objective_dependencies, dim, components, current="x"),
                    "impl_->element_values.get()",
                )),
            )
        )
        objective_steps_cases.append(
            _dual_status_case(
                element,
                "%s_objective_steps_affine_mesh_soa" % stem,
                ", ".join(_nonempty(
                    "nelements, mesh->n_nodes(), domain.block->elements()->data(), %s, determinant%s"
                    % (
                        _affine_geometry_offsets_for(
                            kernel_sources,
                            "%s_objective_affine_mesh_soa" % stem,
                            dim,
                        ),
                        objective_args,
                    ),
                    *_energy_field_args(objective_dependencies, dim, components, current="x"),
                    dim,
                    _offsets("h", components),
                    "nsteps",
                    "steps",
                    "impl_->element_values.get()",
                )),
                "%s_objective_steps_isoparametric_mesh_soa" % stem,
                ", ".join(_nonempty(
                    "nelements, mesh->n_nodes(), domain.block->elements()->data(), points%s"
                    % objective_args,
                    *_energy_field_args(objective_dependencies, dim, components, current="x"),
                    dim,
                    _offsets("h", components),
                    "nsteps",
                    "steps",
                    "impl_->element_values.get()",
                )),
            )
        )
        hessian_state_args = ", ".join(
            _nonempty(
                apply_common_isoparametric_args,
                *_energy_field_args(
                    apply_dependencies,
                    dim,
                    components,
                    current="current",
                ),
            )
        )
        hessian_crs_function = "%s_hessian_crs_isoparametric_mesh_soa" % stem
        if _c_abi_function_exists(kernel_sources, hessian_crs_function):
            hessian_crs_cases.append(
                _case(
                    element,
                    hessian_crs_function,
                    ", ".join(
                        _nonempty(
                            hessian_state_args,
                            "rowptr",
                            "colidx",
                            "values",
                        )
                    ),
                )
            )
        hessian_bsr_function = "%s_hessian_bsr_isoparametric_mesh_soa" % stem
        if _c_abi_function_exists(kernel_sources, hessian_bsr_function):
            hessian_bsr_cases.append(
                _case(
                    element,
                    hessian_bsr_function,
                    ", ".join(
                        _nonempty(
                            hessian_state_args,
                            "rowptr",
                            "colidx",
                            "values",
                        )
                    ),
                )
            )
        hessian_dia_function = "%s_hessian_dia_isoparametric_mesh_soa" % stem
        if _c_abi_function_exists(kernel_sources, hessian_dia_function):
            hessian_dia_cases.append(
                _case(
                    element,
                    hessian_dia_function,
                    ", ".join(
                        _nonempty(
                            hessian_state_args,
                            "diag_offsets",
                            "ndiag",
                            "values",
                        )
                    ),
                )
            )
        hessian_coo_function = "%s_hessian_coo_isoparametric_mesh_soa" % stem
        if _c_abi_function_exists(kernel_sources, hessian_coo_function):
            hessian_coo_cases.append(
                _case(
                    element,
                    hessian_coo_function,
                    ", ".join(
                        _nonempty(
                            hessian_state_args,
                            "nnz",
                            "rows",
                            "cols",
                            "values",
                        )
                    ),
                )
            )
        hessian_patch_function = "%s_hessian_patch_isoparametric_mesh_soa" % stem
        if _c_abi_function_exists(kernel_sources, hessian_patch_function):
            hessian_patch_cases.append(
                _case(
                    element,
                    hessian_patch_function,
                    ", ".join(
                        _nonempty(
                            hessian_state_args,
                            "rowptr",
                            "colidx",
                            "values",
                        )
                    ),
                )
            )
    source = """#include "sfem_%(op)s.hpp"
%(c_abi_include)s
%(packed_scratch_include)s

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_kernel_data.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

%(declaration_block)s

namespace sfem {
    namespace {
        void seed_parameters(Parameters &parameters) {
%(defaults)s
        }

        void seed_material(MultiDomainOp &domains) {
            for (auto &entry : domains.domains()) {
                seed_parameters(*entry.second.parameters);
            }
        }

%(yaml_helpers)s

        smesh::block_idx_t block_id_for_domain(const smesh::Mesh &mesh,
                                               const smesh::Mesh::Block &block) {
            for (size_t i = 0; i < mesh.n_blocks(); ++i) {
                if (mesh.block(i).get() == &block) {
                    return static_cast<smesh::block_idx_t>(i);
                }
            }
            SFEM_ERROR("%(op)s: mesh block pointer not found in mesh.blocks()\\n");
            return 0;
        }

        int packed_block_id_for_domain(const FunctionSpace::PackedMesh &packed,
                                       const smesh::Mesh::Block &block) {
            for (ptrdiff_t i = 0; i < packed.n_blocks(); ++i) {
                if (packed.block_name(i) == block.name()) {
                    return static_cast<int>(i);
                }
            }
            return -1;
        }

        struct AffineGeometryCache {
            std::shared_ptr<smesh::JacobianAdjugateAndDeterminant> jacobian_soa;
            std::shared_ptr<smesh::JacobianAdjugateAndDeterminant> jacobian_aos;
%(metric_cache_field)s%(inexact_cache_field)s        };

        int cache_affine_geometry(const std::shared_ptr<FunctionSpace> &space,
                                  MultiDomainOp &domains) {
            auto mesh = space->mesh_ptr();
            const bool needs_jacobian_aos =
                    %(gradient_affine_uses_jacobian_aos)s ||
                    %(apply_affine_uses_jacobian_aos)s;
            for (auto &entry : domains.domains()) {
                const smesh::block_idx_t block_id =
                        block_id_for_domain(*mesh, *entry.second.block);
                auto cache = std::make_shared<AffineGeometryCache>();
                cache->jacobian_soa = smesh::JacobianAdjugateAndDeterminant::create_SoA(
                        mesh, smesh::MEMORY_SPACE_HOST, block_id);
                if (!cache->jacobian_soa) {
                    return SFEM_FAILURE;
                }
                if (needs_jacobian_aos) {
                    cache->jacobian_aos = smesh::JacobianAdjugateAndDeterminant::create_AoS(
                            mesh, smesh::MEMORY_SPACE_HOST, block_id);
                    if (!cache->jacobian_aos) {
                        return SFEM_FAILURE;
                    }
                }
%(metric_cache_setup)s                entry.second.user_data = std::static_pointer_cast<void>(cache);
            }
            return SFEM_SUCCESS;
        }
    }  // namespace

    class %(op)s::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::unique_ptr<real_t[]> element_values;
        ptrdiff_t element_capacity{0};
        bool objective_uses_affine{false};
        bool gradient_uses_affine{false};
        bool apply_uses_affine{false};
        bool use_packed_two_pass{false};
        std::vector<SharedBuffer<real_t>> packed_ghost_buf;
    };

    std::unique_ptr<Op> %(op)s::create(const std::shared_ptr<FunctionSpace> &space) {
        if (space->block_size() != space->mesh_ptr()->spatial_dimension()) {
            SFEM_ERROR("%(op)s requires block_size=spatial_dimension\\n");
            return nullptr;
        }
        auto op = std::make_unique<%(op)s>(space);
        op->initialize();
        return op;
    }

    %(op)s::%(op)s(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    %(op)s::~%(op)s() = default;

    ptrdiff_t %(op)s::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t %(op)s::n_dofs_image() const { return impl_->space->n_dofs(); }

%(performance_methods)s

    // Establish once, at setup, that this operator's dof graph is well formed:
    // rows in order, every column in range, each row sorted and duplicate free.
    // The assembly kernels assume it -- they locate an entry and write to it
    // without re-checking that it is there -- so this is where the assumption
    // is earned.
    //
    // It used to be earned per element instead: every scatter walked its
    // N_SHAPE x N_SHAPE candidates, tested each with a three-condition branch
    // and reported through std::fprintf from inside the caller's parallel
    // region.  That paid O(elements x N_SHAPE^2) on every assembly for a
    // property of the mesh and the graph together, which cannot change between
    // elements or between calls.  Here it is O(nnz), once.
    //
    // Raw pointers rather than the graph type, so this does not depend on which
    // headers the generated wrapper happens to pull in.
    static int validate_dof_graph(const count_t *const rowptr,
                                  const idx_t *const colidx,
                                  const ptrdiff_t n_nodes,
                                  const ptrdiff_t nnz) {
        if (!rowptr || !colidx || n_nodes < 0) {
            return SFEM_FAILURE;
        }
        if (rowptr[0] != 0 || (ptrdiff_t)rowptr[n_nodes] != nnz) {
            return SFEM_FAILURE;
        }
        for (ptrdiff_t i = 0; i < n_nodes; ++i) {
            const count_t begin = rowptr[i];
            const count_t end = rowptr[i + 1];
            if (end < begin || (ptrdiff_t)end > nnz) {
                return SFEM_FAILURE;
            }
            for (count_t k = begin; k < end; ++k) {
                if (colidx[k] < 0 || (ptrdiff_t)colidx[k] >= n_nodes) {
                    return SFEM_FAILURE;
                }
                if (k > begin && colidx[k] <= colidx[k - 1]) {
                    return SFEM_FAILURE;
                }
            }
        }
        return SFEM_SUCCESS;
    }

    int %(op)s::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("%(op)s::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        {
            auto dof_graph = impl_->space->dof_to_dof_graph();
            if (!dof_graph ||
                validate_dof_graph(dof_graph->rowptr()->data(),
                                   dof_graph->colidx()->data(),
                                   dof_graph->n_nodes(),
                                   dof_graph->nnz()) != SFEM_SUCCESS) {
                SFEM_ERROR("%(op)s::initialize: the dof graph is malformed; the assembly kernels assume it is not\\n");
                return SFEM_FAILURE;
            }
        }
        auto mesh = impl_->space->mesh_ptr();
        const bool needs_affine_geometry =
                impl_->objective_uses_affine ||
                impl_->gradient_uses_affine ||
                impl_->apply_uses_affine;
        for (auto &entry : impl_->domains->domains()) {
            seed_parameters(*entry.second.parameters);
            impl_->element_capacity =
                    std::max(impl_->element_capacity, entry.second.block->n_elements());
            if (needs_affine_geometry) {
                const smesh::block_idx_t block_id =
                        block_id_for_domain(*mesh, *entry.second.block);
                auto cache = std::make_shared<AffineGeometryCache>();
                cache->jacobian_soa = smesh::JacobianAdjugateAndDeterminant::create_SoA(
                        mesh, smesh::MEMORY_SPACE_HOST, block_id);
                if (!cache->jacobian_soa) {
                    return SFEM_FAILURE;
                }
                if ((impl_->gradient_uses_affine && %(gradient_affine_uses_jacobian_aos)s) ||
                    (impl_->apply_uses_affine && %(apply_affine_uses_jacobian_aos)s)) {
                    cache->jacobian_aos = smesh::JacobianAdjugateAndDeterminant::create_AoS(
                            mesh, smesh::MEMORY_SPACE_HOST, block_id);
                    if (!cache->jacobian_aos) {
                        return SFEM_FAILURE;
                    }
                }
                entry.second.user_data = std::static_pointer_cast<void>(cache);
            }
        }
        impl_->element_values.reset(new real_t[impl_->element_capacity]);
%(packed_scratch_prealloc)s
        return SFEM_SUCCESS;
    }

    int %(op)s::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::gradient");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *adjugate_aos = nullptr;
            const geom_t *determinant = nullptr;
%(metric_declaration)s            if (impl_->gradient_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache || !cache->jacobian_soa) {
                    SFEM_ERROR("%(op)s affine gradient requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        cache->jacobian_soa->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        cache->jacobian_soa->jacobian_determinant()->data());
                if (%(gradient_affine_uses_jacobian_aos)s) {
                    if (!cache->jacobian_aos) {
                        SFEM_ERROR("%(op)s affine gradient requires cached AoS geometry\\n");
                        return SFEM_FAILURE;
                    }
                    adjugate_aos = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_adjugate_AoS()->data());
                    determinant = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_determinant()->data());
                }
%(gradient_metric_binding)s            }
%(gradient_packed_dispatch_body)s
%(gradient_dispatch_body)s
        });
    }

    int %(op)s::apply(const real_t *const x,
                      const real_t *const h,
                      real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::apply");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *adjugate_aos = nullptr;
            const geom_t *determinant = nullptr;
%(metric_declaration)s            if (impl_->apply_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache || !cache->jacobian_soa) {
                    SFEM_ERROR("%(op)s affine hessian action requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        cache->jacobian_soa->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        cache->jacobian_soa->jacobian_determinant()->data());
                if (%(apply_affine_uses_jacobian_aos)s) {
                    if (!cache->jacobian_aos) {
                        SFEM_ERROR("%(op)s affine hessian action requires cached AoS geometry\\n");
                        return SFEM_FAILURE;
                    }
                    adjugate_aos = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_adjugate_AoS()->data());
                    determinant = reinterpret_cast<const geom_t *>(
                            cache->jacobian_aos->jacobian_determinant()->data());
                }
%(apply_metric_binding)s            }
%(apply_dispatch_body)s
        });
    }

    int %(op)s::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::value");
        // The objective is the 0-form at one step of length zero.  `value_steps`
        // evaluates at `x + alpha * h`, so alpha = 0 leaves the increment
        // unused and `x` itself can stand in for it -- `x + 0 * x` is `x`
        // exactly in IEEE arithmetic for any finite state, and the kernel then
        // calls the same block function the objective kernel called.
        //
        // Writing it this way is what keeps the two from disagreeing.  They
        // did: this method zeroed `*out` before accumulating while
        // `value_steps` only accumulated, so the same Op answered the same
        // question two ways depending on which entry point was used.  With one
        // implementation there is nothing left to diverge.
        const real_t objective_step = 0;
        *out = 0;
        return value_steps(x, x, 1, &objective_step, out);
    }

    int %(op)s::value_steps(const real_t *x,
                            const real_t *h,
                            const int nsteps,
                            const real_t *const steps,
                            real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::value_steps");
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        if (nsteps <= 0) {
            return SFEM_SUCCESS;
        }
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            const ptrdiff_t nvalues = (ptrdiff_t)nsteps * nelements;
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
%(metric_declaration)s            if (impl_->objective_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache || !cache->jacobian_soa) {
                    SFEM_ERROR("%(op)s affine objective_steps requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        cache->jacobian_soa->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        cache->jacobian_soa->jacobian_determinant()->data());
%(objective_steps_metric_binding)s            }
            if (nvalues > impl_->element_capacity) {
                impl_->element_values.reset(new real_t[nvalues]);
                impl_->element_capacity = nvalues;
            }
            std::fill(impl_->element_values.get(),
                      impl_->element_values.get() + nvalues,
                      real_t(0));
            int status = SFEM_FAILURE;
%(objective_steps_packed_dispatch_body)s
            if (status == SFEM_FAILURE) {
%(objective_steps_dispatch_body)s
            }
            if (status != SFEM_SUCCESS) return status;
            for (int step = 0; step < nsteps; ++step) {
                real_t sum = 0;
#pragma omp simd reduction(+ : sum)
                for (ptrdiff_t element = 0; element < nelements; ++element) {
                    sum += impl_->element_values[(ptrdiff_t)step * nelements + element];
                }
                out[step] += sum;
            }
            return SFEM_SUCCESS;
        });
    }

    int %(op)s::hessian_crs(const real_t *const x,
                            const count_t *const rowptr,
                            const idx_t *const colidx,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_crs");
%(hessian_crs_current_prologue)s
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
%(hessian_crs_dispatch_body)s
        });
    }

    int %(op)s::hessian_bsr(const real_t *const x,
                            const count_t *const rowptr,
                            const idx_t *const colidx,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_bsr");
%(hessian_bsr_current_prologue)s
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
%(hessian_bsr_dispatch_body)s
        });
    }

    int %(op)s::hessian_dia(const real_t *const x,
                            const int *const diag_offsets,
                            const ptrdiff_t ndiag,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_dia");
%(hessian_dia_current_prologue)s
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
%(hessian_dia_dispatch_body)s
        });
    }

    int %(op)s::hessian_coo(const real_t *const x,
                            const ptrdiff_t nnz,
                            const idx_t *const rows,
                            const idx_t *const cols,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_coo");
%(hessian_coo_current_prologue)s
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
%(hessian_coo_dispatch_body)s
        });
    }

    int %(op)s::hessian_patch(const real_t *const x,
                              const count_t *const rowptr,
                              const idx_t *const colidx,
                              real_t *const values) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_patch");
%(hessian_patch_current_prologue)s
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
%(hessian_patch_dispatch_body)s
        });
    }

    int %(op)s::hessian_block_diag_sym(const real_t *const x,
                                       real_t *const values) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_block_diag_sym");
%(hessian_block_diag_sym_current_prologue)s
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
%(hessian_block_diag_sym_dispatch_body)s
        });
    }

    void %(op)s::set_option(const std::string &name, const bool val) {
        SFEM_TRACE_SCOPE("%(op)s::set_option");
        if (name == "PACKED_TWO_PASS" || name == "two_pass") {
            impl_->use_packed_two_pass = val;
            return;
        }
        AffineOption options[] = {
%(affine_options)s
        };
        const bool matched = set_affine_option(name, val, options, sizeof(options) / sizeof(options[0]));
        if (matched && val && impl_->domains) {
            if (cache_affine_geometry(impl_->space, *impl_->domains) != SFEM_SUCCESS) {
                SFEM_ERROR("%(op)s failed to cache affine geometry\\n");
            }
        }
    }

    void %(op)s::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        SFEM_TRACE_SCOPE("%(op)s::set_value_in_block");
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> %(op)s::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
        SFEM_TRACE_SCOPE("%(op)s::create_from_yaml");
        auto ret = std::make_shared<%(op)s>(space);

        std::vector<std::string> block_names;
        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (block.has_child("name")) {
                    block_names.push_back(yaml_read_string(block["name"]));
                }
            }
        }

        AffineOption options[] = {
%(yaml_affine_options)s
        };
        read_affine_options(node, options, sizeof(options) / sizeof(options[0]));

        if (ret->initialize(block_names) != SFEM_SUCCESS) {
            return nullptr;
        }

        real_t defaults[N_MATERIAL_PARAMETERS];
        material_defaults(defaults);
        real_t top_values[N_MATERIAL_PARAMETERS];
        copy_material_parameters(defaults, top_values);
        if (material_from_yaml(node, defaults, top_values)) {
            set_material(*ret->impl_->domains, top_values);
        }

        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (!block.has_child("name")) {
                    continue;
                }

                real_t block_values[N_MATERIAL_PARAMETERS];
                copy_material_parameters(top_values, block_values);
                if (!material_from_yaml(block, top_values, block_values)) {
                    continue;
                }

                const std::string block_name = yaml_read_string(block["name"]);
                set_material_in_block(*ret->impl_->domains, block_name, block_values);
            }
        }

        return ret;
    }
#endif  // SFEM_ENABLE_RYAML
}  // namespace sfem
""" % {
        "op": material.op_name,
        "c_abi_include": '#include "%s"' % c_abi_header if c_abi_header else "",
        "packed_scratch_include": packed_scratch_include,
        "packed_scratch_prealloc": packed_scratch_prealloc,
        "declaration_block": (
            ""
            if c_abi_header
            else (
                'extern "C" {\n%s\n}' % "\n".join(declarations)
                if declarations
                else ""
            )
        ),
        "declarations": "\n".join(declarations),
        "defaults": defaults,
        "yaml_helpers": _yaml_helpers(material.parameter_defaults),
        "gradient_affine_uses_jacobian_aos": _cpp_bool(any(gradient_affine_aos_flags)),
        "apply_affine_uses_jacobian_aos": _cpp_bool(any(apply_affine_aos_flags)),
        "metric_cache_field": _metric_cache_field(any(affine_metric_flags)),
        "metric_cache_setup": _metric_cache_setup(any(affine_metric_flags)),
        "metric_declaration": _metric_declaration(any(affine_metric_flags)),
        "gradient_metric_binding": _metric_binding(
            any(affine_metric_flags), material.op_name, "gradient"
        ),
        "apply_metric_binding": _metric_binding(
            any(affine_metric_flags), material.op_name, "hessian action"
        ),
        "objective_steps_metric_binding": _metric_binding(
            any(affine_metric_flags), material.op_name, "objective_steps"
        ),
        "gradient_cases": "\n".join(gradient_cases),
        "apply_cases": "\n".join(apply_cases),
        "objective_cases": "\n".join(objective_cases),
        "objective_steps_cases": "\n".join(objective_steps_cases),
        "hessian_crs_cases": "\n".join(hessian_crs_cases),
        "hessian_bsr_cases": "\n".join(hessian_bsr_cases),
        "hessian_dia_cases": "\n".join(hessian_dia_cases),
        "hessian_coo_cases": "\n".join(hessian_coo_cases),
        "hessian_patch_cases": "\n".join(hessian_patch_cases),
        "apply_dispatch_body": _hyperelastic_apply_dispatch_body(
            material.name,
            kernel_sources,
            {dim: deps[2] for dim, deps in dependencies_by_dim.items()},
            indent="            ",
            n_field_components_by_dim=n_field_components_by_dim,
        ),
        "gradient_packed_dispatch_body": _hyperelastic_gradient_packed_dispatch_body(
            material.name,
            kernel_sources,
            {dim: deps[1] for dim, deps in dependencies_by_dim.items()},
            indent="            ",
            n_field_components_by_dim=n_field_components_by_dim,
        ),
        "gradient_dispatch_body": _hyperelastic_gradient_dispatch_body(
            material.name,
            kernel_sources,
            {dim: deps[1] for dim, deps in dependencies_by_dim.items()},
            indent="            ",
            n_field_components_by_dim=n_field_components_by_dim,
        ),
        "objective_dispatch_body": _hyperelastic_objective_dispatch_body(
            material.name,
            kernel_sources,
            {dim: deps[0] for dim, deps in dependencies_by_dim.items()},
            indent="            ",
            n_field_components_by_dim=n_field_components_by_dim,
        ),
        "objective_steps_packed_dispatch_body": _hyperelastic_objective_steps_packed_dispatch_body(
            material.name,
            kernel_sources,
            {dim: deps[0] for dim, deps in dependencies_by_dim.items()},
            indent="            ",
            n_field_components_by_dim=n_field_components_by_dim,
        ),
        "objective_steps_dispatch_body": _hyperelastic_objective_steps_dispatch_body(
            material.name,
            kernel_sources,
            {dim: deps[0] for dim, deps in dependencies_by_dim.items()},
            indent="                ",
            n_field_components_by_dim=n_field_components_by_dim,
        ),
        "hessian_crs_dispatch_body": _hyperelastic_hessian_dispatch_body(
            material.name,
            "hessian_crs",
            kernel_sources,
            {dim: deps[2] for dim, deps in dependencies_by_dim.items()},
            ("rowptr", "colidx", "values"),
            indent="            ",
        ),
        "hessian_crs_current_prologue": _hyperelastic_hessian_current_prologue(
            material.op_name,
            "hessian_crs",
            {dim: deps[2] for dim, deps in dependencies_by_dim.items()},
        ),
        "hessian_bsr_dispatch_body": _hyperelastic_hessian_dispatch_body(
            material.name,
            "hessian_bsr",
            kernel_sources,
            {dim: deps[2] for dim, deps in dependencies_by_dim.items()},
            ("rowptr", "colidx", "values"),
            indent="            ",
        ),
        "hessian_bsr_current_prologue": _hyperelastic_hessian_current_prologue(
            material.op_name,
            "hessian_bsr",
            {dim: deps[2] for dim, deps in dependencies_by_dim.items()},
        ),
        "hessian_dia_dispatch_body": _hyperelastic_hessian_dispatch_body(
            material.name,
            "hessian_dia",
            kernel_sources,
            {dim: deps[2] for dim, deps in dependencies_by_dim.items()},
            ("diag_offsets", "ndiag", "values"),
            indent="            ",
        ),
        "hessian_dia_current_prologue": _hyperelastic_hessian_current_prologue(
            material.op_name,
            "hessian_dia",
            {dim: deps[2] for dim, deps in dependencies_by_dim.items()},
        ),
        "hessian_coo_dispatch_body": _hyperelastic_hessian_dispatch_body(
            material.name,
            "hessian_coo",
            kernel_sources,
            {dim: deps[2] for dim, deps in dependencies_by_dim.items()},
            ("nnz", "rows", "cols", "values"),
            indent="            ",
        ),
        "hessian_coo_current_prologue": _hyperelastic_hessian_current_prologue(
            material.op_name,
            "hessian_coo",
            {dim: deps[2] for dim, deps in dependencies_by_dim.items()},
        ),
        "hessian_patch_dispatch_body": _hyperelastic_hessian_dispatch_body(
            material.name,
            "hessian_patch",
            kernel_sources,
            {dim: deps[2] for dim, deps in dependencies_by_dim.items()},
            ("rowptr", "colidx", "values"),
            indent="            ",
        ),
        "hessian_patch_current_prologue": _hyperelastic_hessian_current_prologue(
            material.op_name,
            "hessian_patch",
            {dim: deps[2] for dim, deps in dependencies_by_dim.items()},
        ),
        "hessian_block_diag_sym_dispatch_body": _hyperelastic_hessian_dispatch_body(
            material.name,
            "hessian_block_diag_sym",
            kernel_sources,
            {dim: deps[2] for dim, deps in dependencies_by_dim.items()},
            ("values",),
            indent="            ",
        ),
        "hessian_block_diag_sym_current_prologue": _hyperelastic_hessian_current_prologue(
            material.op_name,
            "hessian_block_diag_sym",
            {dim: deps[2] for dim, deps in dependencies_by_dim.items()},
        ),
        "inexact_cache_field": _inexact_cache_field(material),
        "performance_methods": _performance_methods(material.op_name, material.name, elements, performance_cases),
        "affine_options": _affine_option_entries(
            "objective_uses_affine",
            "gradient_uses_affine",
            "apply_uses_affine",
        ),
        "yaml_affine_options": _affine_option_entries(
            "objective_uses_affine",
            "gradient_uses_affine",
            "apply_uses_affine",
            owner="ret->impl_",
        ),
    }
    # The inexact methods are appended rather than woven into the template: they
    # are additive, reached only through their own entry points, and absent
    # entirely for a material that does not ask for them.
    inexact = _inexact_definitions(
        material,
        form_collections,
        elements,
        kernel_sources,
        apply_dependencies_by_dim={
            dim: deps[2] for dim, deps in dependencies_by_dim.items()
        },
        n_field_components_by_dim=n_field_components_by_dim,
    )
    if inexact:
        marker = "\n}  // namespace sfem"
        source = source.replace(
            marker,
            inexact % {"op": material.op_name} + marker,
            1,
        )
    return _header(material, False), source


def _residual_op(material, elements, c_abi_header=None, form_collections=None, kernel_sources=None):
    mixed_order = _uses_mixed_field_arrays(elements)
    if form_collections is None:
        raise ValueError("residual generated Op requires form collections")
    measures = {collection.measure for collection in form_collections.values()}
    if measures == {"ds"}:
        return _boundary_residual_op(material, elements, c_abi_header, form_collections)
    if "ds" in measures:
        raise ValueError("generated residual Op wrappers cannot mix dx and ds forms yet")

    # A residual whose 0-form is a merit can compute it here: it is half the
    # squared norm of what `gradient` already produces.  One whose 0-form is a
    # potential needs an element kernel this emitter cannot yet build, so it
    # keeps the failing stub rather than being handed a different quantity.
    emits_merit = _residual_zero_form_is_assembled_norm(form_collections)

    defaults = _seed_lines(material.parameter_defaults)
    declarations = []
    residual_cases = []
    action_cases = []
    hessian_crs_cases = []
    hessian_bsr_cases = []
    hessian_dia_cases = []
    performance_cases = {"value": [], "gradient": [], "apply": []}
    dependencies_by_dim = {}
    parameter_names_by_dim = {}
    fields_by_dim = {}
    block_size_by_dim = {}
    residual_affine_metric_flags = []
    action_affine_metric_flags = []
    residual_affine_metric_soa_flags = []
    action_affine_metric_soa_flags = []
    residual_affine_metric_aos_flags = []
    action_affine_metric_aos_flags = []
    residual_affine_metric_aos_elements_by_dim = {}
    action_affine_metric_aos_elements_by_dim = {}
    residual_affine_metric_aos_unit_elements_by_dim = {}
    action_affine_metric_aos_unit_elements_by_dim = {}
    for element in elements:
        dim = _element_dim(element)
        dependencies = dependencies_by_dim.get(dim)
        if dependencies is None:
            collection = form_collections[dim]
            dependencies = (
                collection.form_metadata(_form_order_one()).dependencies,
                collection.form_metadata(_form_order_two()).dependencies,
            )
            dependencies_by_dim[dim] = dependencies
            parameter_names_by_dim[dim] = tuple(str(symbol) for symbol in collection.parameters)
            fields_by_dim[dim] = tuple(collection.fields)
            block_size_by_dim[dim] = sum(int(field.components) for field in collection.fields)
        residual_dependencies, action_dependencies = dependencies
        parameter_index = {
            name: index for index, name in enumerate(parameter_names_by_dim[dim])
        }
        stem = "%s_%s" % (material.name, _element_name(element).lower())
        performance_cases["gradient"].append(
            _performance_case(
                element,
                ("%s_residual_element_soa_diagnostics" % stem,),
                affine_flags=("residual_uses_affine",),
            )
        )
        performance_cases["apply"].append(
            _performance_case(
                element,
                ("%s_jacobian_action_element_soa_diagnostics" % stem,),
                affine_flags=("jacobian_action_uses_affine",),
            )
        )
        residual_pointer_params = []
        if residual_dependencies.current:
            residual_pointer_params.append("const real_t *")
        if residual_dependencies.previous:
            residual_pointer_params.append("const real_t *")
        declarations.append(
            "int %s_residual_isoparametric_mesh_aos("
            "ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, "
            "const real_t *, %sreal_t *);"
            % (stem, "".join("%s, " % param for param in residual_pointer_params))
        )
        action_pointer_params = []
        if action_dependencies.current:
            action_pointer_params.append("const real_t *")
        if action_dependencies.previous:
            action_pointer_params.append("const real_t *")
        if action_dependencies.direction:
            action_pointer_params.append("const real_t *")
        declarations.append(
            "int %s_jacobian_action_isoparametric_mesh_aos("
            "ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, "
            "const real_t *, %sreal_t *);"
            % (stem, "".join("%s, " % param for param in action_pointer_params))
        )
        common_isoparametric = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), points"
        )
        residual_affine_uses_metric = _c_abi_function_uses_cached_metric(
            kernel_sources, "%s_residual_affine_mesh_soa" % stem
        )
        action_affine_uses_metric = _c_abi_function_uses_cached_metric(
            kernel_sources, "%s_jacobian_action_affine_mesh_soa" % stem
        )
        residual_affine_uses_metric_aos = _c_abi_function_exists(
            kernel_sources, "%s_residual_affine_mesh_soa_aos" % stem
        )
        action_affine_uses_metric_aos = _c_abi_function_exists(
            kernel_sources, "%s_jacobian_action_affine_mesh_soa_aos" % stem
        )
        residual_affine_uses_metric_aos_unit = _c_abi_function_exists(
            kernel_sources, "%s_residual_affine_mesh_soa_aos_unit" % stem
        )
        action_affine_uses_metric_aos_unit = _c_abi_function_exists(
            kernel_sources, "%s_jacobian_action_affine_mesh_soa_aos_unit" % stem
        )
        residual_affine_metric_flags.append(residual_affine_uses_metric)
        action_affine_metric_flags.append(action_affine_uses_metric)
        residual_affine_metric_soa_flags.append(
            residual_affine_uses_metric and not residual_affine_uses_metric_aos
        )
        action_affine_metric_soa_flags.append(
            action_affine_uses_metric and not action_affine_uses_metric_aos
        )
        residual_affine_metric_aos_flags.append(residual_affine_uses_metric_aos)
        action_affine_metric_aos_flags.append(action_affine_uses_metric_aos)
        mesh_element = _mesh_element_name(element)
        if residual_affine_uses_metric_aos:
            residual_affine_metric_aos_elements_by_dim.setdefault(dim, []).append(mesh_element)
        if action_affine_uses_metric_aos:
            action_affine_metric_aos_elements_by_dim.setdefault(dim, []).append(mesh_element)
        if residual_affine_uses_metric_aos_unit:
            residual_affine_metric_aos_unit_elements_by_dim.setdefault(dim, []).append(mesh_element)
        if action_affine_uses_metric_aos_unit:
            action_affine_metric_aos_unit_elements_by_dim.setdefault(dim, []).append(mesh_element)
        common_affine_residual = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), %s"
            % (
                _affine_metric_offsets(dim)
                if residual_affine_uses_metric
                else "%s, determinant" % _affine_geometry_offsets(dim)
            )
        )
        common_affine_residual_aos = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), geom_metric_aos"
        )
        common_affine_action = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), %s"
            % (
                _affine_metric_offsets(dim)
                if action_affine_uses_metric
                else "%s, determinant" % _affine_geometry_offsets(dim)
            )
        )
        common_affine_action_aos = (
            "domain.block->n_elements(), mesh->n_nodes(), "
            "domain.block->elements()->data(), geom_metric_aos"
        )
        residual_common_args = []
        residual_common_args.extend(
            _dependency_storage_args(residual_dependencies.parameters, parameter_index)
        )
        residual_setup = []
        if residual_dependencies.current:
            residual_setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    "state",
                    "data",
                    "const real_t",
                )
            )
            residual_common_args.append("FIELD_STRIDE")
            residual_common_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "data", mixed_order)
            )
        if residual_dependencies.previous:
            residual_setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    "previous",
                    "old_data",
                    "const real_t",
                )
            )
            residual_common_args.append("FIELD_STRIDE")
            residual_common_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "old_data", mixed_order)
            )
        residual_setup.extend(
            _residual_soa_view_declarations(
                fields_by_dim[dim],
                "out",
                "out",
                "real_t",
            )
        )
        residual_common_args.append("FIELD_STRIDE")
        residual_common_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "out", mixed_order))
        residual_unit_args = []
        if residual_dependencies.current:
            residual_unit_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "data", mixed_order)
            )
        residual_unit_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "out", mixed_order))
        residual_cases.append(
            _residual_dual_soa_case(
                element,
                "residual_uses_affine",
                (
                    "%s_residual_affine_mesh_soa_aos" % stem
                    if residual_affine_uses_metric_aos
                    else "%s_residual_affine_mesh_soa" % stem
                ),
                ", ".join(
                    (
                        common_affine_residual_aos
                        if residual_affine_uses_metric_aos
                        else common_affine_residual,
                        *residual_common_args,
                    )
                ),
                "%s_residual_isoparametric_mesh_soa" % stem,
                ", ".join((common_isoparametric, *residual_common_args)),
                block_size_by_dim[dim],
                residual_setup,
                affine_unit_function=(
                    "%s_residual_affine_mesh_soa_aos_unit" % stem
                    if residual_affine_uses_metric_aos_unit
                    else None
                ),
                affine_unit_arguments=(
                    ", ".join((common_affine_residual_aos, *residual_unit_args))
                    if residual_affine_uses_metric_aos_unit
                    else None
                ),
                affine_unit_condition=(
                    "storage[0] == real_t(1)"
                    if residual_affine_uses_metric_aos_unit
                    else None
                ),
            )
        )
        action_common_args = []
        action_common_args.extend(
            _dependency_storage_args(action_dependencies.parameters, parameter_index)
        )
        action_setup = []
        if action_dependencies.current:
            action_setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    "current",
                    "data",
                    "const real_t",
                )
            )
            action_common_args.append("FIELD_STRIDE")
            action_common_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "data", mixed_order)
            )
        if action_dependencies.previous:
            action_setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    "previous",
                    "old_data",
                    "const real_t",
                )
            )
            action_common_args.append("FIELD_STRIDE")
            action_common_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "old_data", mixed_order)
            )
        if action_dependencies.direction:
            action_setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    "direction",
                    "direction_data",
                    "const real_t",
                )
            )
            action_common_args.append("FIELD_STRIDE")
            action_common_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "direction_data", mixed_order)
            )
        action_setup.extend(
            _residual_soa_view_declarations(
                fields_by_dim[dim],
                "out",
                "out",
                "real_t",
            )
        )
        action_common_args.append("FIELD_STRIDE")
        action_common_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "out", mixed_order))
        action_unit_args = []
        if action_dependencies.direction:
            action_unit_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "direction_data", mixed_order)
            )
        action_unit_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "out", mixed_order))
        action_cases.append(
            _residual_dual_soa_case(
                element,
                "jacobian_action_uses_affine",
                (
                    "%s_jacobian_action_affine_mesh_soa_aos" % stem
                    if action_affine_uses_metric_aos
                    else "%s_jacobian_action_affine_mesh_soa" % stem
                ),
                ", ".join(
                    (
                        common_affine_action_aos
                        if action_affine_uses_metric_aos
                        else common_affine_action,
                        *action_common_args,
                    )
                ),
                "%s_jacobian_action_isoparametric_mesh_soa" % stem,
                ", ".join((common_isoparametric, *action_common_args)),
                block_size_by_dim[dim],
                action_setup,
                affine_unit_function=(
                    "%s_jacobian_action_affine_mesh_soa_aos_unit" % stem
                    if action_affine_uses_metric_aos_unit
                    else None
                ),
                affine_unit_arguments=(
                    ", ".join((common_affine_action_aos, *action_unit_args))
                    if action_affine_uses_metric_aos_unit
                    else None
                ),
                affine_unit_condition=(
                    "storage[0] == real_t(1)"
                    if action_affine_uses_metric_aos_unit
                    else None
                ),
            )
        )
        hessian_common_args = []
        hessian_common_args.extend(
            _dependency_storage_args(action_dependencies.parameters, parameter_index)
        )
        hessian_setup = []
        if action_dependencies.current:
            hessian_setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    "current",
                    "data",
                    "const real_t",
                )
            )
            hessian_common_args.append("FIELD_STRIDE")
            hessian_common_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "data", mixed_order)
            )
        if action_dependencies.previous:
            hessian_setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    "previous",
                    "old_data",
                    "const real_t",
                )
            )
            hessian_common_args.append("FIELD_STRIDE")
            hessian_common_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "old_data", mixed_order)
            )
        hessian_crs_function = "%s_hessian_crs_isoparametric_mesh_soa" % stem
        if _c_abi_function_exists(kernel_sources, hessian_crs_function):
            hessian_crs_cases.append(
                _residual_soa_case(
                    element,
                    hessian_crs_function,
                    ", ".join(
                        (
                            common_isoparametric,
                            *hessian_common_args,
                            "rowptr",
                            "colidx",
                            "values",
                        )
                    ),
                    block_size_by_dim[dim],
                    hessian_setup,
                )
            )
        hessian_bsr_function = "%s_hessian_bsr_isoparametric_mesh_soa" % stem
        if _c_abi_function_exists(kernel_sources, hessian_bsr_function):
            hessian_bsr_cases.append(
                _residual_soa_case(
                    element,
                    hessian_bsr_function,
                    ", ".join(
                        (
                            common_isoparametric,
                            *hessian_common_args,
                            "rowptr",
                            "colidx",
                            "values",
                        )
                    ),
                    block_size_by_dim[dim],
                    hessian_setup,
                )
            )
        hessian_dia_function = "%s_hessian_dia_isoparametric_mesh_soa" % stem
        if _c_abi_function_exists(kernel_sources, hessian_dia_function):
            hessian_dia_cases.append(
                _case(
                    element,
                    hessian_dia_function,
                    ", ".join(
                        (
                            common_isoparametric,
                            *_dependency_storage_args(
                                action_dependencies.parameters,
                                parameter_index,
                            ),
                            "diag_offsets",
                            "ndiag",
                            "values",
                        )
                    ),
                )
            )

    residual_uses_previous = any(
        dependencies[0].previous for dependencies in dependencies_by_dim.values()
    )
    action_uses_current = any(
        dependencies[1].current for dependencies in dependencies_by_dim.values()
    )
    action_uses_previous = any(
        dependencies[1].previous for dependencies in dependencies_by_dim.values()
    )
    hessian_state_alias = (
        "        const real_t *const current = state ? state : impl_->current;"
        if action_uses_current
        else ""
    )
    hessian_state_check = (
        "        if (%s) {\n"
        '            SFEM_ERROR("%s requires %s\\n");\n'
        "            return SFEM_FAILURE;\n"
        "        }"
        % (
            " || ".join(
                condition
                for condition in (
                    "!current" if action_uses_current else "",
                    "!impl_->previous" if action_uses_previous else "",
                )
                if condition
            ),
            material.op_name,
            (
                "current and previous states"
                if action_uses_current and action_uses_previous
                else (
                    "a current state"
                    if action_uses_current
                    else "a previous state"
                )
            ),
        )
        if action_uses_current or action_uses_previous
        else ""
    )
    hessian_previous_alias = (
        "            const real_t *const previous = impl_->previous;"
        if action_uses_previous
        else ""
    )
    residual_affine_uses_metric = any(residual_affine_metric_flags)
    action_affine_uses_metric = any(action_affine_metric_flags)
    residual_affine_uses_metric_soa = any(residual_affine_metric_soa_flags)
    action_affine_uses_metric_soa = any(action_affine_metric_soa_flags)
    action_packed_affine_uses_metric_soa = any(
        _c_abi_function_uses_cached_metric(
            kernel_sources,
            "%s_jacobian_action_packed_%dd_affine_mesh_soa" % (material.name, dim),
        )
        for dim in (2, 3)
    )
    laplace_tet4_packed_affine_uses_metric_soa = (
        material.name == "laplace"
        and _c_abi_function_exists(
            kernel_sources,
            "laplace_tet4_jacobian_action_packed_affine_mesh_soa",
        )
    )
    laplace_proteus_hex8_packed_metric = (
        material.name == "laplace"
        and _c_abi_function_exists(
            kernel_sources,
            "laplace_proteus_hex8_private_metric_jacobian_action_packed_mesh_soa",
        )
    )
    laplace_tet10_packed_metric = (
        material.name == "laplace"
        and _c_abi_function_exists(
            kernel_sources,
            "laplace_tet10_private_metric_jacobian_action_packed_mesh_soa",
        )
    )
    action_affine_uses_metric_soa = (
        action_affine_uses_metric_soa
        or action_packed_affine_uses_metric_soa
        or laplace_tet4_packed_affine_uses_metric_soa
    )
    residual_affine_uses_metric_aos = any(residual_affine_metric_aos_flags)
    action_affine_uses_metric_aos = (
        any(action_affine_metric_aos_flags)
        or laplace_proteus_hex8_packed_metric
        or laplace_tet10_packed_metric
    )
    residual_affine_uses_jacobian = not all(residual_affine_metric_flags)
    action_affine_uses_jacobian = not all(action_affine_metric_flags)

    max_parameters = max(
        1,
        len(material.parameter_defaults),
        *(len(names) for names in parameter_names_by_dim.values()),
    )
    parameter_lines = _residual_parameter_array_lines(parameter_names_by_dim)
    laplace_packed_apply = material.name == "laplace"
    generated_packed_apply = any("_jacobian_action_packed_" in source for source in kernel_sources.values())
    use_laplace_packed_fast_path = laplace_packed_apply and not generated_packed_apply
    laplace_packed_include = '#include "sfem_PackedLaplacian.hpp"' if use_laplace_packed_fast_path else ""
    packed_scratch_include = '#include "packed_thread_scratch.hpp"' if generated_packed_apply else ""
    packed_scratch_prealloc = (
        """        if (impl_->space->has_packed_mesh()) {
            auto packed = impl_->space->packed_mesh();
            const ptrdiff_t max_nodes_per_pack = packed->max_nodes_per_pack();
            const int dim = impl_->space->mesh_ptr()->spatial_dimension();
            sfem::codegen::prealloc_thread_scratch<real_t>(
                    0, (size_t)dim * (size_t)max_nodes_per_pack);
            sfem::codegen::prealloc_thread_scratch<real_t>(
                    1, (size_t)max_nodes_per_pack);
            sfem::codegen::prealloc_thread_scratch<real_t>(
                    2, (size_t)max_nodes_per_pack);
            sfem::codegen::prealloc_thread_scratch<real_t>(
                    3, (size_t)max_nodes_per_pack);
        }"""
        if generated_packed_apply
        else ""
    )
    laplace_packed_helpers = (
        """
        bool packed_laplacian_apply_supported(const smesh::ElemType element_type) {
            switch (element_type) {
                case smesh::TET4:
                case smesh::TET10:
                case smesh::HEX8:
                    return true;
                default:
                    return false;
            }
        }

        bool can_use_packed_laplacian_apply(const FunctionSpace &space,
                                            MultiDomainOp &domains) {
            if (!space.has_packed_mesh()) {
                return false;
            }

            for (auto &entry : domains.domains()) {
                const OpDomain &domain = entry.second;
                if (!packed_laplacian_apply_supported(domain.element_type)) {
                    return false;
                }
                if (domain.parameters->require_real_value("kappa") != real_t(1)) {
                    return false;
                }
            }

            return true;
        }
"""
        if laplace_packed_apply
        and use_laplace_packed_fast_path
        else ""
    )
    laplace_packed_member = (
        "        std::shared_ptr<Op> packed_affine_apply;"
        if use_laplace_packed_fast_path
        else ""
    )
    laplace_packed_apply_fast_path = (
        """
        if (impl_->jacobian_action_uses_affine &&
            can_use_packed_laplacian_apply(*impl_->space, *impl_->domains)) {
            if (!impl_->packed_affine_apply) {
                impl_->packed_affine_apply = std::make_shared<PackedLaplacian>(impl_->space);
                if (impl_->packed_affine_apply->initialize() != SFEM_SUCCESS) {
                    SFEM_ERROR("%s failed to initialize packed affine apply backend\\n");
                    return SFEM_FAILURE;
                }
            }
            return impl_->packed_affine_apply->apply(current, direction, out);
        }
"""
        % material.op_name
        if use_laplace_packed_fast_path
        else ""
    )
    private_declarations = []
    if laplace_tet4_packed_affine_uses_metric_soa:
        private_declarations.append(
            """int laplace_tet4_jacobian_action_packed_affine_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const geom_t *const SFEM_RESTRICT g_geom_metric3,
        const geom_t *const SFEM_RESTRICT g_geom_metric4,
        const geom_t *const SFEM_RESTRICT g_geom_metric5,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);"""
        )
    for private_name in (
        "laplace_proteus_hex8_private_metric_jacobian_action_packed_mesh_soa",
        "laplace_tet10_private_metric_jacobian_action_packed_mesh_soa",
    ):
        if material.name == "laplace" and _c_abi_function_exists(kernel_sources, private_name):
            private_declarations.append(
                """int %s(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);"""
                % private_name
            )
    if c_abi_header:
        declaration_block = (
            'extern "C" {\n%s\n}' % "\n".join(private_declarations)
            if private_declarations
            else ""
        )
    else:
        declaration_block = 'extern "C" {\n%s\n}' % "\n".join(
            [*declarations, *private_declarations]
        )
    source = """#include "sfem_%(op)s.hpp"
%(c_abi_include)s
%(laplace_packed_include)s
%(packed_scratch_include)s

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_kernel_data.hpp"
#include "smesh_mesh.hpp"

#include <cstring>%(merit_include)s

%(declaration_block)s

namespace sfem {
    namespace {
        constexpr int MAX_PARAMETERS = %(max_parameters)d;

        void seed_parameters(Parameters &parameters) {
%(defaults)s
        }

        void seed_material(MultiDomainOp &domains) {
            for (auto &entry : domains.domains()) {
                seed_parameters(*entry.second.parameters);
            }
        }

%(yaml_helpers)s

        smesh::block_idx_t block_id_for_domain(const smesh::Mesh &mesh,
                                               const smesh::Mesh::Block &block) {
            for (size_t i = 0; i < mesh.n_blocks(); ++i) {
                if (mesh.block(i).get() == &block) {
                    return static_cast<smesh::block_idx_t>(i);
                }
            }
            SFEM_ERROR("%(op)s: mesh block pointer not found in mesh.blocks()\\n");
            return 0;
        }

        int packed_block_id_for_domain(const FunctionSpace::PackedMesh &packed,
                                       const smesh::Mesh::Block &block) {
            for (ptrdiff_t i = 0; i < packed.n_blocks(); ++i) {
                if (packed.block_name(i) == block.name()) {
                    return static_cast<int>(i);
                }
            }
            return -1;
        }

        struct AffineGeometryCache {
            std::shared_ptr<smesh::JacobianAdjugateAndDeterminant> jacobian;
            std::shared_ptr<smesh::FFF> metric_soa;
            std::shared_ptr<smesh::FFF> metric_aos;
        };

        int cache_affine_geometry(const std::shared_ptr<FunctionSpace> &space,
                                  MultiDomainOp &domains,
                                  const bool needs_jacobian,
                                  const bool needs_metric_soa,
                                  const bool needs_metric_aos) {
            auto mesh = space->mesh_ptr();
            for (auto &entry : domains.domains()) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        entry.second.user_data);
                if (!cache) {
                    cache = std::make_shared<AffineGeometryCache>();
                }
                const smesh::block_idx_t block_id =
                        block_id_for_domain(*mesh, *entry.second.block);
                if (needs_jacobian && !cache->jacobian) {
                    cache->jacobian = smesh::JacobianAdjugateAndDeterminant::create_SoA(
                            mesh, smesh::MEMORY_SPACE_HOST, block_id);
                    if (!cache->jacobian) {
                        return SFEM_FAILURE;
                    }
                }
                if (needs_metric_soa && !cache->metric_soa) {
                    cache->metric_soa = smesh::FFF::create_SoA(
                            mesh, smesh::MEMORY_SPACE_HOST, block_id);
                    if (!cache->metric_soa) {
                        return SFEM_FAILURE;
                    }
                }
                if (needs_metric_aos && !cache->metric_aos) {
                    cache->metric_aos = smesh::FFF::create_AoS(
                            mesh, smesh::MEMORY_SPACE_HOST, block_id);
                    if (!cache->metric_aos) {
                        return SFEM_FAILURE;
                    }
                }
                entry.second.user_data = std::static_pointer_cast<void>(cache);
            }
            return SFEM_SUCCESS;
        }

        void parameter_array(const Parameters &parameters,
                             const int dim,
                             real_t *const values) {
            int index = 0;
%(parameter_lines)s
        }

        ptrdiff_t block_size_for_dim(const int dim) {
%(block_size_lines)s
        }
%(laplace_packed_helpers)s
    }  // namespace

    class %(op)s::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
%(laplace_packed_member)s
        std::shared_ptr<Buffer<real_t>> previous_buffer;
        const real_t *previous{nullptr};
        const real_t *current{nullptr};
        bool residual_uses_affine{false};
        bool jacobian_action_uses_affine{false};
    };

    std::unique_ptr<Op> %(op)s::create(const std::shared_ptr<FunctionSpace> &space) {
        const ptrdiff_t expected_block_size =
                block_size_for_dim(space->mesh_ptr()->spatial_dimension());
        if (space->block_size() != expected_block_size) {
            SFEM_ERROR("%(op)s requires block_size=%%ld\\n",
                       static_cast<long>(expected_block_size));
            return nullptr;
        }
        auto op = std::make_unique<%(op)s>(space);
        op->initialize();
        return op;
    }

    %(op)s::%(op)s(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    %(op)s::~%(op)s() = default;

    ptrdiff_t %(op)s::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t %(op)s::n_dofs_image() const { return impl_->space->n_dofs(); }

%(performance_methods)s

    // Establish once, at setup, that this operator's dof graph is well formed:
    // rows in order, every column in range, each row sorted and duplicate free.
    // The assembly kernels assume it -- they locate an entry and write to it
    // without re-checking that it is there -- so this is where the assumption
    // is earned.
    //
    // It used to be earned per element instead: every scatter walked its
    // N_SHAPE x N_SHAPE candidates, tested each with a three-condition branch
    // and reported through std::fprintf from inside the caller's parallel
    // region.  That paid O(elements x N_SHAPE^2) on every assembly for a
    // property of the mesh and the graph together, which cannot change between
    // elements or between calls.  Here it is O(nnz), once.
    //
    // Raw pointers rather than the graph type, so this does not depend on which
    // headers the generated wrapper happens to pull in.
    static int validate_dof_graph(const count_t *const rowptr,
                                  const idx_t *const colidx,
                                  const ptrdiff_t n_nodes,
                                  const ptrdiff_t nnz) {
        if (!rowptr || !colidx || n_nodes < 0) {
            return SFEM_FAILURE;
        }
        if (rowptr[0] != 0 || (ptrdiff_t)rowptr[n_nodes] != nnz) {
            return SFEM_FAILURE;
        }
        for (ptrdiff_t i = 0; i < n_nodes; ++i) {
            const count_t begin = rowptr[i];
            const count_t end = rowptr[i + 1];
            if (end < begin || (ptrdiff_t)end > nnz) {
                return SFEM_FAILURE;
            }
            for (count_t k = begin; k < end; ++k) {
                if (colidx[k] < 0 || (ptrdiff_t)colidx[k] >= n_nodes) {
                    return SFEM_FAILURE;
                }
                if (k > begin && colidx[k] <= colidx[k - 1]) {
                    return SFEM_FAILURE;
                }
            }
        }
        return SFEM_SUCCESS;
    }

    int %(op)s::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("%(op)s::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        {
            auto dof_graph = impl_->space->dof_to_dof_graph();
            if (!dof_graph ||
                validate_dof_graph(dof_graph->rowptr()->data(),
                                   dof_graph->colidx()->data(),
                                   dof_graph->n_nodes(),
                                   dof_graph->nnz()) != SFEM_SUCCESS) {
                SFEM_ERROR("%(op)s::initialize: the dof graph is malformed; the assembly kernels assume it is not\\n");
                return SFEM_FAILURE;
            }
        }
        seed_material(*impl_->domains);
        const bool needs_affine_jacobian =
                (impl_->residual_uses_affine && %(residual_affine_uses_jacobian)s) ||
                (impl_->jacobian_action_uses_affine && %(action_affine_uses_jacobian)s);
        const bool needs_affine_metric =
                (impl_->residual_uses_affine && (%(residual_affine_uses_metric_soa)s || %(residual_affine_uses_metric_aos)s)) ||
                (impl_->jacobian_action_uses_affine && (%(action_affine_uses_metric_soa)s || %(action_affine_uses_metric_aos)s));
        const bool needs_affine_metric_soa =
                (impl_->residual_uses_affine && %(residual_affine_uses_metric_soa)s) ||
                (impl_->jacobian_action_uses_affine && %(action_affine_uses_metric_soa)s);
        const bool needs_affine_metric_aos =
                (impl_->residual_uses_affine && %(residual_affine_uses_metric_aos)s) ||
                (impl_->jacobian_action_uses_affine && %(action_affine_uses_metric_aos)s);
        if (needs_affine_jacobian || needs_affine_metric) {
            const int status = cache_affine_geometry(impl_->space,
                                                     *impl_->domains,
                                                     needs_affine_jacobian,
                                                     needs_affine_metric_soa,
                                                     needs_affine_metric_aos);
            if (status != SFEM_SUCCESS) return status;
        }
%(packed_scratch_prealloc)s
        return SFEM_SUCCESS;
    }

    int %(op)s::update(const real_t *const x) {
        SFEM_TRACE_SCOPE("%(op)s::update");
        impl_->current = x;
        return SFEM_SUCCESS;
    }

    int %(op)s::update(const real_t *const previous,
                       const real_t *const current) {
        SFEM_TRACE_SCOPE("%(op)s::update");
        impl_->previous_buffer.reset();
        impl_->previous = previous;
        impl_->current = current;
        return SFEM_SUCCESS;
    }

    int %(op)s::gradient(const real_t *const state, real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::gradient");
%(gradient_previous_check)s
        impl_->current = state;
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            const geom_t *const *geom_metric = nullptr;
            const geom_t *geom_metric_aos = nullptr;
            if (impl_->residual_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache) {
                    SFEM_ERROR("%(op)s affine residual requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                if (%(residual_affine_uses_jacobian)s) {
                    if (!cache->jacobian) {
                        SFEM_ERROR("%(op)s affine residual requires cached jacobian geometry\\n");
                        return SFEM_FAILURE;
                    }
                    adjugate = reinterpret_cast<const geom_t *const *>(
                            cache->jacobian->jacobian_adjugate_SoA()->data());
                    determinant = reinterpret_cast<const geom_t *>(
                            cache->jacobian->jacobian_determinant()->data());
                }
                if (%(residual_affine_uses_metric_soa)s) {
                    if (!cache->metric_soa) {
                        SFEM_ERROR("%(op)s affine residual requires cached SoA metric geometry\\n");
                        return SFEM_FAILURE;
                    }
                    geom_metric = reinterpret_cast<const geom_t *const *>(
                            cache->metric_soa->fff_SoA()->data());
                }
                if (%(residual_affine_uses_metric_aos)s) {
                    if (!cache->metric_aos) {
                        SFEM_ERROR("%(op)s affine residual requires cached AoS metric geometry\\n");
                        return SFEM_FAILURE;
                    }
                    geom_metric_aos = reinterpret_cast<const geom_t *>(
                            cache->metric_aos->fff_AoS()->data());
                }
            }
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters,
                            mesh->spatial_dimension(),
                            storage);
%(gradient_previous_alias)s
%(residual_dispatch_body)s
        });
    }

    int %(op)s::apply(const real_t *const state,
                      const real_t *const direction,
                      real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::apply");
        const real_t *const current = state ? state : impl_->current;
%(apply_state_check)s
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
%(laplace_packed_apply_fast_path)s
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            const geom_t *const *geom_metric = nullptr;
            const geom_t *geom_metric_aos = nullptr;
            if (impl_->jacobian_action_uses_affine) {
                auto cache = std::static_pointer_cast<AffineGeometryCache>(
                        domain.user_data);
                if (!cache) {
                    SFEM_ERROR("%(op)s affine jacobian action requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                if (%(action_affine_uses_jacobian)s) {
                    if (!cache->jacobian) {
                        SFEM_ERROR("%(op)s affine jacobian action requires cached jacobian geometry\\n");
                        return SFEM_FAILURE;
                    }
                    adjugate = reinterpret_cast<const geom_t *const *>(
                            cache->jacobian->jacobian_adjugate_SoA()->data());
                    determinant = reinterpret_cast<const geom_t *>(
                            cache->jacobian->jacobian_determinant()->data());
                }
                if (%(action_affine_uses_metric_soa)s) {
                    if (!cache->metric_soa) {
                        SFEM_ERROR("%(op)s affine jacobian action requires cached SoA metric geometry\\n");
                        return SFEM_FAILURE;
                    }
                    geom_metric = reinterpret_cast<const geom_t *const *>(
                            cache->metric_soa->fff_SoA()->data());
                }
                if (%(action_affine_uses_metric_aos)s) {
                    if (!cache->metric_aos) {
                        SFEM_ERROR("%(op)s affine jacobian action requires cached AoS metric geometry\\n");
                        return SFEM_FAILURE;
                    }
                    geom_metric_aos = reinterpret_cast<const geom_t *>(
                            cache->metric_aos->fff_AoS()->data());
                }
            }
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters,
                            mesh->spatial_dimension(),
                            storage);
%(apply_previous_alias)s
%(action_dispatch_body)s
        });
    }

    void %(op)s::set_field(const char *name,
                           const std::shared_ptr<Buffer<real_t>> &values,
                           const int component) {
        SFEM_TRACE_SCOPE("%(op)s::set_field");
        if (component != 0 || std::strcmp(name, "previous") != 0) {
            SFEM_ERROR("%(op)s supports set_field(\\"previous\\", buffer, 0)\\n");
            return;
        }
        impl_->previous_buffer = values;
        impl_->previous = values->data();
    }

    void %(op)s::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        SFEM_TRACE_SCOPE("%(op)s::set_value_in_block");
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void %(op)s::set_option(const std::string &name, const bool val) {
        SFEM_TRACE_SCOPE("%(op)s::set_option");
        AffineOption options[] = {
%(affine_options)s
        };
        const bool matched = set_affine_option(name, val, options, sizeof(options) / sizeof(options[0]));
        if (matched && val && impl_->domains) {
            const bool needs_affine_jacobian =
                    (impl_->residual_uses_affine && %(residual_affine_uses_jacobian)s) ||
                    (impl_->jacobian_action_uses_affine && %(action_affine_uses_jacobian)s);
            const bool needs_affine_metric =
                    (impl_->residual_uses_affine && (%(residual_affine_uses_metric_soa)s || %(residual_affine_uses_metric_aos)s)) ||
                    (impl_->jacobian_action_uses_affine && (%(action_affine_uses_metric_soa)s || %(action_affine_uses_metric_aos)s));
            const bool needs_affine_metric_soa =
                    (impl_->residual_uses_affine && %(residual_affine_uses_metric_soa)s) ||
                    (impl_->jacobian_action_uses_affine && %(action_affine_uses_metric_soa)s);
            const bool needs_affine_metric_aos =
                    (impl_->residual_uses_affine && %(residual_affine_uses_metric_aos)s) ||
                    (impl_->jacobian_action_uses_affine && %(action_affine_uses_metric_aos)s);
            if (cache_affine_geometry(impl_->space,
                                      *impl_->domains,
                                      needs_affine_jacobian,
                                      needs_affine_metric_soa,
                                      needs_affine_metric_aos) != SFEM_SUCCESS) {
                SFEM_ERROR("%(op)s failed to cache affine geometry\\n");
            }
        }
    }

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> %(op)s::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
        SFEM_TRACE_SCOPE("%(op)s::create_from_yaml");
        auto ret = std::make_shared<%(op)s>(space);

        std::vector<std::string> block_names;
        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (block.has_child("name")) {
                    block_names.push_back(yaml_read_string(block["name"]));
                }
            }
        }

        AffineOption options[] = {
%(yaml_affine_options)s
        };
        read_affine_options(node, options, sizeof(options) / sizeof(options[0]));

        if (ret->initialize(block_names) != SFEM_SUCCESS) {
            return nullptr;
        }

        real_t defaults[N_MATERIAL_PARAMETERS];
        material_defaults(defaults);
        real_t top_values[N_MATERIAL_PARAMETERS];
        copy_material_parameters(defaults, top_values);
        if (material_from_yaml(node, defaults, top_values)) {
            set_material(*ret->impl_->domains, top_values);
        }

        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (!block.has_child("name")) {
                    continue;
                }

                real_t block_values[N_MATERIAL_PARAMETERS];
                copy_material_parameters(top_values, block_values);
                if (!material_from_yaml(block, top_values, block_values)) {
                    continue;
                }

                const std::string block_name = yaml_read_string(block["name"]);
                set_material_in_block(*ret->impl_->domains, block_name, block_values);
            }
        }

        return ret;
    }
#endif  // SFEM_ENABLE_RYAML

    int %(op)s::hessian_crs(const real_t *const state,
                            const count_t *const rowptr,
                            const idx_t *const colidx,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_crs");
%(hessian_crs_body)s
    }

    int %(op)s::hessian_bsr(const real_t *const state,
                            const count_t *const rowptr,
                            const idx_t *const colidx,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_bsr");
%(hessian_bsr_body)s
    }

    int %(op)s::hessian_dia(const real_t *const state,
                            const int *const diag_offsets,
                            const ptrdiff_t ndiag,
                            real_t *const values) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_dia");
%(hessian_dia_body)s
    }

%(merit_methods)s}  // namespace sfem
""" % {
        "op": material.op_name,
        "c_abi_include": '#include "%s"' % c_abi_header if c_abi_header else "",
        "laplace_packed_include": laplace_packed_include,
        "packed_scratch_include": packed_scratch_include,
        "packed_scratch_prealloc": packed_scratch_prealloc,
        "declaration_block": (
            declaration_block
        ),
        "declarations": "\n".join(declarations),
        "max_parameters": max_parameters,
        "defaults": defaults,
        "yaml_helpers": _yaml_helpers(material.parameter_defaults),
        "parameter_lines": parameter_lines,
        "block_size_lines": _residual_block_size_lines(block_size_by_dim),
        "laplace_packed_helpers": laplace_packed_helpers,
        "laplace_packed_member": laplace_packed_member,
        "laplace_packed_apply_fast_path": laplace_packed_apply_fast_path,
        "inexact_cache_field": _inexact_cache_field(material),
        "performance_methods": _performance_methods(material.op_name, material.name, elements, performance_cases),
        # Only the merit uses std::vector, so only the merit brings its header.
        "merit_include": "\n#include <vector>" if emits_merit else "",
        "merit_methods": (
            _residual_merit_methods(material.op_name)
            if emits_merit
            else """    int %s::value(const real_t *, real_t *const) {
        SFEM_TRACE_SCOPE("%s::value");
        return SFEM_FAILURE;
    }
""" % (material.op_name, material.op_name)
        ),
        "residual_cases": "\n".join(residual_cases),
        "action_cases": "\n".join(action_cases),
        "residual_dispatch_body": _residual_apply_dispatch_body(
            material.name,
            "residual",
            "residual_uses_affine",
            "state",
            kernel_sources,
            {dim: deps[0] for dim, deps in dependencies_by_dim.items()},
            parameter_names_by_dim,
            fields_by_dim,
            block_size_by_dim,
            residual_affine_metric_aos_elements_by_dim,
            residual_affine_metric_aos_unit_elements_by_dim,
            "            ",
                    mixed_order=mixed_order,
                ),
        "action_dispatch_body": _residual_apply_dispatch_body(
            material.name,
            "jacobian_action",
            "jacobian_action_uses_affine",
            "current",
            kernel_sources,
            {dim: deps[1] for dim, deps in dependencies_by_dim.items()},
            parameter_names_by_dim,
            fields_by_dim,
            block_size_by_dim,
            action_affine_metric_aos_elements_by_dim,
            action_affine_metric_aos_unit_elements_by_dim,
            "            ",
                    mixed_order=mixed_order,
                ),
        "hessian_crs_body": (
            "%s\n"
            "%s\n"
            "        auto mesh = impl_->space->mesh_ptr();\n"
            "        auto points = const_cast<const geom_t *const *>(mesh->points()->data());\n"
            "        return impl_->domains->iterate([&](const OpDomain &domain) {\n"
            "            real_t storage[MAX_PARAMETERS];\n"
            "            parameter_array(*domain.parameters,\n"
            "                            mesh->spatial_dimension(),\n"
            "                            storage);\n"
            "%s\n"
            "%s\n"
            "        });"
            % (
                hessian_state_alias,
                hessian_state_check,
                hessian_previous_alias,
                _residual_hessian_dispatch_body(
                    material.name,
                    "hessian_crs",
                    kernel_sources,
                    {dim: deps[1] for dim, deps in dependencies_by_dim.items()},
                    parameter_names_by_dim,
                    fields_by_dim,
                    block_size_by_dim,
                    ("rowptr", "colidx", "values"),
                    "            ",
                    mixed_order=mixed_order,
                ),
            )
            if hessian_crs_cases
            else "        return SFEM_FAILURE;"
        ),
        "hessian_bsr_body": (
            "%s\n"
            "%s\n"
            "        auto mesh = impl_->space->mesh_ptr();\n"
            "        auto points = const_cast<const geom_t *const *>(mesh->points()->data());\n"
            "        return impl_->domains->iterate([&](const OpDomain &domain) {\n"
            "            real_t storage[MAX_PARAMETERS];\n"
            "            parameter_array(*domain.parameters,\n"
            "                            mesh->spatial_dimension(),\n"
            "                            storage);\n"
            "%s\n"
            "%s\n"
            "        });"
            % (
                hessian_state_alias,
                hessian_state_check,
                hessian_previous_alias,
                _residual_hessian_dispatch_body(
                    material.name,
                    "hessian_bsr",
                    kernel_sources,
                    {dim: deps[1] for dim, deps in dependencies_by_dim.items()},
                    parameter_names_by_dim,
                    fields_by_dim,
                    block_size_by_dim,
                    ("rowptr", "colidx", "values"),
                    "            ",
                    mixed_order=mixed_order,
                ),
            )
            if hessian_bsr_cases
            else "        return SFEM_FAILURE;"
        ),
        "hessian_dia_body": (
            "%s\n"
            "%s\n"
            "        auto mesh = impl_->space->mesh_ptr();\n"
            "        auto points = const_cast<const geom_t *const *>(mesh->points()->data());\n"
            "        return impl_->domains->iterate([&](const OpDomain &domain) {\n"
            "            real_t storage[MAX_PARAMETERS];\n"
            "            parameter_array(*domain.parameters,\n"
            "                            mesh->spatial_dimension(),\n"
            "                            storage);\n"
            "%s\n"
            "%s\n"
            "        });"
            % (
                hessian_state_alias,
                hessian_state_check,
                hessian_previous_alias,
                _residual_hessian_dispatch_body(
                    material.name,
                    "hessian_dia",
                    kernel_sources,
                    {dim: deps[1] for dim, deps in dependencies_by_dim.items()},
                    parameter_names_by_dim,
                    fields_by_dim,
                    block_size_by_dim,
                    ("diag_offsets", "ndiag", "values"),
                    "            ",
                    mixed_order=mixed_order,
                ),
            )
            if hessian_dia_cases
            else "        return SFEM_FAILURE;"
        ),
        "affine_options": _affine_option_entries(
            "residual_uses_affine",
            "jacobian_action_uses_affine",
        ),
        "residual_affine_uses_jacobian": _cpp_bool(residual_affine_uses_jacobian),
        "action_affine_uses_jacobian": _cpp_bool(action_affine_uses_jacobian),
        "residual_affine_uses_metric": _cpp_bool(residual_affine_uses_metric),
        "action_affine_uses_metric": _cpp_bool(action_affine_uses_metric),
        "residual_affine_uses_metric_soa": _cpp_bool(residual_affine_uses_metric_soa),
        "action_affine_uses_metric_soa": _cpp_bool(action_affine_uses_metric_soa),
        "residual_affine_uses_metric_aos": _cpp_bool(residual_affine_uses_metric_aos),
        "action_affine_uses_metric_aos": _cpp_bool(action_affine_uses_metric_aos),
        "yaml_affine_options": _affine_option_entries(
            "residual_uses_affine",
            "jacobian_action_uses_affine",
            owner="ret->impl_",
        ),
        "gradient_previous_check": (
            "        if (!impl_->previous) {\n"
            '            SFEM_ERROR("%s requires a previous state\\n");\n'
            "            return SFEM_FAILURE;\n"
            "        }" % material.op_name
            if residual_uses_previous
            else ""
        ),
        "gradient_previous_alias": (
            "            const real_t *const previous = impl_->previous;"
            if residual_uses_previous
            else ""
        ),
        "apply_state_check": (
            "        if (%s) {\n"
            '            SFEM_ERROR("%s requires %s\\n");\n'
            "            return SFEM_FAILURE;\n"
            "        }"
            % (
                " || ".join(
                    condition
                    for condition in (
                        "!current" if action_uses_current else "",
                        "!impl_->previous" if action_uses_previous else "",
                    )
                    if condition
                ),
                material.op_name,
                (
                    "current and previous states"
                    if action_uses_current and action_uses_previous
                    else (
                        "a current state"
                        if action_uses_current
                        else "a previous state"
                    )
                ),
            )
            if action_uses_current or action_uses_previous
            else ""
        ),
        "apply_previous_alias": (
            "            const real_t *const previous = impl_->previous;"
            if action_uses_previous
            else ""
        ),
    }
    return _header(material, True, publishes_value_steps=emits_merit), source


def _boundary_residual_op(material, elements, c_abi_header=None, form_collections=None):
    defaults = _seed_lines(material.parameter_defaults)
    if form_collections is None:
        raise ValueError("boundary residual generated Op requires form collections")

    material_parameter_index = {
        str(name): index
        for index, (name, _) in enumerate(material.parameter_defaults)
    }
    material_parameter_names = set(material_parameter_index)
    parameter_names_by_dim = {}
    fields_by_dim = {}
    block_size_by_dim = {}
    for dim, collection in form_collections.items():
        if collection.measure != "ds":
            raise ValueError("boundary residual generated Op requires ds measure")
        fields = tuple(collection.fields)
        if len(fields) != 1:
            raise ValueError("boundary residual generated Op currently supports one field")
        parameter_names_by_dim[dim] = _boundary_residual_parameter_names(
            collection, material_parameter_names
        )
        fields_by_dim[dim] = fields
        block_size_by_dim[dim] = sum(int(field.components) for field in fields)

    max_parameters = max(1, len(material.parameter_defaults))
    parameter_lines = _residual_parameter_array_lines(parameter_names_by_dim)
    gradient_cases = []
    for element in elements:
        dim = _element_dim(element)
        fields = fields_by_dim[dim]
        block_size = block_size_by_dim[dim]
        stem = "%s_%s_boundary_residual_%dd_sideset_soa" % (
            material.name,
            _boundary_surface_name(element),
            dim,
        )
        setup = _residual_soa_view_declarations(fields, "out", "out", "real_t")
        parameter_args = ", ".join(
            "condition.values->data()[%d]" % material_parameter_index[name]
            for name in parameter_names_by_dim[dim]
        )
        output_args = _boundary_soa_component_argument_names(fields, "out")
        call_args = _nonempty(
            "domain.element_type",
            "real_type",
            "sideset->size()",
            "mesh->n_nodes()",
            "domain.block->elements()->data()",
            "sideset->parent()->data()",
            "sideset->lfi()->data()",
            "points",
            parameter_args,
            "FIELD_STRIDE",
            *output_args,
        )
        gradient_cases.append(
            _boundary_residual_soa_case(
                element,
                stem,
                ", ".join(call_args),
                block_size,
                setup,
            )
        )

    source = """#include "sfem_%(op)s.hpp"
%(c_abi_include)s

#include "sfem_aliases.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_NeumannConditions.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_mesh.hpp"
#include "smesh_sideset.hpp"

#include <cstring>
#include <memory>
#include <vector>

%(declaration_block)s

namespace sfem {
    namespace {
        constexpr int MAX_PARAMETERS = %(max_parameters)d;

        void seed_parameters(Parameters &parameters) {
%(defaults)s
        }

        void seed_material(MultiDomainOp &domains) {
            for (auto &entry : domains.domains()) {
                seed_parameters(*entry.second.parameters);
            }
        }

%(yaml_helpers)s

        void parameter_array(const Parameters &parameters,
                             const int dim,
                             real_t *const values) {
            int index = 0;
%(parameter_lines)s
        }

        ptrdiff_t block_size_for_dim(const int dim) {
%(block_size_lines)s
        }

        smesh::block_idx_t block_id_for_domain(const smesh::Mesh &mesh,
                                               const smesh::Mesh::Block &block) {
            for (size_t i = 0; i < mesh.n_blocks(); ++i) {
                if (mesh.block(i).get() == &block) {
                    return static_cast<smesh::block_idx_t>(i);
                }
            }
            SFEM_ERROR("%(op)s: mesh block pointer not found in mesh.blocks()\\n");
            return 0;
        }

#ifdef SFEM_ENABLE_RYAML
        std::shared_ptr<smesh::Sideset> sideset_from_yaml(
                const std::shared_ptr<FunctionSpace> &space,
                const ryml::ConstNodeRef             &node) {
            const bool is_sideset = node["type"].readable() && node["type"].val() == "sideset";
            const bool is_file    = node["format"].readable() && node["format"].val() == "file";
            const bool is_expr    = node["format"].readable() && node["format"].val() == "expr";

            if (!is_sideset && node.has_child("type")) {
                SFEM_ERROR("%(op)s neumann condition requires type=sideset\\n");
                return nullptr;
            }

            if (is_file || node.has_child("path")) {
                if (!node.has_child("path")) {
                    SFEM_ERROR("%(op)s file sideset condition requires path\\n");
                    return nullptr;
                }
                const std::string path = yaml_read_string(node["path"]);
                return smesh::Sideset::create_from_file(
                        space->mesh_ptr()->comm(), smesh::Path(path));
            }

            if (is_expr || (node.has_child("parent") && node.has_child("lfi"))) {
                if (!node["parent"].is_seq() || !node["lfi"].is_seq()) {
                    SFEM_ERROR("%(op)s expr sideset condition requires parent/lfi sequences\\n");
                    return nullptr;
                }

                const ptrdiff_t size = node["parent"].num_children();
                if (node["lfi"].num_children() != size) {
                    SFEM_ERROR("%(op)s expr sideset parent/lfi length mismatch\\n");
                    return nullptr;
                }

                auto parent = create_host_buffer<element_idx_t>(size);
                auto lfi    = create_host_buffer<int16_t>(size);

                ptrdiff_t parent_count = 0;
                for (auto p : node["parent"].children()) {
                    p >> parent->data()[parent_count++];
                }

                ptrdiff_t lfi_count = 0;
                for (auto p : node["lfi"].children()) {
                    p >> lfi->data()[lfi_count++];
                }

                return std::make_shared<smesh::Sideset>(
                        space->mesh_ptr()->comm(), parent, lfi);
            }

            SFEM_ERROR("%(op)s neumann condition requires format=file or format=expr\\n");
            return nullptr;
        }
#endif  // SFEM_ENABLE_RYAML
    }  // namespace

    class %(op)s::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::vector<NeumannConditions::Condition> conditions;
    };

    std::unique_ptr<Op> %(op)s::create(const std::shared_ptr<FunctionSpace> &space) {
        const ptrdiff_t expected_block_size =
                block_size_for_dim(space->mesh_ptr()->spatial_dimension());
        if (space->block_size() != expected_block_size) {
            SFEM_ERROR("%(op)s requires block_size=%%ld\\n",
                       static_cast<long>(expected_block_size));
            return nullptr;
        }
        auto op = std::make_unique<%(op)s>(space);
        op->initialize();
        return op;
    }

    %(op)s::%(op)s(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    %(op)s::~%(op)s() = default;

    ptrdiff_t %(op)s::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t %(op)s::n_dofs_image() const { return impl_->space->n_dofs(); }

%(performance_methods)s

    // Establish once, at setup, that this operator's dof graph is well formed:
    // rows in order, every column in range, each row sorted and duplicate free.
    // The assembly kernels assume it -- they locate an entry and write to it
    // without re-checking that it is there -- so this is where the assumption
    // is earned.
    //
    // It used to be earned per element instead: every scatter walked its
    // N_SHAPE x N_SHAPE candidates, tested each with a three-condition branch
    // and reported through std::fprintf from inside the caller's parallel
    // region.  That paid O(elements x N_SHAPE^2) on every assembly for a
    // property of the mesh and the graph together, which cannot change between
    // elements or between calls.  Here it is O(nnz), once.
    //
    // Raw pointers rather than the graph type, so this does not depend on which
    // headers the generated wrapper happens to pull in.
    static int validate_dof_graph(const count_t *const rowptr,
                                  const idx_t *const colidx,
                                  const ptrdiff_t n_nodes,
                                  const ptrdiff_t nnz) {
        if (!rowptr || !colidx || n_nodes < 0) {
            return SFEM_FAILURE;
        }
        if (rowptr[0] != 0 || (ptrdiff_t)rowptr[n_nodes] != nnz) {
            return SFEM_FAILURE;
        }
        for (ptrdiff_t i = 0; i < n_nodes; ++i) {
            const count_t begin = rowptr[i];
            const count_t end = rowptr[i + 1];
            if (end < begin || (ptrdiff_t)end > nnz) {
                return SFEM_FAILURE;
            }
            for (count_t k = begin; k < end; ++k) {
                if (colidx[k] < 0 || (ptrdiff_t)colidx[k] >= n_nodes) {
                    return SFEM_FAILURE;
                }
                if (k > begin && colidx[k] <= colidx[k - 1]) {
                    return SFEM_FAILURE;
                }
            }
        }
        return SFEM_SUCCESS;
    }

    int %(op)s::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("%(op)s::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        {
            auto dof_graph = impl_->space->dof_to_dof_graph();
            if (!dof_graph ||
                validate_dof_graph(dof_graph->rowptr()->data(),
                                   dof_graph->colidx()->data(),
                                   dof_graph->n_nodes(),
                                   dof_graph->nnz()) != SFEM_SUCCESS) {
                SFEM_ERROR("%(op)s::initialize: the dof graph is malformed; the assembly kernels assume it is not\\n");
                return SFEM_FAILURE;
            }
        }
        seed_material(*impl_->domains);
        return SFEM_SUCCESS;
    }

    void %(op)s::add_sideset(const std::shared_ptr<smesh::Sideset> &sideset) {
        real_t values[MAX_PARAMETERS];
        material_defaults(values);
        add_sideset(sideset, values);
    }

    void %(op)s::add_sideset(const std::shared_ptr<smesh::Sideset> &sideset,
                             const real_t *const parameters) {
        SFEM_TRACE_SCOPE("%(op)s::add_sideset");
        NeumannConditions::Condition condition;
        condition.sidesets = {sideset};
        condition.values = create_host_buffer<real_t>(MAX_PARAMETERS);
        for (int i = 0; i < MAX_PARAMETERS; ++i) {
            condition.values->data()[i] = parameters[i];
        }
        condition.value = parameters[0];
        condition.component = 0;
        add_condition(condition);
    }

    void %(op)s::add_condition(const NeumannConditions::Condition &condition) {
        SFEM_TRACE_SCOPE("%(op)s::add_condition");
        impl_->conditions.push_back(condition);
    }

    int %(op)s::gradient(const real_t *const, real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::gradient");
        if (impl_->conditions.empty()) {
            return SFEM_SUCCESS;
        }
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const smesh::block_idx_t block_id = block_id_for_domain(*mesh, *domain.block);
            int status = SFEM_SUCCESS;
            for (const auto &condition : impl_->conditions) {
                const auto sideset = condition.sidesets.empty() ? nullptr : condition.sidesets[0];
                if (!sideset || !condition.values || sideset->block_id() != block_id) {
                    continue;
                }
                switch (domain.element_type) {
%(gradient_cases)s
                    default:
                        SFEM_ERROR("%(op)s does not support element type %%d\\n",
                                   domain.element_type);
                        return SFEM_FAILURE;
                }
            }
            return status;
        });
    }

    int %(op)s::apply(const real_t *const,
                      const real_t *const,
                      real_t *const) {
        SFEM_TRACE_SCOPE("%(op)s::apply");
        return SFEM_SUCCESS;
    }

    int %(op)s::value(const real_t *, real_t *const) {
        SFEM_TRACE_SCOPE("%(op)s::value");
        return SFEM_SUCCESS;
    }

    int %(op)s::hessian_crs(const real_t *const,
                            const count_t *const,
                            const idx_t *const,
                            real_t *const) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_crs");
        return SFEM_SUCCESS;
    }

    void %(op)s::set_field(const char *,
                           const std::shared_ptr<Buffer<real_t>> &,
                           const int) {
        SFEM_TRACE_SCOPE("%(op)s::set_field");
    }

    void %(op)s::set_option(const std::string &, const bool) {
        SFEM_TRACE_SCOPE("%(op)s::set_option");
    }

    void %(op)s::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        SFEM_TRACE_SCOPE("%(op)s::set_value_in_block");
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> %(op)s::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
        SFEM_TRACE_SCOPE("%(op)s::create_from_yaml");
        auto ret = std::make_shared<%(op)s>(space);

        std::vector<std::string> block_names;
        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (block.has_child("name")) {
                    block_names.push_back(yaml_read_string(block["name"]));
                }
            }
        }

        if (ret->initialize(block_names) != SFEM_SUCCESS) {
            return nullptr;
        }

        real_t defaults[MAX_PARAMETERS];
        material_defaults(defaults);
        real_t top_values[MAX_PARAMETERS];
        copy_material_parameters(defaults, top_values);
        material_from_yaml(node, defaults, top_values);

        const auto neumann_node =
                node.has_child("neumann_conditions") ? node["neumann_conditions"] :
                 ryml::ConstNodeRef();
        if (neumann_node.readable() && neumann_node.is_seq()) {
            for (auto condition_node : neumann_node.children()) {
                auto sideset = sideset_from_yaml(space, condition_node);
                if (!sideset) {
                    return nullptr;
                }
                real_t condition_values[MAX_PARAMETERS];
                material_from_yaml(condition_node, top_values, condition_values);
                ret->add_sideset(sideset, condition_values);
            }
        }

        return ret;
    }
#endif  // SFEM_ENABLE_RYAML
}  // namespace sfem
""" % {
        "op": material.op_name,
        "c_abi_include": '#include "%s"' % c_abi_header if c_abi_header else "",
        "declaration_block": "",
        "max_parameters": max_parameters,
        "defaults": defaults,
        "yaml_helpers": _yaml_helpers(material.parameter_defaults),
        "parameter_lines": parameter_lines,
        "block_size_lines": _residual_block_size_lines(block_size_by_dim),
        "performance_methods": _performance_methods(material.op_name, material.name, elements, {}),
        "gradient_cases": "\n".join(gradient_cases),
    }
    return _boundary_header(material), source


def _boundary_header(material):
    return """#pragma once

#include "sfem_NeumannConditions.hpp"
#include "sfem_Op.hpp"

namespace smesh {
    class Sideset;
}

namespace sfem {
    class %(op)s final : public Op {
    public:
        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);

        explicit %(op)s(const std::shared_ptr<FunctionSpace> &space);
        ~%(op)s() override;

        const char *name() const override { return "%(op)s"; }
        bool is_linear() const override { return true; }
        ptrdiff_t n_dofs_domain() const override;
        ptrdiff_t n_dofs_image() const override;
        double flops_value() const override;
        double flops_gradient() const override;
        double flops_apply() const override;
        size_t memory_traffic_bytes_value() const override;
        size_t memory_traffic_bytes_gradient() const override;
        size_t memory_traffic_bytes_apply() const override;

        int initialize(const std::vector<std::string> &block_names = {}) override;
        void add_condition(const NeumannConditions::Condition &condition);
        void add_sideset(const std::shared_ptr<smesh::Sideset> &sideset);
        void add_sideset(const std::shared_ptr<smesh::Sideset> &sideset,
                         const real_t *parameters);
        int gradient(const real_t *const x, real_t *const out) override;
        int apply(const real_t *const x,
                  const real_t *const h,
                  real_t *const out) override;
        int value(const real_t *x, real_t *const out) override;
        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override;
        void set_field(const char *name,
                       const std::shared_ptr<Buffer<real_t>> &values,
                       int component) override;
        void set_option(const std::string &name, bool val) override;
        void set_value_in_block(const std::string &block_name,
                                const std::string &var_name,
                                real_t value) override;
#ifdef SFEM_ENABLE_RYAML
        std::shared_ptr<Op> create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                             const ryml::ConstNodeRef             &node) override;
#endif  // SFEM_ENABLE_RYAML

        //! The scalar type the kernels are asked for at run time.
        //!
        //! Mirrors GPULaplacian, which declares the same member with the same
        //! default and hands it to every kernel call.  SMESH_DEFAULT resolves
        //! to the build's real_t, so the default costs a caller nothing and is
        //! the common path rather than a fallback.  The Op interface itself is
        //! unchanged: its methods still take real_t*, which converts to void*
        //! at the call, exactly as gpu_laplacian_block_vector relies on.
        enum smesh::PrimitiveType real_type{smesh::SMESH_DEFAULT};

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem
""" % {"op": material.op_name}


def _coupled_energy_residual_op(
    material,
    elements,
    c_abi_header=None,
    systems_by_dim=None,
    kernel_sources=None,
):
    if systems_by_dim is None:
        systems_by_dim = _systems_by_dim(material, elements)
    equations_by_dim = {
        dim: tuple(system.equations)
        for dim, system in systems_by_dim.items()
    }
    representative = next(iter(equations_by_dim.values()))
    energy_equations = tuple(equation for equation in representative if equation.is_energy)
    residual_equations = tuple(equation for equation in representative if equation.is_residual)
    if len(energy_equations) != 1 or len(residual_equations) != 1:
        raise ValueError(
            "generated coupled Op wrappers currently require one energy and one residual equation"
        )

    energy_name = energy_equations[0].name
    residual_name = residual_equations[0].name
    if not energy_name or not residual_name:
        raise ValueError("coupled generated Op equations must be named")

    defaults = _seed_lines(material.parameter_defaults)
    parameter_index = {
        str(name): index
        for index, (name, _) in enumerate(material.parameter_defaults)
    }
    cases = _coupled_cases(
        material,
        elements,
        systems_by_dim,
        energy_name,
        residual_name,
        parameter_index,
        kernel_sources or {},
    )
    dependency_flags = _coupled_dependency_flags(
        systems_by_dim,
        energy_name,
        residual_name,
    )
    declarations = _extract_c_abi_declarations(kernel_sources or {}, public_only=False)
    max_parameters = max(1, len(material.parameter_defaults))
    source = """#include "sfem_%(op)s.hpp"
%(c_abi_include)s

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_kernel_data.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

%(declaration_block)s

namespace sfem {
    namespace {
        constexpr int MAX_PARAMETERS = %(max_parameters)d;

        void seed_parameters(Parameters &parameters) {
%(defaults)s
        }

        void seed_material(MultiDomainOp &domains) {
            for (auto &entry : domains.domains()) {
                seed_parameters(*entry.second.parameters);
            }
        }

%(yaml_helpers)s

        smesh::block_idx_t block_id_for_domain(const smesh::Mesh &mesh,
                                               const smesh::Mesh::Block &block) {
            for (size_t i = 0; i < mesh.n_blocks(); ++i) {
                if (mesh.block(i).get() == &block) {
                    return static_cast<smesh::block_idx_t>(i);
                }
            }
            SFEM_ERROR("%(op)s: mesh block pointer not found in mesh.blocks()\\n");
            return 0;
        }

        int cache_affine_geometry(const std::shared_ptr<FunctionSpace> &space,
                                  MultiDomainOp &domains) {
            auto mesh = space->mesh_ptr();
            for (auto &entry : domains.domains()) {
                if (entry.second.user_data) {
                    continue;
                }
                const smesh::block_idx_t block_id =
                        block_id_for_domain(*mesh, *entry.second.block);
                auto jacobian = smesh::JacobianAdjugateAndDeterminant::create_SoA(
                        mesh, smesh::MEMORY_SPACE_HOST, block_id);
                if (!jacobian) {
                    return SFEM_FAILURE;
                }
                entry.second.user_data = std::static_pointer_cast<void>(jacobian);
            }
            return SFEM_SUCCESS;
        }

        void parameter_array(const Parameters &parameters,
                             real_t *const values) {
%(parameter_lines)s
        }

        ptrdiff_t block_size_for_dim(const int dim) {
            switch (dim) {
%(block_size_lines)s
                default:
                    SFEM_ERROR("unsupported spatial dimension %%d for generated coupled block size\\n", dim);
                    return 0;
            }
        }
    }  // namespace

    class %(op)s::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::shared_ptr<Buffer<real_t>> previous_buffer;
        std::unique_ptr<real_t[]> element_values;
        ptrdiff_t element_capacity{0};
        const real_t *previous{nullptr};
        const real_t *current{nullptr};
        bool objective_uses_affine{false};
        bool gradient_uses_affine{false};
        bool apply_uses_affine{false};
        bool residual_uses_affine{false};
        bool jacobian_action_uses_affine{false};
    };

    std::unique_ptr<Op> %(op)s::create(const std::shared_ptr<FunctionSpace> &space) {
        const ptrdiff_t expected_block_size =
                block_size_for_dim(space->mesh_ptr()->spatial_dimension());
        if (space->block_size() != expected_block_size) {
            SFEM_ERROR("%(op)s requires block_size=%%ld\\n",
                       static_cast<long>(expected_block_size));
            return nullptr;
        }
        auto op = std::make_unique<%(op)s>(space);
        op->initialize();
        return op;
    }

    %(op)s::%(op)s(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    %(op)s::~%(op)s() = default;

    ptrdiff_t %(op)s::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t %(op)s::n_dofs_image() const { return impl_->space->n_dofs(); }

%(performance_methods)s

    // Establish once, at setup, that this operator's dof graph is well formed:
    // rows in order, every column in range, each row sorted and duplicate free.
    // The assembly kernels assume it -- they locate an entry and write to it
    // without re-checking that it is there -- so this is where the assumption
    // is earned.
    //
    // It used to be earned per element instead: every scatter walked its
    // N_SHAPE x N_SHAPE candidates, tested each with a three-condition branch
    // and reported through std::fprintf from inside the caller's parallel
    // region.  That paid O(elements x N_SHAPE^2) on every assembly for a
    // property of the mesh and the graph together, which cannot change between
    // elements or between calls.  Here it is O(nnz), once.
    //
    // Raw pointers rather than the graph type, so this does not depend on which
    // headers the generated wrapper happens to pull in.
    static int validate_dof_graph(const count_t *const rowptr,
                                  const idx_t *const colidx,
                                  const ptrdiff_t n_nodes,
                                  const ptrdiff_t nnz) {
        if (!rowptr || !colidx || n_nodes < 0) {
            return SFEM_FAILURE;
        }
        if (rowptr[0] != 0 || (ptrdiff_t)rowptr[n_nodes] != nnz) {
            return SFEM_FAILURE;
        }
        for (ptrdiff_t i = 0; i < n_nodes; ++i) {
            const count_t begin = rowptr[i];
            const count_t end = rowptr[i + 1];
            if (end < begin || (ptrdiff_t)end > nnz) {
                return SFEM_FAILURE;
            }
            for (count_t k = begin; k < end; ++k) {
                if (colidx[k] < 0 || (ptrdiff_t)colidx[k] >= n_nodes) {
                    return SFEM_FAILURE;
                }
                if (k > begin && colidx[k] <= colidx[k - 1]) {
                    return SFEM_FAILURE;
                }
            }
        }
        return SFEM_SUCCESS;
    }

    int %(op)s::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("%(op)s::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        {
            auto dof_graph = impl_->space->dof_to_dof_graph();
            if (!dof_graph ||
                validate_dof_graph(dof_graph->rowptr()->data(),
                                   dof_graph->colidx()->data(),
                                   dof_graph->n_nodes(),
                                   dof_graph->nnz()) != SFEM_SUCCESS) {
                SFEM_ERROR("%(op)s::initialize: the dof graph is malformed; the assembly kernels assume it is not\\n");
                return SFEM_FAILURE;
            }
        }
        seed_material(*impl_->domains);
        auto mesh = impl_->space->mesh_ptr();
        const bool needs_affine_geometry =
                impl_->objective_uses_affine ||
                impl_->gradient_uses_affine ||
                impl_->apply_uses_affine ||
                impl_->residual_uses_affine ||
                impl_->jacobian_action_uses_affine;
        for (auto &entry : impl_->domains->domains()) {
            impl_->element_capacity =
                    std::max(impl_->element_capacity, entry.second.block->n_elements());
            if (needs_affine_geometry) {
                const smesh::block_idx_t block_id =
                        block_id_for_domain(*mesh, *entry.second.block);
                auto jacobian = smesh::JacobianAdjugateAndDeterminant::create_SoA(
                        mesh, smesh::MEMORY_SPACE_HOST, block_id);
                if (!jacobian) {
                    return SFEM_FAILURE;
                }
                entry.second.user_data = std::static_pointer_cast<void>(jacobian);
            }
        }
        impl_->element_values.reset(new real_t[impl_->element_capacity]);
        return SFEM_SUCCESS;
    }

    int %(op)s::update(const real_t *const x) {
        SFEM_TRACE_SCOPE("%(op)s::update");
        impl_->current = x;
        return SFEM_SUCCESS;
    }

    int %(op)s::update(const real_t *const previous,
                       const real_t *const current) {
        SFEM_TRACE_SCOPE("%(op)s::update");
        impl_->previous_buffer.reset();
        impl_->previous = previous;
        impl_->current = current;
        return SFEM_SUCCESS;
    }

    int %(op)s::gradient(const real_t *const state, real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::gradient");
%(gradient_previous_check)s
        impl_->current = state;
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->gradient_uses_affine || impl_->residual_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("%(op)s affine gradient/residual requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        jacobian->jacobian_determinant()->data());
            }
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters, storage);
%(gradient_previous_alias)s
            switch (domain.element_type) {
%(gradient_cases)s
                default:
                    SFEM_ERROR("%(op)s does not support element type %%d\\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

    int %(op)s::apply(const real_t *const state,
                      const real_t *const direction,
                      real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::apply");
        const real_t *const current = state ? state : impl_->current;
%(apply_state_check)s
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const geom_t *const *adjugate = nullptr;
            const geom_t *determinant = nullptr;
            if (impl_->apply_uses_affine || impl_->jacobian_action_uses_affine) {
                auto jacobian = std::static_pointer_cast<smesh::JacobianAdjugateAndDeterminant>(
                        domain.user_data);
                if (!jacobian) {
                    SFEM_ERROR("%(op)s affine hessian/jacobian action requires cached geometry\\n");
                    return SFEM_FAILURE;
                }
                adjugate = reinterpret_cast<const geom_t *const *>(
                        jacobian->jacobian_adjugate_SoA()->data());
                determinant = reinterpret_cast<const geom_t *>(
                        jacobian->jacobian_determinant()->data());
            }
            real_t storage[MAX_PARAMETERS];
            parameter_array(*domain.parameters, storage);
%(apply_previous_alias)s
            switch (domain.element_type) {
%(apply_cases)s
                default:
                    SFEM_ERROR("%(op)s does not support element type %%d\\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

%(value_steps_method)s
    void %(op)s::set_field(const char *name,
                           const std::shared_ptr<Buffer<real_t>> &values,
                           const int component) {
        SFEM_TRACE_SCOPE("%(op)s::set_field");
        if (component != 0 || std::strcmp(name, "previous") != 0) {
            SFEM_ERROR("%(op)s supports set_field(\\"previous\\", buffer, 0)\\n");
            return;
        }
        impl_->previous_buffer = values;
        impl_->previous = values->data();
    }

    void %(op)s::set_option(const std::string &name, const bool val) {
        SFEM_TRACE_SCOPE("%(op)s::set_option");
        AffineOption options[] = {
%(affine_options)s
        };
        const bool matched = set_affine_option(name, val, options, sizeof(options) / sizeof(options[0]));
        if (matched && val && impl_->domains &&
            cache_affine_geometry(impl_->space, *impl_->domains) != SFEM_SUCCESS) {
            SFEM_ERROR("%(op)s failed to cache affine geometry\\n");
        }
    }

    void %(op)s::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        SFEM_TRACE_SCOPE("%(op)s::set_value_in_block");
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> %(op)s::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                 const ryml::ConstNodeRef             &node) {
        SFEM_TRACE_SCOPE("%(op)s::create_from_yaml");
        auto ret = std::make_shared<%(op)s>(space);

        std::vector<std::string> block_names;
        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (block.has_child("name")) {
                    block_names.push_back(yaml_read_string(block["name"]));
                }
            }
        }

        AffineOption options[] = {
%(yaml_affine_options)s
        };
        read_affine_options(node, options, sizeof(options) / sizeof(options[0]));

        if (ret->initialize(block_names) != SFEM_SUCCESS) {
            return nullptr;
        }

        real_t defaults[N_MATERIAL_PARAMETERS];
        material_defaults(defaults);
        real_t top_values[N_MATERIAL_PARAMETERS];
        copy_material_parameters(defaults, top_values);
        if (material_from_yaml(node, defaults, top_values)) {
            set_material(*ret->impl_->domains, top_values);
        }

        if (node.has_child("blocks")) {
            for (auto block : node["blocks"].children()) {
                if (!block.has_child("name")) {
                    continue;
                }

                real_t block_values[N_MATERIAL_PARAMETERS];
                copy_material_parameters(top_values, block_values);
                if (!material_from_yaml(block, top_values, block_values)) {
                    continue;
                }

                const std::string block_name = yaml_read_string(block["name"]);
                set_material_in_block(*ret->impl_->domains, block_name, block_values);
            }
        }

        return ret;
    }
#endif  // SFEM_ENABLE_RYAML

    int %(op)s::hessian_crs(const real_t *const,
                            const count_t *const,
                            const idx_t *const,
                            real_t *const) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_crs");
        return SFEM_FAILURE;
    }

    int %(op)s::hessian_bsr(const real_t *const,
                            const count_t *const,
                            const idx_t *const,
                            real_t *const) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_bsr");
        return SFEM_FAILURE;
    }

    int %(op)s::hessian_dia(const real_t *const,
                            const int *const,
                            const ptrdiff_t,
                            real_t *const) {
        SFEM_TRACE_SCOPE("%(op)s::hessian_dia");
        return SFEM_FAILURE;
    }
}  // namespace sfem
""" % {
        "op": material.op_name,
        "c_abi_include": '#include "%s"' % c_abi_header if c_abi_header else "",
        "declaration_block": "" if c_abi_header else "\n".join(declarations),
        "max_parameters": max_parameters,
        "defaults": defaults,
        "yaml_helpers": _yaml_helpers(material.parameter_defaults),
        "parameter_lines": _coupled_parameter_array_lines(material.parameter_defaults),
        "block_size_lines": _coupled_block_size_lines(systems_by_dim),
        "performance_methods": _performance_methods(material.op_name, material.name, elements, cases["performance"]),
        "gradient_previous_check": (
            "        if (!impl_->previous) {\n"
            '            SFEM_ERROR("%s requires a previous state\\n");\n'
            "            return SFEM_FAILURE;\n"
            "        }" % material.op_name
            if dependency_flags["gradient_previous"]
            else ""
        ),
        "gradient_previous_alias": (
            "            const real_t *const previous = impl_->previous;"
            if dependency_flags["gradient_previous"]
            else ""
        ),
        "apply_state_check": _coupled_apply_state_check(
            material.op_name,
            dependency_flags["apply_current"],
            dependency_flags["apply_previous"],
        ),
        "apply_previous_alias": (
            "            const real_t *const previous = impl_->previous;"
            if dependency_flags["apply_previous"]
            else ""
        ),
        "gradient_cases": "\n".join(cases["gradient"]),
        "apply_cases": "\n".join(cases["apply"]),
        "objective_cases": "\n".join(cases["objective"]),
        "value_steps_method": _residual_merit_methods(material.op_name),
        "affine_options": _affine_option_entries(
            "objective_uses_affine",
            "gradient_uses_affine",
            "apply_uses_affine",
            "residual_uses_affine",
            "jacobian_action_uses_affine",
        ),
        "yaml_affine_options": _affine_option_entries(
            "objective_uses_affine",
            "gradient_uses_affine",
            "apply_uses_affine",
            "residual_uses_affine",
            "jacobian_action_uses_affine",
            owner="ret->impl_",
        ),
    }
    # A coupled Op always publishes the line-search 0-form: its merit is built
    # from `gradient`, which every operator has, rather than from the energy
    # block's objective kernels.
    return _header(material, True, publishes_value_steps=True), source


def _coupled_dependency_flags(systems_by_dim, energy_name, residual_name):
    from codegen.framework.symbolic.forms import FormOrder

    flags = {
        "gradient_previous": False,
        "apply_current": False,
        "apply_previous": False,
    }
    for system in systems_by_dim.values():
        energy_equation = next(equation for equation in system.equations if equation.name == energy_name)
        residual_equation = next(equation for equation in system.equations if equation.name == residual_name)
        energy_collection = system.form_collection(energy_equation)
        residual_collection = system.form_collection(
            residual_equation,
            orders=(FormOrder.ONE, FormOrder.TWO),
        )
        energy_apply = energy_collection.form_metadata(FormOrder.TWO).dependencies
        residual_gradient = residual_collection.form_metadata(FormOrder.ONE).dependencies
        residual_apply = residual_collection.form_metadata(FormOrder.TWO).dependencies
        flags["gradient_previous"] |= bool(getattr(residual_gradient, "previous", False))
        flags["apply_current"] |= bool(getattr(energy_apply, "current", False)) or bool(
            getattr(residual_apply, "current", False)
        )
        flags["apply_previous"] |= bool(getattr(residual_apply, "previous", False))
    return flags


def _coupled_apply_state_check(op_name, uses_current, uses_previous):
    conditions = []
    if uses_current:
        conditions.append("!current")
    if uses_previous:
        conditions.append("!impl_->previous")
    if not conditions:
        return ""
    requirement = (
        "current and previous states"
        if uses_current and uses_previous
        else ("a current state" if uses_current else "a previous state")
    )
    return (
        "        if (%s) {\n"
        '            SFEM_ERROR("%s requires %s\\n");\n'
        "            return SFEM_FAILURE;\n"
        "        }"
        % (" || ".join(conditions), op_name, requirement)
    )


def _residual_zero_form_is_assembled_norm(form_collections):
    """Whether this residual system's 0-form is a merit over its assembled residual.

    The form layer answers this by the role it gave the 0-form: `merit` for a
    residual that is the gradient of nothing, `potential` for one that is.  A
    merit is computable here and now -- it is half the squared norm of what
    `gradient` already produces -- while a potential needs an element kernel
    the residual emitter cannot yet build, so the two cases are not
    interchangeable and only the first is emitted.
    """
    from codegen.framework.plans.form_emission import FormReduction, form_reduction
    from codegen.framework.symbolic.forms import FormOrder

    for collection in (form_collections or {}).values():
        for form in getattr(collection, "forms", ()):
            if getattr(form, "order", None) is not FormOrder.ZERO:
                continue
            if form_reduction(form) is not FormReduction.ASSEMBLED_NORM:
                return False
            return True
    return False

def _residual_merit_methods(op_name):
    """The 0-form of a mixed energy/residual system: one residual merit.

    A material that declares both an energy and a residual has, in general, no
    potential: if the residual is not the gradient of anything -- a Kelvin-Voigt
    viscous term is the case that matters -- then neither is the sum of it and
    the energy's gradient.  So the system's 0-form is the merit over its
    assembled residual, `1/2 * ||R||^2`.

    The part worth stating plainly is what happens to the energy.  It does not
    contribute its potential to this, and a potential is never added to a norm:
    those are different kinds of quantity and their sum means nothing.  The
    energy contributes its *gradient*, which is already one of the two terms
    this operator's `gradient` accumulates.  So the merit is computed for the
    whole system including the part that would otherwise supply an energy, and
    the elastic objective kernels play no role in it.

    That is also why this needs no kernel of its own.  `gradient` already
    computes `R = grad(E) + R_residual`; the merit is one dot product over the
    degrees of freedom once it has.  A step of length alpha is evaluated by
    forming `x + alpha * h` and asking for the gradient there, which is the
    same traversal the Newton iteration performs anyway.
    """
    return """
    int %(op)s::value_steps(const real_t *state,
                            const real_t *h,
                            const int nsteps,
                            const real_t *const steps,
                            real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::value_steps");
        if (nsteps <= 0) {
            return SFEM_SUCCESS;
        }
        const ptrdiff_t ndofs = n_dofs_domain();
        std::vector<real_t> stepped(ndofs);
        std::vector<real_t> residual(ndofs);
        for (int step = 0; step < nsteps; ++step) {
            const real_t alpha = steps[step];
            for (ptrdiff_t i = 0; i < ndofs; ++i) {
                stepped[i] = state[i] + alpha * h[i];
            }
            std::fill(residual.begin(), residual.end(), real_t(0));
            const int status = gradient(stepped.data(), residual.data());
            if (status != SFEM_SUCCESS) {
                return status;
            }
            real_t sum = 0;
#pragma omp simd reduction(+ : sum)
            for (ptrdiff_t i = 0; i < ndofs; ++i) {
                sum += residual[i] * residual[i];
            }
            out[step] += real_t(0.5) * sum;
        }
        return SFEM_SUCCESS;
    }

    int %(op)s::value(const real_t *state, real_t *const out) {
        SFEM_TRACE_SCOPE("%(op)s::value");
        // One step of length zero: `state + 0 * h` is `state` exactly, so the
        // increment is unused and `state` can stand in for it.  One
        // implementation, so the two cannot disagree.
        const real_t objective_step = 0;
        *out = 0;
        return value_steps(state, state, 1, &objective_step, out);
    }
""" % {"op": op_name}


def _coupled_cases(
    material,
    elements,
    systems_by_dim,
    energy_name,
    residual_name,
    parameter_index,
    kernel_sources=None,
):
    mixed_order = _uses_mixed_field_arrays(elements)
    from codegen.framework.symbolic.forms import FormOrder

    kernel_sources = kernel_sources or {}
    cases = {
        "gradient": [],
        "apply": [],
        "objective": [],
        "objective_steps": [],
        "performance": {"value": [], "gradient": [], "apply": []},
    }
    for element in elements:
        dim = _element_dim(element)
        system = systems_by_dim[dim]
        energy_equation = next(equation for equation in system.equations if equation.name == energy_name)
        residual_equation = next(equation for equation in system.equations if equation.name == residual_name)
        energy_collection = system.form_collection(energy_equation)
        residual_collection = system.form_collection(
            residual_equation,
            orders=(FormOrder.ONE, FormOrder.TWO),
        )
        energy_field = energy_collection.fields[0]
        residual_fields = tuple(residual_collection.fields)
        block_size = sum(int(field.components) for field in residual_fields)
        energy_element = _compatible_element_for_field(element, energy_field)
        energy_label = energy_element.lower()
        mixed_label = _element_name(element).lower()
        energy_stem = "%s_%s_%s" % (material.name, energy_name, energy_label)
        residual_stem = "%s_%s_%s" % (material.name, residual_name, mixed_label)
        energy_dispatch_stem = "%s_%s" % (material.name, energy_name)
        residual_dispatch_stem = "%s_%s" % (material.name, residual_name)
        has_objective = (
            _c_abi_function_defined(
                kernel_sources,
                "%s_objective_affine_mesh_soa" % energy_stem,
            )
            and _c_abi_function_defined(
                kernel_sources,
                "%s_objective_isoparametric_mesh_soa" % energy_stem,
            )
        )
        has_objective_steps = (
            _c_abi_function_defined(
                kernel_sources,
                "%s_objective_steps_affine_mesh_soa" % energy_stem,
            )
            and _c_abi_function_defined(
                kernel_sources,
                "%s_objective_steps_isoparametric_mesh_soa" % energy_stem,
            )
        )
        has_gradient = (
            _c_abi_function_defined(
                kernel_sources,
                "%s_gradient_affine_mesh_soa" % energy_stem,
            )
            and _c_abi_function_defined(
                kernel_sources,
                "%s_gradient_isoparametric_mesh_soa" % energy_stem,
            )
            and _c_abi_function_defined(
                kernel_sources,
                "%s_residual_affine_mesh_soa" % residual_stem,
            )
            and _c_abi_function_defined(
                kernel_sources,
                "%s_residual_isoparametric_mesh_soa" % residual_stem,
            )
        )
        has_apply = (
            _c_abi_function_defined(
                kernel_sources,
                "%s_apply_affine_mesh_soa" % energy_stem,
            )
            and _c_abi_function_defined(
                kernel_sources,
                "%s_apply_isoparametric_mesh_soa" % energy_stem,
            )
            and _c_abi_function_defined(
                kernel_sources,
                "%s_jacobian_action_affine_mesh_soa" % residual_stem,
            )
            and _c_abi_function_defined(
                kernel_sources,
                "%s_jacobian_action_isoparametric_mesh_soa" % residual_stem,
            )
        )
        if has_objective:
            cases["performance"]["value"].append(
                _performance_case(
                    element,
                    ("%s_objective_soa_diagnostics" % energy_stem,),
                    affine_flags=("objective_uses_affine",),
                )
            )
        if has_gradient:
            diagnostics = ["%s_gradient_soa_diagnostics" % energy_stem]
            affine_flags = ["gradient_uses_affine"]
            if getattr(residual_equation, "diagnostics", True):
                diagnostics.append("%s_residual_element_soa_diagnostics" % residual_stem)
                affine_flags.append("residual_uses_affine")
            cases["performance"]["gradient"].append(
                _performance_case(
                    element,
                    tuple(diagnostics),
                    affine_flags=tuple(affine_flags),
                )
            )
        if has_apply:
            diagnostics = ["%s_apply_soa_diagnostics" % energy_stem]
            affine_flags = ["apply_uses_affine"]
            if getattr(residual_equation, "diagnostics", True):
                diagnostics.append("%s_jacobian_action_element_soa_diagnostics" % residual_stem)
                affine_flags.append("jacobian_action_uses_affine")
            cases["performance"]["apply"].append(
                _performance_case(
                    element,
                    tuple(diagnostics),
                    affine_flags=tuple(affine_flags),
                )
            )
        energy_objective_dependencies = energy_collection.form_metadata(FormOrder.ZERO).dependencies
        energy_gradient_dependencies = energy_collection.form_metadata(FormOrder.ONE).dependencies
        energy_apply_dependencies = energy_collection.form_metadata(FormOrder.TWO).dependencies
        residual_dependencies = residual_collection.form_metadata(FormOrder.ONE).dependencies
        residual_apply_dependencies = residual_collection.form_metadata(FormOrder.TWO).dependencies
        energy_params = _metadata_parameter_args(energy_gradient_dependencies, parameter_index)
        energy_apply_params = _metadata_parameter_args(energy_apply_dependencies, parameter_index)
        energy_objective_params = _metadata_parameter_args(energy_objective_dependencies, parameter_index)
        residual_params = _dependency_parameter_args(
            residual_dependencies.parameters,
            parameter_index,
        )
        residual_apply_params = _dependency_parameter_args(
            residual_apply_dependencies.parameters,
            parameter_index,
        )

        field_offsets = _field_offsets(residual_fields)
        energy_components = tuple(range(int(energy_field.components)))
        energy_data = _component_offsets("state", field_offsets[energy_field.name], energy_components)
        energy_direction = _component_offsets("direction", field_offsets[energy_field.name], energy_components)
        energy_increment = _component_offsets("h", field_offsets[energy_field.name], energy_components)
        energy_out = _component_offsets("out", field_offsets[energy_field.name], energy_components)
        residual_state_setup = _residual_soa_view_declarations(residual_fields, "state", "data", "const real_t")
        residual_current_setup = _residual_soa_view_declarations(residual_fields, "current", "data", "const real_t")
        residual_previous_setup = _residual_soa_view_declarations(residual_fields, "previous", "old_data", "const real_t")
        residual_direction_setup = _residual_soa_view_declarations(residual_fields, "direction", "direction_data", "const real_t")
        residual_out_setup = _residual_soa_view_declarations(residual_fields, "out", "out", "real_t")
        residual_state_args = _residual_soa_field_argument_names(residual_fields, "data", mixed_order)
        residual_previous_args = _residual_soa_field_argument_names(residual_fields, "old_data", mixed_order)
        residual_direction_args = _residual_soa_field_argument_names(residual_fields, "direction_data", mixed_order)
        residual_out_args = _residual_soa_field_argument_names(residual_fields, "out", mixed_order)

        geometry_affine = _affine_geometry_offsets(dim) + ", determinant"
        common_iso = "domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points"
        common_affine = "domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), %s" % geometry_affine
        common_iso_dispatch = "domain.element_type, real_type, %s" % common_iso
        common_affine_dispatch = "domain.element_type, real_type, %s" % common_affine
        energy_grad_args = ", ".join(
            _nonempty(
                *_coupled_energy_field_args(
                    energy_gradient_dependencies,
                    block_size,
                    current=energy_data,
                ),
                block_size,
                energy_out,
            )
        )
        energy_grad_affine = "%s_gradient_%dd_affine_mesh_soa(%s%s, %s)" % (
            energy_dispatch_stem, dim, common_affine_dispatch, energy_params, energy_grad_args
        )
        energy_grad_iso = "%s_gradient_%dd_isoparametric_mesh_soa(%s%s, %s)" % (
            energy_dispatch_stem, dim, common_iso_dispatch, energy_params, energy_grad_args
        )
        residual_gradient_args = []
        residual_gradient_setup = []
        if residual_dependencies.current:
            residual_gradient_setup.extend(residual_state_setup)
            residual_gradient_args.extend((str(block_size), *residual_state_args))
        if residual_dependencies.previous:
            residual_gradient_setup.extend(residual_previous_setup)
            residual_gradient_args.extend((str(block_size), *residual_previous_args))
        residual_gradient_setup.extend(residual_out_setup)
        residual_gradient_args.extend((str(block_size), *residual_out_args))
        residual_args_common = ", ".join(
            _nonempty(
                residual_params[2:] if residual_params.startswith(", ") else residual_params,
                *residual_gradient_args,
            )
        )
        residual_grad_affine = "%s_residual_%dd_affine_mesh_soa(%s, %s)" % (
            residual_dispatch_stem, dim, common_affine_dispatch, residual_args_common
        )
        residual_grad_iso = "%s_residual_%dd_isoparametric_mesh_soa(%s, %s)" % (
            residual_dispatch_stem, dim, common_iso_dispatch, residual_args_common
        )
        if has_gradient:
            cases["gradient"].append(
                _coupled_case(
                    element,
                    block_size,
                    residual_gradient_setup,
                    (
                        "                    int status = impl_->gradient_uses_affine ? %s : %s;\n"
                        "                    if (status != SFEM_SUCCESS) return status;\n"
                        "                    return impl_->residual_uses_affine ? %s : %s;"
                    ) % (energy_grad_affine, energy_grad_iso, residual_grad_affine, residual_grad_iso),
                )
            )

        energy_apply_args = ", ".join(
            _nonempty(
                *_coupled_energy_field_args(
                    energy_apply_dependencies,
                    block_size,
                    current=energy_data,
                    direction=energy_direction,
                ),
                block_size,
                energy_out,
            )
        )
        energy_apply_affine = "%s_apply_%dd_affine_mesh_soa(%s%s, %s)" % (
            energy_dispatch_stem, dim, common_affine_dispatch, energy_apply_params, energy_apply_args
        )
        energy_apply_iso = "%s_apply_%dd_isoparametric_mesh_soa(%s%s, %s)" % (
            energy_dispatch_stem, dim, common_iso_dispatch, energy_apply_params, energy_apply_args
        )
        residual_apply_args = []
        residual_apply_setup = []
        if residual_apply_dependencies.current:
            residual_apply_setup.extend(residual_current_setup)
            residual_apply_args.extend((str(block_size), *residual_state_args))
        if residual_apply_dependencies.previous:
            residual_apply_setup.extend(residual_previous_setup)
            residual_apply_args.extend((str(block_size), *residual_previous_args))
        if residual_apply_dependencies.direction:
            residual_apply_setup.extend(residual_direction_setup)
            residual_apply_args.extend((str(block_size), *residual_direction_args))
        residual_apply_setup.extend(residual_out_setup)
        residual_apply_args.extend((str(block_size), *residual_out_args))
        residual_apply_args_common = ", ".join(
            _nonempty(
                residual_apply_params[2:] if residual_apply_params.startswith(", ") else residual_apply_params,
                *residual_apply_args,
            )
        )
        residual_apply_affine = "%s_jacobian_action_%dd_affine_mesh_soa(%s, %s)" % (
            residual_dispatch_stem, dim, common_affine_dispatch, residual_apply_args_common
        )
        residual_apply_iso = "%s_jacobian_action_%dd_isoparametric_mesh_soa(%s, %s)" % (
            residual_dispatch_stem, dim, common_iso_dispatch, residual_apply_args_common
        )
        if has_apply:
            cases["apply"].append(
                _coupled_case(
                    element,
                    block_size,
                    residual_apply_setup,
                    (
                        "                    int status = impl_->apply_uses_affine ? %s : %s;\n"
                        "                    if (status != SFEM_SUCCESS) return status;\n"
                        "                    return impl_->jacobian_action_uses_affine ? %s : %s;"
                    ) % (energy_apply_affine, energy_apply_iso, residual_apply_affine, residual_apply_iso),
                )
            )

        energy_objective_args = ", ".join(
            _nonempty(
                *_coupled_energy_field_args(
                    energy_objective_dependencies,
                    block_size,
                    current=energy_data,
                ),
                "impl_->element_values.get()",
            )
        )
        energy_objective_affine = "%s_objective_%dd_affine_mesh_soa(%s%s, %s)" % (
            energy_dispatch_stem, dim, common_affine_dispatch, energy_objective_params, energy_objective_args
        )
        energy_objective_iso = "%s_objective_%dd_isoparametric_mesh_soa(%s%s, %s)" % (
            energy_dispatch_stem, dim, common_iso_dispatch, energy_objective_params, energy_objective_args
        )
        if has_objective:
            cases["objective"].append(
                """%(cases)s
                    status = impl_->objective_uses_affine ? %(affine)s : %(isoparametric)s;
                    break;""" % {
                    "cases": _mesh_case_labels(element, "                "),
                    "affine": energy_objective_affine,
                    "isoparametric": energy_objective_iso,
                }
            )

        # The steps kernel is the objective's, with the increment and the step
        # lengths threaded in ahead of the output: the same energy evaluated at
        # `state + steps[s] * h` for every s in one traversal of the mesh.
        energy_objective_steps_args = ", ".join(
            _nonempty(
                *_coupled_energy_field_args(
                    energy_objective_dependencies,
                    block_size,
                    current=energy_data,
                ),
                str(block_size),
                energy_increment,
                "nsteps",
                "steps",
                "impl_->element_values.get()",
            )
        )
        energy_objective_steps_affine = (
            "%s_objective_steps_%dd_affine_mesh_soa(%s%s, %s)"
            % (energy_dispatch_stem, dim, common_affine_dispatch,
               energy_objective_params, energy_objective_steps_args)
        )
        energy_objective_steps_iso = (
            "%s_objective_steps_%dd_isoparametric_mesh_soa(%s%s, %s)"
            % (energy_dispatch_stem, dim, common_iso_dispatch,
               energy_objective_params, energy_objective_steps_args)
        )
        if has_objective_steps:
            cases["objective_steps"].append(
                """%(cases)s
                    status = impl_->objective_uses_affine ? %(affine)s : %(isoparametric)s;
                    break;""" % {
                    "cases": _mesh_case_labels(element, "                "),
                    "affine": energy_objective_steps_affine,
                    "isoparametric": energy_objective_steps_iso,
                }
            )
    return cases


def _coupled_case(element, block_size, setup_lines, body):
    return """%(cases)s {
                    static constexpr ptrdiff_t FIELD_STRIDE = %(block_size)d;
%(setup)s
%(body)s
                }""" % {
        "cases": _mesh_case_labels(element, "                "),
        "block_size": block_size,
        "setup": "\n".join(setup_lines),
        "body": body,
    }


def _coupled_parameter_array_lines(defaults):
    lines = []
    for index, (name, _) in enumerate(defaults):
        lines.append(
            '            values[%d] = parameters.require_real_value("%s");'
            % (index, name)
        )
    if not lines:
        lines.append("            values[0] = 0;")
    return "\n".join(lines)


def _coupled_block_size_lines(systems_by_dim):
    lines = []
    for dim in sorted(systems_by_dim):
        fields = systems_by_dim[dim].fields
        block_size = sum(int(field.components) for field in fields)
        lines.append("                case %d: return %d;" % (dim, block_size))
    return "\n".join(lines)


def _field_offsets(fields):
    offsets = {}
    offset = 0
    for field in fields:
        offsets[field.name] = offset
        offset += int(field.components)
    return offsets


def _component_offsets(base, offset, components):
    return ", ".join("%s + %d" % (base, offset + component) for component in components)


def _coupled_energy_field_args(dependencies, block_size, current=None, direction=None):
    args = []
    if current is not None and getattr(dependencies, "current", False):
        args.extend((str(block_size), current))
    if direction is not None and getattr(dependencies, "direction", False):
        args.extend((str(block_size), direction))
    return tuple(args)


def _metadata_parameter_args(parameters, parameter_index):
    parameters = _dependency_parameters(parameters)
    names = set()
    for parameter in parameters or ():
        name = str(parameter)
        if name in parameter_index:
            names.add(name)
    return "".join(
        ", storage[%d]" % parameter_index[name]
        for name in sorted(names, key=lambda value: parameter_index[value])
    )


def _dependency_parameter_args(parameters, parameter_index):
    parameters = _dependency_parameters(parameters)
    names = []
    for parameter in parameters or ():
        name = str(parameter)
        if name in parameter_index and name not in names:
            names.append(name)
    return "".join(", storage[%d]" % parameter_index[name] for name in names)


def _dependency_storage_args(parameters, parameter_index):
    parameters = _dependency_parameters(parameters)
    names = []
    for parameter in parameters or ():
        name = str(parameter)
        if name in parameter_index and name not in names:
            names.append(name)
    return tuple("storage[%d]" % parameter_index[name] for name in names)


def _dependency_parameters(dependencies):
    return tuple(getattr(dependencies, "parameters", dependencies or ()))


def _dependency_domain_parameter_args(dependencies):
    return tuple(
        'domain.parameters->require_real_value("%s")' % str(parameter)
        for parameter in _dependency_parameters(getattr(dependencies, "parameters", ()))
    )


def _compatible_element_for_field(element, field):
    if hasattr(element, "element_for_field"):
        return element.element_for_field(
            getattr(field, "family", "") or getattr(field, "name", "")
        )
    return _element_name(element)


def _nonempty(*values):
    return tuple(str(value) for value in values if str(value))


def _boundary_residual_parameter_names(collection, available_parameters):
    dependencies = collection.form_metadata(_form_order_one()).dependencies
    used = {
        str(symbol)
        for symbol in dependencies.parameters
        if str(symbol) in available_parameters
    }
    return tuple(
        str(symbol)
        for symbol in collection.parameters
        if str(symbol) in used
    )


def _form_order_zero():
    from codegen.framework.symbolic.forms import FormOrder

    return FormOrder.ZERO


def _form_order_one():
    from codegen.framework.symbolic.forms import FormOrder

    return FormOrder.ONE


def _form_order_two():
    from codegen.framework.symbolic.forms import FormOrder

    return FormOrder.TWO


def _residual_parameter_array_lines(parameter_names_by_dim):
    lines = ["            switch (dim) {"]
    for dim in sorted(parameter_names_by_dim):
        lines.append("                case %d:" % dim)
        for name in parameter_names_by_dim[dim]:
            lines.append(
                '                    values[index++] = parameters.require_real_value("%s");'
                % name
            )
        lines.append("                    break;")
    lines.extend(
        [
            "                default:",
            '                    SFEM_ERROR("unsupported spatial dimension %d for generated residual parameters\\n", dim);',
            "                    break;",
            "            }",
        ]
    )
    return "\n".join(lines)


def _residual_block_size_lines(block_size_by_dim):
    lines = ["            switch (dim) {"]
    for dim in sorted(block_size_by_dim):
        lines.append("                case %d: return %d;" % (dim, block_size_by_dim[dim]))
    lines.extend(
        [
            "                default:",
            '                    SFEM_ERROR("unsupported spatial dimension %d for generated residual block size\\n", dim);',
            "                    return 0;",
            "            }",
        ]
    )
    return "\n".join(lines)


def _residual_soa_view_declarations(fields, base, suffix, scalar_type):
    lines = []
    offset = 0
    for field in fields:
        components = int(field.components)
        name = _safe_identifier("%s_%s" % (field.name, suffix))
        if components == 1:
            lines.append(
                "                    %s *const SFEM_RESTRICT %s = %s + %d;"
                % (scalar_type, name, base, offset)
            )
        else:
            entries = ", ".join("%s + %d" % (base, offset + component) for component in range(components))
            lines.append(
                "                    %s *const SFEM_RESTRICT %s[%d] = {%s};"
                % (scalar_type, name, components, entries)
            )
        offset += components
    return lines


def _uses_mixed_field_arrays(elements):
    """Whether this operator's mesh kernels group a vector field into one array.

    A mixed-order (Taylor-Hood) element context selects the mixed emitter, and
    that emitter keeps a vector field whole: its components live on one space,
    ``MixedFieldLayout`` groups them, and the kernel declares a single
    array-of-pointer parameter, ``const real_t *const u_data[3]``.  Every other
    context goes through the single-space model, where
    ``plans.residual_model._component_field_names`` splits the same field into
    ``u0, u1, u2`` and the kernel takes one pointer each.

    So the convention follows the element, not the field and not the material,
    and ``pipeline.driver._integration_case_for_material_element`` says so
    outright: ``is_mixed_order`` on the element is what returns
    ``"isoparametric_mixed"``.  Asking the element is therefore reading the
    decision rather than reconstructing it -- the wrapper used to answer this
    by scanning the emitted C++ for an array extent, which worked but left L7
    parsing L6's output to learn something L2 had already settled.

    Non-mixed generations pass plain element-name strings, which carry no such
    attribute and are correctly reported as not mixed.
    """
    return any(getattr(element, "is_mixed_order", False) for element in elements)


def _residual_soa_field_argument_names(fields, suffix, mixed_order=False):
    """The field arguments a residual-path mesh kernel is called with.

    One argument per field when the mixed emitter produced the kernel, because
    it declares the field as one array; one per component otherwise, because
    the single-space model split the field before the kernel ever saw it.  The
    setup lines build the array either way, so only the spelling differs.
    """
    names = []
    for field in fields:
        components = int(field.components)
        name = _safe_identifier("%s_%s" % (field.name, suffix))
        if components == 1 or mixed_order:
            if mixed_order and components > 1:
                # The kernel takes `const void *const u_data[3]` now, and the
                # local built above is `const real_t *const u_data[3]`.  A
                # pointer converts to `void *` on its own; an array of them
                # does not, so the element type has to be said explicitly.
                names.append(
                    "(%svoid *const *)%s"
                    % ("" if suffix == "out" else "const ", name)
                )
            else:
                names.append(name)
        else:
            names.extend("%s[%d]" % (name, component) for component in range(components))
    return tuple(names)


def _boundary_soa_component_argument_names(fields, suffix):
    names = []
    for field in fields:
        components = int(field.components)
        name = _safe_identifier("%s_%s" % (field.name, suffix))
        if components == 1:
            names.append(name)
        else:
            names.extend("%s[%d]" % (name, component) for component in range(components))
    return tuple(names)


def _residual_soa_case(element, function, arguments, field_stride, setup_lines):
    return """                case smesh::%(element)s: {
                    static constexpr ptrdiff_t FIELD_STRIDE = %(field_stride)d;
%(setup)s
                    return %(function)s(%(arguments)s);
                }""" % {
        "element": _mesh_element_name(element),
        "function": function,
        "arguments": arguments,
        "field_stride": field_stride,
        "setup": "\n".join(setup_lines),
    }


def _boundary_residual_soa_case(element, function, arguments, field_stride, setup_lines):
    return """                    case smesh::%(element)s: {
                        static constexpr ptrdiff_t FIELD_STRIDE = %(field_stride)d;
%(setup)s
                        status |= %(function)s(%(arguments)s);
                        break;
                    }""" % {
        "element": _mesh_element_name(element),
        "function": function,
        "arguments": arguments,
        "field_stride": field_stride,
        "setup": "\n".join(setup_lines),
    }


def _residual_dual_soa_case(
        element,
        flag,
        affine_function,
        affine_arguments,
        isoparametric_function,
        isoparametric_arguments,
        field_stride,
        setup_lines,
        affine_unit_function=None,
        affine_unit_arguments=None,
        affine_unit_condition=None):
    if affine_unit_function:
        body = """                    if (impl_->%(flag)s) {
                        if (%(affine_unit_condition)s) {
                            return %(affine_unit_function)s(%(affine_unit_arguments)s);
                        }
                        return %(affine_function)s(%(affine_arguments)s);
                    }
                    return %(isoparametric_function)s(%(isoparametric_arguments)s);""" % {
            "flag": flag,
            "affine_unit_condition": affine_unit_condition,
            "affine_unit_function": affine_unit_function,
            "affine_unit_arguments": affine_unit_arguments,
            "affine_function": affine_function,
            "affine_arguments": affine_arguments,
            "isoparametric_function": isoparametric_function,
            "isoparametric_arguments": isoparametric_arguments,
        }
    else:
        body = """                    return impl_->%(flag)s ? %(affine_function)s(%(affine_arguments)s) : %(isoparametric_function)s(%(isoparametric_arguments)s);""" % {
            "flag": flag,
            "affine_function": affine_function,
            "affine_arguments": affine_arguments,
            "isoparametric_function": isoparametric_function,
            "isoparametric_arguments": isoparametric_arguments,
        }
    return """                case smesh::%(element)s: {
                    static constexpr ptrdiff_t FIELD_STRIDE = %(field_stride)d;
%(setup)s
%(body)s
                }""" % {
        "element": _mesh_element_name(element),
        "field_stride": field_stride,
        "setup": "\n".join(setup_lines),
        "body": body,
    }


def _safe_identifier(name):
    return re.sub(r"[^0-9A-Za-z_]", "_", str(name))


def _element_api_sources(material, elements, kernel_sources):
    element_headers = _element_api_headers(material, elements, kernel_sources)
    if not element_headers:
        return {}
    header_path = "op/sfem_%s_element_api.hpp" % material.op_name
    return {header_path: _element_api_dispatch_header(material, element_headers)}


def _element_api_headers(material, elements, kernel_sources):
    entries = []
    for element in elements:
        dim = _element_dim(element)
        label = _element_name(element).lower()
        suffix = "/%s_%s_element.hpp" % (material.name, label)
        for path in sorted(kernel_sources):
            if path.endswith(suffix):
                entries.append(
                    {
                        "element": element,
                        "dim": dim,
                        "label": label,
                        "header": path,
                        "source": kernel_sources[path],
                    }
                )
                break
    return tuple(entries)


def _element_api_dispatch_header(material, entries):
    lines = [
        "#pragma once",
        "",
        "#include <cstddef>",
        "",
        "#ifndef SFEM_SUCCESS",
        "#define SFEM_SUCCESS 0",
        "#endif",
        "#ifndef SFEM_FAILURE",
        "#define SFEM_FAILURE 1",
        "#endif",
        "",
    ]
    for header in sorted({entry["header"] for entry in entries}):
        lines.append('#include "../%s"' % header)
    lines.extend(["", "namespace sfem {", "namespace codegen {", ""])

    for operation in ("energy", "gradient", "hessian"):
        for suffix in ("", "coords", "geometry"):
            for dim in sorted({entry["dim"] for entry in entries}):
                group = tuple(
                    entry for entry in entries
                    if entry["dim"] == dim
                    and _element_api_function_params(
                        entry["source"],
                        _element_api_function_name(material.name, entry["label"], operation, suffix),
                    )
                )
                if not group:
                    continue
                function_name = "%s_%s_%dd_element%s_soa" % (
                    material.name,
                    operation,
                    dim,
                    "_%s" % suffix if suffix else "",
                )
                first_element_function = _element_api_function_name(
                    material.name,
                    group[0]["label"],
                    operation,
                    suffix,
                )
                params = _element_api_function_params(group[0]["source"], first_element_function)
                lines.extend(_element_api_dispatch_function_lines(function_name, operation, suffix, group, params, material.name))
                lines.append("")

    lines.extend(["} // namespace codegen", "} // namespace sfem", ""])
    return "\n".join(lines)


def _element_api_function_name(material_name, label, operation, suffix):
    suffix_part = "_%s" % suffix if suffix else ""
    return "%s_%s_%s_element%s_soa" % (material_name, label, operation, suffix_part)


def _element_api_function_params(source, function_name):
    marker = "static SFEM_INLINE int %s(" % function_name
    start = source.find(marker)
    if start < 0:
        return ()
    start = source.find("\n", start)
    end = source.find("\n) {", start)
    if start < 0 or end < 0:
        return ()
    params = []
    for line in source[start:end].splitlines():
        param = line.strip()
        if not param:
            continue
        if param.endswith(","):
            param = param[:-1].rstrip()
        params.append(param)
    return tuple(params)


def _element_api_dispatch_function_lines(function_name, operation, suffix, entries, params, material_name):
    lines = [
        "template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>",
        "static SFEM_INLINE int %s(" % function_name,
        "        const elem_type_t element_type%s" % ("," if params else ""),
    ]
    lines.extend(parameter_list_lines(params))
    arg_names = ", ".join(_element_api_param_name(param) for param in params)
    lines.extend(
        [
            ") {",
            "    switch ((int)element_type) {",
        ]
    )
    for entry in entries:
        element_function = _element_api_function_name(
            material_name,
            entry["label"],
            operation,
            suffix,
        )
        if not _element_api_function_params(entry["source"], element_function):
            continue
        lines.extend(
            [
                "        case %d:" % _smesh_elem_type_value(_mesh_element_name(entry["element"])),
                "            return %s<scalar_t, VECTOR_SIZE>(%s);" % (element_function, arg_names),
            ]
        )
    lines.extend(
        [
            "        default:",
            "            return SFEM_FAILURE;",
            "    }",
            "}",
        ]
    )
    return lines


def _element_api_param_name(param):
    return param.replace(",", "").split()[-1]


def _smesh_elem_type_value(name):
    values = {
        "NODE1": 1,
        "EDGE2": 2,
        "EDGE3": 11,
        "TRI3": 3,
        "TRI6": 6,
        "TRI10": 1010,
        "QUAD4": 40,
        "QUAD9": 9,
        "TET4": 4,
        "TET10": 10,
        "TET15": 15,
        "TET20": 20,
        "HEX8": 8,
        "HEX27": 27,
        "WEDGE6": 1006,
        "MACRO": 200,
        "MACRO_TRI3": 203,
        "MACRO_TET4": 204,
        "PROTEUS_HEX8": 100008,
        "PROTEUS_HEX27": 270000,
        "PROTEUS_HEX64": 640000,
        "PROTEUS_HEX125": 1250000,
        "PROTEUS_HEX216": 2160000,
        "PROTEUS_HEX343": 3430000,
        "PROTEUS_HEX512": 5120000,
        "PROTEUS_HEX729": 7290000,
        "PROTEUS_HEX4913": 49130000,
        "PROTEUS_QUAD4": 400000,
        "PROTEUS_QUAD9": 900000,
        "PROTEUS_QUAD16": 1600000,
        "PROTEUS_QUAD25": 2500000,
        "PROTEUS_QUAD36": 3600000,
        "PROTEUS_QUAD49": 4900000,
        "PROTEUS_QUAD64": 6400000,
        "PROTEUS_QUAD81": 8100000,
        "PROTEUS_QUAD289": 28900000,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError("unsupported generated element API dispatch element %s" % name) from exc


def _declared_dispatch_signatures(groups):
    """The signatures the dispatch sources are about to be written with.

    These entry points are authored here, in L7, by ``_dispatch_function_lines``
    from ``group["params"]`` -- and they are also the entry points the generated
    wrapper calls.  Recovering their shape by parsing the C++ that this same
    function is about to print is a round trip through text for something
    already held as data, and it is where the round trip lost the array extent
    that made Stokes and poro-hyperelasticity uncallable.

    So the declaration is recorded as it is written.  Parsing remains the
    fallback for the kernels the emitters author, which L7 genuinely does not
    see except as text.
    """
    declared = {}
    for group in groups:
        params = ("const smesh::ElemType element_type",) + tuple(group["params"])
        parsed = [_parse_c_parameter(param) for param in params]
        declared[group["name"]] = CSignature(
            name=group["name"],
            parameters=tuple(parameter for parameter in parsed if parameter),
            declaration=None,
        )
    return declared


def _dispatch_sources(material, elements, c_abi_header, kernel_sources):
    declarations = _extract_c_abi_declarations(kernel_sources, public_only=False)
    groups = _dispatch_groups(material, elements, declarations)
    diagnostic_groups = _diagnostic_dispatch_groups(material, elements, declarations)
    if not groups and not diagnostic_groups:
        return {}, {}

    sources = {}
    for kind, grouped in _dispatch_groups_by_source_kind(groups):
        sources[
            "op/sfem_%s_%s_dispatch.cpp" % (material.op_name, kind)
        ] = _dispatch_source(c_abi_header, grouped)
    if diagnostic_groups:
        sources[
            "op/sfem_%s_diagnostics_dispatch.cpp" % material.op_name
        ] = _diagnostic_dispatch_source(c_abi_header, diagnostic_groups)
    return sources, _declared_dispatch_signatures(groups)


def _dispatch_source(c_abi_header, groups):
    lines = [
        '#include "%s"' % c_abi_header,
        "#include <cstdio>",
        "",
        "#ifndef SFEM_SUCCESS",
        "#define SFEM_SUCCESS 0",
        "#endif",
        "#ifndef SFEM_FAILURE",
        "#define SFEM_FAILURE 1",
        "#endif",
        "#ifndef SFEM_CODEGEN_PUBLIC_C_ABI",
        "#define SFEM_CODEGEN_PUBLIC_C_ABI",
        "#endif",
        "",
    ]
    private_declarations = []
    for group in groups:
        for variant in group["variants"]:
            if "declarations" in variant:
                private_declarations.extend(variant["declarations"])
            else:
                private_declarations.append(variant["declaration"])
    lines.extend(_unique(private_declarations))
    if private_declarations:
        lines.append("")

    for group in groups:
        lines.extend(_dispatch_function_lines(group))

    return "\n".join(lines) + "\n"


def _dispatch_groups_by_source_kind(groups):
    grouped = {kind: [] for kind in _DISPATCH_SOURCE_KIND_ORDER}
    for group in groups:
        kind = _dispatch_source_kind(group["name"])
        if kind not in grouped:
            grouped[kind] = []
        grouped[kind].append(group)

    ordered = []
    for kind in _DISPATCH_SOURCE_KIND_ORDER:
        if grouped[kind]:
            ordered.append((kind, tuple(grouped[kind])))
    for kind in sorted(kind for kind in grouped if kind not in _DISPATCH_SOURCE_KIND_ORDER):
        ordered.append((kind, tuple(grouped[kind])))
    return tuple(ordered)


_DISPATCH_SOURCE_KIND_ORDER = (
    "isoparametric",
    "affine",
    "packed_isoparametric",
    "packed_affine",
)


def _dispatch_source_kind(function_name):
    packed = "_packed_" in function_name
    affine = "_affine_" in function_name
    isoparametric = "_isoparametric_" in function_name
    if packed and isoparametric:
        return "packed_isoparametric"
    if packed and affine:
        return "packed_affine"
    if isoparametric:
        return "isoparametric"
    if affine:
        return "affine"
    if "_sideset_" in function_name:
        return "sideset"
    return "other"


def _dispatch_groups(material, elements, declarations):
    element_names = _dispatch_element_names(elements)
    groups = {}
    for declaration in declarations:
        name = _c_abi_function_name(declaration)
        if not name or not declaration.startswith('extern "C" int '):
            continue
        mapped = _dispatch_mapping(material.name, name, element_names)
        if mapped is None:
            continue
        dispatch_name, mesh_element, dim = mapped
        params = _c_abi_parameters(declaration)
        if not params:
            continue
        key = (dispatch_name, tuple(params))
        groups.setdefault(
            key,
            {
                "name": dispatch_name,
                "params": tuple(params),
                "dim": dim,
                "variants": [],
            },
        )["variants"].append(
            {
                "mesh_element": mesh_element,
                "function": name,
                "declaration": declaration,
            }
        )

    ordered = []
    signatures_by_name = {}
    for key in groups:
        signatures_by_name.setdefault(key[0], []).append(key)
    for _, group in sorted(groups.items(), key=lambda item: item[0][0]):
        if len(signatures_by_name[group["name"]]) > 1:
            # Elements of one dimension can want different geometry -- an
            # affine simplex contracts through the symmetric metric where a
            # hexahedron needs the full adjugate -- and that is two ABIs, not
            # one.  They used to collide here and the second was dropped
            # silently, which left the metric elements with no affine entry
            # point at all and a runtime `default:` failure as the only sign.
            # Naming the geometry keeps both.
            group["name"] = _geometry_qualified_dispatch_name(
                group["name"], group["params"]
            )
        group["variants"] = tuple(
            sorted(group["variants"], key=lambda item: item["mesh_element"])
        )
        ordered.append(group)
    return _merge_precision_groups(tuple(ordered))


def _geometry_qualified_dispatch_name(name, params):
    """`name` with the geometry it takes spelled in it.

    Only the metric is qualified; the adjugate keeps the plain name because it
    is the shape every element can be handed.
    """
    if not any("g_geom_metric0" in parameter for parameter in params):
        return name
    marker = "_mesh_"
    index = name.rfind(marker)
    if index < 0:
        return "%s_metric" % name
    return "%s_metric%s" % (name[:index], name[index:])


def _dispatch_mapping(material_name, function_name, element_names):
    prefix = "%s_" % material_name
    if not function_name.startswith(prefix):
        return None
    suffix = function_name[len(prefix) :]
    for element_name, (mesh_element, dim) in sorted(
        element_names.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        element_prefix = "%s_" % element_name
        dispatch_prefix = ""
        if suffix.startswith(element_prefix):
            op_suffix = suffix[len(element_prefix) :]
        else:
            element_marker = "_%s_" % element_name
            marker_index = suffix.find(element_marker)
            if marker_index < 0:
                continue
            dispatch_prefix = suffix[:marker_index]
            op_suffix = suffix[marker_index + len(element_marker) :]
        repeated_prefix = "%s_" % element_name
        if op_suffix.startswith(repeated_prefix):
            op_suffix = op_suffix[len(repeated_prefix) :]
        dispatch_suffix = _insert_dispatch_dimension(op_suffix, dim)
        if dispatch_suffix is None:
            return None
        if dispatch_prefix:
            dispatch_suffix = "%s_%s" % (dispatch_prefix, dispatch_suffix)
        return "%s_%s" % (material_name, dispatch_suffix), mesh_element, dim
    return None


def _insert_dispatch_dimension(op_suffix, dim):
    for marker in ("_affine_", "_isoparametric_", "_sideset_"):
        index = op_suffix.find(marker)
        if index >= 0:
            return "%s_%dd%s" % (op_suffix[:index], dim, op_suffix[index:])
    return None


def _c_abi_parameters(declaration):
    begin = declaration.find("(")
    end = declaration.rfind(")")
    if begin < 0 or end < begin:
        return ()
    body = declaration[begin + 1 : end].strip()
    if not body or body == "void":
        return ()
    return tuple(_split_c_parameters(body))


def _split_c_parameters(body):
    params = []
    current = []
    depth = 0
    for char in body:
        if char == "," and depth == 0:
            params.append("".join(current).strip())
            current = []
            continue
        current.append(char)
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
    if current:
        params.append("".join(current).strip())
    return params


#: Spelled once, next to the code that emits it.
_RUNTIME_TYPE_PARAMETER = "const enum smesh::PrimitiveType real_type"
_RUNTIME_TYPE_ARGUMENT = "real_type"
_RESOLVED_RUNTIME_TYPE = "resolved_real_type"


def _runtime_typed_parameter(param):
    """One parameter, as the merged entry point declares it.

    A pointer whose element type is the scalar being dispatched crosses as
    ``void *``; a plain scalar crosses as ``real_t``.  Both are SFEM's own
    choices -- ``cu_tet4_laplacian_apply`` takes ``const void *const x``, and
    ``cu_linear_elasticity_apply`` takes ``const real_t mu`` rather than
    putting the material parameter on the runtime axis.
    """
    if "*" in param:
        return re.sub(r"\b(double|float)\b", "void", param, count=1)
    return re.sub(r"\b(double|float)\b", "real_t", param, count=1)


def _runtime_typed_argument(param, scalar_type):
    """The same parameter, cast back to a concrete type at the call.

    Two shapes reach here.  A plain buffer is one pointer and casts to one.  A
    mixed-order field is an array of pointers -- ``const real_t *const
    u_data[3]``, because its components live on different spaces -- and decays
    to a pointer-to-pointer, so it casts to ``const float *const *`` rather
    than to ``const float *``.
    """
    name = _c_parameter_name(param)
    if "*" not in param:
        return name
    const = "const " if param.lstrip().startswith("const ") else ""
    if re.search(r"\[\s*\d+\s*\]\s*$", param):
        return "(%s%s *const *)%s" % (const, scalar_type, name)
    return "(%s%s *)%s" % (const, scalar_type, name)


def _merge_precision_pair(base, twin):
    """One runtime-typed group from a ``(double, float)`` pair of groups.

    The two declarations differ in exactly the parameters that carry the scalar
    being dispatched, so diffing them identifies those parameters rather than
    requiring a rule about which ones they ought to be.  Everything that agrees
    -- the geometry, the strides, the connectivity -- is left alone, which is
    what makes this safe: geometry is not on this axis in SFEM either, where
    ``fff`` is cast to a fixed ``cu_jacobian_t``.

    ``real_type`` is inserted immediately before the first pointer it governs,
    the position ``cu_laplacian_apply`` puts ``real_type_xy`` in.
    """
    base_params, twin_params = list(base["params"]), list(twin["params"])
    if len(base_params) != len(twin_params):
        return None
    typed = [i for i, (a, b) in enumerate(zip(base_params, twin_params)) if a != b]
    if not typed:
        return None
    pointers = [i for i in typed if "*" in base_params[i]]
    if not pointers:
        return None
    params = [_runtime_typed_parameter(p) if i in set(typed) else p
              for i, p in enumerate(base_params)]
    # Immediately after ``element_type``, which the emitter prepends.
    #
    # SFEM places this parameter next to the buffers it governs, and its
    # position varies by operator -- mid-list in ``cu_laplacian_apply``, after
    # the material scalars in ``cu_linear_elasticity_apply``.  Generated code
    # cannot afford that latitude: every call site would have to work out where
    # the field-stream group begins, in a signature whose shape depends on the
    # form.  One fixed slot, next to the other "what kind of thing is this"
    # parameter, is emittable identically from both sides.
    params.insert(0, _RUNTIME_TYPE_PARAMETER)

    by_element = {v["mesh_element"]: v for v in twin["variants"]}
    variants = []
    for variant in base["variants"]:
        other = by_element.get(variant["mesh_element"])
        if other is None:
            return None
        variants.append(
            {
                "mesh_element": variant["mesh_element"],
                "by_scalar_type": {
                    "double": variant["function"],
                    "float": other["function"],
                },
                # Both leaves still need forward-declaring in the dispatch
                # source; merging the public symbol does not merge them.
                "declarations": (variant["declaration"], other["declaration"]),
            }
        )
    return {
        "name": base["name"],
        "params": tuple(params),
        "dim": base["dim"],
        "variants": tuple(variants),
        # By name.  The emitter sees the transformed parameters -- `void *`
        # where these were `double *` -- so matching on the original text
        # would never fire, which is exactly the bug the compile gate caught.
        "runtime_typed": tuple(
            _c_parameter_name(base_params[i])
            for i in typed
            if "*" in base_params[i]
        ),
    }


def _merge_precision_groups(groups):
    """Collapse every ``(name, name_float)`` pair into one runtime-typed group.

    Pairs that cannot be merged -- a missing twin, a mismatched parameter list,
    nothing actually differing -- are left exactly as they were rather than
    forced, so an unexpected shape degrades to the old two-symbol form instead
    of producing something that will not compile.
    """
    by_name = {group["name"]: group for group in groups}
    merged, consumed = [], set()
    for group in groups:
        name = group["name"]
        if name in consumed or name.endswith("_float"):
            continue
        twin = by_name.get("%s_float" % name)
        if twin is None:
            merged.append(group)
            continue
        pair = _merge_precision_pair(group, twin)
        if pair is None:
            merged.append(group)
            continue
        consumed.add(twin["name"])
        merged.append(pair)
    return tuple(
        group for group in merged if group["name"] not in consumed
    ) + tuple(
        group for group in groups
        if group["name"].endswith("_float") and group["name"] not in consumed
        and "%s" % group["name"][:-6] not in by_name
    )


def _runtime_typed_dispatch_function_lines(group):
    """One entry point where there were two, taking the scalar type as a value.

    This is the shape ``operators/tet4/cuda/cu_tet4_laplacian.cu`` uses: the
    buffers cross as ``void *`` and a ``smesh::PrimitiveType`` says what they
    hold, the body switches, casts, and calls the concrete implementation.
    SFEM splits the two switches across two functions -- element type in
    ``cu_laplacian_apply``, scalar type in the leaf -- and they are nested here
    only because this layer emits a single dispatcher.

    ``SMESH_DEFAULT`` is resolved through ``smesh::TypeToEnum<real_t>``, so it
    means what it means in ``GPULaplacian``: the build's own ``real_t``, not a
    fixed width.  It is a caller's default rather than a fallback, so it is
    resolved once, up front, rather than repeated in every element case.
    """
    runtime = set(group["runtime_typed"])
    params = ("const smesh::ElemType element_type",) + tuple(group["params"])
    lines = ['SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int %s(' % group["name"]]
    lines.extend(parameter_list_lines(params))
    lines.extend(
        [
            ") {",
            "    const enum smesh::PrimitiveType %s =" % _RESOLVED_RUNTIME_TYPE,
            "            (%s == smesh::SMESH_DEFAULT)" % _RUNTIME_TYPE_ARGUMENT,
            "                    ? smesh::TypeToEnum<real_t>::value()",
            "                    : %s;" % _RUNTIME_TYPE_ARGUMENT,
            "    switch (element_type) {",
        ]
    )
    cases = (("smesh::SMESH_FLOAT64", "double"), ("smesh::SMESH_FLOAT32", "float"))
    for variant in group["variants"]:
        lines.extend(
            [
                "        case smesh::%s: {" % variant["mesh_element"],
                "            switch (%s) {" % _RESOLVED_RUNTIME_TYPE,
            ]
        )
        for enum_value, scalar_type in cases:
            args = [
                _runtime_typed_argument(param, scalar_type)
                if _c_parameter_name(param) in runtime
                else _c_parameter_name(param)
                for param in group["params"]
                if param != _RUNTIME_TYPE_PARAMETER
            ]
            lines.extend(
                [
                    "                case %s:" % enum_value,
                    "                    return %s(%s);"
                    % (variant["by_scalar_type"][scalar_type], ", ".join(args)),
                ]
            )
        lines.extend(
            [
                "                default:",
                "                    break;",
                "            }",
                "            break;",
                "        }",
            ]
        )
    lines.extend(
        [
            "        default:",
            "            break;",
            "    }",
            '    std::fprintf(stderr,',
            '            "%s does not support element type %%d with real type %%d\\n",'
            % group["name"],
            "            (int)element_type,",
            "            (int)%s);" % _RUNTIME_TYPE_ARGUMENT,
            "    return SFEM_FAILURE;",
            "}",
            "",
        ]
    )
    return lines


def _dispatch_function_lines(group):
    if group.get("runtime_typed"):
        return _runtime_typed_dispatch_function_lines(group)
    params = ("const smesh::ElemType element_type",) + tuple(group["params"])
    arg_names = tuple(_c_parameter_name(param) for param in group["params"])
    lines = [
        'SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int %s(' % group["name"],
    ]
    lines.extend(parameter_list_lines(params))
    lines.extend(
        [
            ") {",
            "    switch (element_type) {",
        ]
    )
    for variant in group["variants"]:
        lines.extend(
            [
                "        case smesh::%s:" % variant["mesh_element"],
                "            return %s(%s);" % (variant["function"], ", ".join(arg_names)),
            ]
        )
    lines.extend(
        [
            "        default:",
            '            std::fprintf(stderr, "%s does not support element type %%d\\n", (int)element_type);'
            % group["name"],
            "            return SFEM_FAILURE;",
            "    }",
            "}",
            "",
        ]
    )
    return lines


def _c_parameter_name(param):
    cleaned = param.strip()
    cleaned = cleaned.replace(" SFEM_RESTRICT", "")
    cleaned = cleaned.replace("SFEM_RESTRICT ", "")
    cleaned = cleaned.rstrip()
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?$", cleaned)
    if not match:
        raise ValueError("could not extract C parameter name from '%s'" % param)
    return match.group(1)


def _diagnostic_dispatch_groups(material, elements, declarations):
    element_names = _dispatch_element_names(elements)
    groups = {}
    for declaration in declarations:
        name = _c_abi_function_name(declaration)
        if (
            not name
            or not name.endswith("_soa_diagnostics")
            or "KernelDiagnostics *" not in declaration
        ):
            continue
        mapped = _diagnostic_dispatch_mapping(material.name, name, element_names)
        if mapped is None:
            continue
        dispatch_name, mesh_element, dim = mapped
        groups.setdefault(
            dispatch_name,
            {
                "name": dispatch_name,
                "dim": dim,
                "variants": [],
            },
        )["variants"].append(
            {
                "mesh_element": mesh_element,
                "function": name,
                "declaration": declaration,
            }
        )

    ordered = []
    for _, group in sorted(groups.items(), key=lambda item: item[0]):
        group["variants"] = tuple(
            sorted(group["variants"], key=lambda item: item["mesh_element"])
        )
        ordered.append(group)
    return tuple(ordered)


def _diagnostic_dispatch_mapping(material_name, function_name, element_names):
    prefix = "%s_" % material_name
    if not function_name.startswith(prefix) or not function_name.endswith("_soa_diagnostics"):
        return None
    suffix = function_name[len(prefix) :]
    for element_name, (mesh_element, dim) in sorted(
        element_names.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        element_prefix = "%s_" % element_name
        dispatch_prefix = ""
        if suffix.startswith(element_prefix):
            op_suffix = suffix[len(element_prefix) :]
        else:
            element_marker = "_%s_" % element_name
            marker_index = suffix.find(element_marker)
            if marker_index < 0:
                continue
            dispatch_prefix = suffix[:marker_index]
            op_suffix = suffix[marker_index + len(element_marker) :]
        repeated_prefix = "%s_" % element_name
        if op_suffix.startswith(repeated_prefix):
            op_suffix = op_suffix[len(repeated_prefix) :]
        marker = "_soa_diagnostics"
        if marker not in op_suffix:
            return None
        dispatch_suffix = op_suffix.replace(marker, "_%dd%s" % (dim, marker), 1)
        if dispatch_prefix:
            dispatch_suffix = "%s_%s" % (dispatch_prefix, dispatch_suffix)
        return "%s_%s" % (material_name, dispatch_suffix), mesh_element, dim
    return None


def _diagnostic_dispatch_source(c_abi_header, groups):
    lines = [
        '#include "%s"' % c_abi_header,
        "#include <cstdio>",
        "",
        "#ifndef SFEM_CODEGEN_PUBLIC_C_ABI",
        "#define SFEM_CODEGEN_PUBLIC_C_ABI",
        "#endif",
        "",
    ]
    private_declarations = []
    for group in groups:
        for variant in group["variants"]:
            if "declarations" in variant:
                private_declarations.extend(variant["declarations"])
            else:
                private_declarations.append(variant["declaration"])
    lines.extend(_unique(private_declarations))
    if private_declarations:
        lines.append("")

    for group in groups:
        lines.extend(_diagnostic_dispatch_function_lines(group))
    return "\n".join(lines) + "\n"


def _diagnostic_dispatch_function_lines(group):
    lines = [
        "SFEM_CODEGEN_PUBLIC_C_ABI extern \"C\" const sfem::codegen::KernelDiagnostics *%s("
        % group["name"],
        "        const smesh::ElemType element_type) {",
        "    switch (element_type) {",
    ]
    for variant in group["variants"]:
        lines.extend(
            [
                "        case smesh::%s:" % variant["mesh_element"],
                "            return %s();" % variant["function"],
            ]
        )
    lines.extend(
        [
            "        default:",
            '            std::fprintf(stderr, "%s does not support element type %%d\\n", (int)element_type);'
            % group["name"],
            "            return nullptr;",
            "    }",
            "}",
            "",
        ]
    )
    return lines


def _unique(values):
    seen = set()
    ret = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ret.append(value)
    return ret


def _c_abi_header(material, kernel_sources):
    declarations = _extract_c_abi_declarations(kernel_sources, public_only=True)
    body = "\n\n".join(declarations)
    if body:
        body += "\n"
    matrix_formats_include = (
        '#include "../matrix_formats.hpp"\n'
        if "sfem_MatrixAssemblyDiagnostics" in body
        else ""
    )
    return """#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_CODEGEN_OP_HAS_SFEM_BASE
#endif
#endif

#ifndef SFEM_CODEGEN_OP_HAS_SFEM_BASE
typedef ptrdiff_t idx_t;
typedef ptrdiff_t element_idx_t;
typedef ptrdiff_t count_t;
typedef double real_t;
typedef double geom_t;
#endif

#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif

#include "../kernel_diagnostics.hpp"
%(matrix_formats_include)s
%(smesh_include)s

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

%(body)s""" % {
        "body": body,
        "matrix_formats_include": matrix_formats_include,
        "smesh_include": '#include "smesh_mesh.hpp"' if "smesh::ElemType" in body else "",
    }


def _registration_source(material, wrapper_header):
    function = _registration_function(material)
    return """#include "%(header)s"
#include "sfem_OpFactory.hpp"

namespace sfem {
    void %(function)s() {
        Factory::register_op("%(op)s", %(op)s::create);
        Factory::register_op("ss:%(op)s", %(op)s::create);
    }
}  // namespace sfem
""" % {
        "header": os.path.basename(wrapper_header),
        "function": function,
        "op": material.op_name,
    }


def _registration_function(material):
    return "register_%s_generated_op" % _safe_identifier(material.op_name)


def _op_manifest(
    material,
    kernel_sources,
    wrapper_header,
    wrapper_source,
    registration_source,
    c_abi_header,
    element_api_sources=None,
):
    declarations = _extract_c_abi_declarations(kernel_sources, public_only=True)
    c_abi = [
        {
            "name": _c_abi_function_name(declaration),
            "declaration": declaration,
        }
        for declaration in declarations
    ]
    manifest = {
        "schema": "sfem.generated_op_manifest.v1",
        "material": material.name,
        "op_name": material.op_name,
        "wrapper": {
            "header": wrapper_header,
            "source": wrapper_source,
            "c_abi_header": c_abi_header,
        },
        "registration": {
            "source": registration_source,
            "function": "sfem::%s" % _registration_function(material),
            "operator_name": material.op_name,
        },
        "factory": {
            "class": "sfem::%s" % material.op_name,
            "create": "sfem::%s::create" % material.op_name,
            "create_from_yaml": "sfem::%s::create_from_yaml" % material.op_name,
        },
        "generated_include_paths": _generated_include_paths(kernel_sources),
        "header_api": _header_api_sources(kernel_sources, element_api_sources or {}),
        "matrix_formats": _matrix_format_sources(kernel_sources),
        "runtime_operations": _runtime_operations(c_abi),
        "c_abi": c_abi,
    }
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def _header_api_sources(kernel_sources, element_api_sources):
    headers = []
    for path in sorted(kernel_sources):
        if path.endswith("_element.hpp"):
            headers.append(
                {
                    "kind": "dense_element",
                    "header": path,
                    "layout": "soa_streams",
                    "scope": "element_local",
                    "gather_scatter": "external",
                }
            )
    for path in sorted(element_api_sources):
        if path.endswith("_element_api.hpp"):
            headers.append(
                {
                    "kind": "dense_element_dispatch",
                    "header": path,
                    "layout": "soa_streams",
                    "scope": "element_local",
                    "gather_scatter": "external",
                }
            )
    return tuple(headers)


def _generated_include_paths(kernel_sources):
    paths = set([".", "op"])
    for path in kernel_sources:
        if path.startswith("op/"):
            continue
        if not path.endswith((".hpp", ".cuh", ".cpp", ".cu")):
            continue
        dirname = os.path.dirname(path)
        paths.add(dirname if dirname else ".")
    return tuple(sorted(paths))


def _matrix_format_sources(kernel_sources):
    entries = []
    for path in sorted(kernel_sources):
        if path.endswith("_matrix_format_operator.cpp"):
            entries.append(
                {
                    "source": path,
                    "header": "matrix_formats.hpp",
                }
            )
    return tuple(entries)


#: One parameter of a generated kernel's C entry point.
#:
#: ``extent`` is the array bound for a parameter declared as
#: ``const real_t *const SFEM_RESTRICT u_data[3]`` and ``None`` for a plain
#: pointer or scalar.  It is the fact that decides whether a caller passes the
#: array or its elements, and it used to be discarded: parameter names were
#: recovered as "the last identifier in the declaration", which reads `u_data`
#: out of both forms and cannot tell them apart.  Stokes and
#: poro-hyperelasticity were miscalled for exactly that reason.
CParameter = collections.namedtuple("CParameter", "name extent text")

#: One generated entry point: its name, its parameters in ABI order, and the
#: declaration it was recovered from.
CSignature = collections.namedtuple("CSignature", "name parameters declaration")


def _split_c_parameters(text):
    """Split a parameter list on top-level commas.

    ``(void)`` is C for an empty parameter list, so it splits to nothing.  The
    declaration parser and the call scanner both go through here, which is what
    keeps them agreeing: counting ``void`` as a parameter on one side and not
    the other made every diagnostics accessor look like a call with the wrong
    arity.
    """
    if text.strip() == "void":
        return []
    # Only brackets nest.  Angle brackets must not be counted: a C linkage
    # signature cannot carry a template argument, while the calls scanned
    # through here are full of `packed->n_packs(...)`, whose `>` would open a
    # depth that never closes and make the argument count meaningless.
    parameters, depth, current = [], 0, []
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parameters.append("".join(current))
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        parameters.append(tail)
    return [parameter.strip() for parameter in parameters if parameter.strip()]


def _parse_c_parameter(text):
    """One parameter declaration, as a name and an optional array extent."""
    extent = None
    match = re.search(r"\[\s*(\d+)\s*\]\s*$", text)
    if match:
        extent = int(match.group(1))
        text_without_extent = text[: match.start()]
    else:
        text_without_extent = text
    identifiers = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text_without_extent)
    if not identifiers:
        return None
    return CParameter(name=identifiers[-1], extent=extent, text=text.strip())


def _parse_c_declaration(declaration):
    """A recovered ``extern "C"`` declaration, as a structured signature.

    The parse is required to be total.  It used to drop any parameter it could
    not read a name out of, which is the shape of failure that has to be
    excluded here rather than merely made unlikely: a signature short one
    parameter is not obviously wrong, it is a signature, and the consumer
    checking arity against it would then be checking against a number the
    parse invented.  The whole reason this layer parses at all is that the ABI
    reaches it as text (ARCHITECTURE.html OP 16); while that is true, the parse
    has to fail loudly rather than quietly return something plausible.
    """
    name = _c_abi_function_name(declaration)
    if not name:
        return None
    match = re.search(
        r"\b%s\s*\((.*)\)\s*;" % re.escape(name), declaration, re.S
    )
    if not match:
        return CSignature(name=name, parameters=(), declaration=declaration)
    texts = _split_c_parameters(match.group(1))
    parsed = [_parse_c_parameter(parameter) for parameter in texts]
    unreadable = [
        text for text, parameter in zip(texts, parsed) if parameter is None
    ]
    if unreadable:
        raise ValueError(
            "cannot read the parameters of generated entry point '%s': %s"
            % (name, "; ".join(unreadable))
        )
    return CSignature(
        name=name,
        parameters=tuple(parsed),
        declaration=declaration,
    )


_SIGNATURE_CACHE = {}


def _c_abi_signatures(kernel_sources, public_only=False):
    """Every generated entry point this material publishes, by name.

    This is the one place the emitted C++ is read.  It used to be eight
    functions and twenty scattered regexes, each recovering the part of a
    declaration it happened to need and each free to disagree with the others
    about what a declaration says -- which is how a parameter's array extent
    came to be visible to the header writer and invisible to the call writer.

    Parsing the printed text at all is the layering defect described in
    ARCHITECTURE.html OP 16; emission holds this in ``mesh_kernel_stream_plans``
    and ``MixedFieldLayout`` and should publish it rather than have L7 recover
    it.  Confining the recovery to one function is what makes that swap a
    change of producer instead of a rewrite of every consumer.
    """
    if not kernel_sources:
        return {}
    key = (id(kernel_sources), bool(public_only))
    cached = _SIGNATURE_CACHE.get(key)
    if cached is None:
        cached = {}
        for declaration in _extract_c_abi_declarations(
            kernel_sources, public_only=public_only
        ):
            signature = _parse_c_declaration(declaration)
            if signature:
                cached[signature.name] = signature
        _SIGNATURE_CACHE[key] = cached
    return cached


def _extract_c_abi_declarations(kernel_sources, public_only=False):
    declarations = {}
    for path, source in sorted(kernel_sources.items()):
        if not path.endswith((".cpp", ".hpp")) or path.startswith("op/"):
            if not (public_only and path.endswith("_dispatch.cpp")):
                continue
        offset = 0
        while True:
            start = source.find('extern "C"', offset)
            if start < 0:
                break
            brace = source.find("{", start)
            semicolon = source.find(";", start)
            if semicolon >= 0 and (brace < 0 or semicolon < brace):
                declaration = source[start:semicolon + 1]
                offset = semicolon + 1
            elif brace >= 0:
                declaration = source[start:brace].rstrip() + ";"
                offset = brace + 1
            else:
                break
            name = _c_abi_function_name(declaration)
            declaration_prefix = source[max(0, start - 32) : start]
            if public_only and "SFEM_CODEGEN_PUBLIC_C_ABI" not in declaration_prefix:
                is_int_kernel = declaration.startswith('extern "C" int ')
                is_metadata = name and "_matrix_assembly_" in name
                if not is_metadata:
                    continue
            if name and name not in declarations:
                declarations[name] = declaration
    return tuple(declarations[name] for name in sorted(declarations))


def _c_abi_function_name(declaration):
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", declaration)
    return match.group(1) if match else None


def _c_abi_function_exists(kernel_sources, function_name, public_only=False):
    return function_name in _c_abi_signatures(kernel_sources, public_only=public_only)


def _c_abi_public_dispatch_case_elements(kernel_sources, function_name):
    if not kernel_sources:
        return ()
    for source in kernel_sources.values():
        marker = "%s(" % function_name
        start = source.find(marker)
        while start >= 0:
            declaration_prefix = source[max(0, start - 64) : start]
            if "SFEM_CODEGEN_PUBLIC_C_ABI" not in declaration_prefix:
                start = source.find(marker, start + len(marker))
                continue
            body_start = source.find("{", start)
            if body_start < 0:
                return ()
            depth = 0
            for idx in range(body_start, len(source)):
                char = source[idx]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        body = source[body_start : idx + 1]
                        # A dispatch switches on the element type and then
                        # on the real type, and both are `case smesh::`.  Only
                        # the element types answer the question asked here;
                        # the `SMESH_`-prefixed names are precisions, and
                        # comparing an element type against one is nonsense.
                        return tuple(
                            sorted(
                                name
                                for name in set(
                                    re.findall(
                                        r"\bcase\s+smesh::([A-Z0-9_]+)\s*:", body
                                    )
                                )
                                if not name.startswith("SMESH_")
                            )
                        )
            break
    return ()


def _c_abi_function_declaration(kernel_sources, function_name, public_only=False):
    signature = _c_abi_signatures(kernel_sources, public_only=public_only).get(
        function_name
    )
    return signature.declaration if signature else None


def _c_abi_parameter_names(kernel_sources, function_name, public_only=False):
    signature = _c_abi_signatures(kernel_sources, public_only=public_only).get(
        function_name
    )
    if not signature:
        return ()
    return tuple(parameter.name for parameter in signature.parameters)


def _c_abi_ordered_domain_parameter_args(kernel_sources, function_name, dependencies):
    parameters = tuple(str(parameter) for parameter in _dependency_parameters(dependencies))
    if not parameters:
        return ()
    remaining = list(parameters)
    ordered = []
    for name in _c_abi_parameter_names(kernel_sources, function_name, public_only=True):
        if name in remaining:
            ordered.append(name)
            remaining.remove(name)
    ordered.extend(remaining)
    return tuple(
        'domain.parameters->require_real_value("%s")' % parameter
        for parameter in ordered
    )


def _c_abi_function_defined(kernel_sources, function_name):
    if not kernel_sources:
        return False
    pattern = re.compile(
        r'extern\s+"C"\s+[A-Za-z_][A-Za-z0-9_:<>\s\*&]*\b'
        + re.escape(function_name)
        + r"\s*\([^;{}]*\)\s*\{",
        re.S,
    )
    for path, source in kernel_sources.items():
        normalized = str(path).replace("\\", "/")
        basename = os.path.basename(normalized)
        if basename.startswith("sfem_Generated") and basename.endswith(".cpp"):
            continue
        if _is_replaced_tensor_product_source(normalized, kernel_sources):
            continue
        if pattern.search(source):
            return True
    return False


def _is_replaced_tensor_product_source(path, kernel_sources):
    if not path.endswith(("_operator.cpp", "_boundary_operator.cpp")):
        return False
    keys = {str(key).replace("\\", "/") for key in kernel_sources}
    aliases = (
        ("d2/quad4/", "d2/proteus_quad4/", "_quad4_", "_proteus_quad4_"),
        ("d3/hex8/", "d3/proteus_hex8/", "_hex8_", "_proteus_hex8_"),
        ("d3/hex27/", "d3/proteus_hex27/", "_hex27_", "_proteus_hex27_"),
    )
    for source_prefix, target_prefix, source_suffix, target_suffix in aliases:
        prefix_index = path.find(source_prefix)
        if prefix_index < 0:
            continue
        relative = path[prefix_index:]
        proteus_relative = relative.replace(source_prefix, target_prefix, 1)
        proteus_relative = proteus_relative.replace(source_suffix, target_suffix, 1)
        proteus_path = path[:prefix_index] + proteus_relative
        if proteus_path in keys or proteus_relative in keys:
            return True
        if any(key.endswith("/%s" % proteus_relative) for key in keys):
            return True
    return False


_RUNTIME_OPERATION_MARKERS = (
    ("jacobian_action", "_jacobian_action_"),
    ("hessian_block_diag_sym", "_hessian_block_diag_sym_"),
    ("hessian_bsr", "_hessian_bsr_"),
    ("hessian_coo_triplet", "_hessian_coo_triplet_"),
    ("hessian_coo", "_hessian_coo_"),
    ("hessian_crs", "_hessian_crs_"),
    ("hessian_dia", "_hessian_dia_"),
    ("hessian_patch", "_hessian_patch_"),
    ("bsr_apply", "_bsr_apply_"),
    ("dia_apply", "_dia_apply_"),
    ("patch_apply", "_patch_apply_"),
    ("boundary_residual", "_boundary_residual_"),
    ("objective_steps", "_objective_steps_"),
    ("objective", "_objective_"),
    ("gradient", "_gradient_"),
    ("apply", "_apply_"),
    ("residual", "_residual_"),
)

_RUNTIME_VARIANT_SUFFIXES = (
    ("affine", "_affine_mesh_soa"),
    ("isoparametric", "_isoparametric_mesh_soa"),
    ("sideset", "_sideset_soa"),
)

_AFFINE_OPTION_ALIASES = {
    "objective_uses_affine": (
        "ASSUME_AFFINE_OBJECTIVE",
        "objective_assume_affine",
    ),
    "gradient_uses_affine": (
        "ASSUME_AFFINE_GRADIENT",
        "gradient_assume_affine",
    ),
    "apply_uses_affine": (
        "ASSUME_AFFINE_HESSIAN_ACTION",
        "hessian_action_assume_affine",
        "ASSUME_AFFINE_APPLY",
        "apply_assume_affine",
    ),
    "residual_uses_affine": (
        "ASSUME_AFFINE_RESIDUAL",
        "residual_assume_affine",
        "ASSUME_AFFINE_GRADIENT",
        "gradient_assume_affine",
    ),
    "jacobian_action_uses_affine": (
        "ASSUME_AFFINE_JACOBIAN_ACTION",
        "jacobian_action_assume_affine",
        "ASSUME_AFFINE_APPLY",
        "apply_assume_affine",
    ),
}


def _runtime_operations(c_abi):
    variants_by_operation = {}
    seen = set()
    for entry in c_abi:
        name = entry["name"]
        operation, target = _runtime_operation_and_target(name)
        variant, scalar_type = _runtime_variant_and_scalar_type(name)
        if operation is None or variant is None:
            continue
        key = (operation, variant, scalar_type, name)
        if key in seen:
            continue
        seen.add(key)
        variants_by_operation.setdefault(operation, []).append(
            {
                "variant": variant,
                "scalar_type": scalar_type,
                "target": target,
                "function": name,
            }
        )
    return tuple(
        {
            "name": operation,
            "variants": tuple(
                sorted(
                    variants,
                    key=lambda item: (
                        item["variant"],
                        item["scalar_type"],
                        item["target"],
                        item["function"],
                    ),
                )
            ),
        }
        for operation, variants in sorted(variants_by_operation.items())
    )


def _runtime_operation_and_target(name):
    for operation, marker in _RUNTIME_OPERATION_MARKERS:
        marker_index = name.find(marker)
        if marker_index >= 0:
            return operation, name[:marker_index]
    return None, None


def _runtime_variant_and_scalar_type(name):
    for variant, suffix in _RUNTIME_VARIANT_SUFFIXES:
        if name.endswith(suffix):
            return variant, "real_t"
        if name.endswith("%s_float" % suffix):
            return variant, "float"
    return None, None


def _performance_case(
    element,
    diagnostics,
    count_expression="domain.block->n_elements()",
    affine_flags=None,
):
    diagnostics = tuple(diagnostics)
    if affine_flags is None:
        affine_flags = (None,) * len(diagnostics)
    else:
        affine_flags = tuple(affine_flags)
    if len(affine_flags) != len(diagnostics):
        raise ValueError("performance diagnostics and affine flags length mismatch")
    diagnostic_entries = tuple(
        {"name": diagnostic, "affine_flag": affine_flag}
        for diagnostic, affine_flag in dict.fromkeys(zip(diagnostics, affine_flags))
    )
    return {
        "element": element,
        "diagnostics": diagnostic_entries,
        "count": count_expression,
    }


def _performance_methods(op_name, material_name, elements, cases_by_method):
    methods = []
    element_names = _dispatch_element_names(elements)
    for method in ("value", "gradient", "apply"):
        cases = tuple(cases_by_method.get(method, ()))
        cases = _performance_dispatch_cases(material_name, element_names, cases)
        methods.append(_performance_flops_method(op_name, method, cases))
        methods.append(_performance_bytes_method(op_name, method, cases))
    return "\n\n".join(methods)


def _dispatch_element_names(elements):
    names = {}
    for element in elements:
        element_name = _element_name(element)
        mesh_element = _mesh_element_name(element)
        dim = _element_dim(element)
        names[element_name.lower()] = (mesh_element, dim)
        primary = _primary_element_alias(element_name)
        if primary:
            names.setdefault(primary.lower(), (mesh_element, dim))
    return names


def _primary_element_alias(element_name):
    name = str(element_name).upper()
    prefixes = (
        "PROTEUS_QUAD4",
        "PROTEUS_HEX729",
        "PROTEUS_HEX512",
        "PROTEUS_HEX343",
        "PROTEUS_HEX216",
        "PROTEUS_HEX125",
        "PROTEUS_HEX64",
        "PROTEUS_HEX27",
        "PROTEUS_HEX8",
        "HEX27",
        "TET10",
        "QUAD4",
        "TRI6",
        "HEX8",
        "TET4",
        "TRI3",
    )
    for prefix in prefixes:
        if name.startswith("%s_" % prefix):
            return prefix
    return None


def _performance_dispatch_cases(material_name, element_names, cases):
    mapped = []
    for case in cases:
        entries = []
        for diagnostic in case["diagnostics"]:
            name = diagnostic["name"]
            dispatch_name = _diagnostic_dispatch_mapping(material_name, name, element_names)
            if dispatch_name is None:
                entries.append(diagnostic)
            else:
                entries.append(
                    {
                        "name": dispatch_name[0],
                        "affine_flag": diagnostic["affine_flag"],
                    }
                )
        mapped.append(
            {
                "element": case["element"],
                "diagnostics": tuple(dict.fromkeys((entry["name"], entry["affine_flag"]) for entry in entries)),
                "count": case["count"],
            }
        )

    by_dim = {}
    for case in mapped:
        dim = _element_dim(case["element"])
        diagnostics = by_dim.setdefault(dim, [])
        for name, affine_flag in case["diagnostics"]:
            item = {"name": name, "affine_flag": affine_flag}
            if item not in diagnostics:
                diagnostics.append(item)
    return tuple(
        {
            "dim": dim,
            "diagnostics": tuple(by_dim[dim]),
        }
        for dim in sorted(by_dim)
    )


def _performance_flops_method(op_name, method, cases):
    return """    double %(op)s::flops_%(method)s() const {
        double total = 0;
        if (!impl_->domains) {
            return total;
        }

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
%(cases)s
            return SFEM_SUCCESS;
        });

        return total;
    }""" % {
        "op": op_name,
        "method": method,
        "cases": _performance_flops_cases(cases),
    }


def _performance_bytes_method(op_name, method, cases):
    return """    size_t %(op)s::memory_traffic_bytes_%(method)s() const {
        size_t total = 0;
        if (!impl_->domains) {
            return total;
        }

        const int dim = impl_->space->mesh_ptr()->spatial_dimension();
        impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
%(cases)s
            return SFEM_SUCCESS;
        });

        return total;
    }""" % {
        "op": op_name,
        "method": method,
        "cases": _performance_bytes_cases(cases),
    }


def _performance_flops_cases(cases):
    lines = []
    for case in cases:
        lines.append("            if (dim == %d) {" % case["dim"])
        for diagnostic in case["diagnostics"]:
            name = diagnostic["name"]
            affine_flag = diagnostic["affine_flag"]
            lines.append("                {")
            lines.append(
                "                    const sfem::codegen::KernelDiagnostics *const diagnostics = %s(domain.element_type);"
                % name
            )
            lines.append("                    if (diagnostics) {")
            if affine_flag is None:
                lines.append(
                    "                        total += sfem::codegen::KernelDiagnostics_total_flops(diagnostics, nelements);"
                )
            else:
                lines.append(
                    "                        total += impl_->%s ? sfem::codegen::KernelDiagnostics_total_flops_affine_mesh(diagnostics, nelements) : sfem::codegen::KernelDiagnostics_total_flops_isoparametric_mesh(diagnostics, nelements);"
                    % affine_flag
                )
            lines.append("                    }")
            lines.append("                }")
        lines.append("            }")
    return "\n".join(lines)


def _performance_bytes_cases(cases):
    lines = []
    for case in cases:
        lines.append("            if (dim == %d) {" % case["dim"])
        for diagnostic in case["diagnostics"]:
            name = diagnostic["name"]
            affine_flag = diagnostic["affine_flag"]
            lines.append("                {")
            lines.append(
                "                    const sfem::codegen::KernelDiagnostics *const diagnostics = %s(domain.element_type);"
                % name
            )
            lines.append("                    if (diagnostics) {")
            if affine_flag is None:
                lines.append(
                    "                        total += sfem::codegen::KernelDiagnostics_total_bytes(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));"
                )
            else:
                lines.append(
                    "                        total += impl_->%s ? sfem::codegen::KernelDiagnostics_total_bytes_affine_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t)) : sfem::codegen::KernelDiagnostics_total_bytes_isoparametric_mesh(diagnostics, nelements, sizeof(geom_t), sizeof(real_t), sizeof(real_t));"
                    % affine_flag
                )
            lines.append("                    }")
            lines.append("                }")
        lines.append("            }")
    return "\n".join(lines)


def _affine_option_entries(*flags, owner="impl_"):
    lines = []
    for flag in flags:
        for alias in _AFFINE_OPTION_ALIASES[flag]:
            lines.append('            {"%s", &%s->%s},' % (alias, owner, flag))
    return "\n".join(lines)


def _energy_field_args(dependencies, dim, components, current=None, direction=None):
    args = []
    if current is not None and (
        getattr(dependencies, "current", False) if dependencies is not None else True
    ):
        args.extend((dim, _offsets(current, components)))
    if direction is not None and (
        getattr(dependencies, "direction", False) if dependencies is not None else True
    ):
        args.extend((dim, _offsets(direction, components)))
    return tuple(args)


def _energy_output_args(dim, components):
    return (dim, _offsets("out", components))


def _energy_declaration_field_args(dependencies, dim, components, current=False, direction=False):
    args = []
    vectors = "".join(", const real_t *" for _ in components)
    if current and (
        getattr(dependencies, "current", False) if dependencies is not None else True
    ):
        args.append("ptrdiff_t%s" % vectors)
    if direction and (
        getattr(dependencies, "direction", False) if dependencies is not None else True
    ):
        args.append("ptrdiff_t%s" % vectors)
    return "".join(", %s" % arg for arg in args)


def _hyperelastic_declarations(stem, dim, parameters, dependencies=None, n_field_components_by_dim=None):
    components = _components((n_field_components_by_dim or {}).get(dim, dim))
    if dependencies is None:
        dependencies = (None, None, None)
    objective_dependencies, gradient_dependencies, apply_dependencies = dependencies
    parameter_decl = "".join(", const real_t %s" % name for name in parameters)
    vectors = "".join(", const real_t *" for _ in components)
    outputs = "".join(", real_t *" for _ in components)
    isoparametric_common = (
        "ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *"
        + parameter_decl
    )
    affine_common = (
        "ptrdiff_t, ptrdiff_t, idx_t **"
        + "".join(", const geom_t *" for _ in range(dim * dim))
        + ", const geom_t *"
        + parameter_decl
    )
    affine_aos_unit_common = (
        "ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *, const geom_t *"
        + parameter_decl
    )
    objective_steps_field_decl = (
        _energy_declaration_field_args(
            objective_dependencies,
            dim,
            components,
            current=True,
        )
        + _energy_declaration_field_args(
            apply_dependencies,
            dim,
            components,
            direction=True,
        )
    )
    return (
        "int %s_objective_isoparametric_mesh_soa(%s%s, real_t *);"
        % (
            stem,
            isoparametric_common,
            _energy_declaration_field_args(
                objective_dependencies,
                dim,
                components,
                current=True,
            ),
        ),
        "int %s_gradient_isoparametric_mesh_soa(%s%s, ptrdiff_t%s);"
        % (
            stem,
            isoparametric_common,
            _energy_declaration_field_args(
                gradient_dependencies,
                dim,
                components,
                current=True,
            ),
            outputs,
        ),
        "int %s_apply_isoparametric_mesh_soa(%s%s, ptrdiff_t%s);"
        % (
            stem,
            isoparametric_common,
            _energy_declaration_field_args(
                apply_dependencies,
                dim,
                components,
                current=True,
                direction=True,
            ),
            outputs,
        ),
        "int %s_objective_steps_isoparametric_mesh_soa(%s%s, ptrdiff_t, const real_t *, real_t *);"
        % (
            stem,
            isoparametric_common,
            objective_steps_field_decl,
        ),
        "int %s_objective_affine_mesh_soa(%s%s, real_t *);"
        % (
            stem,
            affine_common,
            _energy_declaration_field_args(
                objective_dependencies,
                dim,
                components,
                current=True,
            ),
        ),
        "int %s_gradient_affine_mesh_soa(%s%s, ptrdiff_t%s);"
        % (
            stem,
            affine_common,
            _energy_declaration_field_args(
                gradient_dependencies,
                dim,
                components,
                current=True,
            ),
            outputs,
        ),
        "int %s_gradient_affine_mesh_soa_aos_unit(%s%s, ptrdiff_t%s);"
        % (
            stem,
            affine_aos_unit_common,
            _energy_declaration_field_args(
                gradient_dependencies,
                dim,
                components,
                current=True,
            ),
            outputs,
        ),
        "int %s_apply_affine_mesh_soa(%s%s, ptrdiff_t%s);"
        % (
            stem,
            affine_common,
            _energy_declaration_field_args(
                apply_dependencies,
                dim,
                components,
                current=True,
                direction=True,
            ),
            outputs,
        ),
        "int %s_apply_affine_mesh_soa_aos_unit(%s%s, ptrdiff_t%s);"
        % (
            stem,
            affine_aos_unit_common,
            _energy_declaration_field_args(
                apply_dependencies,
                dim,
                components,
                current=True,
                direction=True,
            ),
            outputs,
        ),
        "int %s_objective_steps_affine_mesh_soa(%s%s, ptrdiff_t, const real_t *, real_t *);"
        % (
            stem,
            affine_common,
            objective_steps_field_decl,
        ),
    )


def _case(element, function, arguments):
    return """%(cases)s
                    return %(function)s(%(arguments)s);""" % {
        "cases": _mesh_case_labels(element, "                "),
        "function": function,
        "arguments": arguments,
    }


def _residual_hessian_dispatch_body(
    material_name,
    operation,
    kernel_sources,
    action_dependencies_by_dim,
    parameter_names_by_dim,
    fields_by_dim,
    block_size_by_dim,
    tail_args,
    indent,
    mixed_order=False,
):
    lines = ["%sconst int dim = mesh->spatial_dimension();" % indent]
    for dim in (2, 3):
        dependencies = action_dependencies_by_dim.get(dim)
        if dependencies is None:
            continue
        function = "%s_%s_%dd_isoparametric_mesh_soa" % (
            material_name,
            operation,
            dim,
        )
        prefix = "if" if not any(line.endswith("{") for line in lines) else "else if"
        lines.append("%s%s (dim == %d) {" % (indent, prefix, dim))
        lines.append(
            "%s    static constexpr ptrdiff_t FIELD_STRIDE = %d;"
            % (indent, block_size_by_dim[dim])
        )
        setup = []
        args = [
            "domain.element_type",
            "real_type",
            "domain.block->n_elements()",
            "mesh->n_nodes()",
            "domain.block->elements()->data()",
            "points",
        ]
        parameter_index = {
            name: index for index, name in enumerate(parameter_names_by_dim[dim])
        }
        args.extend(_dependency_storage_args(dependencies.parameters, parameter_index))
        fields = fields_by_dim[dim]
        if dependencies.current:
            setup.extend(
                _residual_soa_view_declarations(
                    fields,
                    "current",
                    "data",
                    "const real_t",
                )
            )
            args.append("FIELD_STRIDE")
            args.extend(_residual_soa_field_argument_names(fields, "data", mixed_order))
        if dependencies.previous:
            setup.extend(
                _residual_soa_view_declarations(
                    fields,
                    "previous",
                    "old_data",
                    "const real_t",
                )
            )
            args.append("FIELD_STRIDE")
            args.extend(_residual_soa_field_argument_names(fields, "old_data", mixed_order))
        args.extend(tail_args)
        for line in setup:
            lines.append(line)
        if _c_abi_function_exists(kernel_sources, function, public_only=True):
            lines.append("%s    return %s(%s);" % (indent, function, ", ".join(args)))
        else:
            lines.append(
                '%s    SFEM_ERROR("%s %s %dd dispatch was not generated\\n");'
                % (indent, material_name, operation, dim)
            )
            lines.append("%s    return SFEM_FAILURE;" % indent)
        lines.append("%s}" % indent)
    lines.extend(
        [
            '%sSFEM_ERROR("%s %s does not support spatial dimension %%d\\n", dim);'
            % (indent, material_name, operation),
            "%sreturn SFEM_FAILURE;" % indent,
        ]
    )
    return "\n".join(lines)


def _element_condition(variable_name, mesh_elements):
    names = tuple(sorted(set(mesh_elements)))
    if not names:
        return "false"
    return " || ".join("%s == smesh::%s" % (variable_name, name) for name in names)


def _residual_apply_dispatch_body(
    material_name,
    operation,
    affine_flag,
    current_base,
    kernel_sources,
    dependencies_by_dim,
    parameter_names_by_dim,
    fields_by_dim,
    block_size_by_dim,
    affine_aos_elements_by_dim,
    affine_aos_unit_elements_by_dim,
    indent,
    mixed_order=False,
):
    lines = ["%sconst int dim = mesh->spatial_dimension();" % indent]
    for dim in (2, 3):
        dependencies = dependencies_by_dim.get(dim)
        if dependencies is None:
            continue
        prefix = "if" if not any(line.endswith("{") for line in lines) else "else if"
        lines.append("%s%s (dim == %d) {" % (indent, prefix, dim))
        lines.append(
            "%s    static constexpr ptrdiff_t FIELD_STRIDE = %d;"
            % (indent, block_size_by_dim[dim])
        )
        setup = []
        parameter_index = {
            name: index for index, name in enumerate(parameter_names_by_dim[dim])
        }
        common_args = [
            "domain.element_type",
            "real_type",
            "domain.block->n_elements()",
            "mesh->n_nodes()",
            "domain.block->elements()->data()",
        ]
        field_args = []
        unit_field_args = []
        if dependencies.current:
            setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    current_base,
                    "data",
                    "const real_t",
                )
            )
            field_args.append("FIELD_STRIDE")
            field_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "data", mixed_order))
            unit_field_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "data", mixed_order))
        if dependencies.previous:
            setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    "previous",
                    "old_data",
                    "const real_t",
                )
            )
            field_args.append("FIELD_STRIDE")
            field_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "old_data", mixed_order))
        if dependencies.direction:
            setup.extend(
                _residual_soa_view_declarations(
                    fields_by_dim[dim],
                    "direction",
                    "direction_data",
                    "const real_t",
                )
            )
            field_args.append("FIELD_STRIDE")
            field_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "direction_data", mixed_order)
            )
            unit_field_args.extend(
                _residual_soa_field_argument_names(fields_by_dim[dim], "direction_data", mixed_order)
            )
        setup.extend(
            _residual_soa_view_declarations(
                fields_by_dim[dim],
                "out",
                "out",
                "real_t",
            )
        )
        field_args.append("FIELD_STRIDE")
        field_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "out", mixed_order))
        unit_field_args.extend(_residual_soa_field_argument_names(fields_by_dim[dim], "out", mixed_order))
        for line in setup:
            lines.append(line)

        storage_args = list(
            _dependency_storage_args(dependencies.parameters, parameter_index)
        )
        affine_soa = "%s_%s_%dd_affine_mesh_soa" % (material_name, operation, dim)
        affine_aos = "%s_%s_%dd_affine_mesh_soa_aos" % (material_name, operation, dim)
        affine_aos_unit = "%s_%s_%dd_affine_mesh_soa_aos_unit" % (
            material_name,
            operation,
            dim,
        )
        isop = "%s_%s_%dd_isoparametric_mesh_soa" % (material_name, operation, dim)
        packed_affine = "%s_%s_packed_%dd_affine_mesh_soa" % (
            material_name,
            operation,
            dim,
        )
        packed = "%s_%s_packed_%dd_isoparametric_mesh_soa" % (
            material_name,
            operation,
            dim,
        )
        aos_condition = _element_condition(
            "domain.element_type", affine_aos_elements_by_dim.get(dim, ())
        )
        unit_condition = _element_condition(
            "domain.element_type", affine_aos_unit_elements_by_dim.get(dim, ())
        )

        lines.append("%s    if (impl_->%s) {" % (indent, affine_flag))
        if operation == "jacobian_action" and _c_abi_function_exists(
            kernel_sources, packed_affine, public_only=True
        ):
            packed_affine_geometry_args = (
                _affine_metric_offsets(dim).split(", ")
                if _c_abi_function_uses_cached_metric(kernel_sources, packed_affine)
                else [*_affine_geometry_offsets(dim).split(", "), "determinant"]
            )
            lines.extend(
                [
                    "%s        if (impl_->space->has_packed_mesh()) {" % indent,
                    "%s            auto packed = impl_->space->packed_mesh();" % indent,
                    "%s            const int packed_block = packed_block_id_for_domain(*packed, *domain.block);" % indent,
                    "%s            if (packed_block >= 0) {" % indent,
                    "%s                auto packed_elements = packed->elements(packed_block);" % indent,
                    "%s                auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);" % indent,
                    "%s                auto n_shared_nodes = packed->n_shared(packed_block);" % indent,
                    "%s                auto ghost_ptr = packed->ghost_ptr(packed_block);" % indent,
                    "%s                auto ghost_idx = packed->ghost_idx(packed_block);" % indent,
                    *(
                        [
                            "%s                if (domain.element_type == smesh::TET4) {" % indent,
                            "%s                    return laplace_tet4_jacobian_action_packed_affine_mesh_soa(%s);"
                            % (
                                indent,
                                ", ".join(
                                    [
                                        "packed->n_packs(packed_block)",
                                        "packed->n_elements_per_pack(packed_block)",
                                        "domain.block->n_elements()",
                                        "mesh->n_nodes()",
                                        "packed->max_nodes_per_pack()",
                                        "packed_elements->data()",
                                        "owned_nodes_ptr->data()",
                                        "n_shared_nodes->data()",
                                        "ghost_ptr->data()",
                                        "ghost_idx->data()",
                                        "geom_metric[0]",
                                        "geom_metric[1]",
                                        "geom_metric[2]",
                                        "geom_metric[3]",
                                        "geom_metric[4]",
                                        "geom_metric[5]",
                                        *storage_args,
                                        *field_args,
                                    ]
                                ),
                            ),
                            "%s                }" % indent,
                        ]
                        if material_name == "laplace"
                        and operation == "jacobian_action"
                        and dim == 3
                        and _c_abi_function_exists(
                            kernel_sources,
                            "laplace_tet4_jacobian_action_packed_affine_mesh_soa",
                        )
                        else []
                    ),
                    *(
                        [
                            "%s                if (domain.element_type == smesh::HEX8) {" % indent,
                            "%s                    uint16_t *proteus_elements[8] = {packed_elements->data()[0], packed_elements->data()[1], packed_elements->data()[3], packed_elements->data()[2], packed_elements->data()[4], packed_elements->data()[5], packed_elements->data()[7], packed_elements->data()[6]};" % indent,
                            "%s                    return laplace_proteus_hex8_private_metric_jacobian_action_packed_mesh_soa(%s);"
                            % (
                                indent,
                                ", ".join(
                                    [
                                        "packed->n_packs(packed_block)",
                                        "packed->n_elements_per_pack(packed_block)",
                                        "domain.block->n_elements()",
                                        "mesh->n_nodes()",
                                        "packed->max_nodes_per_pack()",
                                        "proteus_elements",
                                        "owned_nodes_ptr->data()",
                                        "n_shared_nodes->data()",
                                        "ghost_ptr->data()",
                                        "ghost_idx->data()",
                                        "geom_metric_aos",
                                        *storage_args,
                                        *field_args,
                                    ]
                                ),
                            ),
                            "%s                }" % indent,
                            "%s                if (domain.element_type == smesh::PROTEUS_HEX8) {" % indent,
                            "%s                    return laplace_proteus_hex8_private_metric_jacobian_action_packed_mesh_soa(%s);"
                            % (
                                indent,
                                ", ".join(
                                    [
                                        "packed->n_packs(packed_block)",
                                        "packed->n_elements_per_pack(packed_block)",
                                        "domain.block->n_elements()",
                                        "mesh->n_nodes()",
                                        "packed->max_nodes_per_pack()",
                                        "packed_elements->data()",
                                        "owned_nodes_ptr->data()",
                                        "n_shared_nodes->data()",
                                        "ghost_ptr->data()",
                                        "ghost_idx->data()",
                                        "geom_metric_aos",
                                        *storage_args,
                                        *field_args,
                                    ]
                                ),
                            ),
                            "%s                }" % indent,
                        ]
                        if material_name == "laplace"
                        and operation == "jacobian_action"
                        and dim == 3
                        and _c_abi_function_exists(
                            kernel_sources,
                            "laplace_proteus_hex8_private_metric_jacobian_action_packed_mesh_soa",
                        )
                        else []
                    ),
                    *(
                        [
                            "%s                if (domain.element_type == smesh::TET10) {" % indent,
                            "%s                    return laplace_tet10_private_metric_jacobian_action_packed_mesh_soa(%s);"
                            % (
                                indent,
                                ", ".join(
                                    [
                                        "packed->n_packs(packed_block)",
                                        "packed->n_elements_per_pack(packed_block)",
                                        "domain.block->n_elements()",
                                        "mesh->n_nodes()",
                                        "packed->max_nodes_per_pack()",
                                        "packed_elements->data()",
                                        "owned_nodes_ptr->data()",
                                        "n_shared_nodes->data()",
                                        "ghost_ptr->data()",
                                        "ghost_idx->data()",
                                        "geom_metric_aos",
                                        *storage_args,
                                        *field_args,
                                    ]
                                ),
                            ),
                            "%s                }" % indent,
                        ]
                        if material_name == "laplace"
                        and operation == "jacobian_action"
                        and dim == 3
                        and _c_abi_function_exists(
                            kernel_sources,
                            "laplace_tet10_private_metric_jacobian_action_packed_mesh_soa",
                        )
                        else []
                    ),
                    "%s                return %s(%s);"
                    % (
                        indent,
                        packed_affine,
                        ", ".join(
                            [
                                "domain.element_type",
                                "real_type",
                                "packed->n_packs(packed_block)",
                                "packed->n_elements_per_pack(packed_block)",
                                "domain.block->n_elements()",
                                "mesh->n_nodes()",
                                "packed->max_nodes_per_pack()",
                                "packed_elements->data()",
                                "owned_nodes_ptr->data()",
                                "n_shared_nodes->data()",
                                "ghost_ptr->data()",
                                "ghost_idx->data()",
                                *packed_affine_geometry_args,
                                *storage_args,
                                *field_args,
                            ]
                        ),
                    ),
                    "%s            }" % indent,
                    "%s        }" % indent,
                ]
            )
        if _c_abi_function_exists(kernel_sources, affine_aos_unit, public_only=True):
            lines.append(
                "%s        if ((%s) && storage[0] == real_t(1)) {"
                % (indent, unit_condition)
            )
            lines.append(
                "%s            return %s(%s);"
                % (
                    indent,
                    affine_aos_unit,
                    ", ".join(
                        [
                            *common_args,
                            "geom_metric_aos",
                            *unit_field_args,
                        ]
                    ),
                )
            )
            lines.append("%s        }" % indent)
        if _c_abi_function_exists(kernel_sources, affine_aos, public_only=True):
            lines.append("%s        if (%s) {" % (indent, aos_condition))
            lines.append(
                "%s            return %s(%s);"
                % (
                    indent,
                    affine_aos,
                    ", ".join(
                        [
                            *common_args,
                            "geom_metric_aos",
                            *storage_args,
                            *field_args,
                        ]
                    ),
                )
            )
            lines.append("%s        }" % indent)
        if _c_abi_function_exists(kernel_sources, affine_soa, public_only=True):
            lines.append(
                "%s        return %s(%s);"
                % (
                    indent,
                    affine_soa,
                    ", ".join(
                        [
                            *common_args,
                            # The geometry this entry point takes, not the one
                            # its neighbours take.  A linear simplex Laplacian
                            # contracts two reference gradients, so its kernel
                            # asks for the symmetric gradient metric rather
                            # than the adjugate and determinant -- exactly as
                            # the packed branch above already works out for
                            # itself.  This branch did not, and passed five
                            # geometry arguments to a kernel taking three.
                            #
                            # It only shows when the metric form is the only
                            # form: generate laplace for TRI3 or TET4 alone and
                            # the wrapper does not compile.  Add any
                            # tensor-product element and the shared entry point
                            # becomes the adjugate form, so the unconditional
                            # spelling is right again and the defect hides.
                            *_affine_dispatch_geometry_args(
                                kernel_sources, affine_soa, dim
                            ),
                            *storage_args,
                            *field_args,
                        ]
                    ),
                )
            )
        else:
            lines.append(
                '%s        SFEM_ERROR("%s %s affine %dd dispatch was not generated\\n");'
                % (indent, material_name, operation, dim)
            )
            lines.append("%s        return SFEM_FAILURE;" % indent)
        lines.append("%s    }" % indent)
        if operation == "jacobian_action" and _c_abi_function_exists(
            kernel_sources, packed, public_only=True
        ):
            lines.extend(
                [
                    "%s    if (impl_->space->has_packed_mesh()) {" % indent,
                    "%s        auto packed = impl_->space->packed_mesh();" % indent,
                    "%s        const int packed_block = packed_block_id_for_domain(*packed, *domain.block);" % indent,
                    "%s        if (packed_block >= 0) {" % indent,
                    "%s            auto packed_elements = packed->elements(packed_block);" % indent,
                    "%s            auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);" % indent,
                    "%s            auto n_shared_nodes = packed->n_shared(packed_block);" % indent,
                    "%s            auto ghost_ptr = packed->ghost_ptr(packed_block);" % indent,
                    "%s            auto ghost_idx = packed->ghost_idx(packed_block);" % indent,
                    "%s            return %s(%s);"
                    % (
                        indent,
                        packed,
                        ", ".join(
                            [
                                "domain.element_type",
                                "real_type",
                                "packed->n_packs(packed_block)",
                                "packed->n_elements_per_pack(packed_block)",
                                "domain.block->n_elements()",
                                "mesh->n_nodes()",
                                "packed->max_nodes_per_pack()",
                                "packed_elements->data()",
                                "owned_nodes_ptr->data()",
                                "n_shared_nodes->data()",
                                "ghost_ptr->data()",
                                "ghost_idx->data()",
                                "points",
                                *storage_args,
                                *field_args,
                            ]
                        ),
                    ),
                    "%s        }" % indent,
                    "%s    }" % indent,
                ]
            )
        if _c_abi_function_exists(kernel_sources, isop, public_only=True):
            lines.append(
                "%s    return %s(%s);"
                % (
                    indent,
                    isop,
                    ", ".join(
                        [
                            *common_args,
                            "points",
                            *storage_args,
                            *field_args,
                        ]
                    ),
                )
            )
        else:
            lines.append(
                '%s    SFEM_ERROR("%s %s isoparametric %dd dispatch was not generated\\n");'
                % (indent, material_name, operation, dim)
            )
            lines.append("%s    return SFEM_FAILURE;" % indent)
        lines.append("%s}" % indent)
    lines.extend(
        [
            '%sSFEM_ERROR("%s %s does not support spatial dimension %%d\\n", dim);'
            % (indent, material_name, operation),
            "%sreturn SFEM_FAILURE;" % indent,
        ]
    )
    return "\n".join(lines)


def _packed_two_pass_function(function):
    if "_packed_two_pass_" in function:
        return function
    return function.replace("_packed_", "_packed_two_pass_", 1)


def _packed_call_args_common():
    return [
        "packed->n_packs(packed_block)",
        "packed->n_elements_per_pack(packed_block)",
        "domain.block->n_elements()",
        "mesh->n_nodes()",
        "packed->max_nodes_per_pack()",
        "packed_elements->data()",
        "owned_nodes_ptr->data()",
        "n_shared_nodes->data()",
        "ghost_ptr->data()",
        "ghost_idx->data()",
    ]


def _packed_two_pass_extra_args():
    return [
        "packed->n_ghost_entries(packed_block)",
        "packed->n_ghost_reduce_rows(packed_block)",
        "ghost_reduce_ptr->data()",
        "ghost_reduce_idx->data()",
        "ghost_reduce_dest->data()",
        "impl_->packed_ghost_buf[packed_block]->data()",
    ]


def _with_runtime_type_argument(leading_args):
    """``leading_args`` with the runtime scalar type after the element type.

    The merged entry point takes it second, so every caller has to supply it
    there.  Doing it here rather than at each packed call site keeps the one
    rule in one place.
    """
    args = list(leading_args)
    if _RUNTIME_TYPE_ARGUMENT in args:
        return tuple(args)
    for index, arg in enumerate(args):
        if arg == "domain.element_type":
            args.insert(index + 1, _RUNTIME_TYPE_ARGUMENT)
            return tuple(args)
    return tuple(args)


def _hyperelastic_packed_return(indent, function, leading_args, trailing_args, kernel_sources):
    """leading_args usually ['domain.element_type']; trailing follows ghost_idx.

    `trailing_args` may be a callable taking the callee's name, for the same
    reason `_affine_dispatch_call_lines` takes one: a metric-geometry sibling
    takes different geometry from the plain dispatch, and each is asked what it
    takes.  A plain sequence is accepted unchanged, for the callers whose
    arguments do not depend on the callee.
    """
    metric = _metric_dispatch_name(function)
    elements = ()
    if _c_abi_function_exists(kernel_sources, metric, public_only=True):
        elements = _metric_dispatch_elements(kernel_sources, metric)
    lines = []
    if elements:
        lines.extend(
            [
                "%sif (%s) {"
                % (
                    indent,
                    " || ".join(
                        "domain.element_type == smesh::%s" % element
                        for element in elements
                    ),
                ),
                *_packed_return_lines(
                    indent + "    ", metric, leading_args, trailing_args, kernel_sources
                ),
                "%s}" % indent,
            ]
        )
    lines.extend(
        _packed_return_lines(indent, function, leading_args, trailing_args, kernel_sources)
    )
    return lines


def _packed_return_lines(indent, function, leading_args, trailing_args, kernel_sources):
    """One packed dispatch call, one-pass or two-pass as the operator asks."""
    trailing = (
        list(trailing_args(function)) if callable(trailing_args) else list(trailing_args)
    )
    two_pass = _packed_two_pass_function(function)
    leading_args = _with_runtime_type_argument(leading_args)
    base = list(leading_args) + _packed_call_args_common()
    one_call = ", ".join(base + trailing)
    if not _c_abi_function_exists(kernel_sources, two_pass, public_only=True):
        return ["%sreturn %s(%s);" % (indent, function, one_call)]
    two_trailing = (
        list(trailing_args(two_pass)) if callable(trailing_args) else trailing
    )
    two_call = ", ".join(
        list(leading_args)
        + _packed_call_args_common()
        + _packed_two_pass_extra_args()
        + two_trailing
    )
    return [
        "%sif (impl_->use_packed_two_pass) {" % indent,
        "%s    return %s(%s);" % (indent, two_pass, two_call),
        "%s}" % indent,
        "%sreturn %s(%s);" % (indent, function, one_call),
    ]


def _hyperelastic_gradient_dispatch_body(material_name, kernel_sources, gradient_dependencies_by_dim, indent, n_field_components_by_dim=None):
    lines = ["%sconst int dim = mesh->spatial_dimension();" % indent]
    emitted = False
    for dim in (2, 3):
        dependencies = gradient_dependencies_by_dim.get(dim)
        if dependencies is None:
            continue
        prefix = "if" if not emitted else "else if"
        emitted = True
        components = _components((n_field_components_by_dim or {}).get(dim, dim))
        current_args = [str(arg) for arg in _energy_field_args(dependencies, dim, components, current="x")]
        output_args = [str(arg) for arg in _energy_output_args(dim, components)]
        affine = "%s_gradient_%dd_affine_mesh_soa" % (material_name, dim)
        affine_aos_unit = "%s_gradient_%dd_affine_mesh_soa_aos_unit" % (material_name, dim)
        isop = "%s_gradient_%dd_isoparametric_mesh_soa" % (material_name, dim)
        lines.append("%s%s (dim == %d) {" % (indent, prefix, dim))
        lines.append("%s    if (impl_->gradient_uses_affine) {" % indent)
        affine_aos_unit_elements = _c_abi_public_dispatch_case_elements(
            kernel_sources,
            affine_aos_unit,
        )
        if affine_aos_unit_elements:
            lines.append(
                "%s        if (adjugate_aos && (%s)) {"
                % (
                    indent,
                    _element_condition("domain.element_type", affine_aos_unit_elements),
                )
            )
            lines.append(
                "%s            return %s(%s);"
                % (
                    indent,
                    affine_aos_unit,
                    ", ".join(
                        [
                            "domain.element_type",
                            "real_type",
                            "domain.block->n_elements()",
                            "mesh->n_nodes()",
                            "domain.block->elements()->data()",
                            "adjugate_aos",
                            "determinant",
                            *_c_abi_ordered_domain_parameter_args(
                                kernel_sources,
                                affine_aos_unit,
                                dependencies,
                            ),
                            *current_args,
                            *output_args,
                        ]
                    ),
                )
            )
            lines.append("%s        }" % indent)
        if _c_abi_function_exists(kernel_sources, affine, public_only=True):
            lines.extend(
                _affine_dispatch_call_lines(
                    kernel_sources,
                    affine,
                    indent + "        ",
                    lambda callee: ", ".join(
                        [
                            "domain.element_type",
                            "real_type",
                            "domain.block->n_elements()",
                            "mesh->n_nodes()",
                            "domain.block->elements()->data()",
                            *_affine_geometry_call_args(kernel_sources, callee, dim),
                            *_c_abi_ordered_domain_parameter_args(
                                kernel_sources,
                                callee,
                                dependencies,
                            ),
                            *current_args,
                            *output_args,
                        ]
                    ),
                )
            )
        else:
            lines.append('%s        SFEM_ERROR("%s affine gradient %dd dispatch was not generated\\n");' % (indent, material_name, dim))
            lines.append("%s        return SFEM_FAILURE;" % indent)
        lines.append("%s    }" % indent)
        if _c_abi_function_exists(kernel_sources, isop, public_only=True):
            lines.append(
                "%s    return %s(%s);"
                % (
                    indent,
                    isop,
                    ", ".join(
                        [
                            "domain.element_type",
                            "real_type",
                            "domain.block->n_elements()",
                            "mesh->n_nodes()",
                            "domain.block->elements()->data()",
                            "points",
                            *_c_abi_ordered_domain_parameter_args(
                                kernel_sources,
                                isop,
                                dependencies,
                            ),
                            *current_args,
                            *output_args,
                        ]
                    ),
                )
            )
        else:
            lines.append('%s    SFEM_ERROR("%s isoparametric gradient %dd dispatch was not generated\\n");' % (indent, material_name, dim))
            lines.append("%s    return SFEM_FAILURE;" % indent)
        lines.append("%s}" % indent)
    lines.extend(
        [
            '%sSFEM_ERROR("%s gradient does not support spatial dimension %%d\\n", dim);' % (indent, material_name),
            "%sreturn SFEM_FAILURE;" % indent,
        ]
    )
    return "\n".join(lines)


def _hyperelastic_objective_dispatch_body(material_name, kernel_sources, objective_dependencies_by_dim, indent, n_field_components_by_dim=None):
    lines = ["%sconst int dim = mesh->spatial_dimension();" % indent]
    emitted = False
    emitted_dims = []
    for dim in (2, 3):
        dependencies = objective_dependencies_by_dim.get(dim)
        if dependencies is None:
            continue
        prefix = "if" if not emitted else "else if"
        emitted = True
        emitted_dims.append(dim)
        components = _components((n_field_components_by_dim or {}).get(dim, dim))
        parameter_args = list(_dependency_domain_parameter_args(dependencies))
        current_args = [str(arg) for arg in _energy_field_args(dependencies, dim, components, current="x")]
        affine = "%s_objective_%dd_affine_mesh_soa" % (material_name, dim)
        isop = "%s_objective_%dd_isoparametric_mesh_soa" % (material_name, dim)
        lines.append("%s%s (dim == %d) {" % (indent, prefix, dim))
        lines.append("%s    if (impl_->objective_uses_affine) {" % indent)
        if _c_abi_function_exists(kernel_sources, affine, public_only=True):
            lines.append(
                "%s        status = %s(%s);"
                % (
                    indent,
                    affine,
                    ", ".join(
                        [
                            "domain.element_type",
                            "real_type",
                            "nelements",
                            "mesh->n_nodes()",
                            "domain.block->elements()->data()",
                            *_affine_geometry_call_args(kernel_sources, affine, dim),
                            *parameter_args,
                            *current_args,
                            "impl_->element_values.get()",
                        ]
                    ),
                )
            )
        else:
            lines.append('%s        SFEM_ERROR("%s affine objective %dd dispatch was not generated\\n");' % (indent, material_name, dim))
            lines.append("%s        return SFEM_FAILURE;" % indent)
        lines.append("%s    } else {" % indent)
        if _c_abi_function_exists(kernel_sources, isop, public_only=True):
            lines.append(
                "%s        status = %s(%s);"
                % (
                    indent,
                    isop,
                    ", ".join(
                        [
                            "domain.element_type",
                            "real_type",
                            "nelements",
                            "mesh->n_nodes()",
                            "domain.block->elements()->data()",
                            "points",
                            *parameter_args,
                            *current_args,
                            "impl_->element_values.get()",
                        ]
                    ),
                )
            )
        else:
            lines.append('%s        SFEM_ERROR("%s isoparametric objective %dd dispatch was not generated\\n");' % (indent, material_name, dim))
            lines.append("%s        return SFEM_FAILURE;" % indent)
        lines.append("%s    }" % indent)
        lines.append("%s}" % indent)
    unsupported_condition = " && ".join("dim != %d" % dim for dim in emitted_dims) or "true"
    lines.extend(
        [
            '%sif (%s) {' % (indent, unsupported_condition),
            '%s    SFEM_ERROR("%s objective does not support spatial dimension %%d\\n", dim);' % (indent, material_name),
            "%s    return SFEM_FAILURE;" % indent,
            "%s}" % indent,
        ]
    )
    return "\n".join(lines)


def _hyperelastic_objective_steps_dispatch_body(material_name, kernel_sources, objective_dependencies_by_dim, indent, n_field_components_by_dim=None):
    lines = ["%sconst int dim = mesh->spatial_dimension();" % indent]
    emitted = False
    emitted_dims = []
    for dim in (2, 3):
        dependencies = objective_dependencies_by_dim.get(dim)
        if dependencies is None:
            continue
        prefix = "if" if not emitted else "else if"
        emitted = True
        emitted_dims.append(dim)
        components = _components((n_field_components_by_dim or {}).get(dim, dim))
        parameter_args = list(_dependency_domain_parameter_args(dependencies))
        current_args = [str(arg) for arg in _energy_field_args(dependencies, dim, components, current="x")]
        direction_args = [str(dim), _offsets("h", components)]
        affine = "%s_objective_steps_%dd_affine_mesh_soa" % (material_name, dim)
        isop = "%s_objective_steps_%dd_isoparametric_mesh_soa" % (material_name, dim)
        lines.append("%s%s (dim == %d) {" % (indent, prefix, dim))
        lines.append("%s    if (impl_->objective_uses_affine) {" % indent)
        if _c_abi_function_exists(kernel_sources, affine, public_only=True):
            lines.extend(
                _affine_dispatch_status_lines(
                    kernel_sources,
                    affine,
                    indent + "        ",
                    lambda callee: ", ".join(
                        [
                            "domain.element_type",
                            "real_type",
                            "nelements",
                            "mesh->n_nodes()",
                            "domain.block->elements()->data()",
                            *_affine_geometry_call_args(kernel_sources, callee, dim),
                            *parameter_args,
                            *current_args,
                            *direction_args,
                            "nsteps",
                            "steps",
                            "impl_->element_values.get()",
                        ]
                    ),
                )
            )
        else:
            lines.append('%s        SFEM_ERROR("%s affine objective_steps %dd dispatch was not generated\\n");' % (indent, material_name, dim))
            lines.append("%s        return SFEM_FAILURE;" % indent)
        lines.append("%s    } else {" % indent)
        if _c_abi_function_exists(kernel_sources, isop, public_only=True):
            lines.append(
                "%s        status = %s(%s);"
                % (
                    indent,
                    isop,
                    ", ".join(
                        [
                            "domain.element_type",
                            "real_type",
                            "nelements",
                            "mesh->n_nodes()",
                            "domain.block->elements()->data()",
                            "points",
                            *parameter_args,
                            *current_args,
                            *direction_args,
                            "nsteps",
                            "steps",
                            "impl_->element_values.get()",
                        ]
                    ),
                )
            )
        else:
            lines.append('%s        SFEM_ERROR("%s isoparametric objective_steps %dd dispatch was not generated\\n");' % (indent, material_name, dim))
            lines.append("%s        return SFEM_FAILURE;" % indent)
        lines.append("%s    }" % indent)
        lines.append("%s}" % indent)
    unsupported_condition = " && ".join("dim != %d" % dim for dim in emitted_dims) or "true"
    lines.extend(
        [
            '%sif (%s) {' % (indent, unsupported_condition),
            '%s    SFEM_ERROR("%s objective_steps does not support spatial dimension %%d\\n", dim);' % (indent, material_name),
            "%s    return SFEM_FAILURE;" % indent,
            "%s}" % indent,
        ]
    )
    return "\n".join(lines)


def _hyperelastic_apply_dispatch_body(material_name, kernel_sources, apply_dependencies_by_dim, indent, n_field_components_by_dim=None):
    lines = [
        "%sconst int dim = mesh->spatial_dimension();" % indent,
    ]
    for dim in (2, 3):
        prefix = "if" if dim == 2 else "else if"
        dependencies = apply_dependencies_by_dim.get(dim)
        uses_current = getattr(dependencies, "current", True)
        uses_direction = getattr(dependencies, "direction", True)
        parameter_args = list(_dependency_domain_parameter_args(dependencies))
        # The stride and the offsets are the field's component count, not the
        # spatial dimension: one array per component of the field being applied.
        n_components = (n_field_components_by_dim or {}).get(dim, dim)
        current_args = ([str(n_components)] + ["x + %d" % d for d in range(n_components)]) if uses_current else []
        direction_args = ([str(n_components)] + ["h + %d" % d for d in range(n_components)]) if uses_direction else []
        output_args = [str(n_components)] + ["out + %d" % d for d in range(n_components)]
        affine = "%s_apply_%dd_affine_mesh_soa" % (material_name, dim)
        isop = "%s_apply_%dd_isoparametric_mesh_soa" % (material_name, dim)
        packed = "%s_apply_packed_%dd_isoparametric_mesh_soa" % (material_name, dim)
        packed_affine = "%s_apply_packed_%dd_affine_mesh_soa" % (material_name, dim)
        lines.append("%s%s (dim == %d) {" % (indent, prefix, dim))
        lines.append("%s    if (impl_->apply_uses_affine) {" % indent)
        if _c_abi_function_exists(kernel_sources, affine, public_only=True):
            if _c_abi_function_exists(kernel_sources, packed_affine, public_only=True):
                lines.extend(
                    [
                        "%s        if (impl_->space->has_packed_mesh()) {" % indent,
                        "%s            auto packed = impl_->space->packed_mesh();" % indent,
                        "%s            const int packed_block = packed_block_id_for_domain(*packed, *domain.block);" % indent,
                        "%s            if (packed_block >= 0) {" % indent,
                        "%s                auto packed_elements = packed->elements(packed_block);" % indent,
                        "%s                auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);" % indent,
                        "%s                auto n_shared_nodes = packed->n_shared(packed_block);" % indent,
                        "%s                auto ghost_ptr = packed->ghost_ptr(packed_block);" % indent,
                        "%s                auto ghost_idx = packed->ghost_idx(packed_block);" % indent,
                        "%s                auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);" % indent,
                        "%s                auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);" % indent,
                        "%s                auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);" % indent,
                    ]
                )
                lines.extend(
                    _hyperelastic_packed_return(
                        indent + "                ",
                        packed_affine,
                        ["domain.element_type"],
                        lambda callee: [
                            *_affine_geometry_call_args(kernel_sources, callee, dim),
                            *parameter_args,
                            *current_args,
                            *direction_args,
                            *output_args,
                        ],
                        kernel_sources,
                    )
                )
                lines.extend(
                    [
                        "%s            }" % indent,
                        "%s        }" % indent,
                    ]
                )
            lines.extend(
                _affine_dispatch_call_lines(
                    kernel_sources,
                    affine,
                    indent + "        ",
                    lambda callee: ", ".join(
                        [
                            "domain.element_type",
                            "real_type",
                            "domain.block->n_elements()",
                            "mesh->n_nodes()",
                            "domain.block->elements()->data()",
                            *_affine_geometry_call_args(kernel_sources, callee, dim),
                            *parameter_args,
                            *current_args,
                            *direction_args,
                            *output_args,
                        ]
                    ),
                )
            )
        else:
            lines.append('%s        SFEM_ERROR("%s affine apply %dd dispatch was not generated\\n");' % (indent, material_name, dim))
            lines.append("%s        return SFEM_FAILURE;" % indent)
        lines.append("%s    }" % indent)
        if _c_abi_function_exists(kernel_sources, packed, public_only=True):
            lines.extend(
                [
                    "%s    if (impl_->space->has_packed_mesh()) {" % indent,
                    "%s        auto packed = impl_->space->packed_mesh();" % indent,
                    "%s        const int packed_block = packed_block_id_for_domain(*packed, *domain.block);" % indent,
                    "%s        if (packed_block >= 0) {" % indent,
                    "%s            auto packed_elements = packed->elements(packed_block);" % indent,
                    "%s            auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);" % indent,
                    "%s            auto n_shared_nodes = packed->n_shared(packed_block);" % indent,
                    "%s            auto ghost_ptr = packed->ghost_ptr(packed_block);" % indent,
                    "%s            auto ghost_idx = packed->ghost_idx(packed_block);" % indent,
                    "%s            auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);" % indent,
                    "%s            auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);" % indent,
                    "%s            auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);" % indent,
                ]
            )
            lines.extend(
                _hyperelastic_packed_return(
                    indent + "            ",
                    packed,
                    ["domain.element_type"],
                    [
                        "points",
                        *parameter_args,
                        *current_args,
                        *direction_args,
                        *output_args,
                    ],
                    kernel_sources,
                )
            )
            lines.extend(
                [
                    "%s        }" % indent,
                    "%s    }" % indent,
                ]
            )
        if _c_abi_function_exists(kernel_sources, isop, public_only=True):
            lines.append(
                "%s    return %s(%s);"
                % (
                    indent,
                    isop,
                    ", ".join(
                        [
                            "domain.element_type",
                            "real_type",
                            "domain.block->n_elements()",
                            "mesh->n_nodes()",
                            "domain.block->elements()->data()",
                            "points",
                            *parameter_args,
                            *current_args,
                            *direction_args,
                            *output_args,
                        ]
                    ),
                )
            )
        else:
            lines.append('%s    SFEM_ERROR("%s isoparametric apply %dd dispatch was not generated\\n");' % (indent, material_name, dim))
            lines.append("%s    return SFEM_FAILURE;" % indent)
        lines.append("%s}" % indent)
    lines.extend(
        [
            '%sSFEM_ERROR("%s apply does not support spatial dimension %%d\\n", dim);' % (indent, material_name),
            "%sreturn SFEM_FAILURE;" % indent,
        ]
    )
    return "\n".join(lines)


def _hyperelastic_gradient_packed_dispatch_body(material_name, kernel_sources, gradient_dependencies_by_dim, indent, n_field_components_by_dim=None):
    affine_lines = [
        "%sif (impl_->gradient_uses_affine && impl_->space->has_packed_mesh()) {" % indent,
        "%s    auto packed = impl_->space->packed_mesh();" % indent,
        "%s    const int packed_block = packed_block_id_for_domain(*packed, *domain.block);" % indent,
        "%s    if (packed_block >= 0) {" % indent,
        "%s        auto packed_elements = packed->elements(packed_block);" % indent,
        "%s        auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);" % indent,
        "%s        auto n_shared_nodes = packed->n_shared(packed_block);" % indent,
        "%s        auto ghost_ptr = packed->ghost_ptr(packed_block);" % indent,
        "%s        auto ghost_idx = packed->ghost_idx(packed_block);" % indent,
        "%s        auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);" % indent,
        "%s        auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);" % indent,
        "%s        auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);" % indent,
        "%s        const int dim = mesh->spatial_dimension();" % indent,
    ]
    emitted_affine = False
    for dim in (2, 3):
        dependencies = gradient_dependencies_by_dim.get(dim)
        if dependencies is None:
            continue
        function = "%s_gradient_packed_%dd_affine_mesh_soa" % (material_name, dim)
        if not _c_abi_function_exists(kernel_sources, function, public_only=True):
            continue
        prefix = "if" if not emitted_affine else "else if"
        emitted_affine = True
        parameter_args = list(_dependency_domain_parameter_args(dependencies))
        current_args = (
            [str(_packed_n_components(n_field_components_by_dim, dim))] + ["x + %d" % d for d in range(_packed_n_components(n_field_components_by_dim, dim))]
            if getattr(dependencies, "current", True)
            else []
        )
        output_args = [str(_packed_n_components(n_field_components_by_dim, dim))] + ["out + %d" % d for d in range(_packed_n_components(n_field_components_by_dim, dim))]
        affine_lines.append("%s        %s (dim == %d) {" % (indent, prefix, dim))
        affine_lines.extend(
            _hyperelastic_packed_return(
                indent + "            ",
                function,
                ["domain.element_type"],
                lambda callee: [
                    *_affine_geometry_call_args(kernel_sources, callee, dim),
                    *parameter_args,
                    *current_args,
                    *output_args,
                ],
                kernel_sources,
            )
        )
        affine_lines.append("%s        }" % indent)
    affine_lines.extend(
        [
            "%s    }" % indent,
            "%s}" % indent,
        ]
    )
    lines = [
        "%sif (!impl_->gradient_uses_affine && impl_->space->has_packed_mesh()) {" % indent,
        "%s    auto packed = impl_->space->packed_mesh();" % indent,
        "%s    const int packed_block = packed_block_id_for_domain(*packed, *domain.block);" % indent,
        "%s    if (packed_block >= 0) {" % indent,
        "%s        auto packed_elements = packed->elements(packed_block);" % indent,
        "%s        auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);" % indent,
        "%s        auto n_shared_nodes = packed->n_shared(packed_block);" % indent,
        "%s        auto ghost_ptr = packed->ghost_ptr(packed_block);" % indent,
        "%s        auto ghost_idx = packed->ghost_idx(packed_block);" % indent,
        "%s        auto ghost_reduce_ptr = packed->ghost_reduce_ptr(packed_block);" % indent,
        "%s        auto ghost_reduce_idx = packed->ghost_reduce_idx(packed_block);" % indent,
        "%s        auto ghost_reduce_dest = packed->ghost_reduce_dest(packed_block);" % indent,
        "%s        const int dim = mesh->spatial_dimension();" % indent,
    ]
    emitted = False
    for dim in (2, 3):
        dependencies = gradient_dependencies_by_dim.get(dim)
        if dependencies is None:
            continue
        function = "%s_gradient_packed_%dd_isoparametric_mesh_soa" % (material_name, dim)
        if not _c_abi_function_exists(kernel_sources, function, public_only=True):
            continue
        emitted = True
        prefix = "if" if emitted and not any("if (dim ==" in line for line in lines) else "else if"
        parameter_args = list(_dependency_domain_parameter_args(dependencies))
        current_args = (
            [str(_packed_n_components(n_field_components_by_dim, dim))] + ["x + %d" % d for d in range(_packed_n_components(n_field_components_by_dim, dim))]
            if getattr(dependencies, "current", True)
            else []
        )
        output_args = [str(_packed_n_components(n_field_components_by_dim, dim))] + ["out + %d" % d for d in range(_packed_n_components(n_field_components_by_dim, dim))]
        lines.append("%s        %s (dim == %d) {" % (indent, prefix, dim))
        lines.extend(
            _hyperelastic_packed_return(
                indent + "            ",
                function,
                ["domain.element_type"],
                [
                    "points",
                    *parameter_args,
                    *current_args,
                    *output_args,
                ],
                kernel_sources,
            )
        )
        lines.append("%s        }" % indent)
    lines.extend(
        [
            "%s    }" % indent,
            "%s}" % indent,
        ]
    )
    packed_blocks = []
    if emitted_affine:
        packed_blocks.append("\n".join(affine_lines))
    if emitted:
        packed_blocks.append("\n".join(lines))
    return "\n".join(packed_blocks)


def _hyperelastic_objective_steps_packed_dispatch_body(material_name, kernel_sources, apply_dependencies_by_dim, indent, n_field_components_by_dim=None):
    affine_lines = [
        "%sif (impl_->objective_uses_affine && impl_->space->has_packed_mesh()) {" % indent,
        "%s    auto packed = impl_->space->packed_mesh();" % indent,
        "%s    const int packed_block = packed_block_id_for_domain(*packed, *domain.block);" % indent,
        "%s    if (packed_block >= 0) {" % indent,
        "%s        auto packed_elements = packed->elements(packed_block);" % indent,
        "%s        auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);" % indent,
        "%s        auto n_shared_nodes = packed->n_shared(packed_block);" % indent,
        "%s        auto ghost_ptr = packed->ghost_ptr(packed_block);" % indent,
        "%s        auto ghost_idx = packed->ghost_idx(packed_block);" % indent,
        "%s        const int dim = mesh->spatial_dimension();" % indent,
    ]
    emitted_affine = False
    for dim in (2, 3):
        dependencies = apply_dependencies_by_dim.get(dim)
        if dependencies is None:
            continue
        function = "%s_objective_steps_packed_%dd_affine_mesh_soa" % (material_name, dim)
        if not _c_abi_function_exists(kernel_sources, function, public_only=True):
            continue
        prefix = "if" if not emitted_affine else "else if"
        emitted_affine = True
        parameter_args = list(_dependency_domain_parameter_args(dependencies))
        current_args = (
            [str(_packed_n_components(n_field_components_by_dim, dim))] + ["x + %d" % d for d in range(_packed_n_components(n_field_components_by_dim, dim))]
            if getattr(dependencies, "current", True)
            else []
        )
        direction_args = [str(_packed_n_components(n_field_components_by_dim, dim))] + ["h + %d" % d for d in range(_packed_n_components(n_field_components_by_dim, dim))]
        affine_lines.extend(
            [
                "%s        %s (dim == %d) {" % (indent, prefix, dim),
                "%s            status = %s(%s);" % (
                    indent,
                    function,
                    ", ".join(
                        [
                            "domain.element_type",
                            "real_type",
                            "packed->n_packs(packed_block)",
                            "packed->n_elements_per_pack(packed_block)",
                            "domain.block->n_elements()",
                            "mesh->n_nodes()",
                            "packed->max_nodes_per_pack()",
                            "packed_elements->data()",
                            "owned_nodes_ptr->data()",
                            "n_shared_nodes->data()",
                            "ghost_ptr->data()",
                            "ghost_idx->data()",
                            *_affine_geometry_call_args(kernel_sources, function, dim),
                            *parameter_args,
                            *current_args,
                            *direction_args,
                            "nsteps",
                            "steps",
                            "impl_->element_values.get()",
                        ]
                    ),
                ),
                "%s        }" % indent,
            ]
        )
    affine_lines.extend(
        [
            "%s    }" % indent,
            "%s}" % indent,
        ]
    )
    lines = [
        "%sif (!impl_->objective_uses_affine && impl_->space->has_packed_mesh()) {" % indent,
        "%s    auto packed = impl_->space->packed_mesh();" % indent,
        "%s    const int packed_block = packed_block_id_for_domain(*packed, *domain.block);" % indent,
        "%s    if (packed_block >= 0) {" % indent,
        "%s        auto packed_elements = packed->elements(packed_block);" % indent,
        "%s        auto owned_nodes_ptr = packed->owned_nodes_ptr(packed_block);" % indent,
        "%s        auto n_shared_nodes = packed->n_shared(packed_block);" % indent,
        "%s        auto ghost_ptr = packed->ghost_ptr(packed_block);" % indent,
        "%s        auto ghost_idx = packed->ghost_idx(packed_block);" % indent,
        "%s        const int dim = mesh->spatial_dimension();" % indent,
    ]
    emitted = False
    for dim in (2, 3):
        dependencies = apply_dependencies_by_dim.get(dim)
        if dependencies is None:
            continue
        function = "%s_objective_steps_packed_%dd_isoparametric_mesh_soa" % (material_name, dim)
        if not _c_abi_function_exists(kernel_sources, function, public_only=True):
            continue
        emitted = True
        prefix = "if" if emitted and not any("if (dim ==" in line for line in lines) else "else if"
        parameter_args = list(_dependency_domain_parameter_args(dependencies))
        current_args = (
            [str(_packed_n_components(n_field_components_by_dim, dim))] + ["x + %d" % d for d in range(_packed_n_components(n_field_components_by_dim, dim))]
            if getattr(dependencies, "current", True)
            else []
        )
        direction_args = (
            [str(_packed_n_components(n_field_components_by_dim, dim))] + ["h + %d" % d for d in range(_packed_n_components(n_field_components_by_dim, dim))]
        )
        lines.extend(
            [
                "%s        %s (dim == %d) {" % (indent, prefix, dim),
                "%s            status = %s(%s);" % (
                    indent,
                    function,
                    ", ".join(
                        [
                            "domain.element_type",
                            "real_type",
                            "packed->n_packs(packed_block)",
                            "packed->n_elements_per_pack(packed_block)",
                            "domain.block->n_elements()",
                            "mesh->n_nodes()",
                            "packed->max_nodes_per_pack()",
                            "packed_elements->data()",
                            "owned_nodes_ptr->data()",
                            "n_shared_nodes->data()",
                            "ghost_ptr->data()",
                            "ghost_idx->data()",
                            "points",
                            *parameter_args,
                            *current_args,
                            *direction_args,
                            "nsteps",
                            "steps",
                            "impl_->element_values.get()",
                        ]
                    ),
                ),
                "%s        }" % indent,
            ]
        )
    lines.extend(
        [
            "%s    }" % indent,
            "%s}" % indent,
        ]
    )
    packed_blocks = []
    if emitted_affine:
        packed_blocks.append("\n".join(affine_lines))
    if emitted:
        packed_blocks.append("\n".join(lines))
    return "\n".join(packed_blocks)


def _hyperelastic_hessian_dispatch_body(material_name, operation, kernel_sources, apply_dependencies_by_dim, tail_args, indent):
    lines = ["%sconst int dim = mesh->spatial_dimension();" % indent]
    for dim in (2, 3):
        prefix = "if" if dim == 2 else "else if"
        function = "%s_%s_%dd_isoparametric_mesh_soa" % (
            material_name,
            operation,
            dim,
        )
        dependencies = apply_dependencies_by_dim.get(dim)
        parameter_args = list(_dependency_domain_parameter_args(dependencies))
        current_args = (
            [str(dim)] + ["current + %d" % d for d in range(dim)]
            if getattr(dependencies, "current", True)
            else []
        )
        lines.append("%s%s (dim == %d) {" % (indent, prefix, dim))
        if _c_abi_function_exists(kernel_sources, function, public_only=True):
            lines.append(
                "%s    return %s(%s);"
                % (
                    indent,
                    function,
                    ", ".join(
                        [
                            "domain.element_type",
                            "real_type",
                            "domain.block->n_elements()",
                            "mesh->n_nodes()",
                            "domain.block->elements()->data()",
                            "points",
                            *parameter_args,
                            *current_args,
                            *tail_args,
                        ]
                    ),
                )
            )
        else:
            lines.append('%s    SFEM_ERROR("%s %s %dd dispatch was not generated\\n");' % (indent, material_name, operation, dim))
            lines.append("%s    return SFEM_FAILURE;" % indent)
        lines.append("%s}" % indent)
    lines.extend(
        [
            '%sSFEM_ERROR("%s %s does not support spatial dimension %%d\\n", dim);' % (indent, material_name, operation),
            "%sreturn SFEM_FAILURE;" % indent,
        ]
    )
    return "\n".join(lines)


def _hyperelastic_hessian_current_prologue(op_name, operation, apply_dependencies_by_dim):
    uses_current = any(
        bool(getattr(dependencies, "current", False))
        for dependencies in apply_dependencies_by_dim.values()
    )
    if not uses_current:
        return "        (void)x;"
    return "\n".join(
        (
            "        const real_t *const current = x;",
            "        if (!current) {",
            '            SFEM_ERROR("%s::%s requires a current state\\n");'
            % (op_name, operation),
            "            return SFEM_FAILURE;",
            "        }",
        )
    )


def _dual_case(element, flag, affine_function, affine_arguments, isoparametric_function, isoparametric_arguments):
    return """%(cases)s
                    return impl_->%(flag)s ? %(affine_function)s(%(affine_arguments)s) : %(isoparametric_function)s(%(isoparametric_arguments)s);""" % {
        "cases": _mesh_case_labels(element, "                "),
        "flag": flag,
        "affine_function": affine_function,
        "affine_arguments": affine_arguments,
        "isoparametric_function": isoparametric_function,
        "isoparametric_arguments": isoparametric_arguments,
    }


def _dual_aos_unit_case(
    element,
    flag,
    affine_aos_function,
    affine_aos_arguments,
    affine_function,
    affine_arguments,
    isoparametric_function,
    isoparametric_arguments,
):
    return """%(cases)s
                    if (impl_->%(flag)s) {
                        return adjugate_aos ? %(affine_aos_function)s(%(affine_aos_arguments)s) : %(affine_function)s(%(affine_arguments)s);
                    }
                    return %(isoparametric_function)s(%(isoparametric_arguments)s);""" % {
        "cases": _mesh_case_labels(element, "                "),
        "flag": flag,
        "affine_aos_function": affine_aos_function,
        "affine_aos_arguments": affine_aos_arguments,
        "affine_function": affine_function,
        "affine_arguments": affine_arguments,
        "isoparametric_function": isoparametric_function,
        "isoparametric_arguments": isoparametric_arguments,
    }


def _dual_status_case(element, affine_function, affine_arguments, isoparametric_function, isoparametric_arguments):
    return """%(cases)s
                    status = impl_->objective_uses_affine ? %(affine_function)s(%(affine_arguments)s) : %(isoparametric_function)s(%(isoparametric_arguments)s);
                    break;""" % {
        "cases": _mesh_case_labels(element, "                "),
        "affine_function": affine_function,
        "affine_arguments": affine_arguments,
        "isoparametric_function": isoparametric_function,
        "isoparametric_arguments": isoparametric_arguments,
    }


def _seed_lines(defaults):
    return "\n".join(
        '            parameters.set_value("%s", %.17g);' % (name, value)
        for name, value in defaults
    )


def _yaml_helpers(defaults):
    nparameters = len(defaults)
    storage_size = max(1, nparameters)
    names = ", ".join('"%s"' % name for name, _ in defaults) or "nullptr"
    default_lines = []
    for i, (_, value) in enumerate(defaults):
        default_lines.append("            values[%d] = %.17g;" % (i, value))
    if not default_lines:
        default_lines.append("            values[0] = 0;")
    return """        struct AffineOption {
            const char *name;
            bool       *flag;
        };

        inline bool set_affine_option(const std::string &name,
                                      const bool val,
                                      const AffineOption *const options,
                                      const int n_options) {
            if (name == "ASSUME_AFFINE" || name == "assume_affine") {
                for (int i = 0; i < n_options; ++i) {
                    *options[i].flag = val;
                }
                return true;
            }
            bool matched = false;
            for (int i = 0; i < n_options; ++i) {
                if (name == options[i].name) {
                    *options[i].flag = val;
                    matched = true;
                }
            }
            return matched;
        }

        void material_defaults(real_t *const values) {
%(default_lines)s
        }

#ifdef SFEM_ENABLE_RYAML
        constexpr int N_DEFINED_MATERIAL_PARAMETERS = %(nparameters)d;
        constexpr int N_MATERIAL_PARAMETERS = %(storage_size)d;
        static const char *const MATERIAL_PARAMETER_NAMES[N_MATERIAL_PARAMETERS] = {%(names)s};

        bool yaml_read_real(const ryml::ConstNodeRef &node,
                            const char *const key,
                            real_t &value) {
            if (!node.has_child(key)) {
                return false;
            }
            node[key] >> value;
            return true;
        }

        bool yaml_read_parameter(const ryml::ConstNodeRef &node,
                                 const char *const key,
                                 real_t &value) {
            if (yaml_read_real(node, key, value)) {
                return true;
            }
            if (node.has_child("parameters") &&
                yaml_read_real(node["parameters"], key, value)) {
                return true;
            }
            if (node.has_child("material") &&
                yaml_read_real(node["material"], key, value)) {
                return true;
            }
            return false;
        }

        std::string yaml_read_string(const ryml::ConstNodeRef &node) {
            const auto value = node.val();
            return std::string(value.str, value.len);
        }

        void copy_material_parameters(const real_t *const src,
                                      real_t *const dst) {
            for (int i = 0; i < N_MATERIAL_PARAMETERS; ++i) {
                dst[i] = src[i];
            }
        }

        bool material_from_yaml(const ryml::ConstNodeRef &node,
                                const real_t *const base,
                                real_t *const values) {
            copy_material_parameters(base, values);
            bool changed = false;
            for (int i = 0; i < N_DEFINED_MATERIAL_PARAMETERS; ++i) {
                changed |= yaml_read_parameter(node,
                                               MATERIAL_PARAMETER_NAMES[i],
                                               values[i]);
            }
            return changed;
        }

        void set_material(MultiDomainOp &domains,
                          const real_t *const values) {
            for (auto &entry : domains.domains()) {
                for (int i = 0; i < N_DEFINED_MATERIAL_PARAMETERS; ++i) {
                    entry.second.parameters->set_value(MATERIAL_PARAMETER_NAMES[i],
                                                       values[i]);
                }
            }
        }

        void set_material_in_block(MultiDomainOp &domains,
                                   const std::string &block_name,
                                   const real_t *const values) {
            for (int i = 0; i < N_DEFINED_MATERIAL_PARAMETERS; ++i) {
                domains.set_value_in_block(block_name,
                                           MATERIAL_PARAMETER_NAMES[i],
                                           values[i]);
            }
        }

        bool yaml_read_bool(const ryml::ConstNodeRef &node,
                            const char *const key,
                            bool &value) {
            if (!node.has_child(key)) {
                return false;
            }
            int raw = value ? 1 : 0;
            node[key] >> raw;
            value = raw != 0;
            return true;
        }

        inline void read_affine_options(const ryml::ConstNodeRef &node,
                                        const AffineOption *const options,
                                        const int n_options) {
            bool all = true;
            for (int i = 0; i < n_options; ++i) {
                all = all && *options[i].flag;
            }
            if (yaml_read_bool(node, "ASSUME_AFFINE", all) ||
                yaml_read_bool(node, "assume_affine", all)) {
                for (int i = 0; i < n_options; ++i) {
                    *options[i].flag = all;
                }
            }
            for (int i = 0; i < n_options; ++i) {
                yaml_read_bool(node, options[i].name, *options[i].flag);
            }
        }
#endif  // SFEM_ENABLE_RYAML""" % {
        "nparameters": nparameters,
        "storage_size": storage_size,
        "names": names,
        "default_lines": "\n".join(default_lines),
    }


def _parameter_args(parameters):
    return "".join(
        ', domain.parameters->require_real_value("%s")' % name
        for name in parameters
    )


def _energy_field_component_count(collection, dim):
    """How many components the energy's field has, in this dimension.

    The wrapper spelled this `dim`, naming one array per spatial direction.
    That is right for a displacement -- which is every energy material the
    framework had -- and wrong for a scalar field, whose gradient has `dim`
    directions but whose value has one component.  It has to be asked per
    dimension rather than once: a displacement really does have two components
    in 2D and three in 3D.
    """
    fields = tuple(getattr(collection, "fields", ()) or ())
    if not fields:
        return dim
    return max(1, int(getattr(fields[0], "components", dim)))

def _packed_n_components(n_field_components_by_dim, dim):
    """The field's component count for this dimension, defaulting to `dim`."""
    return (n_field_components_by_dim or {}).get(dim, dim)

def _components(dim):
    return ("x", "y", "z")[:dim]


def _offsets(name, components):
    return ", ".join("%s + %d" % (name, i) for i, _ in enumerate(components))




def _metric_dispatch_name(name):
    """The metric-geometry sibling of an affine dispatch name."""
    marker = "_mesh_"
    index = name.rfind(marker)
    if index < 0:
        return "%s_metric" % name
    return "%s_metric%s" % (name[:index], name[index:])


def _metric_dispatch_elements(kernel_sources, name):
    """The element types a metric dispatch covers, from its own switch."""
    return _c_abi_public_dispatch_case_elements(kernel_sources, name)


def _affine_dispatch_call_lines(kernel_sources, name, indent, arguments):
    """A call to an affine dispatch, routing metric elements to their own.

    Elements of one dimension can want different geometry, so `name` may have
    a metric-geometry sibling covering the elements whose flux factors through
    it.  Those elements are named here rather than discovered at run time: the
    plain dispatch has no case for them and would answer SFEM_FAILURE.

    `arguments` is called with the callee, because the two take different
    geometry and each is asked what it takes.
    """
    lines = []
    metric = _metric_dispatch_name(name)
    elements = ()
    if _c_abi_function_exists(kernel_sources, metric, public_only=True):
        elements = _metric_dispatch_elements(kernel_sources, metric)
    if elements:
        lines.extend(
            [
                "%sif (%s) {"
                % (
                    indent,
                    " || ".join(
                        "domain.element_type == smesh::%s" % element
                        for element in elements
                    ),
                ),
                "%s    return %s(%s);" % (indent, metric, arguments(metric)),
                "%s}" % indent,
            ]
        )
    lines.append("%sreturn %s(%s);" % (indent, name, arguments(name)))
    return lines



def _affine_dispatch_status_lines(kernel_sources, name, indent, arguments):
    """`status = name(args);`, routing metric elements to their own dispatch.

    The assignment form of `_affine_dispatch_call_lines`, for the 0-form: it
    folds a status across domains rather than returning one, so it cannot use
    the returning shape and would otherwise be the one member of the
    matrix-free triple left unrouted.
    """
    lines = []
    metric = _metric_dispatch_name(name)
    elements = ()
    if _c_abi_function_exists(kernel_sources, metric, public_only=True):
        elements = _metric_dispatch_elements(kernel_sources, metric)
    if elements:
        lines.extend(
            [
                "%sif (%s) {"
                % (
                    indent,
                    " || ".join(
                        "domain.element_type == smesh::%s" % element
                        for element in elements
                    ),
                ),
                "%s    status = %s(%s);" % (indent, metric, arguments(metric)),
                "%s} else {" % indent,
                "%s    status = %s(%s);" % (indent, name, arguments(name)),
                "%s}" % indent,
            ]
        )
        return lines
    lines.append("%sstatus = %s(%s);" % (indent, name, arguments(name)))
    return lines


def _metric_cache_field(uses_metric):
    """The cache slot for the gradient metric, where anything reads it.

    An operator whose flux does not factor through the metric never asks for
    one, so it gets no slot, no allocation and no dead branch guarding a
    pointer nothing uses.
    """
    if not uses_metric:
        return ""
    return "            std::shared_ptr<smesh::FFF> metric_soa;\n"


def _metric_cache_setup(uses_metric):
    """Building that metric once per domain, alongside the Jacobian."""
    if not uses_metric:
        return ""
    return (
        "                cache->metric_soa = smesh::FFF::create_SoA(\n"
        "                        mesh, smesh::MEMORY_SPACE_HOST, block_id);\n"
        "                if (!cache->metric_soa) {\n"
        "                    return SFEM_FAILURE;\n"
        "                }\n"
    )


def _metric_declaration(uses_metric):
    """The metric pointer an operation body passes to its affine kernels."""
    if not uses_metric:
        return ""
    return "            const geom_t *const *geom_metric = nullptr;\n"


def _metric_binding(uses_metric, op_name, label):
    """Reading that pointer out of the cache, under the affine guard.

    Emitted text rather than a template hole, so the operator name is spelled
    here: this string is a substitution value and is not itself substituted.
    """
    if not uses_metric:
        return ""
    return (
        "                if (!cache->metric_soa) {\n"
        '                    SFEM_ERROR("%s affine %s requires cached metric geometry\\n");\n'
        "                    return SFEM_FAILURE;\n"
        "                }\n"
        "                geom_metric = reinterpret_cast<const geom_t *const *>(\n"
        "                        cache->metric_soa->fff_SoA()->data());\n"
    ) % (op_name, label)


def _cpp_bool(value):
    return "true" if value else "false"


def _affine_geometry_offsets(dim):
    return ", ".join("adjugate[%d]" % i for i in range(dim * dim))


def _affine_dispatch_parameters(kernel_sources, name):
    """The parameter list of a declared entry point, or None if undeclared.

    Matched on the declaration rather than on the name appearing anywhere in a
    source, because a source that merely calls the function would otherwise
    answer for it.
    """
    # Matched on `int <name>(`, which every declaration and definition of an
    # entry point spells and no call site does, so an element-level kernel --
    # declared in its own header without `extern "C"` -- answers here too.
    pattern = re.compile(
        r"\bint\s+" + re.escape(name) + r"\s*\(([^;{}]*)\)",
        re.S,
    )
    for source in (kernel_sources or {}).values():
        found = pattern.search(source)
        if found:
            return found.group(1)
    return None


def _affine_dispatch_uses_metric(kernel_sources, name):
    """Whether a declared affine entry point takes the gradient metric.

    The wrapper does not re-derive whether this operator's contraction factors
    through the metric.  The kernel already says so -- it declares
    `g_geom_metric0` or it declares `g_jacobian_adjugate0` -- and reading that
    is what keeps the two sides from agreeing only by coincidence.  See
    ARCHITECTURE.html OP 16 for what the wrapper deriving geometry
    independently cost the last time.
    """
    parameters = _affine_dispatch_parameters(kernel_sources, name)
    return bool(parameters) and "g_geom_metric0" in parameters


def _affine_geometry_call_args(kernel_sources, name, dim):
    """The geometry arguments an affine entry point takes, in ABI order."""
    if _affine_dispatch_uses_metric(kernel_sources, name):
        # The metric carries the determinant, so there is none to pass.
        return tuple(
            "geom_metric[%d]" % index
            for index in range(symmetric_metric_component_count(dim))
        )
    return tuple(
        ["adjugate[%d]" % index for index in range(dim * dim)] + ["determinant"]
    )


def _affine_geometry_offsets_for(kernel_sources, name, dim):
    """Those same arguments, spelled as one comma-separated list."""
    return ", ".join(_affine_geometry_call_args(kernel_sources, name, dim))

def _affine_metric_offsets(dim):
    # The count is the plan's, not a second copy of the arithmetic.  Both sides
    # of this boundary have to agree on how many components a symmetric metric
    # has, and they used to agree only by both spelling dim * (dim + 1) // 2 --
    # which is the shape of agreement that stops holding the moment one side
    # changes.  See ARCHITECTURE.html OP 16 for what the wrapper deriving
    # geometry independently already cost once.
    return ", ".join(
        "geom_metric[%d]" % i
        for i in range(symmetric_metric_component_count(dim))
    )


def _affine_dispatch_geometry_args(kernel_sources, function_name, dim):
    """The geometry arguments one affine entry point takes, as a list."""
    if _c_abi_function_uses_cached_metric(kernel_sources, function_name):
        return _affine_metric_offsets(dim).split(", ")
    return [*_affine_geometry_offsets(dim).split(", "), "determinant"]


def _c_abi_function_uses_cached_metric(kernel_sources, function_name):
    """Whether this kernel takes a cached gradient metric, not the adjugate.

    A linear simplex Laplacian contracts two reference gradients, so the
    geometry it needs is the symmetric metric -- three components in 2D, six in
    3D -- rather than the adjugate and determinant.  The wrapper builds one
    through ``smesh::FFF`` and caches it, and has to know which of the two the
    kernel is asking for.

    Both views, because they cover different files.  This used to read the
    private view alone, which skips everything under `op/`, and the affine mesh
    entry points are defined in the generated `_dispatch.cpp` sources, which
    live there.  So it answered "no metric" for any build where the metric form
    is the only form -- laplace generated for TRI3 or TET4 without a
    tensor-product element alongside it -- and the wrapper passed the adjugate
    and determinant to a kernel taking three metric components.  That wrapper
    could not compile, and generating laplace for a single simplex element is
    an ordinary thing to do.  With a tensor-product element also present the
    shared entry point does take the adjugate, so the wrong answer was the
    right one and the defect stayed hidden.
    """
    signatures = dict(_c_abi_signatures(kernel_sources))
    signatures.update(_c_abi_signatures(kernel_sources, public_only=True))
    signature = signatures.get(function_name)
    if not signature:
        return False
    return any(
        parameter.name == "g_geom_metric0" for parameter in signature.parameters
    )


def _boundary_surface_name(element):
    name = _element_name(element)
    surface_by_cell = {
        "TRI3": "edgeshell2",
        "QUAD4": "edgeshell2",
        "PROTEUS_QUAD4": "edgeshell2",
        "TET4": "trishell3",
        "TET10": "trishell6",
        "HEX8": "quadshell4",
        "HEX27": "quadshell9",
        "PROTEUS_HEX8": "proteus_quadshell4",
        "PROTEUS_HEX27": "proteus_quadshell9",
        "PROTEUS_HEX64": "proteus_quadshell16",
        "PROTEUS_HEX125": "proteus_quadshell25",
        "PROTEUS_HEX216": "proteus_quadshell36",
        "PROTEUS_HEX343": "proteus_quadshell49",
        "PROTEUS_HEX512": "proteus_quadshell64",
        "PROTEUS_HEX729": "proteus_quadshell81",
    }
    try:
        return surface_by_cell[name]
    except KeyError as exc:
        raise ValueError("unsupported generated boundary Op element %s" % element) from exc


def _element_dim(element):
    name = _element_name(element)
    if name in ("TRI3", "TRI6", "QUAD4", "PROTEUS_QUAD4") or name.startswith(("TRI6_", "QUAD4_", "PROTEUS_QUAD4_")):
        return 2
    if (
        name in ("TET4", "TET10", "HEX8", "HEX27")
        or name.startswith(("TET10_", "HEX27_", "PROTEUS_HEX"))
    ):
        return 3
    raise ValueError("unsupported generated Op element %s" % element)


def _mesh_element_name(element):
    return getattr(element, "cell_element_type", _element_name(element))


def _mesh_case_elements(element):
    name = _mesh_element_name(element)
    aliases = {
        "PROTEUS_QUAD4": ("QUAD4", "PROTEUS_QUAD4"),
        "PROTEUS_HEX8": ("HEX8", "PROTEUS_HEX8"),
        "PROTEUS_HEX27": ("HEX27", "PROTEUS_HEX27"),
    }
    return aliases.get(name, (name,))


def _mesh_case_labels(element, indent):
    return "\n".join("%scase smesh::%s:" % (indent, name) for name in _mesh_case_elements(element))


def _element_name(element):
    return getattr(element, "name", str(element).upper())
