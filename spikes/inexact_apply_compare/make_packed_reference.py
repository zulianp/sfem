"""The packed inexact apply, derived from the emitted standard one.

This is the reference implementation the plan calls for: fast first, by hand, on
emitted C++, before anything goes into the generator.  It is a script rather than
a pasted file because the arithmetic it rewrites is 900 lines long and is not the
part being changed -- what changes is the gather and the scatter, and a
transformation states that far more clearly than a copy does.

    python3 make_packed_reference.py <inexact_apply_inline.hpp> <out.hpp> <prefix>

What it does to the stored apply:

* the flat element loop `for (evb = 0; evb < nelements; evb += VS)` becomes a pack
  loop with the element block nested inside it, so each node is fetched once per
  *pack* rather than once per element that touches it;
* the connectivity becomes pack-local `uint16_t`, and the gather reads thread-private
  scratch instead of a global array through an indirection;
* the scatter accumulates into that same scratch with **no atomic at all** -- the
  scratch is thread-private, so nothing else can be writing it -- and only the pack
  boundary reaches global memory, ghosts through a buffer that a second disjoint
  pass reduces;
* the arithmetic between them is copied across untouched.  That is the point: if
  the packed kernel is faster, it is faster for the reason claimed.

The tangent store needs no transformation.  Packs are contiguous element ranges, so
`tangent + evb + k * tangent_component_stride` addresses the same thing it always did,
and the geometry is likewise still a pointer bump.
"""

import re
import sys


def function_body(source, signature_prefix):
    start = source.find(signature_prefix)
    if start < 0:
        raise SystemExit("not found: %s" % signature_prefix)
    open_brace = source.find("{", source.find(")", start))
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise SystemExit("unterminated body")


def between(body, opening, closing):
    start = body.index(opening) + len(opening)
    return body[start:body.index(closing, start)]


PACKED_SIGNATURE = """template <typename s_t, typename tangent_t, int VS>
static SFEM_INLINE int %(name)s(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t n_ghost_entries,
    const ptrdiff_t n_ghost_reduce_rows,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const ptrdiff_t *const RSTR ghost_reduce_ptr,
    const ptrdiff_t *const RSTR ghost_reduce_idx,
    const idx_t *const RSTR ghost_reduce_dest,
    s_t *const RSTR ghost_buf,
    const ptrdiff_t tangent_component_stride,
    const tangent_t *const RSTR tangent,
    const ptrdiff_t h_stride,
    const s_t *const RSTR hx,
    const s_t *const RSTR hy,
    const s_t *const RSTR hz,
    const ptrdiff_t out_stride,
    s_t *const RSTR outx,
    s_t *const RSTR outy,
    s_t *const RSTR outz
) {
  static constexpr int NC = %(nc)d;
  const s_t *const h_components[NC] = {%(h_list)s};
  s_t *const out_components[NC] = {%(out_list)s};

#pragma omp parallel
  {
    s_t *const RSTR pk_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
      const ptrdiff_t e_start = pack * n_elements_per_pack;
      const ptrdiff_t e_end = (nelements < (pack + 1) * n_elements_per_pack)
                                  ? nelements
                                  : (pack + 1) * n_elements_per_pack;
      const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
      const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
      const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
      const ptrdiff_t ghost_off = ghost_ptr[pack];
      const idx_t *const RSTR ghosts = &ghost_idx[ghost_off];

      // Once per pack, not once per element: this is the whole of the change.
      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_h_component = pk_h + d * max_nodes_per_pack;
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        const s_t *const RSTR h_component = h_components[d];
        for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) pk_component_out[k] = s_t(0);
        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
          pk_h_component[k] = h_component[(owned_nodes_ptr[pack] + k) * h_stride];
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
          pk_h_component[n_contiguous + k] = h_component[ghosts[k] * h_stride];
        }
      }

      for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
        const int ne = (int)((e_end - evb) < (ptrdiff_t)VS ? (e_end - evb) : (ptrdiff_t)VS);
%(block)s
      }

      // Two-pass: owned nodes go straight out, ghosts into the buffer for the
      // disjoint reduction below.  No atomic anywhere in this kernel.
      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        s_t *const RSTR global_out = out_components[d];
        s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
          global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
          ghost_component[ghost_off + k] = pk_component_out[n_contiguous + k];
        }
      }
    }
  }

  // Rows have distinct destinations, so this needs no atomic either.
#pragma omp parallel for schedule(static)
  for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
    const idx_t dest = ghost_reduce_dest[row];
    const ptrdiff_t begin = ghost_reduce_ptr[row];
    const ptrdiff_t end = ghost_reduce_ptr[row + 1];
    for (int d = 0; d < NC; ++d) {
      const s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
      s_t sum = s_t(0);
      for (ptrdiff_t j = begin; j < end; ++j) sum += ghost_component[ghost_reduce_idx[j]];
      out_components[d][dest * out_stride] += sum;
    }
  }
  return SFEM_SUCCESS;
}
"""


def packed_stored(source, prefix, components=("x", "y", "z")):
    standard = "static SFEM_INLINE int %s_inexact_apply_stored_a_msoa_impl(" % prefix
    body = function_body(source, standard)
    inner = between(body, "(ptrdiff_t)VS);", "\n  }\n\n  return SFEM_SUCCESS;")

    # The connectivity is now pack-local, and the gather reads the scratch.
    inner = inner.replace("idx_t bev", "uint16_t bev")
    for index, name in enumerate(components):
        inner = re.sub(
            r"\bb(h%s_\d+)\[lane\] = h%s\[bev(\d+)\[lane\] \* h_stride\];" % (name, name),
            r"b\1[lane] = pk_h[%d * max_nodes_per_pack + bev\2[lane]];" % index,
            inner,
        )
    # The scatter lands in thread-private scratch, so it drops the atomic with it.
    inner = re.sub(
        r"[ \t]*#pragma omp atomic update\n[ \t]*out(\w)\[bev(\d+)\[lane\] \* out_stride\] \+= (\w+)\[lane\];",
        lambda m: "        pk_out[%d * max_nodes_per_pack + bev%s[lane]] += %s[lane];"
        % (components.index(m.group(1)), m.group(2), m.group(3)),
        inner,
    )
    if "atomic" in inner:
        raise SystemExit("an atomic survived the transformation")
    if "h_stride" in inner or "out_stride" in inner:
        raise SystemExit("a global stride survived the transformation")

    return PACKED_SIGNATURE % {
        # A distinct name so the reference and the generated kernel can live in
        # one binary and be timed against each other in one run.  Comparing them
        # across runs measures the machine's mood as much as the kernels.
        "name": "%s_inexact_apply_stored_packed_two_pass_reference_impl" % prefix,
        "nc": len(components),
        "h_list": ", ".join("h%s" % name for name in components),
        "out_list": ", ".join("out%s" % name for name in components),
        "block": "\n".join("  " + line if line.strip() else line
                           for line in inner.strip("\n").split("\n")),
    }


def main(argv):
    if len(argv) != 4:
        raise SystemExit(__doc__.strip().splitlines()[2])
    source = open(argv[1]).read()
    prefix = argv[3]
    out = [
        "// Generated by make_packed_reference.py -- the hand-written packed reference.",
        "// Not a generated-tree artifact: this is the experiment the emitter port is",
        "// measured against.  See the script for what it does and why.",
        "#pragma once",
        '#include "packed_thread_scratch.hpp"',
        "//",
        "// The standard inline header is deliberately not included: it carries no",
        "// include guard, so a second inclusion redefines every kernel in it.  The",
        "// benchmark includes it once, before this.  (That missing guard is an",
        "// emitter gap -- it has never bitten because exactly one translation unit",
        "// includes the header today.)",
        "",
        "namespace sfem {",
        "namespace codegen {",
        "",
        packed_stored(source, prefix),
        "}  // namespace codegen",
        "}  // namespace sfem",
        "",
    ]
    open(argv[2], "w").write("\n".join(out))
    print("wrote %s" % argv[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
