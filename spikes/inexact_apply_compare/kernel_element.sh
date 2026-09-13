# Which element owns a micro-kernel, asked of the generator rather than listed here.
#
# A mesh-ordered tensor-product element publishes no kernel body of its own.  It
# forwards to its lexicographic twin -- HEX8 to PROTEUS_HEX8, QUAD4 to
# PROTEUS_QUAD4 -- so the two names a benchmark needs are different things:
#
#   ELEMENT          the mesh, and the `extern "C"` symbols the library links
#                    against, which stay this element's whatever it forwards to
#   KERNEL_ELEMENT   the element whose header holds the templated `..._impl`
#                    bodies a benchmark instantiates directly
#
# For a simplex the two are the same and nothing below changes.
#
# `plans/layout.cartesian_twin` is where that pairing is decided.  This asks it
# rather than repeating the table, because a spike carrying its own copy is how
# the two would come to disagree -- and the disagreement would be silent: the
# benchmark would still build, against whichever element's kernel it found.
#
# Usage, after WORKTREE (and ideally SFEM and PYTHON) are set:
#
#     . "$HERE/kernel_element.sh"
#     KERNEL_ELEMENT="$(kernel_element "$ELEMENT")"
#     KLOWER="$(echo "$KERNEL_ELEMENT" | tr '[:upper:]' '[:lower:]')"
#     SHAPE_ORDER="$(kernel_shape_order "$ELEMENT")"
#
# `kernel_shape_order` is the permutation that goes with it, empty when the
# element owns its kernel.  A benchmark that calls the generated `extern "C"`
# symbol needs only the first; one that instantiates the `..._impl` template
# directly bypasses the forwarder that permutes, so it needs both.

# The interpreter candidates, in order.  A bare `python3` is last and is not
# assumed to work: importing the plan layer reaches `codegen.framework.__init__`,
# which imports sympy, so an interpreter without it has to be skipped rather
# than picked and then failed on.  The scripts here that have no venv discovery
# of their own get one from this, rather than growing a fourth copy of the loop.
kernel_element() {
    _ke_element="$1"
    _ke_field="${2:-name}"
    _ke_query="${TMPDIR:-/tmp}/sfem_kernel_element.$$.py"
    cat > "$_ke_query" <<'QUERY'
import sys

sys.path.insert(0, "%s/python" % sys.argv[1])
from codegen.framework.plans.layout import cartesian_twin

element, field = sys.argv[2], sys.argv[3]
twin = cartesian_twin(element)
if field == "order":
    print("" if twin is None else ",".join(str(node) for node in twin.shape_order))
else:
    print((twin.twin_name if twin is not None else element).upper())
QUERY
    for _ke_candidate in "${PYTHON:-}" "${SFEM_PYTHON:-}" \
        "${SFEM:-/nonexistent}/.venv/bin/python" "${SFEM:-/nonexistent}/venv/bin/python" \
        "${WORKTREE:-/nonexistent}/.venv/bin/python" "${WORKTREE:-/nonexistent}/venv/bin/python" \
        python3 python; do
        [ -n "$_ke_candidate" ] || continue
        command -v "$_ke_candidate" >/dev/null 2>&1 || continue
        _ke_answer="$("$_ke_candidate" "$_ke_query" "$WORKTREE" "$_ke_element" "$_ke_field" 2>/dev/null)" || continue
        if [ -n "$_ke_answer" ] || [ "$_ke_field" = order ]; then
            rm -f "$_ke_query"
            printf '%s' "$_ke_answer"
            return 0
        fi
    done
    rm -f "$_ke_query"
    echo "kernel_element: no interpreter could import the generator's plan layer;" >&2
    echo "  set SFEM_PYTHON to the venv the kernels were generated with" >&2
    return 1
}


# The permutation that pairs with it: entry `i` is the mesh node carrying the
# kernel's node `i`.  Empty when the element owns its kernel.
kernel_shape_order() {
    kernel_element "$1" order
}
