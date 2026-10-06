#!/usr/bin/env bash
# The distributed setup describes the same problem as the serial one.
#
# Everything before the solve is meant to be correct multi-rank already: the decomposition,
# the ownership metadata, the constraint set, the counts. Nothing checked that, and the
# failure mode is quiet -- mesh->n_nodes() is the LOCAL count on a distributed mesh, so a
# quantity summed over it counts every shared, ghost and aura entry once per rank holding it
# and comes out larger the more ranks you use, while every array is still the right length and
# nothing fails.
#
# Measured on the cavity at N=2 over two ranks before this gate existed: the ranks reported 27
# and 18 local nodes against a serial 27, and each printed its own coordinate checksum,
# 13.5/13.5/13.5 and 9/9/13.5. Neither is the mesh's checksum, and their sum is not either.
#
# WHAT IS COMPARED, AND WHAT DELIBERATELY IS NOT.
#
# Integer counts only: global nodes, global elements, and constrained dofs. Those partition
# the problem exactly once, so they must agree with the serial run EXACTLY, and an exact
# integer comparison is a real test rather than a tolerance dressed up as one.
#
# The coordinate checksum is NOT compared across rank counts, and that is a considered
# omission rather than an oversight. It is a sum of floating-point coordinates, and floating
# point addition does not associate: a four-way partial-sum tree and a sequential sum need not
# produce identical doubles even when both are correct. It remains what its own comment in the
# driver says it is -- a guard that the node NUMBERING has not changed between runs at the same
# rank count, which is what a restart needs.
set -u

DRIVER=${1:?usage: cvfem_distributed_setup_test.sh <cvfem_hex8_ns_ssgmg> <mpiexec>}
MPIEXEC=${2:?usage: cvfem_distributed_setup_test.sh <cvfem_hex8_ns_ssgmg> <mpiexec>}

[ -x "$DRIVER" ] || { echo "distributed_setup: no driver at $DRIVER" >&2; exit 1; }

WORK=$(mktemp -d 2>/dev/null || mktemp -d -t cvfem_dist_setup)
trap 'rm -rf "$WORK"' EXIT

# Whether this machine can place several ranks is a property of the machine, not of the code.
LAUNCH_OPTS=""
if ! "$MPIEXEC" -n 2 true > "$WORK/probe.log" 2>&1; then
    LAUNCH_OPTS="--oversubscribe"
    if ! "$MPIEXEC" -n 2 $LAUNCH_OPTS true >> "$WORK/probe.log" 2>&1; then
        echo "distributed_setup: SKIP -- $MPIEXEC cannot launch 2 ranks here"
        sed 's/^/    /' "$WORK/probe.log" | head -10
        exit 77
    fi
fi

fail=0
say() { printf '%-58s %s\n' "$1" "$2"; [ "$2" = OK ] || fail=1; }

# N=4 rather than the N=2 used while developing this: at N=2 one rank's local set happens to
# cover the entire cube, so the partition is degenerate and proves very little. Refine level 1
# keeps it to well under a second.
#
# SFEM_FGMRES=1 even though no solve runs: the driver is never invoked in this spike without
# it, and a configuration that differs from the real one is the wrong thing to gate.
# Two configurations, because they reach different code and only one of them reaches the
# two-pass nodal gradient.
#
#   flat -- refine level 1, the flat packed path. This is the arm that caught the
#           constrained-dof defect at four ranks.
#   ss   -- refine level 2 with semi-structured packing on. The only configuration in the
#           suite where the packing covers just PART of the mesh, so the only one that enters
#           sscvfem_nodal_grad_strided's two-pass branch rather than its covers_all fast path.
#           Confirmed from pack stats rather than assumed: owned_ptr[n] is 324 and 187 against
#           local node counts of 405 and 567 at two ranks, while serial is 729 == 729 and so
#           stays on the fast path.
#
# SFEM_FGMRES=1 even though no solve runs: the driver is never invoked in this spike without
# it, and a configuration that differs from the real one is the wrong thing to gate.
COMMON="SFEM_CASE=cavity SFEM_N=4 SFEM_FGMRES=1 SFEM_GMG=0 SFEM_PRECOND=direct
        SFEM_ENABLE_OUTPUT=0 SFEM_SETUP_ONLY=1"

CASE_flat="$COMMON SFEM_ELEMENT_REFINE_LEVEL=1"
CASE_ss="$COMMON SFEM_ELEMENT_REFINE_LEVEL=2 SFEM_SS_PACK_SIZE=8"

# grep -a throughout: a failing rank can put non-text bytes in the log, and grep then refuses
# to print matching lines at all rather than saying why. That silently produced two empty
# "results" while this was being written.
run() {   # label, nranks, logfile
    local lbl=$1 n=$2 log=$3
    if [ "$n" = 1 ]; then
        env $CASE "$DRIVER" "$WORK/out_${lbl}_$n" > "$log" 2>&1
    else
        env $CASE SFEM_ALLOW_MPI=1 "$MPIEXEC" -n "$n" $LAUNCH_OPTS "$DRIVER" "$WORK/out_${lbl}_$n" > "$log" 2>&1
    fi
    return $?
}

field() {   # logfile, regex, awk-index
    grep -a -E "$2" "$1" | tail -1 | awk "{print \$$3}"
}

# One configuration, compared against its own serial run. $CASE selects which.
check_config() {   # label
    local lbl=$1
    local rc1 rc n
    local ser_nodes ser_elems ser_con
    local g_nodes g_sum_nodes g_elems g_sum_elems g_con

    run "$lbl" 1 "$WORK/${lbl}_r1.log"
    rc1=$?
    [ $rc1 -eq 0 ] && say "$lbl: serial setup completes" OK \
                   || { say "$lbl: serial setup completes" FAIL; tail -8 "$WORK/${lbl}_r1.log"; }

    # Serial prints no "global:" line, because there is no decomposition to describe. Its local
    # counts ARE the global ones, which is exactly why it is the reference.
    ser_nodes=$(field "$WORK/${lbl}_r1.log" "^nnodes: " 2)
    ser_elems=$(field "$WORK/${lbl}_r1.log" "^nnodes: " 4)
    ser_con=$(field "$WORK/${lbl}_r1.log" "^constrained dofs: " 3)

    [ -n "$ser_nodes" ] && [ -n "$ser_elems" ] && [ -n "$ser_con" ] \
        && say "$lbl: serial reports node, element and constrained counts" OK \
        || { say "$lbl: serial reports node, element and constrained counts" FAIL; tail -8 "$WORK/${lbl}_r1.log"; }

    for n in 2 4; do
        run "$lbl" $n "$WORK/${lbl}_r$n.log"
        rc=$?
        [ $rc -eq 0 ] && say "$lbl $n-rank setup completes" OK \
                      || { say "$lbl $n-rank setup completes" FAIL; grep -a -iE "error|abort|does not support" "$WORK/${lbl}_r$n.log" | head -4; }

        g_nodes=$(field "$WORK/${lbl}_r$n.log" "^global: nnodes " 3)
        g_sum_nodes=$(field "$WORK/${lbl}_r$n.log" "^global: nnodes " 5)
        g_elems=$(field "$WORK/${lbl}_r$n.log" "^global: nnodes " 7)
        g_sum_elems=$(field "$WORK/${lbl}_r$n.log" "^global: nnodes " 9)
        g_con=$(field "$WORK/${lbl}_r$n.log" "^constrained dofs: " 3)

        if [ -z "$g_nodes" ] || [ -z "$g_con" ]; then
            say "$lbl $n ranks: global counts are reported" FAIL
            grep -a -E "^global:|^nnodes:|^constrained" "$WORK/${lbl}_r$n.log" | head -6
            continue
        fi

        # The decomposition's own record against what the owned ranges actually sum to. These
        # come from independent places -- smesh recorded one when it partitioned, the driver
        # computed the other by reducing over owned ranges -- so agreement means the ownership
        # metadata still describes the mesh. This is the assertion that would catch the class of
        # corruption the smesh packing work was about.
        [ "$g_nodes" = "$g_sum_nodes" ] \
            && say "$lbl $n ranks: recorded global nodes == summed owned" OK \
            || say "$lbl $n ranks: global nodes $g_nodes != summed $g_sum_nodes" FAIL

        [ "$g_elems" = "$g_sum_elems" ] \
            && say "$lbl $n ranks: recorded global elements == summed owned" OK \
            || say "$lbl $n ranks: global elements $g_elems != summed $g_sum_elems" FAIL

        # And against serial. Exact, because these are integers: the number of nodes in a mesh
        # does not depend on how the mesh was cut.
        [ "$g_nodes" = "$ser_nodes" ] \
            && say "$lbl $n ranks: global nodes match serial ($ser_nodes)" OK \
            || say "$lbl $n ranks: global nodes $g_nodes != serial $ser_nodes" FAIL

        [ "$g_elems" = "$ser_elems" ] \
            && say "$lbl $n ranks: global elements match serial ($ser_elems)" OK \
            || say "$lbl $n ranks: global elements $g_elems != serial $ser_elems" FAIL

        # The constraint set is the one most exposed to partitioning, and this assertion has
        # already earned its place: on first run it caught a real defect at four ranks, 245
        # against a serial 258.
        #
        # The obvious guess was wrong and is recorded here so nobody spends the afternoon on it
        # again. A rank-interface node mistaken for a boundary node would make this count come
        # out HIGHER; it came out LOWER, and the cause is not the constraint set at all. The
        # count runs over [0, ndof_owned), which assumes the owned nodes are the first n_owned
        # slots. Packing renumbers the nodes in place, and a packing that permutes shared, ghost
        # and aura nodes along with the owned ones destroys that prefix, after which the loop
        # counts the wrong dofs -- and, far worse than a wrong count, leaves genuinely
        # constrained dofs unconstrained on the rank that owns them.
        #
        # Isolated by rerunning with SFEM_PACK_SIZE=0: packing off gives 258 at every rank
        # count, packing on gives 258, 258, 245 at one, two and four. So this assertion depends
        # on the installed smesh restricting its permutation to the owned-not-shared prefix. If
        # it fails here again with packing on and passes with packing off, the installed SFEM
        # predates that fix -- check the install timestamp against the smesh commits before
        # suspecting the constraints.
        #
        # Two ranks passes either way on this mesh, which is why the gate runs four.
        [ "$g_con" = "$ser_con" ] \
            && say "$lbl $n ranks: constrained dofs match serial ($ser_con)" OK \
            || say "$lbl $n ranks: constrained dofs $g_con != serial $ser_con" FAIL
    done
}

CASE="$CASE_flat"
check_config flat

CASE="$CASE_ss"
check_config ss

echo
[ $fail -eq 0 ] && echo PASSED || echo FAILED
exit $fail
