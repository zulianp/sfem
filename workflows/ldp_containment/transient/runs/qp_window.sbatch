#!/bin/bash
#SBATCH --job-name=qpwindow
#SBATCH --account=c40
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/qp_window_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The QP gate of the stored tangents (P-qp): FF200 simulation 2 over its first 30 ms (300 steps), one
# H100 and 72 cores per run, four at once, A B B A per tangent format: build A's playground (A, the
# reference) against build B's (B), each twice, so both get two GPU slots on one node.
#   ELEMENT=TET4|TET10 A=<playground> B=<playground> TAG=<name> TANGENTS="bsr32 qbsr16 ..."
#     [HISTORY=full] [T_END=0.03]
# Folders $CASE/qpw_<tag>_<s0|p2>_<A|B><k>_<tangent>__<history>_a; collect with collect_qp_window.py. A alone
# (B unset) runs A four times: the baseline's own spread. No frames are written (the shared file
# system's load is not the QP's).
set -u
T=/ritom/scratch/cscs/zulianp/sfem-prony-md/workflows/ldp_containment/transient
source $T/runs/tangent9_valves.sbatch
mesh=s0; [ "$ELEMENT" = TET10 ] && mesh=p2
B=${B:-$A}
for t in $TANGENTS; do
  for slot in A1 B1 B2 A2; do
    bin=$A; [ ${slot:0:1} = B ] && bin=$B
    ( PLAYGROUND=$bin PREFIX=qpw_${TAG}_${mesh}_${slot} EXTRA="--set time.t_end=${T_END:-0.03} --set output.export_freq=1000000" \
        go "${t}__${HISTORY:-full}_a" ) &
  done
  wait
done
