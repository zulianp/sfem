#!/usr/bin/env bash

set -e
set -x

export PATH=../../python/mesh/:$PATH

nsteps=`ls out/u0.*.raw | wc -l | awk '{print $1}'`
raw_to_db.py ./mesh u.xmf  \
 --transient --time_whole_txt="out/time.txt" \
 --point_data="out/u0.*.raw,out/u1.*.raw" 

raw_to_db.py ./mesh/p1 p.xmf \
 --transient --time_whole_txt="out/time.txt"\
 --point_data="out/p.*.raw,out/div.*.raw,out/div_pre.*.raw" 
