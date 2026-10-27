#!/usr/bin/env bash

set -e
set -x

if [ ! -d "./downloads/" ]; then
	mkdir ./downloads
fi

remotefolder=/scratch/zulian/xdns/fe_hydros/sfem/tests/compare/mesh-multi-outlet-better/
scp zulian@hpc.ics.usi.ch:$remotefolder/\{p,lhs.colindex,lhs.rowindex,lhs.value,on,zd,rhs\}.raw ./downloads/
