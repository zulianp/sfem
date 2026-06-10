#!/usr/bin/env bash

set -e

if [[ -z "$SFEM_PATH" ]]
then
	echo "SFEM_PATH=</path/to/sfem/installation> must be defined"
	exit 1
fi

export PATH=$SFEM_PATH/bin:$PATH

if [[ $# -le "1" ]]
then
	printf "usage: $0 <mesh> <out_folder>\n" 1>&2
	exit -1
fi

set -x

db_in=$1
db_out=$2

workspace=$db_out/__workspace
skinned=$workspace/skinned

rm -rf $workspace

mkdir -p $db_out
mkdir -p $workspace
mkdir -p $skinned

skin $db_in $skinned
raw_to_db $skinned $db_out/skinned.e

extract_sharp_features $skinned $db_out
select_submesh $skinned $db_out/disconnected_faces/i0.int32 $db_out/disconnected

raw_to_db $db_out/edges        $db_out/sharp_edges.e  --coords=$skinned
raw_to_db $db_out/corners      $db_out/corners.e      --coords=$skinned
raw_to_db $db_out/disconnected $db_out/disconnected.e
