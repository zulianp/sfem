#!/usr/bin/env bash

set -e
set -x

# mesh=quad4_example.e
# folder=quad4_example_raw

# mesh=tri3_example.e
# folder=tri3_example_raw

mesh=tet4_example.e
folder=tet4_example_raw

./convert_exodus_to_raw.sh $mesh $folder
echo "folder raw content"
ls -la $folder