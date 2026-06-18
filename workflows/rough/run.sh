#!/usr/bin/env bash

set -e

rm -rf  rough_surface surf_tri3

db_to_raw rock.vtk rock
db_to_raw rough_surface.vtk rough_surface
echo "element_type: QUAD4" > rough_surface/meta.yaml

mesh_convert TRI3 rough_surface surf_tri3

raw_to_db surf_tri3 surf_tri3.vtk

mkdir -p sdf
cd sdf
../../../python/sfem/sdf/mesh_to_sdf.py ../surf_tri3.vtk sdf.float32 --hmax=0.005 --margin=0.1
cp metadata_sdf.yml meta.yaml
echo "spatial_dimension: 3" >> meta.yaml
raw_to_xdmf metadata_sdf.yml 
cd ..


create_sideset rock 0.5 0.5 -0.1 0.9999 rock/sidesets
create_sideset rock 0.5 0.5 -0.0 0.8 rock/contact_boundary


surface_from_sideset rock rock/sidesets dboundary 
surface_from_sideset rock rock/contact_boundary contact_boundary 

raw_to_db dboundary dboundary.vtk --coords=rock --cell_type=QUAD4
raw_to_db contact_boundary contact_boundary.vtk  --coords=rock --cell_type=QUAD4


./sim.sh
