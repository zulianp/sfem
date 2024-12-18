#!/usr/bin/env bash

set -e
set -x

c++ -o viz quad4_viz.exe.cpp  -framework GLUT -framework OpenGL -DGL_SILENCE_DEPRECATION -g -O0 -std=c++11
./viz ../tests/sdf_obstacle/cases/2_highfreq/mesh/skin  ../tests/sdf_obstacle/cases/2_highfreq/output/disp.0.raw  ../tests/sdf_obstacle/cases/2_highfreq/output/disp.1.raw  ../tests/sdf_obstacle/cases/2_highfreq/output/disp.2.raw