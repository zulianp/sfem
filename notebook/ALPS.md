# Troubleshooting on new Alps Piz Daint

In order to work with cupy we need the following


```sh
module load cray 
module load PrgEnv-cray cray-python craype-arm-grace cray-hdf5-parallel

# In the sfem dir
python -mvenv venv
source venv/bin/activate
python -m pip install -U setuptools pip
pip install cupy-cuda12x
```

This environment also supports
```sh
pip install matplotlib numpy rich scipy sympy nanobind meshio netCDF4
```

# Unsupported python packages

- taichi
- gmsh
- gmsh-interop
- h5
- h5py


# Confiuring SFEM

```sh
# In the sfem dir
mkdir build
cd build
cmake .. -DSFEM_ENABLE_CUDA=ON -DSFEM_ENABLE_PYTHON=OFF -DSFEM_ENABLE_OPNEMP=ON
make -j12
```