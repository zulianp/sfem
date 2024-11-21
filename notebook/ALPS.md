# Troubleshooting on new Alps Piz Daint

```bash
uenv image pull prgenv-gnu/24.7:v3
uenv start prgenv-gnu/24.7:v3
uenv view default
```

# Unsupported python packages

- taichi
- gmsh
- gmsh-interop
- h5
- h5py


# Configuring SFEM

```sh
# In the sfem dir
mkdir build
cd build
cmake .. -DSFEM_ENABLE_CUDA=ON -DSFEM_ENABLE_PYTHON=OFF -DSFEM_ENABLE_OPNEMP=ON
make -j12
```
