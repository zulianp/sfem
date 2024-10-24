# Simple FEM #

# Requirements

SFEM depends on the following base technologies

- C/C++ Compiler
- CMake				(e.g., `brew install cmake`)
- MPI 				(e.g., `brew install open-mpi`)
- Python 3 			(e.g., `brew install python@3.12`)

The following technologies are also supported
- OpenMP 			(e.g., `brew install libomp`)

# Installation guide

First we install the `C` code base
Go to a directory of your choosing and type

```bash
git clone https://bitbucket.com/zulianp/isolver.git && \
git clone https://bitbucket.com/zulianp/matrix.io && \
git clone https://github.com/zulianp/sfem.git && \
cd matrix.io && make && \
cd ../sfem && git submodule update --init --recursive && \
mkdir build && cd build && cmake .. -DSFEM_ENABLE_OPENMP=ON && make
```


In the `sfem` folder

- Enter `python3 -m venv venv` (**Optional/Recommended**)
- Enter `source venv/bin/activate` (to be activated for every new command-line window) (**Optional/Recommended**)
- Type `pip install -r python/requirements.txt` (Python 3 is assumed here) to install `Python3` dependencies.

Both makefiles allow to pass options such as 
`MPICC=<path_to_your_mpicc_compiler>` and `MPICXX=<path_to_your_mpicxx_compiler>`.


# Installing the Python frontend

1. downloading sfem and submodules and installing the python env requirements.
2. Activate python env
3. Install nanobind in the environment `pip install nanobind`

Then compile and install sfem
```
export INSTALL_DIR=<path_to_your_libs>
mkdir -p build && \
cd build && \
cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/sfem -DSFEM_ENABLE_PYTHON=ON && \
make && make install
export PYTHONPATH=$INSTALL_DIR/sfem/lib:$INSTALL_DIR/sfem/scripts:$PYTHONPATH
```

# New Piz daint

Install the `uenv` machinery (https://confluence.cscs.ch/display/KB/UENV+user+environments)

```bash
uenv image pull prgenv-gnu/24.7:v3
uenv start prgenv-gnu/24.7:v3
uenv view default

# Remeber to compile matrix.io

# In the sfem folder
mkdir build && \
cd build && \
cmake .. -DSFEM_ENABLE_CUDA=ON -DSFEM_ENABLE_PYTHON=ON -DSFEM_ENABLE_OPENMP=ON -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx && \
make -j12
```


# OLD Piz Daint (XC50/XC40)

On Piz Daint the ideal environment is obtained with
```bash
source $APPS/UES/anfink/gpu/environment
```

# Cite SFEM

Cite SFEM if you use it for your work:

```bibtex
@misc{sfemgit,
	author = {Zulian, Patrick and Riva, Simone},
	title = {{SFEM}: Simple {FEM}},
	url = {https://bitbucket.org/zulianp/sfem},
	howpublished = {https://bitbucket.org/zulianp/sfem},
	year = {2024}
}
```
