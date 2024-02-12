# Simple FEM #

# Installation guide

First we install the `C` code base
Go to a directory of your choosing and type

```bash
git clone https://bitbucket.com/zulianp/matrix.io && \
git clone https://bitbucket.com/zulianp/sfem  && \
git submodule update --init --recursive

cd matrix.io
make
cd ../sfem
make
```

In the `sfem` folder
type `pip install -r requirements.txt` (Python 3 is assumed here) to install `Python3` dependencies.

Both makefiles allow to pass options such as 
`MPICC=<path_to_your_mpicc_compiler>` and `MPICXX=<path_to_your_mpicxx_compiler>`.

