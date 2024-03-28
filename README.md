# Simple FEM #

# Installation guide

First we install the `C` code base
Go to a directory of your choosing and type

```bash
git clone https://bitbucket.com/zulianp/isolver.git && \
git clone https://bitbucket.com/zulianp/matrix.io && \
git clone https://bitbucket.com/zulianp/sfem && \
cd matrix.io && make && \
cd ../sfem && git submodule update --init --recursive && \
make
```

In the `sfem` folder

- Enter `python3 -m venv venv`
- Enter `source venv/bin/activate` (to be activated for every new command-line window)
- Type `pip install -r python/requirements.txt` (Python 3 is assumed here) to install `Python3` dependencies.

Both makefiles allow to pass options such as 
`MPICC=<path_to_your_mpicc_compiler>` and `MPICXX=<path_to_your_mpicxx_compiler>`.

