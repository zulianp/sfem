export SFEM_DIR=$INSTALL_DIR/sfem_amg
# export SFEM_DIR=$INSTALL_DIR/sfem_amg_32
# export SFEM_DIR=$INSTALL_DIR/sfem_asan
export PATH=$SFEM_DIR/bin:$PATH
export PATH=$SFEM_DIR/scripts/sfem/mesh:$PATH
export PATH=$SFEM_DIR/scripts/sfem/sdf:$PATH
export PATH=$SFEM_DIR/scripts/sfem/grid:$PATH
export PYTHONPATH=$SFEM_DIR/lib:$SFEM_DIR/scripts:$PYTHONPATH
