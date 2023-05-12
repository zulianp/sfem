#!/usr/bin/env bash

set -e
set -x

if [[ -z "$INSTALL_DIR" ]]
then
	echo "Define INSTALL_DIR"
	exit 1
fi

mkdir -p $INSTALL_DIR/sources
export GKLIB_DIR=$INSTALL_DIR/gklib
export METIS_DIR=$INSTALL_DIR/metis
export PARMETIS_DIR=$INSTALL_DIR/parmetis

# Clean-up
rm -rf $INSTALL_DIR/sources/gklib
rm -rf $INSTALL_DIR/sources/metis
rm -rf $INSTALL_DIR/sources/parmetis	

# If we do not want to clean-up

if [ ! -d "$INSTALL_DIR/sources/metis" ]
then
	cd $INSTALL_DIR/sources
	git clone https://github.com/KarypisLab/METIS.git metis
	git clone https://github.com/KarypisLab/GKlib.git gklib
	git clone https://github.com/KarypisLab/ParMETIS.git parmetis
fi

# DEBUG_OPTIONS='debug=1 assert=1 assert2=1'
export DEBUG_OPTIONS='debug=1 assert=1'


cd $INSTALL_DIR/sources/gklib
make config shared=0 r64=1 prefix=$INSTALL_DIR/gklib  $DEBUG_OPTIONS
make -j8
make install 

cd $INSTALL_DIR/sources/metis
make config prefix=$INSTALL_DIR/metis gklib_path=$INSTALL_DIR/gklib $DEBUG_OPTIONS
make -j8
make install 

cd $INSTALL_DIR/sources/parmetis
make config prefix=$INSTALL_DIR/parmetis gklib_path=$INSTALL_DIR/gklib metis_path=$INSTALL_DIR/metis $DEBUG_OPTIONS
make -j8
make install 
