#!/usr/bin/env bash

set -e

# code gets checked out at below:
# $SOURCE_DIR ===> <anaconda_root>/conda-bld/protobuf_<timestamp>/work
#
# build directory gets created at $SOURCE_DIR/build
#
# CONDA environment for debugging:
# cd <anaconda_root>/conda-bld/protobuf_<timestamp>
# source activate ./_h_env_......    # long placeholders
#
# $CONDA_PREFIX and $PREFIX are set to the same value i.e. the environment value
#
# Installation happens in the $PREFIX which is the environment and rpath is set
# to that
#
# For tests, a new environment _test_env_.... is created
# During the tests, you will see that the protobuf package gets checked out

CMAKE_VERSION=${CMAKE_VERSION:="`which cmake3 || which cmake`"}
VERBOSE=${VERBOSE:=0}

CC=${CC:="`which gcc`"}
CXX=${CXX:="`which g++`"}

export INSTALL_PREFIX=$PREFIX
echo "CONDA_PREFIX: ${CONDA_PREFIX}"
echo "PREFIX: ${PREFIX}"

echo "Building Protobuf conda package"

./autogen.sh
./configure --prefix="${PREFIX}" \
            --with-pic \
            --enable-shared \
            --enable-static \
	    CC="${CC}" \
	    CXX="${CXX}" \
	    CXXFLAGS="${CXXFLAGS} -O2" \
	    LDFLAGS="${LDFLAGS}"

VERBOSE=${VERBOSE} make -j"$(nproc)"
make check -j"$(nproc)"
VERBOSE=${VERBOSE} make install

echo "Successfully built protobuf conda package"
