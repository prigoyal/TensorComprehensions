#!/usr/bin/env bash

set -e

echo "Packaging TC"

ANACONDA_USER=prigoyal

# set the anaconda upload to NO for now
conda config --set anaconda_upload no

##############################################################################
# ISL settings
ISL_TC_BUILD_VERSION="0.2.1"
ISL_TC_BUILD_NUMBER=1
ISL_TC_GIT_HASH="67f217662681e479bc1d143a5da9caaa5ed501a1"

echo "Packaging ISL-TC first"
echo "ISL_TC_BUILD_VERSION: ${ISL_TC_BUILD_VERSION} ISL_TC_BUILD_NUMBER: ${ISL_TC_BUILD_NUMBER}"

export ISL_TC_BUILD_VERSION=$ISL_TC_BUILD_VERSION
export ISL_TC_BUILD_NUMBER=$ISL_TC_BUILD_NUMBER
export ISL_TC_GIT_HASH=$ISL_TC_GIT_HASH

time conda build -c $ANACONDA_USER --python 3.6 isl-tc --keep-old-work --no-remove-work-dir

echo "ISL-TC packaged Successfully"

###############################################################################
# CLANG+LLVM settings
CLANG_LLVM_BUILD_VERSION="0.2.1"
CLANG_LLVM_BUILD_NUMBER=1
CLANG_LLVM_GIT_HASH="65dd2568aa09fcf5498c0fe9adfd92608dbf4d5c"

echo "Building clang+llvm-tapir5.0"
echo "CLANG_LLVM_BUILD_VERSION: $CLANG_LLVM_BUILD_VERSION CLANG_LLVM_BUILD_NUMBER: ${CLANG_LLVM_BUILD_NUMBER}"

export CLANG_LLVM_BUILD_VERSION=$CLANG_LLVM_BUILD_VERSION
export CLANG_LLVM_BUILD_NUMBER=$CLANG_LLVM_BUILD_NUMBER
export CLANG_LLVM_GIT_HASH=$CLANG_LLVM_GIT_HASH

time conda build -c $ANACONDA_USER --python 3.6 clang+llvm-tapir5.0 --keep-old-work --no-remove-work-dir

echo "clang+llvm-tapir5.0 packaged Successfully, Hooray!!"

###############################################################################
# Gflags settings
GFLAGS_BUILD_VERSION="2.4.4"
GFLAGS_BUILD_NUMBER=1
GFLAGS_GIT_HASH="4663c80d3ab19fc7d9408fe8fb22b07b87c76e5a"

echo "Packaging GFLAGS ==> GFLAGS_BUILD_VERSION: ${GFLAGS_BUILD_VERSION} GFLAGS_BUILD_NUMBER: ${GFLAGS_BUILD_NUMBER}"

export GFLAGS_BUILD_VERSION=$GFLAGS_BUILD_VERSION
export GFLAGS_BUILD_NUMBER=$GFLAGS_BUILD_NUMBER
export GFLAGS_GIT_HASH=$GFLAGS_GIT_HASH

time conda build -c $ANACONDA_USER --python 3.6 gflags --keep-old-work --no-remove-work-dir

echo "GFLAGS packaged Successfully"

###############################################################################
# Gflags settings
GLOG_BUILD_VERSION="0.3.9"
GLOG_BUILD_NUMBER=1
GLOG_GIT_HASH="d0531421fd5437ae3e5249106c6fc4247996e526"

echo "Packaging GLOG ==> GLOG_BUILD_VERSION: ${GLOG_BUILD_VERSION} GLOG_BUILD_NUMBER: ${GLOG_BUILD_NUMBER}"

export GLOG_BUILD_VERSION=$GLOG_BUILD_VERSION
export GLOG_BUILD_NUMBER=$GLOG_BUILD_NUMBER
export GLOG_GIT_HASH=$GLOG_GIT_HASH

time conda build -c $ANACONDA_USER --python 3.6 glog --keep-old-work --no-remove-work-dir

echo "GLOG packaged Successfully"

##############################################################################
# Protobuf settings
PROTO_BUILD_VERSION="3.4.1"
PROTO_BUILD_NUMBER=1

echo "Packaging Protobuf ==> PROTO_BUILD_VERSION: ${PROTO_BUILD_VERSION} PROTO_BUILD_NUMBER: ${PROTO_BUILD_NUMBER}"

export PROTO_BUILD_VERSION=$PROTO_BUILD_VERSION
export PROTO_BUILD_NUMBER=$PROTO_BUILD_NUMBER

time conda build -c $ANACONDA_USER --python 3.6 protobuf --keep-old-work --no-remove-work-dir

echo "Protobuf packaged Successfully"

##############################################################################
# Halide settings
HALIDE_BUILD_VERSION="0.2.1"
HALIDE_BUILD_NUMBER=1
HALIDE_GIT_HASH="5e432c6c5e4bfc85a158bff12ce093812074ade9"

echo "Packaging HALIDE ==> HALIDE_BUILD_VERSION: ${HALIDE_BUILD_VERSION} HALIDE_BUILD_NUMBER: ${HALIDE_BUILD_NUMBER}"

export HALIDE_BUILD_VERSION=$HALIDE_BUILD_VERSION
export HALIDE_BUILD_NUMBER=$HALIDE_BUILD_NUMBER
export HALIDE_GIT_HASH=$HALIDE_GIT_HASH

time conda build -c $ANACONDA_USER --python 3.6 halide --keep-old-work --no-remove-work-dir

echo "HALIDE packaged Successfully"

###############################################################################
# Tensor Comprehensions settings
TC_BUILD_VERSION="0.2.1"
TC_BUILD_NUMBER=1
# TAG: tc-v0.2.1
TC_GIT_HASH="101ed677f87f59f0da73c718a4c4d44ae3e6d6b5"

echo "Packaging TC ==> TC_BUILD_VERSION: ${TC_BUILD_VERSION} TC_BUILD_NUMBER: ${TC_BUILD_NUMBER}"

export TC_BUILD_VERSION=$TC_BUILD_VERSION
export TC_BUILD_NUMBER=$TC_BUILD_NUMBER
export TC_GIT_HASH=$TC_GIT_HASH

# We specify channel `soumith` because that contains the pytorch v0.3.1 package we need
time conda build -c soumith --python 3.6 tensor_comprehensions --keep-old-work --no-remove-work-dir

echo "Tensor Comprehensions packaged Successfully"
