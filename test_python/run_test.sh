#!/usr/bin/env bash
set -e

PYTHON=${PYTHON:="`which python3`"}

pushd "$(dirname "$0")"

echo "Running TC imports test"
$PYTHON test_tc_imports.py -v

echo "Running Mapping options test"
$PYTHON test_mapping_options.py -v

echo "Running normal TC test"
$PYTHON test_tc.py -v

echo "Running debug init test"
$PYTHON test_debug_init.py -v

echo "Running Batchmatmul test"
$PYTHON test_batchmatmul.py -v

echo "Running Batchnorm test"
$PYTHON test_batchnorm.py -v

echo "Running C3 test"
$PYTHON test_C3.py -v

echo "Running Group Convolution test"
$PYTHON test_group_convolution.py -v

echo "Running MLP test"
$PYTHON test_mlp.py -v

echo "Running TMM test"
$PYTHON test_tmm.py -v

echo "Running all PyTorch tests"
$PYTHON test_tc_torch.py -v

popd
