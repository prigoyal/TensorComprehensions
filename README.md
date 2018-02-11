# Using Tensor Comprehensions for writing a layer in PyTorch

We provide integration of Tensor Comprehensions (TC) with PyTorch for both training
and inference purpose. Using TC, you can express any operator using [Einstein
notation](https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/)
and generate the fast cuda implementation code for that layer with just 3-4 lines
of code. A few examples are shown below and a detailed tutorial will be provided for this.


## Installation and prerequisites

For native and python TC development, follow the following installation instructions **step-wise**. If you have already done some steps (in-order), you can skip them.

### **Step 1:** Install Clang+LLVM

For building TC, you also need to install a custom clang+llvm. For that, follow the instructions below:

```Shell
# now check your gcc/g++ and make sure they are in system path, somewhere like /usr/bin
which gcc
which g++

export CC=$(which gcc)
export CXX=$(which g++)
export LLVM_SOURCES=/tmp/llvm_sources-tapir5.0
export CLANG_PREFIX=/usr/local/clang+llvm-tapir5.0  # change this to whatever path you want
export CMAKE_VERSION=cmake

mkdir -p $LLVM_SOURCES
cd $LLVM_SOURCES

# clone the repo now
git clone --recursive https://github.com/wsmoses/Tapir-LLVM llvm
mkdir -p ${LLVM_SOURCES}/llvm_build && cd ${LLVM_SOURCES}/llvm_build

# cmake and install
${CMAKE_VERSION} -DCMAKE_INSTALL_PREFIX=${CLANG_PREFIX} -DLLVM_TARGETS_TO_BUILD=X86 -DCOMPILER_RT_BUILD_CILKTOOLS=OFF -DLLVM_ENABLE_CXX1Y=ON -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_BUILD_TESTS=OFF -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_BUILD_LLVM_DYLIB=ON  -DLLVM_ENABLE_RTTI=ON ../llvm/

make -j"$(nproc)" -s
make install -j"$(nproc)" -s

cd /usr/local
rm -rf $LLVM_SOURCES
```

### **Step 2:** Install Anaconda3
In order to contribute to TC python/C++ API, you need to install TC from source. For this,
anaconda3 is required. Install anaconda3 by following the instructions below:

```Shell
cd /usr/local
wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O anaconda3.sh
chmod +x anaconda3.sh
./anaconda3.sh -b -p /usr/local/anaconda3
rm anaconda3.sh
```

Now add anaconda3 to your PATH so that you can use it. For that run the following command:

```Shell
export PATH=/usr/local/anaconda3/bin:$PATH
```

Now, verify your conda installation and check the version:

```Shell
which conda
```

This command should print the path of your conda bin. If it doesn't, make sure conda is
in your $PATH.

Now, let's create a conda environment which we will work in.

```Shell
conda create -y --name tc-build python=3.6
source activate tc-build
```

### **Step 3:** Install PyTorch

Now for PyTorch, we will install it from conda. Run the following command:

```Shell
conda install -y -c soumith pytorch
```

Wait for this to finish and verify your PyTorch installed correctly. For that run following on your command line:

```python
python -c "import torch"
```

### **Step 4:** Get CUDA and CUDNN
In order to build TC, you also need to have CUDA and cudnn. If you already have it
you can just export the PATH, LD_LIBRARY_PATH simply (see the end of this step). If you don't have CUDA/CUDNN, then follow the instructions below:

```Shell
# install CUDA Toolkit v8.0
# instructions from https://developer.nvidia.com/cuda-downloads (linux -> x86_64 -> Ubuntu -> 16.04 -> deb (network))
CUDA_REPO_PKG="cuda-repo-ubuntu1604_8.0.61-1_amd64.deb"
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${CUDA_REPO_PKG}
sudo dpkg -i ${CUDA_REPO_PKG}
sudo apt-get update
sudo apt-get -y install cuda

# install cuDNN v6.0
CUDNN_TAR_FILE="cudnn-8.0-linux-x64-v6.0.tgz"
wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/${CUDNN_TAR_FILE}
tar -xzvf ${CUDNN_TAR_FILE}
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-8.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/
sudo chmod a+r /usr/local/cuda-8.0/lib64/libcudnn*

# set environment variables
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/local/cuda-8.0/bin:$PATH
```

### **Step 5:** Installing TC

Now, you need to install TC from source (conda packages coming soon). For installing
TC from source, checkout the c2isl repo and run the following commands:

```Shell
cd /usr/local && git clone git@github.com:nicolasvasilache/c2isl.git --recursive
cd c2isl
git submodule update --init --recursive
export CC=$(which gcc)
export CXX=$(which g++)
export TC_DIR=$(pwd)

# install pyyaml now, this is needed for some dependencies
conda install -y pyyaml
```

**NOTE 1:**: Please make sure that you are using system gcc/g++ and not the ones that
are provided by conda. TC supports gcc 4.8.* and gcc 5 builds at the moment.

**NOTE 2:**: Please also make sure that you don't have gflags or glog in your system path. Those might conflict with the TC gflags/glog.

Now, let's run the TC installation.

1. If you want to install Caffe2 as well, run the following:

```Shell
# you set the CLANG_PREFIX variable in your CLANG+LLVM install above, use that
WITH_PYTHON_C2=OFF CLANG_PREFIX=/usr/local/clang+llvm-tapir5.0 ./build.sh --all
```

**NOTE**: This turns off the Caffe2 python build. If you want to turn on the Caffe2
python build, see next step:

2. For installing python binaries as well of Caffe2 with TC:

```Shell
WITH_PYTHON_C2=ON CLANG_PREFIX=/usr/local/clang+llvm-tapir5.0 ./build.sh --all
```

**NOTE**: Caffe2 doesn't provide support for pip/conda at the moment and this means
in order to use the caffe2 python, you might need to set $PYTHONPATH. Normally,
it could be

```Shell
${TC_DIR}/third-party-install/
```

However, please check caffe2 official instructions [here](https://caffe2.ai/docs/getting-started.html?platform=mac&configuration=compile#test-the-caffe2-installation). TC doesn't yet provide support for caffe2 python usage.

3. For installing TC without Caffe2:

```Shell
WITH_CAFFE2=OFF CLANG_PREFIX=/usr/local/clang+llvm-tapir5.0 ./build.sh --all
```

### **Step 6:** Verify TC installation:

```Shell
cd /usr/local/c2isl
./test.sh                 # if you have GPU
./test_cpu.sh             # if you have only CPU
./test_python/run_test.sh   # if you have GPU
```

Make sure all the tests pass here. Now you are ready to start contributing to the C++/Python API of TC.

### Examples
A very simple example of how writing new layers with TC in PyTorch looks like:

Now, let's see how to write a matmul example with TC and use it as a PyTorch Layer:

```python
# first import tc and torch
import tensor_comprehensions as tc
import torch

# use Einstein notation to write the TC expression for layer
lang = """
def matmul(float(M,N) A, float(N,K) B) -> (output) {
  output(i, j) += A(i, kk) * B(kk, j)
}
"""

# now we register the op with TC using tc.define()
matmul = tc.define(lang, name="matmul")

# create input tensors and run layer on it
mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
out = matmul(mat1, mat2)
```

For using TC for training purposes, you need write the expression for backward
layer following Einstein notation. Let's take a look at one example:

```python
import tc
import torch

# define the forward and backward operations
lang = """
def MY_LAYER(float(D2, N2) W2, float(M, N0, N1, N2) X) -> (XW2) {
   XW2(m, n0, n1, d2)   += X(m, n0, n1, n2_red) * W2(d2, n2_red)
}
def MY_LAYER_GRAD(float(D2, N2) W2, float(M, N0, N1, N2) X, float(M, N0, N1, D2) XW2_grad) -> (W2_grad, X_grad) {
   W2_grad(d2, n2)   += XW2_grad(m_red, n0_red, n1_red, d2) * X(m_red, n0_red, n1_red, n2)
   X_grad(m, n0, n1, n2) += XW2_grad(m, n0, n1, d2_red) * W2(d2_red, n2)
}
"""

MY_LAYER = tc.define(lang, training=True, name="MY_LAYER", backward="MY_LAYER_GRAD")
X = Variable(torch.randn(256, 16, 16, 16), requires_grad=True).cuda()
W2 = Parameter(torch.randn(32, 16)).cuda()
out = MY_LAYER(W2, X)

# get the gradients by simply calling backward
out[0].sum().backward()
```

These were simple examples of how to use TC with PyTorch. Please refer to the documentation
for various rules around how to pass the inputs, creating tensors etc.

More details will be available in TC documentation.
