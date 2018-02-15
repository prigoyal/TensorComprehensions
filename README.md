# ![Tensor Comprehensions](docs/source/_static/img/tc-logo-full-color-with-text-2.png)

# Using Tensor Comprehensions with PyTorch

We provide integration of Tensor Comprehensions (TC) with PyTorch for both training
and inference purpose. Using TC, you can express an operator using [Einstein
notation](https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/)
and get the fast CUDA code for that layer with just 3-4 lines of code. By providing
TC integration with PyTorch, we hope to make it further easy to write new
operations with TC.

To make it easy to use TC, we provide conda packages for it. Follow the instructions
below on how to install the conda package.

## Installation
You will need anaconda to install conda packages of TC. If you don't have it, follow the next step, otherwise verify conda is in your **$PATH**.

### **Step 1:** Anaconda3
Install anaconda3 by following the instructions below:

```Shell
cd $HOME && wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O anaconda3.sh
chmod +x anaconda3.sh && ./anaconda3.sh -b -p $HOME/anaconda3 && rm anaconda3.sh
```

Now add anaconda3 to your PATH so that you can use it. For that run the following command:

```Shell
export PATH=$HOME/anaconda3/bin:$PATH
```

Now, verify your conda installation and check the version:

```Shell
which conda
```

This command should print the path of your conda bin. If it doesn't, make sure conda is
in your $PATH.

### **Step 2**: Conda Install TC

Now, go ahead and install TC by running following commands.

```Shell
conda create -y --name tc-test python=3.6
source activate tc-test
conda install -y -c pytorch -c https://conda.anaconda.org/t/oJuz1IosRLQ5/prigoyal tensor_comprehensions
```

Now, clone the repo to see bunch of examples and run a few tests:

```Shell
cd $HOME && git clone git@github.com:prigoyal/TensorComprehensions.git --recursive
./TensorComprehensions/test_python/run_test.sh
```

### **Step 3**: Explore TC

In order to explore TC, there are few helpful resources to get started:

1. We provide **examples** of TC definitions covering wide range of Deep Learning layers.
Those example TC can be found at the repo we just checked out in previous step
`$HOME/TensorComprehensions/test_python/layers`. These examples can serve as a helpful reference
for writing TC for a new operation.

2. TC is based on **Einstein notation** which is very well explained [here](https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/). This notation is
also widely used in Numpy. If you don't know the notation, I recommend doing a 5 minute read of the above link.

3. [TC Documentation](https://facebookresearch.github.io/TensorComprehensions/index.html)
is a very helpful resource to understand how TC is expressed. The sections on
[introduction](https://facebookresearch.github.io/TensorComprehensions/introduction.html),
[range inference](https://facebookresearch.github.io/TensorComprehensions/inference.html),
[semantics](https://facebookresearch.github.io/TensorComprehensions/semantics.html)
are particularly helpful to get insights into writing TC.

4. **Autotuner**: TC provides an evolutionary search based algorithm to automatically tune the kernel.
You can read briefly about autotuner [here]() and look at various examples of autotuning
in `$HOME/TensorComprehensions/test_python/layers/test_autotuner`

### Examples

Let's see few examples of what features TC has and what you can do as a starter. I'll pick a simple layer `matmul`
for the purpose of examples:

TODO (prigoyal)

### What Layers we can't express right now?

TODO (prigoyal)
