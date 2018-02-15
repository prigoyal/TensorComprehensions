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

## Examples

Let's see few examples of what features TC has and what you can do as a starter. I'll pick a simple layer `matmul`
for the purpose of examples:

1. New TC

```python
lang = """
def matmul(float(M,N) A, float(N,K) B) -> (output) {
  output(i, j) +=! A(i, kk) * B(kk, j)
}
"""
matmul = tc.define(lang, name="matmul")
mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
out = matmul(mat1, mat2)
```

2. New TC, Autotune it, run it

```python
lang = """
def matmul(float(M,N) A, float(N,K) B) -> (output) {
  output(i, j) +=! A(i, kk) * B(kk, j)
}
"""
matmul = tc.define(lang, name="matmul")
mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
best_options = matmul.autotune(mat1, mat2, **tc.autotuner_default_options)
out = matmul(mat1, mat2, options=best_options)
```

3. New TC, Autotune it, save cache

```python
lang = """
def matmul(float(M,N) A, float(N,K) B) -> (output) {
  output(i, j) +=! A(i, kk) * B(kk, j)
}
"""
matmul = tc.define(lang, name="matmul")
matmul.autotune((3, 4), (4, 5), cache=True, **tc.small_size_autotuner_options)
matmul.autotune((100, 400), (400, 500), cache=True, **tc.autotuner_default_options)
```

3. Train layer with TC, autotune it and run it

```python
lang = """
def KRU(float(D2, N2) W2, float(M, N0, N1, N2) X) -> (XW2) {
   XW2(m, n0, n1, d2)   +=! X(m, n0, n1, n2_red) * W2(d2, n2_red)
}
def KRU_grad(float(D2, N2) W2, float(M, N0, N1, N2) X, float(M, N0, N1, D2) XW2_grad) -> (W2_grad, X_grad)
{
   W2_grad(d2, n2)   +=! XW2_grad(m_red, n0_red, n1_red, d2) * X(m_red, n0_red, n1_red, n2)
   X_grad(m, n0, n1, n2) +=! XW2_grad(m, n0, n1, d2_red) * W2(d2_red, n2)
}
"""
KRU = tc.define(lang, training=True, name="KRU", backward="KRU_grad")
X = Variable(torch.randn(256, 16, 16, 16).cuda(), requires_grad=True)
W2 = Parameter(torch.randn(32, 16)).cuda()
options = KRU.autotune(W2, X, **tc.autotuner_default_options)
out = KRU(W2, X, options=options)
out[0].sum().backward()
```

4. Get your CUDA code

```python
import tensor_comprehensions as tc

tc.GlobalDebugInit(["tc", "--dump_cuda=true"])

lang = """
def matmul(float(M,N) A, float(N,K) B) -> (output) {
  output(i, j) +=! A(i, kk) * B(kk, j)
}
"""
matmul = tc.define(lang, name="matmul")
mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
out = matmul(mat1, mat2)
```

5. much more....

Now go ahead, write a new TC (maybe fusions :) )

## What Layers we can't express right now?

Currently, we don't support layers:

1. Reshape
2. Dropout
3. Strided "tensors"
4. Padding

We are actively working on these and many more features. If there is some feature that can be very helpful
to you, please send your request our way.


## Communication

* **GitHub issues**: bug reports, feature requests, install issues, RFCs, thoughts, etc.
* **Slack**: For discussion around framework integration, build support, collaboration, etc. join our slack channel https://tensorcomprehensions.herokuapp.com/.
* **Email**: prigoyal@fb.com
