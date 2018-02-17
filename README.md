# ![Tensor Comprehensions](docs/source/_static/img/tc-logo-full-color-with-text-2.png)

# Using Tensor Comprehensions with PyTorch

## Table of Contents

- [Installation](#installation)
- [Examples and Documentation](#examples-and-documentation)
- [Going through basics: by example](#going-through-basics-by-example)
- [Layers that can't be expressed right now](#layers-that-cant-be-expressed-right-now)
- [Note about performance / tuning](#note-about-performance--tuning)
- [Communication](#communication)

A blogpost on Tensor Comprehensions can be read [here](https://research.fb.com/announcing-tensor-comprehensions/).

We provide integration of Tensor Comprehensions (TC) with PyTorch for both training
and inference purposes. Using TC, you can express an operator using [Einstein
notation](https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/)
and get the fast CUDA code for that layer with a few lines of code. By providing
TC integration with PyTorch, we hope to make it further easy to write new
operations with TC.

Here is what the PyTorch-TC package provides:

- inputs and outputs to functions are are `torch.*Tensor`s
- Integration with PyTorch `autograd`: if you specify forward and backward functions, you get an autograd function that takes `Variable` as input and returns `Variable` as output. Here's an [example](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_convolution_train-py).
- autotuner results can be cached to a file (for reuse)

To make it easy to use TC, we provide conda packages for it. Follow the instructions
below on how to install the conda package.
Building from source is not easy, because of large dependencies like llvm, so using the conda package is ideal.

## Installation
You will need anaconda to install conda packages of TC. If you don't have it, follow the next step, otherwise verify conda is in your **$PATH** and proceed to Step 2.

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

### **Step 2**: Conda Install Tensor Comprehensions

Now, go ahead and install Tensor Comprehensions by running following commands.

```Shell
conda install -y -c pytorch -c https://conda.anaconda.org/t/oJuz1IosRLQ5/prigoyal tensor_comprehensions
```

## Examples and documentation

In order to explore Tensor Comprehensions (TC), there are few helpful resources to get started:

1. We provide **examples** of TC definitions covering wide range of Deep Learning layers.

The list of examples we provide are: [avgpool](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_avgpool_autotune-py), [maxpool](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_maxpool-py), [matmul](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_matmul-py), [matmul - give output buffers](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_matmul_reuse_outputs-py) and [batch-matmul](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_batchmatmul-py), [convolution](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_convolution-py), [strided-convolution](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_convolution_strided-py), [batchnorm](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_batchnorm-py), [copy](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_copy-py), [cosine similarity](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_cosine_similarity-py), [Fully-connected](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_fc-py), [fused FC + ReLU](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_fusion_fcrelu-py), [group-convolutions](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_group_convolution-py), [strided group-convolutions](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_group_convolution_strided-py), [indexing](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_indexing-py), [Embedding (lookup table)](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_lookup_table-py), [small-mobilenet](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_small_mobilenet-py), [softmax](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_softmax-py), [tensordot](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_tensordot-py), [transpose](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_transpose-py)

2. Tensor Comprehensions are based on **Einstein notation** which is very well explained [here](https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/). This notation is
also widely used in Numpy. If you don't know the notation, we recommend doing a 5 minute read of the above link.

3. [TC Documentation](https://facebookresearch.github.io/TensorComprehensions/index.html)
is a very helpful resource to understand how Tensor Comprehensions are expressed. The sections on
[introduction](https://facebookresearch.github.io/TensorComprehensions/introduction.html),
[range inference](https://facebookresearch.github.io/TensorComprehensions/inference.html),
[semantics](https://facebookresearch.github.io/TensorComprehensions/semantics.html)
are particularly helpful to get insights into writing Tensor Comprehensions.

4. **Autotuner**: TC provides an evolutionary search based algorithm to automatically tune the kernel.
You can read briefly about autotuner [here](https://facebookresearch.github.io/TensorComprehensions/autotuner.html) and look at various [examples of autotuning](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_autotuner-py).

5. To construct a TC autograd function, [here](https://gist.github.com/anonymous/dc0cd7de343922a8c0c0636ccc4889a9#file-test_convolution_train-py) is one self-descriptive example.

## Going through basics: by example

Let's see few examples of what features Tensor Comprehensions has and what you can do as a starter. I'll pick a simple layer `matmul`
for the purpose of examples and start with describing reduction operator we will use.

**Reduction Operator**:

`+=!` operator with `!` means that the output tensor will be initialized to reduction identity i.e. `0` for `+`.

Now, let's cover the basics:

1. New Tensor Comprehension:

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

2. New Tensor Comprehension, Autotune it, run it:

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

3. New Tensor Comprehension, Autotune it, save cache:

```python
lang = """
def matmul(float(M,N) A, float(N,K) B) -> (output) {
  output(i, j) +=! A(i, kk) * B(kk, j)
}
"""
matmul = tc.define(lang, name="matmul")
matmul.autotune((3, 4), (4, 5), cache="matmul_345.tc", **tc.small_size_autotuner_options)
matmul.autotune((100, 400), (400, 500), cache="matmul_100400500.tc", **tc.autotuner_default_options)
```

**The big advantage of specifying `cache` is that the next time you run the program, the cached autotuned values are used.**
Beware that if you move to a significantly different type of GPU, then you might want to tune again for maximum performance.

3. Train layer with TC, Autotune it and run it:

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

4. Dump out generated CUDA code (for fun?):

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

5. Inject your own CUDA code and run it (because you might have faster code):

```python
lang = """
def add(float(N) A, float(N) B) -> (output) {
    output(i) = A(i) + B(i) + 1
}
"""

cuda_code = """
extern "C"{
__global__ void my_add(float* __restrict__ output, const float* __restrict__ A, const float* __restrict B) {
    int t = threadIdx.x;
    output[t] = A[t] + B[t];
}
}
"""

add = tc.define(lang, name="add", inject_kernel="my_add", cuda_code=cuda_code)
a, b = torch.randn(100).cuda(), torch.randn(100).cuda()
out = add(a, b, grid=[1, 1, 1], block=[100, 1, 1])    # change grid/block for adjusting kernel performance
```

## Layers that can't be expressed right now

1. Reshaping Tensors inside the language
2. Dropout : RNGs are not suppported inside TC language, because TC doesn't do internal allocations
3. Strided "tensors" : input Tensors have to be contiguous. If they are not contiguous, they are made contiguous before passing to the TC backend.
4. RNNs : TC language doesn't have loops yet. You can write them unrolled if you want :)

**We are actively working on these and many more features. If there is some feature that can be very helpful
to you, please send your request our way.**

## Note about performance / tuning

Tensor Comprehensions have an autotuner that uses evolutionary search to find faster kernels.
Here is what you should know about the polyhederal exploration / evolutionary search:

### Static sizes for autotuning

- The autotuner needs static input sizes (for now). You can not tune a kernel, for say: batchsize between `16 and 32`
  - you can autotune `avgpool2x2` for input shape `(16, 32, 24, 23)`:
    ```
    avgpool.autotune((16, 32, 24, 23), **tc.small_size_autotuner_options, cache="16x32x24x23.tc")
    ```
  - if you want to target multiple input shapes, run multiple autotune calls:
    ```
    avgpool.autotune((16, 32, 24, 23), **tc.small_size_autotuner_options, cache="mysize1.tc")
    avgpool.autotune((32, 1, 128, 128), **tc.small_size_autotuner_options, cache="mysize2.tc")
    ```
  - The more static we make the sizes, the better and faster the search procedure. Hence, we made this trade-off of only supporting static sizes in the initial release.
  
### Autotuning options primer

By **default**, `tc.autotuner_default_options` is:

```
options = {
    "threads": 32, "generations": 2, "pop_size": 10, "number_elites": 1
}
```

Good for quick autotuning (2 generations finish quickly)

**good default that runs for a bit longer (maybe in exchange for better performance)**

```
options = {
    "threads": 32, "generations": 5, "pop_size": 10, "number_elites": 1
}
```

**good default that runs for a LOT longer**

```
options = {
    "threads": 32, "generations": 25, "pop_size": 100, "number_elites": 10
}
```


**brief explanation**

- `threads` - set this to number of CPU cores available.
- `generations` - 5 to 10 generations is a good number.
- `pop_size` - 10 is usually reasonable. You can try 10 to 20.
- `number_elites` - number of candidates preserved intact between generations. `1` is usually sufficient.
- `min_launch_total_threads` - If you have really input small sizes, set this to `1`.
- `gpus`: Number of gpus to use for autotuning. Default value is "0". Set this to "0,1" if you wish to use two gpus.

Look at [docs](https://facebookresearch.github.io/TensorComprehensions/autotuner.html) for more details

## Communication

* **Email**: prigoyal@fb.com
* **[GitHub](https://github.com/facebookresearch/TensorComprehensions/) issues**: bug reports, feature requests, install issues, RFCs, thoughts, etc.
* **Slack**: For discussion around framework integration, build support, collaboration, etc. join our slack channel https://tensorcomprehensions.herokuapp.com/.
