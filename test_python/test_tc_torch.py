# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

import os, unittest, time, pdb

import torch
import torch.cuda
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import tensor_comprehensions as tc
from tensor_comprehensions.mapping_options import Options
from common import TestCase, run_tests

tc.GlobalDebugInit(["--dump_cuda=false"])


MATMUL_LANG = """
def matmul(float(M,N) A, float(N,K) B) -> (output) {
  output(i, j) +=! A(i, kk) * B(kk, j)
}
"""

MATMUL_KRU1_LANG = """
def matmul(float(M,N) A, float(N,K) B) -> (output) {
  output(i, j) +=! A(i, kk) * B(kk, j)
}
def KRU3_1(float(D2, N2) W2, float(M, N0, N1, N2) X) -> (XW2) {
   XW2(m, n0, n1, d2)   +=! X(m, n0, n1, n2_red) * W2(d2, n2_red)
}
"""

KRU3_1_TRAINING = """
def KRU3_1(float(D2, N2) W2, float(M, N0, N1, N2) X) -> (XW2) {
   XW2(m, n0, n1, d2)   +=! X(m, n0, n1, n2_red) * W2(d2, n2_red)
}
def KRU3_1_GRAD(float(D2, N2) W2, float(M, N0, N1, N2) X, float(M, N0, N1, D2) XW2_grad) -> (W2_grad, X_grad) {
   W2_grad(d2, n2)   +=! XW2_grad(m_red, n0_red, n1_red, d2) * X(m_red, n0_red, n1_red, n2)
   X_grad(m, n0, n1, n2) +=! XW2_grad(m, n0, n1, d2_red) * W2(d2_red, n2)
}
"""

# PATH_PREFIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tc_test")
PATH_PREFIX = os.path.join("/tmp/", "tc_test")

if not os.path.exists(PATH_PREFIX):
    os.makedirs(PATH_PREFIX)


###########################################################################
# Manual cuda injection tests
###########################################################################
class TestManualCudaInjection(unittest.TestCase):
    def test_simple_cuda_injection(self):
        lang = """
        def add(float(N) A, float(N) B) -> (output) {
            output(i) = A(i) + B(i) + 1
        }
        """

        cuda_code = """
        extern "C"{
        __global__ void my_add(float* __restrict__ output, const float* __restrict__ A, const float* __restrict B)
        {
            int t = threadIdx.x;
            output[t] = A[t] + B[t];
        }
        }
        """

        add = tc.define(lang, name="add", inject_kernel="my_add", cuda_code=cuda_code)
        a, b = torch.randn(100).cuda(), torch.randn(100).cuda()
        out = add(a, b, grid=[1, 1, 1], block=[100, 1, 1])

###########################################################################
# TC tests without autotuning
###########################################################################
class TestTC(unittest.TestCase):
    def test_indexing(self):
        LANG = """
        def indexing(float(H, W) input, int32(L) index) -> (output) {{
            output(l, w) = input(index(l), w) where l in 0:{L}
        }}
        """
        indexing = tc.define(LANG, name="indexing", constants={"L":2})
        inp = torch.arange(0, 16).view(4, 4).cuda()
        idx = torch.IntTensor([1, 1]).cuda()
        out = indexing(inp, idx)

    def test_avgpool(self):
        # NOTE: take note of use of {{ }}
        LANG="""
        def avgpool(float(B, C, H, W) input) -> (output) {{
            output(b, c, h, w) += input(b, c, h * {sH} + kh, w * {sW} + kw) where kh in 0:{kH}, kw in 0:{kW}
        }}
        """
        avgpool = tc.define(LANG, name="avgpool", constants={"sH":1, "sW":1, "kH":2, "kW":2})
        inp = torch.ones(1, 1, 4, 4).cuda()
        out = avgpool(inp)

    def test_matmul(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        out = matmul(mat1, mat2)

    def test_manual_options(self):
        lang = MATMUL_KRU1_LANG
        KRU3_1 = tc.define(lang, name="KRU3_1")
        W2, X = torch.randn(32, 16).cuda(), torch.randn(256, 16, 16, 16).cuda()
        options = Options("naive")
        out = KRU3_1(W2, X, options=options)

    def test_different_input_sizes(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        out1 = matmul(mat1, mat2)

        # if the inputs sizes are different, re-compilation will happen
        mat3, mat4 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        out2 = matmul(mat3, mat4)

    def test_same_tc_reuse_outputs(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        out = matmul(mat1, mat2)
        # reuse the same outputs now instad of allocating again, so we save
        # overhead of allocating storage again. Also, if the input sizes are same
        # we skip the compilation and run directly
        mat3, mat4 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        matmul(mat3, mat4, outputs=out)

    def test_multiple_tc(self):
        lang = MATMUL_KRU1_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        out = matmul(mat1, mat2)

        KRU3_1 = tc.define(lang, name="KRU3_1")
        W2, X = torch.randn(32, 16).cuda(), torch.randn(256, 16, 16, 16).cuda()
        out = KRU3_1(W2, X)

    def test_matmul_variable(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = Variable(torch.randn(3, 4).cuda(), requires_grad=True), Variable(torch.randn(4, 5).cuda(), requires_grad=True)
        out = matmul(mat1, mat2)

    def test_matmul_variable_reuse_outputs(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = Variable(torch.randn(3, 4).cuda(), requires_grad=True), Variable(torch.randn(4, 5).cuda(), requires_grad=True)
        out = matmul(mat1, mat2)

        mat3, mat4 = Variable(torch.randn(3, 4).cuda(), requires_grad=True), Variable(torch.randn(4, 5).cuda(), requires_grad=True)
        matmul(mat3, mat4, outputs=out)

    def test_kru_backward(self):
        lang = KRU3_1_TRAINING
        KRU3_1 = tc.define(lang, training=True, name="KRU3_1", backward="KRU3_1_GRAD")
        X = Variable(torch.randn(256, 16, 16, 16).cuda(), requires_grad=True)
        W2 = Parameter(torch.randn(32, 16)).cuda()
        out = KRU3_1(W2, X)
        out[0].sum().backward()

    def test_kru_backward_pass_options(self):
        lang = KRU3_1_TRAINING
        forward_options = Options("naive")
        backward_options = Options("naive")
        KRU3_1 = tc.define(lang, training=True, name="KRU3_1", backward="KRU3_1_GRAD")
        X = Variable(torch.randn(256, 16, 16, 16).cuda(), requires_grad=True)
        W2 = Parameter(torch.randn(32, 16).cuda())
        out = KRU3_1(W2, X, options=[forward_options, backward_options])
        out[0].sum().backward()

###########################################################################
# Autotuner tests
###########################################################################
class TestAutotuner(unittest.TestCase):

    # For such small sizes, autotuner will probably not help.
    # NOTE: Use "--tuner_min_launch_total_threads=1" for running small sizes
    # tc.small_size_autotuner_options has this option set already
    def test_autotuner_no_cache_small_size(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        options = matmul.autotune(mat1, mat2, **tc.small_size_autotuner_options)

    def test_autotuner_no_cache_medium_size(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(72, 26).cuda(), torch.randn(26, 72).cuda()
        options = matmul.autotune(mat1, mat2, **tc.autotuner_default_options)

    def test_autotuner_no_cache(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        options = matmul.autotune(mat1, mat2, **tc.autotuner_default_options)

    def test_autotuner_no_cache_explicit_set(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        options = matmul.autotune(mat1, mat2, cache=False, **tc.autotuner_default_options)

    def test_autotuner_cache_to_default(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        matmul.autotune(mat1, mat2, cache=True, **tc.autotuner_default_options)

    def test_autotuner_cachefile_first(self):
        cache_file = "{}/matmul_100_400_500".format(PATH_PREFIX)    # use argparse if input from command line
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        matmul.autotune(mat1, mat2, cache=cache_file, **tc.autotuner_default_options)

    def test_autotuner_cachefile_load(self):
        lang = MATMUL_LANG
        cache_file = "{}/matmul_100_400_500".format(PATH_PREFIX)    # use argparse if input from command line
        assert os.path.isfile("{}.cuda".format(cache_file)), "looks like the cache_file doesn't exist"

        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        out = matmul(mat1, mat2, cache=cache_file)

    def test_autotuner_cachefile_load_automatic(self):
        lang = MATMUL_LANG
        cache_file = "{}/matmul_100_400_500".format(PATH_PREFIX)    # use argparse if input from command line
        assert os.path.isfile("{}.cuda".format(cache_file)), "looks like the cache_file doesn't exist"

        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        out1 = matmul(mat1, mat2, cache=cache_file)
        # the second time we run the kernel, we skip the compilation since it was
        # already compiled earlier
        out2 = matmul(mat1, mat2)

    def test_autotuner_no_cache_and_run_kernel(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        options = matmul.autotune(mat1, mat2, **tc.autotuner_default_options)
        out = matmul(mat1, mat2, options=options)

    def test_autotuner_no_cache_and_run_kernel_automatic(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        matmul.autotune(mat1, mat2, **tc.autotuner_default_options)
        out = matmul(mat1, mat2)

    def test_autotuner_start_options_and_run_kernel(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        options = Options("mlp")
        best_options = matmul.autotune(mat1, mat2, cache=True, options=options, **tc.autotuner_default_options)
        out = matmul(mat1, mat2, options=best_options)

    def test_autotuner_multiple_tc(self):
        lang = MATMUL_KRU1_LANG
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(72, 26).cuda(), torch.randn(26, 72).cuda()
        matmul.autotune(mat1, mat2, cache=True, **tc.autotuner_default_options)
        out = matmul(mat1, mat2)

        KRU3_1 = tc.define(lang, name="KRU3_1")
        W2, X = torch.randn(32, 16).cuda(), torch.randn(256, 16, 16, 16).cuda()
        KRU3_1.autotune(W2, X, cache=True, **tc.autotuner_default_options)
        out = KRU3_1(W2, X)

    ###########################################################################
    # Pass tuple inputs for autotuning
    ###########################################################################
    def test_autotuner_tuple_size_no_cache(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        matmul.autotune((3, 4), (4, 5), **tc.small_size_autotuner_options)
        matmul.autotune((100, 400), (400, 500), **tc.autotuner_default_options)

    def test_autotuner_tuple_size_cache_to_default(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        matmul.autotune((3, 4), (4, 5), cache=True, **tc.small_size_autotuner_options)
        matmul.autotune((100, 400), (400, 500), cache=True, **tc.autotuner_default_options)

    def test_autotuner_tuple_size_cache_to_file_run_kernel(self):
        lang = MATMUL_LANG
        matmul = tc.define(lang, name="matmul")
        cache1 = "{}/matmul_3_4_5".format(PATH_PREFIX)
        cache2 = "{}/matmul_100_400_500".format(PATH_PREFIX)
        matmul.autotune((3, 4), (4, 5), cache=cache1, **tc.small_size_autotuner_options)
        matmul.autotune((100, 400), (400, 500), cache=cache2, **tc.autotuner_default_options)

        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        out = matmul(mat1, mat2, cache=cache1)

        mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
        out = matmul(mat1, mat2, cache=cache2)

    ##########################################################################
    # Training layer autotuning
    ##########################################################################
    def test_kru_train_autotune_no_cache_no_options(self):
        lang = KRU3_1_TRAINING
        KRU3_1 = tc.define(lang, training=True, name="KRU3_1", backward="KRU3_1_GRAD")
        W2, X = torch.randn(32, 16).cuda(), torch.randn(256, 16, 16, 16).cuda()
        options = KRU3_1.autotune(W2, X, **tc.autotuner_default_options)

    def test_kru_train_autotune_no_cache_no_options_seed(self):
        lang = KRU3_1_TRAINING
        KRU3_1 = tc.define(lang, training=True, name="KRU3_1", backward="KRU3_1_GRAD")
        W2, X = torch.randn(32, 16).cuda(), torch.randn(256, 16, 16, 16).cuda()
        KRU3_1.autotune(W2, X, **tc.autotuner_default_options)
        # on the second call, autotuning will be seeded from previous best options
        KRU3_1.autotune(W2, X, **tc.autotuner_default_options)

    def test_kru_train_autotune_cache_no_options_seed(self):
        lang = KRU3_1_TRAINING
        KRU3_1 = tc.define(lang, training=True, name="KRU3_1", backward="KRU3_1_GRAD")
        W2, X = torch.randn(32, 16).cuda(), torch.randn(256, 16, 16, 16).cuda()
        KRU3_1.autotune(W2, X, cache=True, **tc.autotuner_default_options)
        # on the second call, autotuning will be seeded from previous best options
        KRU3_1.autotune(W2, X, cache=True, **tc.autotuner_default_options)

    def test_kru_train_autotune_cache_to_default(self):
        lang = KRU3_1_TRAINING
        KRU3_1 = tc.define(lang, training=True, name="KRU3_1", backward="KRU3_1_GRAD")
        W2, X = torch.randn(32, 16).cuda(), torch.randn(256, 16, 16, 16).cuda()
        options = KRU3_1.autotune(W2, X, cache=True, **tc.autotuner_default_options)

    def test_kru_train_autotune_to_cache_file_seed(self):
        lang = KRU3_1_TRAINING
        cache_file = "{}/KRU3_1_32_16_16_16_32".format(PATH_PREFIX)
        KRU3_1 = tc.define(lang, training=True, name="KRU3_1", backward="KRU3_1_GRAD")
        W2, X = torch.randn(32, 16).cuda(), torch.randn(256, 16, 16, 16).cuda()
        KRU3_1.autotune(W2, X, cache=cache_file, **tc.autotuner_default_options)
        # the second call should be seeded from the previous call
        KRU3_1.autotune(W2, X, cache=cache_file, **tc.autotuner_default_options)

    def test_kru_train_autotune_to_cache_file_seed(self):
        lang = KRU3_1_TRAINING
        cache_file = "{}/KRU3_1_32_16_16_16_32".format(PATH_PREFIX)
        KRU3_1 = tc.define(lang, training=True, name="KRU3_1", backward="KRU3_1_GRAD")
        W2, X = torch.randn(32, 16).cuda(), torch.randn(256, 16, 16, 16).cuda()
        # we already did autotuning to this file in previous test, now we will
        # seed the current autotuning with the previous options
        options = KRU3_1.autotune(W2, X, cache=cache_file, **tc.autotuner_default_options)

    def test_kru_train_autotune_manual_options(self):
        lang = KRU3_1_TRAINING
        options = Options("mlp")
        KRU3_1 = tc.define(lang, training=True, name="KRU3_1", backward="KRU3_1_GRAD")
        W2, X = torch.randn(32, 16).cuda(), torch.randn(256, 16, 16, 16).cuda()
        options = KRU3_1.autotune(W2, X, options=options, cache=True, **tc.autotuner_default_options)

    def test_kru_train_autotune_manual_options_both(self):
        lang = KRU3_1_TRAINING
        forward_options, backward_options = Options("mlp"), Options("mlp")
        KRU3_1 = tc.define(lang, training=True, name="KRU3_1", backward="KRU3_1_GRAD")
        W2, X = torch.randn(32, 16).cuda(), torch.randn(256, 16, 16, 16).cuda()
        options = KRU3_1.autotune(W2, X, options=[forward_options, backward_options], cache=True, **tc.autotuner_default_options)

    def test_kru_train_autotune_to_cache_file(self):
        lang = KRU3_1_TRAINING
        cache_file = "{}/KRU3_1_32_16_16_16_32".format(PATH_PREFIX)
        KRU3_1 = tc.define(lang, training=True, name="KRU3_1", backward="KRU3_1_GRAD")
        W2, X = torch.randn(32, 16).cuda(), torch.randn(256, 16, 16, 16).cuda()
        options = KRU3_1.autotune(W2, X, cache=cache_file, **tc.autotuner_default_options)

    def test_kru_train_autotune_no_cache_and_run(self):
        lang = KRU3_1_TRAINING
        KRU3_1 = tc.define(lang, training=True, name="KRU3_1", backward="KRU3_1_GRAD")
        X = Variable(torch.randn(256, 16, 16, 16).cuda(), requires_grad=True)
        W2 = Parameter(torch.randn(32, 16)).cuda()
        options = KRU3_1.autotune(W2, X, **tc.autotuner_default_options)
        out = KRU3_1(W2, X, options=options)
        out[0].sum().backward()

    def test_kru_train_autotune_no_cache_and_run_automatic(self):
        lang = KRU3_1_TRAINING
        KRU3_1 = tc.define(lang, training=True, name="KRU3_1", backward="KRU3_1_GRAD")
        X = Variable(torch.randn(256, 16, 16, 16).cuda(), requires_grad=True)
        W2 = Parameter(torch.randn(32, 16)).cuda()
        KRU3_1.autotune(W2, X, **tc.autotuner_default_options)
        out = KRU3_1(W2, X)
        out[0].sum().backward()

    def test_kru_train_autotune_cache_to_default_and_run_layer(self):
        lang = KRU3_1_TRAINING
        KRU3_1 = tc.define(lang, training=True, name="KRU3_1", backward="KRU3_1_GRAD")
        X = Variable(torch.randn(256, 16, 16, 16).cuda(), requires_grad=True)
        W2 = Parameter(torch.randn(32, 16)).cuda()
        options = KRU3_1.autotune(W2, X, cache=True, **tc.autotuner_default_options)
        out = KRU3_1(W2, X, options=options)
        out[0].sum().backward()

    def test_kru_train_backward_use_cache_file(self):
        lang = KRU3_1_TRAINING
        cache_file = "{}/KRU3_1_32_16_16_16_32".format(PATH_PREFIX)

        KRU3_1 = tc.define(lang, training=True, name="KRU3_1", backward="KRU3_1_GRAD")
        X = Variable(torch.randn(256, 16, 16, 16).cuda(), requires_grad=True)
        W2 = Parameter(torch.randn(32, 16)).cuda()
        out = KRU3_1(W2, X, cache=cache_file)
        out[0].sum().backward()

    def test_kru_train_backward_use_cache_file_twice_automatic(self):
        lang = KRU3_1_TRAINING
        cache_file = "{}/KRU3_1_32_16_16_16_32".format(PATH_PREFIX)

        KRU3_1 = tc.define(lang, training=True, name="KRU3_1", backward="KRU3_1_GRAD")
        X = Variable(torch.randn(256, 16, 16, 16).cuda(), requires_grad=True)
        W2 = Parameter(torch.randn(32, 16)).cuda()
        out = KRU3_1(W2, X, cache=cache_file)
        out[0].sum().backward()
        # we should hit the cache when we run and directly run
        out = KRU3_1(W2, X)
        out[0].sum().backward()

    def test_kru_train_autotune_use_reorder_function(self):
        lang = KRU3_1_TRAINING
        cache_file = "{}/KRU3_1_32_16_16_16_32".format(PATH_PREFIX)
        # the custom_function can also take inputs here if there are backwards
        # specific inputs needed
        def reorder():
            def reorder_function(grad_outputs):
                return grad_outputs
            return reorder_function

        KRU3_1 = tc.define(lang, training=True, name="KRU3_1", backward="KRU3_1_GRAD")
        W2, X = torch.randn(32, 16).cuda(), torch.randn(256, 16, 16, 16).cuda()
        KRU3_1.autotune(W2, X, cache=cache_file, reorder_function=reorder(), **tc.autotuner_default_options)

    def test_kru_train_backward_use_reorder_function(self):
        lang = KRU3_1_TRAINING
        cache_file = "{}/KRU3_1_32_16_16_16_32".format(PATH_PREFIX)
        # the custom_function can also take inputs here if there are backwards
        # specific inputs needed
        def reorder():
            def reorder_function(grad_outputs):
                return grad_outputs
            return reorder_function

        KRU3_1 = tc.define(lang, training=True, name="KRU3_1", backward="KRU3_1_GRAD")
        X = Variable(torch.randn(256, 16, 16, 16).cuda(), requires_grad=True)
        W2 = Parameter(torch.randn(32, 16).cuda())
        out = KRU3_1(W2, X, cache=cache_file, reorder_function=reorder())
        out[0].sum().backward()


if __name__ == '__main__':
    unittest.main()
