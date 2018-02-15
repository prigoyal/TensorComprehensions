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

import os, pdb

import torch
import torch.cuda

from tensor_comprehensions.mapping_options import Options
from test_python.common import TestCase, run_tests


class TestMLP(TestCase):
    def test_mlp(self):
        # define TC
        lang = """
        def mlp3(float(B,N) I, float(O,N) W2, float(O) B2, float(P,O) W3, float(P) B3, float(Q,P) W4, float(Q) B4) -> (O2, O3, O4) {
            O2(b, o) +=! I(b, n) * W2(o, n)
            O2(b, o) = O2(b, o) + B2(o)
            O2(b, o) = fmax(O2(b, o), 0)
            O3(b, p) +=! O2(b, o) * W3(p, o)
            O3(b, p) = O3(b, p) + B3(p)
            O3(b, p) = fmax(O3(b, p), 0)
            O4(b, q) +=! O3(b, p) * W4(q, p)
            O4(b, q) = O4(b, q) + B4(q)
            O4(b, q) = fmax(O4(b, q), 0)
        }
        """

        # create input tensors
        B, N, O, P, Q = 128, 128, 64, 32, 2
        I = torch.randn(B, N).cuda()
        W2 = torch.randn(O, N).cuda()
        B2 = torch.randn(O).cuda()
        W3 = torch.randn(P, O).cuda()
        B3 = torch.randn(P).cuda()
        W4 = torch.randn(Q, P).cuda()
        B4 = torch.randn(Q).cuda()
        inputs = [I, W2, B2, W3, B3, W4, B4]

        # define the mapping_options
        options = Options("naive")
        options.useSharedMemory(False)
        options.usePrivateMemory(False)
        options.unrollCopyShared(True)
        options.outerScheduleFusionStrategy("Max")
        options.fixParametersBeforeScheduling(False)
        options.tile([4])
        options.tileImperfectlyNested(False)
        options.mapToBlocks([128])
        options.mapToThreads([64])
        options.unroll(128)

        # run with TC, get the outputs and check against reference implementation
        outputs = self.check(lang, "mlp3", options, inputs)


if __name__ == '__main__':
    run_tests()
