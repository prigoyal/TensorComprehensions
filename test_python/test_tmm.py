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

import os

import torch
import torch.cuda

from tensor_comprehensions.mapping_options import Options
from common import TestCase, run_tests


class TestTransposedMatmul(TestCase):
    def test_tmm(self):
        # define TC
        lang = """
        def tmm(float(M,K) A, float(N,K) B) -> (C) {
          C(m, n) +=! A(m, kk) * B(n, kk)
        }
        """

        # create input tensors
        M, N, K = 128, 256, 32
        A = torch.randn(M, K).cuda()
        B = torch.randn(N, K).cuda()
        inputs = [A, B]

        # define the mapping_options
        options = Options("naive")
        options.useSharedMemory(True)
        options.usePrivateMemory(True)
        options.unrollCopyShared(False)
        options.outerScheduleFusionStrategy("Preserve3Coincident")
        options.fixParametersBeforeScheduling(False)
        options.tile([4, 32])
        options.tileImperfectlyNested(False)
        options.mapToBlocks([64, 128])
        options.mapToThreads([1, 32])
        options.unroll(4)

        # run with TC, get the outputs and check against reference implementation
        outputs = self.check(lang, "tmm", options, inputs)
        expected = torch.mm(A, torch.transpose(B, 0, 1))
        diff = outputs[0] - expected
        self.assert_almost_equal(diff, inputs, M * N, 3e-7)


if __name__ == '__main__':
    run_tests()
