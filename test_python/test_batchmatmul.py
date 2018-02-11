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


class TestBatchMatmul(TestCase):
    def test_batchmatmul(self):
        # define TC
        lang = """
        def batch_matmul(float(B, N, M) X, float(B, M, K) Y) -> (Z) {
          Z(b, n, k) +=! X(b, n, mm) * Y(b, mm, k)
        }
        """

        # create input tensors
        B, K, M, N = 500, 26, 72, 26
        X = torch.randn(B, N, M).cuda()
        Y = torch.randn(B, M, K).cuda()
        inputs = [X, Y]

        # define the mapping_options
        options = Options("naive")
        options.useSharedMemory(True)
        options.usePrivateMemory(True)
        options.unrollCopyShared(True)
        options.outerScheduleFusionStrategy("Preserve3Coincident")
        options.fixParametersBeforeScheduling(True)
        options.tile([1])
        options.tileImperfectlyNested(False)
        options.mapToBlocks([72, 16, 1])
        options.mapToThreads([7, 26])
        options.unroll(128)

        # run with TC, get the outputs and check against reference implementation
        outputs = self.check(lang, "batch_matmul", options, inputs)


if __name__ == '__main__':
    run_tests()
