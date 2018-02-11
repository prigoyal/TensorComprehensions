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


class TestGroupConvolution(TestCase):
    def test_group_convolution(self):
        # define TC
        lang = """
        def group_convolution(float(N,G,C,H,W) I, float(G,F,C,KH,KW) W1, float(G,F) B)
        -> (O)
        {
          O(n, g, f, h, w) +=! I(n, g, c, h + kh, w + kw) * W1(g, f, c, kh, kw)
          O(n, g, f, h, w) = O(n, g, f, h, w) + B(g, f)
        }
        """

        # create input tensors
        N, G, C, F, H, W, KH, KW = 32, 32, 32, 32, 7, 7, 3, 3
        tI = torch.randn(N, G, C, H, W).cuda()
        tW = torch.randn(G, F, C, KH, KW).cuda()
        tB = torch.randn(G, F).cuda()
        inputs = [tI, tW, tB]

        # define the mapping_options
        options = Options("naive")
        options.useSharedMemory(True)
        options.usePrivateMemory(False)
        options.unrollCopyShared(True)
        options.outerScheduleFusionStrategy("Preserve3Coincident")
        options.fixParametersBeforeScheduling(False)
        options.tile([1, 1])
        options.tileImperfectlyNested(False)
        options.mapToBlocks([32, 32, 3])
        options.mapToThreads([8, 7, 7])
        options.unroll(256)

        # run with TC, get the outputs and check against reference implementation
        outputs = self.check(lang, "group_convolution", options, inputs)


if __name__ == '__main__':
    run_tests()
