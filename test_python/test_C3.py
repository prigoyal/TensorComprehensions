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


class TestC3(TestCase):
    def test_C3(self):
        # define TC
        lang = """
        def _C3(float(B, WX) I, float(WY, WX) W) -> (C3) {
            C3(b, wy) +=! I(b, wxx) * W(wy, wxx)
        }
        """

        # create input tensors
        B, WX, WY = 128, 1000, 1024
        I = torch.randn(B, WX).cuda()
        W = torch.randn(WY, WX).cuda()
        inputs = [I, W]

        # define the mapping_options
        options = Options("naive")
        options.useSharedMemory(True)
        options.usePrivateMemory(True)
        options.unrollCopyShared(True)
        options.outerScheduleFusionStrategy("Preserve3Coincident")
        options.fixParametersBeforeScheduling(True)
        options.tile([8, 32, 32])
        options.tileImperfectlyNested(False)
        options.mapToBlocks([128, 128])
        options.mapToThreads([1, 32])
        options.unroll(256)

        # run with TC, get the outputs and check against reference implementation
        outputs = self.check(lang, "_C3", options, inputs)


if __name__ == '__main__':
    run_tests()
