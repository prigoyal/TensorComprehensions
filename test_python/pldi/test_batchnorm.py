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
from test_python.common import TestCase, run_tests


class TestBatchnorm(TestCase):
    def test_batchnorm(self):
        # define TC
        lang = """
        def spatial_batch_norm(float(1) momentum, float(1) eps, float(N,C,H,W) I, float(C) rMeanIn, float(C) rVarIn)
        -> (O, rMeanOut, rVarOut, mean, centered, variance, expectedVariance, normalizedOut)
        {
             mean(c) +=! I(nn, c, hh, ww)
             mean(c)  = mean(c) / (N * H * W)
             rMeanOut(c) = (1 - momentum(0)) * rMeanIn(c) + momentum(0) * mean(c)
             centered(n, c, h, w) = I(n, c, h, w) - rMeanOut(c)
             variance(n, c, h, w) = centered(n, c, h, w) * centered(n, c, h, w)
             expectedVariance(c) +=! (variance(n, c, h, w) + eps(0)) / (N * H * W)
             rVarOut(c) = rsqrt(
               (1 - momentum(0)) * rVarIn(c) + momentum(0) * expectedVariance(c))
             O(n, c, h, w) = centered(n, c, h, w) * rVarOut(c)
             normalizedOut(n, c, h, w) = O(n, c, h, w)
         }
        """

        # create input tensors
        N, C, H, W = 32, 4, 56, 56
        I = torch.randn(N, C, H, W).cuda()
        running_mean = torch.randn(C).cuda()
        running_var = torch.randn(C).cuda()
        momentum = torch.randn(1).fill_(1.0).cuda()
        epsilon = torch.randn(1).fill_(1.0).cuda()
        inputs = [momentum, epsilon, I, running_mean, running_var]

        # define the mapping_options
        options = Options("naive")
        options.scheduleFusionStrategy("Max")

        # run with TC, get the outputs and check against reference implementation
        outputs = self.check(lang, "spatial_batch_norm", options, inputs)


if __name__ == '__main__':
    run_tests()
