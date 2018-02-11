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
from tensor_comprehensions import TcCompilationUnit
from common import TestCase, run_tests


class TestKRU(TestCase):
    def test_KRU3(self):
        # define TC
        lang = """
        def KRU3_1(float(D2, N2) W2, float(M, N0, N1, N2) X) -> (XW2) {
           XW2(m, n0, n1, d2)   +=! X(m, n0, n1, n2_red) * W2(d2, n2_red)
        }
        def KRU3_2(float(D1, N1) W1, float(M, N0, N1, D2) XW2) -> (XW2W1) {
           XW2W1(m, n0, d1, d2) +=! XW2(m, n0, n1_red, d2) * W1(d1, n1_red)
        }
        def KRU3_3(float(D0, N0) W0, float(M, N0, D1, D2) XW2W1) -> (Y) {
           Y(m, d0, d1, d2)     +=! XW2W1(m, n0_red, d1, d2) * W0(d0, n0_red)
        }
        """

        # create input tensors
        M, D0, D1, D2, N0, N1, N2, max_factors = 256, 32, 32, 32, 16, 16, 16, 3
        W0 = torch.randn(D0, N0).cuda()
        W1 = torch.randn(D1, N1).cuda()
        W2 = torch.randn(D2, N2).cuda()
        X = torch.randn(M, N0, N1, N2).cuda()

        # define the mapping_options
        options = Options("naive")
        options.useSharedMemory(True)
        options.usePrivateMemory(True)
        options.tile([4, 1, 1, 8, 16])
        options.mapToBlocks([64, 16, 16])
        options.mapToThreads([8, 4, 8])
        options.unroll(128)

        # create TC compilation unit object and define the TC language
        cu = TcCompilationUnit()
        cu.define(lang)

        print("Running KRU3_1")
        inputs1 = [W2, X]
        outputs1 = cu.compile_and_run("KRU3_1", inputs1, options=options)

        print("Running KRU3_2")
        XW2 = outputs1[0]
        inputs2 = [W1, XW2]
        outputs2 = cu.compile_and_run("KRU3_2", inputs2, options=options)

        print("Running KRU3_3")
        XW2W1 = outputs2[0]
        inputs3 = [W0, XW2W1]
        outputs3 = cu.compile_and_run("KRU3_3", inputs3, options=options)


if __name__ == '__main__':
    run_tests()
