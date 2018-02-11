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

import os, sys, pdb, uuid, logging

import torch
from torch.autograd import Variable

from tensor_comprehensions.tc import ATenCompilationUnit
from tensor_comprehensions.tc import global_debug_init as GlobalDebugInit
from tensor_comprehensions.torch_tc.tc_function import TCFunction, unpack_variables, get_tensors, make_contiguous
from tensor_comprehensions.autotuner import ATenAutotuner
from tensor_comprehensions.mapping_options import Options

FORMAT = '[%(levelname)s]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


# these are quick options for finishing autotuning
autotuner_default_options = {
    "threads": 32, "generations": 1, "pop_size": 10, "number_elites": 1
}

# TC prunes autotuning for kernels which require < 256 threads. So to tune small
# size kernels, we set the min kernel threads to 1
small_size_autotuner_options = {
    "threads": 32, "generations": 5, "pop_size": 10, "number_elites": 1,
    "tuner_min_launch_total_threads": 1
}

###############################################################################
# Some helper functions
###############################################################################
def get_options_from_cache_file(name, *inputs, **kwargs):
    options = None
    if "cache" in kwargs and kwargs["cache"] and isinstance(kwargs["cache"], str):
        cache_file = kwargs["cache"]
        assert "type" in kwargs, "layer type not specified: forward/backward"
        if "training" in kwargs and kwargs["training"]:
            if (kwargs["type"] == "backward"):
                cache_file = cache_file + "_backward"
        if "tuner" in kwargs:
            tuner = kwargs["tuner"]
            options = tuner.load(cache_file, name, list(inputs), 1)[0]
        else:
            tuner = TcAutotuner(kwargs["tc_lang"])
            options = tuner.load(cache_file, name, list(inputs))
    return options


# get the options from kwargs or construct the naive options
# argument type="forward" or "backward"
def get_options_from_kwargs(name, *inputs, **kwargs):
    # now the options can be a tuple (if training) or it will be just options
    # (only forward)
    options = None
    if "options" in kwargs and kwargs["options"] is not None:
        options = kwargs["options"]
        assert "type" in kwargs, "layer type not specified: forward/backward"
        if "training" in kwargs and kwargs["training"]:
            assert len(options) == 2, \
                "For a training layer, pass the options for forward/backward both."
            options = options[0] if (kwargs["type"] == "forward") else options[1]
    else:
        options = get_options_from_cache_file(name, *inputs, **kwargs)

    if options is None:
        options = Options("naive")
        logger.warning("No mapping options supplied. 'Naive' options will be used which might fail compilation")
    if not isinstance(options, Options):
        options = Options(options)
    return options


def get_tc_hash_key(name, *inputs):
    sizes_key = "_".join(["_".join(map(str, list(inp.size()))) for inp in inputs])
    hash_key = "{}_{}".format(name, sizes_key)
    return hash_key


def get_tc_names_from_kwargs(**kwargs):
    backward, backward_name = False, None
    if "training" in kwargs and kwargs["training"]:
        backward = True
        assert "backward" in kwargs and kwargs["backward"] is not None, \
            "You forgot to specify the name of backward TC. Training requires backward layer TC as well."
        backward_name = kwargs["backward"]
    assert "name" in kwargs and kwargs["name"] is not None, \
        "You forgot to specify which TC to run, please pass the name in define()"
    name = kwargs["name"]
    return name, backward_name


def validate_input(*inputs):
    # at the moment, TC can only take tensors as the input, we validate that
    # the inputs are all tensors
    for inp in inputs:
        assert torch.is_tensor(inp) or isinstance(inp, Variable), \
            "Incorrect input type: One of the inputs is not a tensor / Variable"


def validate_autotuner_input(*inputs):
    # for autotuning, we accept tensors, Variable, tuple as inputs
    for inp in inputs:
        assert torch.is_tensor(inp) or isinstance(inp, Variable) or isinstance(inp, tuple), \
            "Incorrect input type: One of the inputs is not a tensor/Variable/tuple"

###############################################################################
# TC autotuner class - ATen
###############################################################################
class TcAutotuner(object):
    def __init__(
        self,
        tc_lang,
        pop_size=100,
        crossover_rate=80,
        mutation_rate=7,
        generations=25,
        number_elites=10,
        threads=1,
        gpus="0",
        proto="/tmp/tuner.txt",
        restore_from_proto=False,
        restore_number=10,
        log_generations=False,
        tuner_min_launch_total_threads=256,
        **kwargs
    ):
        self.kwargs = kwargs
        self.tc_lang = tc_lang
        self.autotuner = ATenAutotuner(tc_lang)
        self.set_autotuner_options(
            pop_size, crossover_rate, mutation_rate, generations, number_elites,
            threads, gpus, proto, restore_from_proto, restore_number,
            log_generations, tuner_min_launch_total_threads
        )

    def set_autotuner_options(
        self, pop_size, crossover_rate, mutation_rate, generations, number_elites,
        threads, gpus, proto, restore_from_proto, restore_number, log_generations,
        tuner_min_launch_total_threads,
    ):
        self.autotuner.pop_size(pop_size)
        self.autotuner.crossover_rate(crossover_rate)
        self.autotuner.mutation_rate(mutation_rate)
        self.autotuner.generations(generations)
        self.autotuner.number_elites(number_elites)
        self.autotuner.threads(threads)
        self.autotuner.gpus(gpus)
        self.autotuner.proto(proto)
        self.autotuner.restore_from_proto(restore_from_proto)
        self.autotuner.restore_number(restore_number)
        self.autotuner.log_generations(log_generations)
        self.autotuner.tuner_min_launch_total_threads(tuner_min_launch_total_threads)

    # We need to pass the inputs so that we can load the correct options from
    # the cache that correspond to the inputs sizes. This is useful when the
    # cache may contain multiple kernels and multiple sizes for each kernel
    def load(self, filename, tc_name, inputs, num_candidates=1):
        best_options = self.autotuner.load(filename, tc_name, inputs, num_candidates)
        if num_candidates == 1:
            return best_options[0]
        return best_options

    # if the cache_file is not "" then the tuning results would be saved to file
    def tune_and_store(self, tc_name, inputs, mapping_options, cache_file=""):
        options = mapping_options
        if not isinstance(options, Options):
            options = Options(options)
        best_options = self.autotuner.tune(
            cache_file, tc_name, inputs, options, [options]
        )
        return best_options

    def autotune(self, *inputs, **kwargs):
        input_tensors = get_tensors(list(inputs))

        kwargs.update(self.kwargs)
        name, backward_name = get_tc_names_from_kwargs(**kwargs)
        kwargs.pop("name", None)
        backward = True if backward_name is not None else False

        cache_file = ""
        if "cache" in kwargs:
            if isinstance(kwargs["cache"], bool) and kwargs["cache"]:
                hash_key = get_tc_hash_key(name, *input_tensors)
                cache_file = "/tmp/{}_{}".format(hash_key, str(uuid.uuid4()))
            elif isinstance(kwargs["cache"], str) and kwargs["cache"] != "":
                cache_file = kwargs["cache"]
            logger.info('Autotuning cache will be saved to: {}.cuda/options'.format(cache_file))
        else:
            logger.warning("Autotuning results won't be cached. 'cache' option is not specified")

        # we will first run the autotuning on the forward layer, the inputs are given
        # for that, we will tune those
        if "options" in kwargs and kwargs["options"] is not None:
            options = kwargs["options"]
        else:
            options = Options("naive")
            logger.warning("Using naive options for autotuning")
        forward_best_options = self.tune_and_store(
            name, input_tensors, mapping_options=options, cache_file=cache_file
        )
        if not backward:
            return forward_best_options

        # now, we have to tune the backward layer, for that, we need to run
        # the forward layer first, get it's output,
        logger.info('Autotuning the backward layer now')
        cu = TcCompilationUnit()
        cu.define(self.tc_lang)
        kwargs["tuner"] = self.autotuner
        outputs = cu.compile_and_run(name, input_tensors, **kwargs)
        # now that we have the outputs of the forward pass, we have the inputs
        # for the backward layer and we can now tune the backward layer
        reorder_function = kwargs["reorder_function"] if "reorder_function" in kwargs else None
        rearranged_outputs = list(outputs)
        if reorder_function is not None:
            rearranged_outputs = reorder_function(list(outputs))
        inputs = make_contiguous(unpack_variables(input_tensors + list(rearranged_outputs)))
        if cache_file:
            cache_file = cache_file + "_backward"
            logger.info('Backwards autotuning cache will be saved to: {}.cuda/options'.format(cache_file))
        backward_best_options = self.tune_and_store(
            backward_name, inputs, mapping_options=options, cache_file=cache_file
        )
        return [forward_best_options, backward_best_options]


###############################################################################
# TC engine - ATen based
###############################################################################
class TcCompilationUnit(object):

    def __init__(self):
        self.cu = ATenCompilationUnit()
        self.tc_lang = None
        self.compilation_cache = {}

    def define(self, tc_lang):
        self.tc_lang = tc_lang
        self.cu.define(tc_lang)

    # we could have multiple TC strings and want to run one of them
    def compile(self, name, inputs, **kwargs):
        # append the language so that we can use it for creating Autotuner object
        # to load options cache
        kwargs["tc_lang"] = self.tc_lang
        if "type" not in kwargs:
            kwargs["type"] = 'forward'
        options = get_options_from_kwargs(name, *inputs, **kwargs)
        handle = self.cu.compile(name, inputs, options)
        return handle

    def run(self, handle, name, inputs, **kwargs):
        outputs = []
        if "outputs" in kwargs and kwargs["outputs"] is not None:
            outputs = kwargs["outputs"]
        self.cu.run(name, inputs, outputs, handle)
        return outputs

    def compile_and_run(self, name, inputs, **kwargs):
        handle = self.compile(name, inputs, **kwargs)
        return self.run(handle, name, inputs, **kwargs)

    def manual_cuda_injection(
        self, name, injected_kernel_name, cuda_code, inputs, grid, block
    ):
        self.cu.inject_cuda(
            name, injected_kernel_name, cuda_code, inputs, grid, block
        )


###############################################################################
# User Facing Proxy object
###############################################################################
class TcUnit(object):

    def __init__(self, lang, **kwargs_define):
        self.cu = TcCompilationUnit()
        self.cu.define(lang)
        self.kwargs_define = kwargs_define
        self.lang = lang
        self.tuner = None
        # TODO: we should build a tuner cache here which looks like:
        # hash_key -> options

    def __call__(self, *inputs, **kwargs):
        validate_input(*inputs)
        kwargs.update(self.kwargs_define)
        name, backward_name = get_tc_names_from_kwargs(**kwargs)
        kwargs.pop("name", None)
        backward = True if backward_name is not None else False

        hash_key = get_tc_hash_key(name, *inputs)
        if hash_key in self.cu.compilation_cache:
            tc_info = self.cu.compilation_cache[hash_key]
        else:
            tc_info = {}
            kwargs["type"] = "forward"
            input_tensors = unpack_variables(list(inputs))

            if "inject_kernel" in kwargs and "cuda_code" in kwargs:
                assert "grid" in kwargs and "block" in kwargs, \
                    "For manual cuda injection, please specify the grid and block settings"
                self.cu.manual_cuda_injection(
                    name, kwargs["inject_kernel"], kwargs["cuda_code"],
                    input_tensors, kwargs["grid"], kwargs["block"]
                )
            handle_forward = self.cu.compile(name, input_tensors, **kwargs)
            outputs = self.cu.run(handle_forward, name, input_tensors, **kwargs)
            tc_info["forward_name"], tc_info["handle_forward"] = name, handle_forward

            if backward:
                tc_info["backward_name"] = backward_name
            self.cu.compilation_cache[hash_key] = tc_info

        if "outputs" in kwargs and kwargs["outputs"] is not None:
            tc_info["outputs"] = kwargs["outputs"]
        out = TCFunction.apply(self.cu, tc_info, kwargs, *inputs)
         # conversion to list is needed since tuple is returned from Function
        return list(out)

    def autotune(self, *inputs, **kwargs):
        validate_autotuner_input(*inputs)
        kwargs.update(self.kwargs_define)
        if self.tuner is None:
            self.tuner = TcAutotuner(self.lang, **kwargs)
        return self.tuner.autotune(*inputs, **kwargs)

###############################################################################
# User Facing TC
###############################################################################
def define(lang, **kwargs_define):
    if "constants" in kwargs_define and kwargs_define["constants"]:
        # there are some scalars in the lang, replace them with constants
        lang = lang.format(**kwargs_define["constants"])
    tc_unit = TcUnit(lang, **kwargs_define)
    return tc_unit
