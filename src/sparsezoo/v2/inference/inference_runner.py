# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Helper class for running inference using
the selected engine and input/output files
"""
import os
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Generator, List

import numpy
import onnx

import onnxruntime as ort
from sparsezoo.utils.numpy import save_numpy
from sparsezoo.v2.inference.engines import ENGINES
from sparsezoo.v2.objects.file import File
from sparsezoo.v2.objects.model_objects import NumpyDirectory


__all__ = ["InferenceRunner"]


class InferenceRunner:
    """
    Helper class for running inference
    given `sample_inputs`, `sample_outputs` and the onnx model.

    This is intended to be used by the Model class object.

    :params sample_inputs: File object containing sample inputs to the inference engine
    :params sample_outputs: File object containing sample outputs the inference engine
    :params onnx_model: File object holding the onnx model
    :params supported_engines: List of the names of supported engines
        (e.g. onnxruntime, deepsparse etc.)
    """

    def __init__(
        self,
        sample_inputs: NumpyDirectory,
        sample_outputs: NumpyDirectory,
        onnx_file: File,
        supported_engines: List[str] = ENGINES,
    ):
        self.sample_inputs = sample_inputs
        self.sample_outputs = sample_outputs
        self.onnx_file = onnx_file
        self.supported_engines = supported_engines

        self.engine_type_to_iterator = {
            "onnxruntime": self._run_with_onnx_runtime,
            "deepsparse": self._run_with_deepsparse,
        }

    def generate_outputs(
        self, engine_type: str, save_to_tar: bool
    ) -> Generator[List[numpy.ndarray], None, None]:
        """
        Chooses the appropriate engine type to load the onnx model
        Then, feeds the data (sample inputs)
        to generate model outputs (in the iterative fashion).

        This is a general method to obtain `sample_outputs` from
        `sample_inputs`, given the inference engine.

        :params engine_type: name of the inference engine
        :params save_to_tar: boolean flag; if True, the output generated
            by the engine from `sample_inputs` directory will be saved to
            the archive file `sample_outputs_{engine_type}.tar.gz
            (located in the same folder as `sample_inputs`).
            If False, the function will yield model outputs in the
            iterative fashion (one per input)
        :returns Sequentially return a tuple of list
            containing numpy arrays, representing the output
            from the inference engine
        """
        if engine_type not in self.supported_engines:
            raise KeyError(
                f"The argument `engine_type` must be one of {self.supported_engines}"
            )

        iterator = self.engine_type_to_iterator.get(engine_type)
        if iterator is None:
            raise KeyError(
                f"Cannot generate outputs using engine type: {engine_type}. "
                f"Supported engines: {self.engine_type_to_iterator.keys()}."
            )

        if save_to_tar:
            self._save_outputs_to_tar(iterator, engine_type)

        for output in iterator():
            yield output

    def validate_with_onnx_runtime(self) -> bool:
        """
        Validates that output from the onnxruntime
        engine matches the expected output.

        :return boolean flag; if True, outputs match expected outputs. False otherwise
        """
        validation = []

        sample_outputs = self._check_and_fetch_outputs(engine_name="onnxruntime")
        for target_output, output in zip(sample_outputs, self._run_with_onnx_runtime()):
            target_output = list(target_output.values())
            for o1, o2 in zip(target_output, output):
                if o2.ndim != o1.ndim:
                    o2 = o2.squeeze(0)
                validation.append(numpy.allclose(o1, o2, atol=1e-5))
        return all(validation)

    def _save_outputs_to_tar(self, iterator: Callable, engine_type: str):
        output_files = []

        path = os.path.join(
            os.path.dirname(self.sample_inputs.path),
            f"sample_outputs_{engine_type}",
        )
        if not os.path.exists(path):
            os.mkdir(path)

        for input_file, output in zip(self.sample_inputs.files, iterator()):
            # if input's name is `inp-XXXX.npz`
            # output's name should be `out-XXXX.npz`
            name = input_file.name.replace("inp", "out")
            # we need to remove `.npz`, this is
            # required by save_numpy() function
            save_numpy(array=output, export_dir=path, name=Path(name).stem)
            output_files.append(File(name=name, path=os.path.join(path, name)))

        output_directory = NumpyDirectory(
            name=os.path.basename(path), path=path, files=output_files
        )
        output_directory.gzip()

    def _run_with_deepsparse(self):
        try:
            import deepsparse  # noqa F401
        except ModuleNotFoundError as e:  # noqa F841
            raise e

        from deepsparse import compile_model

        engine = compile_model(self.onnx_file.path, batch_size=1)

        for index, input_data in enumerate(self.sample_inputs):
            model_input = [numpy.expand_dims(x, 0) for x in input_data.values()]
            output = engine.run(model_input)
            yield output

    def _run_with_onnx_runtime(self):

        ort_sess = ort.InferenceSession(self.onnx_file.path)
        model = onnx.load(self.onnx_file.path)
        input_names = [inp.name for inp in model.graph.input]

        for index, input_data in enumerate(self.sample_inputs):
            model_input = OrderedDict(
                [
                    (k, numpy.expand_dims(v, 0))
                    for k, v in zip(input_names, input_data.values())
                ]
            )
            output = ort_sess.run(None, model_input)
            yield output

    def _check_and_fetch_outputs(self, engine_name: str) -> NumpyDirectory:
        sample_outputs = self.sample_outputs.get(engine_name)
        if sample_outputs is None:
            raise KeyError(
                f"Attempting to run validation with {engine_name} engine "
                f"but the ground truth folder `sample_outputs_{engine_name}` "
                f"has not been provided."
            )
        return sample_outputs
