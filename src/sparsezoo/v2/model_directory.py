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

import glob
import logging
import os
import re
import typing
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import onnx

import onnxruntime as ort
from sparsezoo.v2.directory import Directory
from sparsezoo.v2.file import File
from sparsezoo.v2.model_objects import FrameworkFiles, NumpyDirectory, SampleOriginals


FRAMEWORKS = ["pytorch", "keras", "tensorflow"]
ENGINES = ["onnxruntime", "deepsparse"]


def file_dictionary(**kwargs):
    return kwargs


class ModelDirectory(Directory):
    """
    Object to represent SparseZoo Model Directory.

    The suggested way to create a class object is to use
    any of the two factory methods:

    - `from_zoo_api()` -> by using the output from the
        `src.sparsezoo.requests.download.download_model_get_request()` function.

    - `from_directory()` -> by using the directory to the model on your machine.

    :param files: list of files, where every file
        is represented by a dictionary (note: not Fil object)
    :directory_path: if ModelDirectory created using method `from_directory()`,
        it points to the directory of the ModelDirectory. By default: None
    """

    def __init__(
        self,
        files: List[Dict[str, Any]],
        name: str,
        path: Optional[str] = None,
        url: Optional[str] = None,
    ):

        self.framework_files: FrameworkFiles = self._directory_from_files(
            files,
            directory_class=FrameworkFiles,
            display_name="framework-files",
        )
        self.sample_originals: SampleOriginals = self._directory_from_files(
            files,
            directory_class=SampleOriginals,
            display_name="sample-originals",
        )
        self.sample_inputs: NumpyDirectory = self._directory_from_files(
            files,
            directory_class=NumpyDirectory,
            display_name="sample-inputs",
        )
        # TODO: Ignoring the dictionary type for now.
        self.sample_outputs: Dict[str, NumpyDirectory] = self._directory_from_files(
            files,
            directory_class=NumpyDirectory,
            display_name="sample-outputs",
        )  # key by engine name.
        self.sample_labels: Directory = self._directory_from_files(
            files, directory_class=Directory, display_name="sample-labels"
        )
        self.onnx_model: File = self._file_from_files(
            files, display_name="model.onnx"
        )  # model.onnx

        self.onnx_models: List[File] = self._file_from_files(
            files, display_name="model.(.*).onnx", regex=True
        )  # model{.opset}.onnx

        self.analysis: File = self._file_from_files(
            files, display_name="analysis.yaml"
        )  # analysis.yaml
        self.benchmarks: File = self._file_from_files(
            files, display_name="benchmarks.yaml"
        )  # benchmarks.yaml
        self.eval_results: File = self._file_from_files(
            files, display_name="eval.yaml"
        )  # eval.yaml
        self.model_card: File = self._file_from_files(
            files, display_name="model.md"
        )  # model.md
        self.recipes: List[File] = self._file_from_files(
            files, display_name="recipe(.*).md", regex=True
        )  # recipe{_tag}.md

        files = [
            self.framework_files,
            self.sample_originals,
            self.sample_inputs,
            self.sample_outputs,
            self.sample_labels,
            self.onnx_model,
            self.onnx_models,
            self.analysis,
            self.benchmarks,
            self.eval_results,
            self.model_card,
            self.recipes,
        ]

        super().__init__(files=files, name=name, path=path, url=url)

    @classmethod
    def from_zoo_api(cls, request_json: List[Dict]) -> "ModelDirectory":
        """
        Factory method for creating ModelDirectory class object
        from the output of the NeuralMagic API.
        :param request_json: output of the NeuralMagic API; list of file dictionaries
        :return: ModelDirectory class object
        """
        files = request_json
        return ModelDirectory(files=files, name="model_directory")

    @classmethod
    def from_directory(cls, directory_path: str) -> "ModelDirectory":
        """
        Factory method for creating ModelDirectory class object
        from the local directory.
        :param directory_path: path to the local directory
        :return: ModelDirectory class object
        """
        files = []
        paths = glob.glob(os.path.join(directory_path, "*"))
        for path in paths:
            display_name = os.path.basename(path)
            files.append(file_dictionary(display_name=display_name, path=path))

        return ModelDirectory(files=files, name="model_directory", path=directory_path)

    def download(self, directory_path: str, override: bool = False) -> bool:
        """
        Attempt to download the files given the `url` attribute
        of the files inside the ModelDirectory.
        
        :param directory_path: directory to download files to
        :param override: if True, the method can override old `directory_path`
        :return: boolean flag; was download successful or not.
        """
        if self.path is not None and not override:
            raise ValueError(
                "ModelDirectory class object was either created "
                "using 'from_directory` factory method or "
                "`download()` method already invoked."
                "Set `override` = True to override."
            )
        else:
            downloads = []
            for file in self.files:
                downloads.append(self._download(file, directory_path))

        return all(downloads)

    def validate(self) -> bool:
        """
        Validate the ModelDirectory class object:
        1. Validate that the sample inputs and outputs work with ONNX Runtime
        2. Validate all the folders

        return: a boolean flag; if True, the validation has been successful
        """

        if not self._validate_with_onnx_runtime():
            logging.warning(
                "Failed to validate the compatibility of "
                "`sample_inputs` files with the `model.onnx` model."
            )
            return False

        # TODO: This is a hack for now,
        #  some files cannot be validated
        #  using dummy inputs (see respective tests)
        SKIP_ATTRIBUTES = ["framework_files"]

        if self.path is None:
            raise ValueError(
                "Cannot validate the ModelDirectory. "
                "If created using method `from_directory`, "
                "please make sure that the `directory_path` is correct. "
                "If created using method `from_zoo_api`, "
                "call `download()` method prior to `validate()`"
            )

        validations = {}
        for attribute, file in self.__iter__():
            # TODO: Continuing with the hack
            if attribute in SKIP_ATTRIBUTES:
                validations[attribute] = True
            elif isinstance(file, File):
                validations[attribute] = file.validate()
            elif isinstance(file, list):
                validations[attribute] = all([_file.validate() for _file in file])
            elif isinstance(file, dict):
                raise NotImplementedError()

        return all(validations.values())

    def analyze(self):
        # TODO: This will be the onboarding task for Kyle
        pass

    def generate_outputs(
        self, engine_type: str
    ) -> Union[List[np.ndarray], typing.OrderedDict[str, np.ndarray]]:
        """
        Chooses the appropriate engine type to load the onnx model.
        Then, feeds the data (sample original inputs)
        to generate model outputs (in the iterative fashion).
        :params engine_type: name of the inference engine
        :returns engine output.
        """
        if engine_type not in ENGINES:
            raise ValueError(f"The argument `engine_type` must be one of {ENGINES}")
        elif engine_type == "onnxruntime":
            for output in self._run_with_onnx_runtime():
                yield output
        else:
            for output in self._run_with_deepsparse():
                yield output

    def _get_directory(
        self,
        file: Dict[str, Any],
        directory_class: Directory,
        display_name: Optional[str] = None,
        regex: Optional[bool] = False,
    ) -> Union[Directory, None]:
        # Takes a file dictionary and returns a Directory() object, if successful.
        # Optionally, can do:
        #   string matching (if `display_name` not None)
        #   regex (if `display_name` not None and `regex`)
        #   file validation (if file's path is not None)

        match = True
        if display_name:
            if regex:
                pattern = re.compile(display_name)
                match = re.search(pattern, file["display_name"])

            else:
                match = display_name == file["display_name"]

        if not match:
            logging.warning(
                "Could not find a directory with "
                f"display_name / regex_pattern: {display_name}"
            )
            return None

        name, path, url = file.get("display_name"), file.get("path"), file.get("url")

        # directory is a tar file
        if self._is_file_tar(file):
            directory = directory_class(files=[], name=name, path=path, url=url)
            return directory

        # directory is folder
        else:
            # is directory using the 'contents' key.
            # this is a placeholder for the nested files that
            # the directory contains
            if file.get("contents"):
                files_within = file["contents"]

            # is directory locally on the machine
            else:
                paths_within = glob.glob(os.path.join(path, "*"))
                files_within = [
                    file_dictionary(display_name=os.path.basename(path), path=path)
                    for path in paths_within
                ]

            files = [self._get_file(file=file) for file in files_within]
            directory = directory_class(files=files, name=name, path=path, url=url)
            return directory

    @staticmethod
    def _get_file(
        file: Dict[str, Any],
        display_name: Optional[str] = None,
        regex: Optional[bool] = False,
    ) -> Union[File, None]:
        # Takes a file dictionary and returns a File() object, if successful.
        # Optionally, can do:
        #   string matching (if `display_name` not None)
        #   regex (if `display_name` not None and `regex`)
        #   file validation (if file's path is not None)

        match = True
        if display_name:
            if regex:
                pattern = re.compile(display_name)
                match = re.search(pattern, file["display_name"])
            else:

                match = display_name == file["display_name"]

        if not match:
            logging.warning(
                "Could not find a file with "
                f"display_name / regex_pattern: {display_name}"
            )
            return None
        else:
            file = File.from_dict(file)

            return file

    def _file_from_files(
        self,
        files: List[Dict[str, Any]],
        display_name: Optional[str] = None,
        regex: Optional[bool] = False,
    ) -> Union[File, List[File]]:
        # Parses a list of file dictionaries and returns
        # a File() object or a list of File() objects, if successful,
        # otherwise None.
        files_found = []
        for file in files:
            file = self._get_file(file=file, display_name=display_name, regex=regex)
            if file is not None:
                files_found.append(file)

        if not files_found:
            return None
        if len(files_found) == 1:
            return files_found[0]
        else:
            raise ValueError(
                f"Found more than one File for `display_name` {display_name}."
            )

    def _directory_from_files(
        self,
        files: List[Dict[str, Any]],
        directory_class: Union[
            Directory, NumpyDirectory, FrameworkFiles, SampleOriginals
        ] = Directory,
        display_name: Optional[str] = None,
        regex: Optional[bool] = True,
    ) -> Union[Directory, None]:
        # Takes a list of file dictionaries and returns
        # a Directory() object, if successful,
        # otherwise None.
        directories_found = []
        for file in files:
            directory = self._get_directory(
                file=file,
                directory_class=directory_class,
                display_name=display_name,
                regex=regex,
            )
            if directory is not None:
                directories_found.append(directory)

        if not directories_found:
            return None
        # For now, following the logic of this class,
        # it is prohibitive for find more than
        # one directory
        elif len(directories_found) != 1:
            raise ValueError(
                f"Found more than one Directory for `display_name` {display_name}."
            )
        else:
            return directories_found[0]

    def _validate_with_onnx_runtime(self):
        sample_outputs = self.sample_outputs

        if sample_outputs.is_archive:
            sample_outputs.unzip()

        validation = []  # validation boolean per output
        for target_output, output in zip(sample_outputs, self._run_with_onnx_runtime()):
            target_output = list(target_output.values())
            # TODO: This does not work for now, the
            #  outputs are quite different. To be investigated.
            validation.append(
                all([np.array_equal(x, y) for x, y in zip(target_output, output)])
            )
        return all(validation)

    def _run_with_deepsparse(self):
        try:
            import deepsparse  # noqa F401
        except ValueError:
            print(
                "Could not load deepsparse library. "
                "Make sure that deepsparse is installed "
                "(e.g. run `pip install deepsparse`)"
            )

        from deepsparse import compile_model

        sample_inputs = self.sample_inputs
        onnx_model = self.onnx_model

        if sample_inputs.is_archive:
            sample_inputs = sample_inputs.unzip(sample_inputs)
            sample_inputs = NumpyDirectory(
                files=sample_inputs.files,
                name=sample_inputs.name,
                path=sample_inputs.path,
            )

        engine = compile_model(onnx_model.path, batch_size=1)

        for index, input_data in enumerate(sample_inputs):
            model_input = [np.expand_dims(x, 0) for x in input_data.values()]
            output = engine.run(model_input)
            yield output

    def _run_with_onnx_runtime(self):
        sample_inputs = self.sample_inputs
        onnx_model = self.onnx_model

        if sample_inputs.is_archive:
            sample_inputs.unzip()

        ort_sess = ort.InferenceSession(onnx_model.path)
        model = onnx.load(onnx_model.path)
        input_names = [inp.name for inp in model.graph.input]

        for index, input_data in enumerate(sample_inputs):
            model_input = OrderedDict(
                [
                    (k, np.expand_dims(v, 0))
                    for k, v in zip(input_names, input_data.values())
                ]
            )
            output = ort_sess.run(None, model_input)
            yield output

    def _download(
        self, file: Union[File, List[File], Dict[Any, File]], directory_path: str
    ) -> bool:

        # TODO: This is a hack for now,
        #  some files cannot be downloaded if
        #  ModelDirectory created .from_zoo_api()

        SKIP_ATTRIBUTES = [
            "framework-files",
            "analysis.yaml",
            "benchmarks.yaml",
            "eval.yaml",
            "recipe_foo.md",
            "recipe_bar.md",
            "sample-labels.tar.gz",
        ]
        if isinstance(file, File):
            # TODO: Continuing with the hack
            if file.name in SKIP_ATTRIBUTES:
                return True

            if file.url:
                file.download(destination_path=directory_path)
                return True
            else:
                logging.warning(
                    f"Failed to download file {file.name}. The url of the file is None."
                )
                return False

        elif isinstance(file, list):
            validations = [self._download(_file, directory_path) for _file in file]
            return all(validations)

        else:
            raise NotImplementedError()

    @staticmethod
    def _is_file_tar(file):
        extension = file["display_name"].split(".")[-2:]
        return extension == ["tar", "gz"]
