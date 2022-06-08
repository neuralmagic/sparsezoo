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
from typing import Any, Dict, List, Optional, Union
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import onnxruntime as ort
from sparsezoo.utils.numpy import save_numpy
from sparsezoo.v2.directory import Directory
from sparsezoo.v2.file import File
from sparsezoo.v2.inference_runner import InferenceRunner
from sparsezoo.v2.model_objects import FrameworkFiles, NumpyDirectory, SampleOriginals


__all__ = ["ModelDirectory"]

ENGINES = ["onnxruntime", "deepsparse", "torch", "keras", "tensorflow_v1"]


def file_dictionary(**kwargs):
    return kwargs


class ModelDirectory(Directory):
    """
    Object to represent SparseZoo Model Directory.

    The suggested way to create a class object is to use
    any of the two factory methods:

    - `from_zoo_api()` -> by using the output from the
        `src.sparsezoo.requests.download.download_model_get_request()` function.

    - `from_directory()` -> by using the path to the model directory on your machine.

    :param files: list of files, where every file
        is represented by a dictionary (note: not File object)
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

        self.training: FrameworkFiles = self._directory_from_files(
            files,
            directory_class=FrameworkFiles,
            display_name="training",
        )
        self.sample_originals: SampleOriginals = self._directory_from_files(
            files,
            directory_class=SampleOriginals,
            display_name="sample_originals",
        )
        self.sample_inputs: NumpyDirectory = self._directory_from_files(
            files,
            directory_class=NumpyDirectory,
            display_name="sample_inputs",
        )
        self.sample_outputs: Dict[
            str, NumpyDirectory
        ] = self._sample_outputs_list_to_dict(
            self._directory_from_files(
                files,
                directory_class=NumpyDirectory,
                display_name="sample_outputs",
                allow_multiple_outputs=True,
            )
        )  # key by engine name.

        self.sample_labels: Directory = self._directory_from_files(
            files, directory_class=Directory, display_name="sample_labels"
        )

        self.onnx_folder: Directory = self._directory_from_files(
            files, display_name="onnx", regex=False
        )  # onnx folder

        self.logs: Directory = self._directory_from_files(
            files, display_name="logs", regex=False
        )  # logs folder

        self.onnx_model: File = self._file_from_files(
            files, display_name="model.onnx"
        )  # model.onnx

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

        self.sample_inputs.files.sort(key=lambda x: x.name)
        [
            sample_outputs.files.sort(key=lambda x: x.name)
            for sample_outputs in self.sample_outputs.values()
        ]

        files = [
            self.training,
            self.sample_originals,
            self.sample_inputs,
            self.sample_outputs,
            self.sample_labels,
            self.onnx_folder,
            self.logs,
            self.onnx_model,
            self.analysis,
            self.benchmarks,
            self.eval_results,
            self.model_card,
            self.recipes,
        ]

        self.inference_runner = InferenceRunner(
            sample_inputs=self.sample_inputs,
            sample_outputs=self.sample_outputs,
            onnx_file=self.onnx_model,
        )

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
        if not paths:
            raise ValueError(
                "The directory path is empty. "
                "Check whether the indicated directory exists."
            )
        for path in paths:
            display_name = os.path.basename(path)
            files.append(file_dictionary(display_name=display_name, path=path))

        return ModelDirectory(files=files, name="model_directory", path=directory_path)

    def generate_outputs(
        self, engine_type: str, save_to_tar: bool = False
    ) -> Union[List[np.ndarray], typing.OrderedDict[str, np.ndarray], None]:
        """
        Chooses the appropriate engine type to obtain inference outputs
        from the `InferenceRunner` class object. The function yields model
        outputs sequentially (in the iterative fashion)

        :params engine_type: name of the inference engine
        :params save_to_tar: boolean flag; if True, the output generated
            by the `inference_runner` will be additionally saved to
            the archive file `sample_outputs_{engine_type}.tar.gz
            (located in the `self.path` directory).
        :returns returns a data structure
            containing numpy arrays, representing the output
            from the inference engine
        """

        for output in self.inference_runner.generate_outputs(
            engine_type=engine_type, save_to_tar=save_to_tar
        ):
            yield output

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

        if not self.inference_runner.validate_with_onnx_runtime():
            logging.warning(
                "Failed to validate the compatibility of "
                "`sample_inputs` files with the `model.onnx` model."
            )
            return False

        # TODO: This is a hack for now,
        #  some files cannot be validated
        #  using dummy inputs (see respective tests)
        SKIP_ATTRIBUTES = ["training"]

        if self.path is None:
            raise ValueError(
                "Cannot validate the ModelDirectory. "
                "If created using method `from_directory`, "
                "please make sure that the `directory_path` is correct. "
                "If created using method `from_zoo_api`, "
                "call `download()` method prior to `validate()`"
            )

        validations = {}
        for file in self.files:
            if isinstance(file, File):
                # TODO: Continuing with the hack
                if file.name in SKIP_ATTRIBUTES:
                    validations[file.name] = True
                else:
                    validations[file.name] = file.validate()
            elif isinstance(file, list):
                for _file in file:
                    validations[_file.name] = _file.validate()
            elif isinstance(file, dict):
                raise NotImplementedError()

        return all(validations.values())

    def analyze(self):
        # TODO: This will be the onboarding task for Kyle
        pass

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
                files_within = (
                    file_dictionary(display_name=os.path.basename(path), path=path)
                    for path in paths_within
                )

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
            return files_found

    def _directory_from_files(
        self,
        files: List[Dict[str, Any]],
        directory_class: Union[
            Directory, NumpyDirectory, FrameworkFiles, SampleOriginals
        ] = Directory,
        display_name: Optional[str] = None,
        regex: Optional[bool] = True,
        allow_multiple_outputs: Optional[bool] = False,
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
            if allow_multiple_outputs:
                return directories_found
            raise ValueError(
                f"Found more than one Directory for `display_name`: {display_name}."
            )
        else:
            return directories_found[0]

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
            validations = (self._download(_file, directory_path) for _file in file)
            return all(validations)

        else:
            raise NotImplementedError()

    @staticmethod
    def _sample_outputs_list_to_dict(
        directories: List[NumpyDirectory],
    ) -> Dict[str, NumpyDirectory]:
        engine_to_numpydir_map = {}
        for directory in directories:
            engine_name = directory.name.split("_")[-1]
            if engine_name not in ENGINES:
                raise ValueError(
                    f"The name of the 'sample_outputs' directory should "
                    f"end with an engine name (one of the {ENGINES}). "
                    f"However, the name is {directory.name}."
                )
            engine_to_numpydir_map[engine_name] = directory
        return engine_to_numpydir_map

    @staticmethod
    def _is_file_tar(file):
        extension = file["display_name"].split(".")[-2:]
        return extension == ["tar", "gz"]
