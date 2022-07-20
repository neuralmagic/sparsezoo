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

import logging
import os
import re
from typing import Any, Dict, Generator, List, Optional, Union

import numpy

from sparsezoo.v2.inference.engines import ENGINES
from sparsezoo.v2.inference.inference_runner import InferenceRunner
from sparsezoo.v2.objects.directory import Directory, is_directory
from sparsezoo.v2.objects.file import File
from sparsezoo.v2.objects.model_objects import NumpyDirectory


__all__ = ["ModelDirectory"]


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

        self.training: Directory = self._directory_from_files(
            files, directory_class=Directory, display_name="training"
        )
        self.sample_originals: Directory = self._directory_from_files(
            files,
            directory_class=Directory,
            display_name="sample_originals",
        )
        self.sample_inputs: NumpyDirectory = self._directory_from_files(
            files,
            directory_class=NumpyDirectory,
            display_name="sample_inputs",
        )

        self.model_card: File = self._file_from_files(files, display_name="model.md")

        self.sample_outputs: Dict[
            str, NumpyDirectory
        ] = self._sample_outputs_list_to_dict(
            self._directory_from_files(
                files,
                directory_class=NumpyDirectory,
                display_name="sample_outputs",
                allow_multiple_outputs=True,
                regex=True,
            )
        )

        self.sample_labels: Directory = self._directory_from_files(
            files, directory_class=Directory, display_name="sample_labels"
        )

        self.deployment: Directory = self._directory_from_files(
            files, display_name="deployment"
        )

        self.onnx_folder: Directory = self._directory_from_files(
            files,
            display_name="onnx",
        )

        self.logs: Directory = self._directory_from_files(files, display_name="logs")

        self.onnx_model: File = self._file_from_files(files, display_name="model.onnx")

        self.analysis: File = self._file_from_files(files, display_name="analysis.yaml")
        self.benchmarks: File = self._file_from_files(
            files, display_name="benchmarks.yaml"
        )
        self.eval_results: File = self._file_from_files(files, display_name="eval.yaml")

        self.recipes: List[File] = self._file_from_files(
            files, display_name="recipe(.*).md", regex=True
        )

        # sorting name of `sample_inputs` and `sample_output` files,
        # so that they have same one-to-one correspondence when we jointly
        # iterate over them
        self.sample_inputs.files.sort(key=lambda x: x.name)
        [
            sample_outputs.files.sort(key=lambda x: x.name)
            for sample_outputs in self.sample_outputs.values()
        ]

        self._files_dictionary = {
            "training": self.training,
            "deployment": self.deployment,
            "onnx_folder": self.onnx_folder,
            "logs": self.logs,
            "sample_originals": self.sample_originals,
            "sample_inputs": self.sample_inputs,
            "sample_outputs": self.sample_outputs,
            "sample_labels": self.sample_labels,
            "model_card": self.model_card,
            "recipes": self.recipes,
            "onnx_model": self.onnx_model,
            "analysis": self.analysis,
            "benchmarks": self.benchmarks,
            "eval_results": self.eval_results,
        }

        self.inference_runner = InferenceRunner(
            sample_inputs=self.sample_inputs,
            sample_outputs=self.sample_outputs,
            onnx_file=self.onnx_model,
        )

        super().__init__(
            files=self._files_dictionary.values(), name=name, path=path, url=url
        )

        # importing the class here, otherwise a circular import error is being raised
        # (IntegrationValidator script also imports ModelDirectory class object)
        from sparsezoo.v2.validation import IntegrationValidator

        self.integration_validator = IntegrationValidator(model_directory=self)

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
        display_names = os.listdir(directory_path)
        if not display_names:
            raise ValueError(
                "The directory path is empty. "
                "Check whether the indicated directory exists."
            )
        for display_name in display_names:
            files.append(
                file_dictionary(
                    display_name=display_name,
                    path=os.path.join(directory_path, display_name),
                )
            )

        return ModelDirectory(files=files, name="model_directory", path=directory_path)

    def generate_outputs(
        self, engine_type: str, save_to_tar: bool = False
    ) -> Generator[List[numpy.ndarray], None, None]:
        """
        Chooses the appropriate engine type to obtain inference outputs
        from the `InferenceRunner` class object. The function yields model
        outputs sequentially (in the iterative fashion)

        :params engine_type: name of the inference engine
        :params save_to_tar: boolean flag; if True, the output generated
            by the `inference_runner` will be additionally saved to
            the archive file `sample_outputs_{engine_type}.tar.gz
            (located in the `self.path` directory).
        :returns list containing numpy arrays, representing the output
            from the inference engine
        """

        for output in self.inference_runner.generate_outputs(
            engine_type=engine_type, save_to_tar=save_to_tar
        ):
            yield output

    def download(
        self, directory_path: str, override: bool = False, strict_mode: bool = False
    ) -> bool:
        """
        Attempt to download the files given the `url` attribute
        of the files inside the ModelDirectory.

        :param directory_path: directory to download files to
        :param override: if True, the method can override old `directory_path`
        :param strict_mode: if True, will throw error if any file, that is
            attempted to be downloaded, turns out to be `None`.
            By default, False.
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
            for key, file in self._files_dictionary.items():
                if file is not None:
                    downloads.append(self._download(file, directory_path))
                else:
                    if strict_mode:
                        raise ValueError(
                            f"Attempted to download file {key}, " f"but it is `None`!"
                        )

                    else:
                        logging.warning(
                            f"Attempted to download file {key}, "
                            f"but it is `None`. The file is being skipped..."
                        )
        return all(downloads)

    def validate(
        self, validate_onnxruntime: bool = True, minimal_validation: bool = False
    ) -> bool:
        """
        Validate the ModelDirectory class object:
        1. Validate that the sample inputs and outputs work with ONNX Runtime
            (if `validate_onnxruntime=True`)
        2. Validate all the folders (this is done by a separate helper class
            IntegrationValidator)

        :param validate_onnxruntime: boolean flag; if True, validate that the
            sample inputs and outputs work with ONNX Runtime
        :param minimal_validation: boolean flag; if True, only the essential files
            in the `training` folder are validated. Else, the `training` folder is
            expected to contain a full set of framework files.
        return: a boolean flag; if True, the validation has been successful
        """

        if validate_onnxruntime:
            if not self.inference_runner.validate_with_onnx_runtime():
                logging.warning(
                    "Failed to validate the compatibility of "
                    "`sample_inputs` files with the `model.onnx` model."
                )
                return False

        if self.model_card.path is None:
            raise ValueError(
                "It seems like the `ModelDirectory` was created using "
                "the `from_zoo_api()` method. Before running `validate()` "
                "method, the respective files must be present locally. "
                "The solution is to call `download()` first."
            )

        return self.integration_validator.validate(minimal_validation)

    def analyze(self):
        raise NotImplementedError()

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
                match = (display_name == file["display_name"]) or (
                    display_name + ".tar.gz" == file["display_name"]
                )

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
            # directory is locally on the machine
            files_in_directory = []
            for file_name in os.listdir(path):
                file = File(name=file_name, path=os.path.join(path, file_name))
                if is_directory(file):
                    file = Directory.from_file(file)

                files_in_directory.append(file)

            directory = directory_class(
                files=files_in_directory, name=name, path=path, url=url
            )
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
        elif len(files_found) == 1:
            return files_found[0]

        elif display_name == "model.onnx" and len(files_found) == 2:
            # `model.onnx` file may be found twice:
            #   - directly in the root directory
            #   - inside `deployment` directory
            # if this is the case, return the first file
            # (the other one is a duplicate)
            return files_found[0]
        else:
            return files_found

    def _directory_from_files(
        self,
        files: List[Dict[str, Any]],
        directory_class: Union[Directory, NumpyDirectory] = Directory,
        display_name: Optional[str] = None,
        regex: Optional[bool] = False,
        allow_multiple_outputs: Optional[bool] = False,
    ) -> Union[Directory, None]:

        # Takes a list of file dictionaries and returns
        # a Directory() object, if successful,
        # otherwise None.
        if all(["file_type" in file_dict for file_dict in files]):
            # if file_dict is retrieved from the API as `request_json`
            # first check if a directory can be created from the
            # "loose" files (alternative to parsing a .tar.gz file as
            # a directory)
            directory = self._get_directory_from_loose_api_files(
                files=files, directory_class=directory_class, display_name=display_name
            )
        else:
            directory = None

        if directory:
            directories_found = [directory]
        else:
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
        # one directory (unless `allow-multiple_outputs`=True)
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

        if isinstance(file, File):
            if file.url or (
                file.url is None
                and any(
                    isinstance(file, _class) for _class in [Directory, NumpyDirectory]
                )
            ):
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
            validations = (
                self._download(_file, directory_path) for _file in file.values()
            )
            return all(validations)

    def _sample_outputs_list_to_dict(
        self,
        directories: Union[List[NumpyDirectory], NumpyDirectory],
    ) -> Dict[str, NumpyDirectory]:
        engine_to_numpydir_map = {}
        if not isinstance(directories, list):
            # if found a single 'sample_outputs' directory,
            # assume it should be mapped to its the native framework
            expected_name = "sample_outputs"
            if directories.name not in [expected_name, expected_name + ".tar.gz"]:
                raise ValueError(
                    "Found single folder (or tar.gz archive)"
                    f"with expected name `{expected_name}`. However,"
                    f"detected a name {directories.name}."
                )
            engine_to_numpydir_map["framework"] = directories

        else:
            # if found multiple 'sample_outputs' directories,
            # use directory name to relate it with the appropriate
            # inference engine
            for directory in directories:
                engine_name = directory.name.split("_")[-1]
                if engine_name.endswith(".tar.gz"):
                    engine_name = engine_name.replace(".tar.gz", "")
                if engine_name not in ENGINES:
                    raise ValueError(
                        f"The name of the 'sample_outputs' directory should "
                        f"end with an engine name (one of the {ENGINES}). "
                        f"However, the name is {directory.name}."
                    )
                engine_to_numpydir_map[engine_name] = directory
        return engine_to_numpydir_map

    @staticmethod
    def _get_directory_from_loose_api_files(files, directory_class, display_name):
        # fetch all the loose files that belong to the directory (use `file_type` key
        # from the `request_json` for proper mapping)
        files = [
            file_dict for file_dict in files if file_dict["file_type"] == display_name
        ]
        if display_name == "onnx":
            # when searching for files to put inside the `onnx` directory,
            # remove the 'model.onnx' file. While all the onnx models
            # share the same `file_type = "onnx"`, the 'onnx' directory should
            # only contain `model.{opset_version}.onnx` files.
            files = [
                file_dict
                for file_dict in files
                if file_dict["display_name"] != "model.onnx"
            ]

        # we want to only process "loose" files here,
        # the `.tar.gz` directories get parsed using
        # a separate pathway
        files = [
            file_dict
            for file_dict in files
            if not file_dict["display_name"].endswith(".tar.gz")
        ]
        if not files:
            return None
        else:
            files = [File.from_dict(file) for file in files]
            directory = directory_class(
                files=files, name=display_name, path=None, url=None
            )
            return directory

    @staticmethod
    def _is_file_tar(file):
        extension = file["display_name"].split(".")[-2:]
        return extension == ["tar", "gz"]
