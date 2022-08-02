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
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy

from sparsezoo.inference import ENGINES, InferenceRunner
from sparsezoo.model.utils import (
    CACHE_DIR,
    ZOO_STUB_PREFIX,
    load_files_from_directory,
    load_files_from_stub,
    save_outputs_to_tar,
)
from sparsezoo.objects import (
    Directory,
    File,
    NumpyDirectory,
    SelectDirectory,
    is_directory,
)
from sparsezoo.utils import DataLoader
from sparsezoo.validation import IntegrationValidator


__all__ = ["Model"]

ALLOWED_CHECKPOINT_VALUES = {"prepruning", "postpruning", "preqat", "postqat"}
ALLOWED_RECIPE_VALUES = {"original", "transfer_learn"}
ALLOWED_DEPLOYMENT_VALUES = {"default"}

PARAM_DICT = {
    "checkpoint": ALLOWED_CHECKPOINT_VALUES,
    "recipe": ALLOWED_RECIPE_VALUES,
    "deployment": ALLOWED_DEPLOYMENT_VALUES,
    "recipe_type": None,  # backwards compatibility with v1 stubs
}


class Model(Directory):
    """
    Object to represent SparseZoo Model Directory.

    :param path: a string argument that can be one of two things:
        a) a SparseZoo model stub:
            i) without additional string arguments
                e.g. 'zoo:model/stub/path'
            ii) with additional string arguments (that will restrict
                the set of file/directory files handled by this
                `Model` object)
                e.g. 'zoo:model/stub/path?param1=value1&param2=value2'
        b) a local directory path
            e.g. `/home/user/model_path`
    """

    def __init__(self, source: str):

        self.source = source
        self._stub_params = {}

        if self.source.startswith(ZOO_STUB_PREFIX):
            # initializing the files and params from the stub
            files, path, url = self.initialize_model_from_stub(self.source)
        else:
            # initializing the model from the path
            files, path, url = self.initialize_model_from_directory(self.source)

        self.path = path

        self.training: SelectDirectory = self._directory_from_files(
            files,
            directory_class=SelectDirectory,
            display_name="training",
            stub_params=self.stub_params,
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

        self.sample_outputs = self._directory_from_files(
            files,
            directory_class=NumpyDirectory,
            display_name="sample_outputs",
            allow_multiple_outputs=True,
            regex=True,
        )
        if self.sample_outputs is not None:
            self.sample_outputs: Dict[
                str, NumpyDirectory
            ] = self._sample_outputs_list_to_dict(self.sample_outputs)

        self.sample_labels: Directory = self._directory_from_files(
            files, directory_class=Directory, display_name="sample_labels"
        )

        self.deployment: SelectDirectory = self._directory_from_files(
            files,
            directory_class=SelectDirectory,
            display_name="deployment",
            stub_params=self.stub_params,
        )

        self.onnx_folder: Directory = self._directory_from_files(
            files,
            display_name="onnx",
        )

        self.logs: Directory = self._directory_from_files(files, display_name="logs")

        self.recipes: SelectDirectory = self._directory_from_files(
            files,
            directory_class=SelectDirectory,
            display_name="recipe",
            stub_params=self.stub_params,
        )

        self.onnx_model: File = self._file_from_files(files, display_name="model.onnx")

        self.analysis: File = self._file_from_files(files, display_name="analysis.yaml")
        self.benchmarks: File = self._file_from_files(
            files, display_name="benchmarks.yaml"
        )
        self.eval_results: File = self._file_from_files(files, display_name="eval.yaml")

        # sorting name of `sample_inputs` and `sample_output` files,
        # so that they have same one-to-one correspondence when we jointly
        # iterate over them
        if self.sample_inputs is not None:
            self.sample_inputs.files.sort(key=lambda x: x.name)

        if self.sample_outputs is not None:
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
            files=self._files_dictionary.values(),
            name=os.path.basename(self._path),
            path=self._path,
            url=url,
        )

        self.integration_validator = IntegrationValidator(model=self)

    @property
    def stub_params(self) -> Dict[str, str]:
        """
        :return: mapping of variable name to value for query params in zoo stub
            this Model was initialized from
        """
        return self._stub_params

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
            (located in the `self._path` directory).
        :returns list containing numpy arrays, representing the output
            from the inference engine
        """
        if save_to_tar:
            iterator = self.inference_runner.engine_type_to_iterator.get(engine_type)
            save_outputs_to_tar(self.sample_inputs, iterator, engine_type)

        for output in self.inference_runner.generate_outputs(engine_type=engine_type):
            yield output

    def download(
        self,
        directory_path: Optional[str] = None,
        override: bool = False,
        strict_mode: bool = False,
    ) -> bool:
        """
        Attempt to download the files given the `url` attribute
        of the files inside the Model.

        :param directory_path: directory to download files to. If not specified,
            will use the `self._path` attribute.
        :param override: if True, the method can override old `directory_path`
        :param strict_mode: if True, will throw error if any file, that is
            attempted to be downloaded, turns out to be `None`.
            By default, False.
        :return: boolean flag; was download successful or not.
        """
        if directory_path is None:
            directory_path = self._path

        # if directory exists and is not empty, make sure that
        # downloading is not possible unless `override` is True
        if (
            os.path.isdir(directory_path)
            and os.listdir(directory_path)
            and not override
        ):
            raise ValueError(
                "Model class object was either created "
                "using path that points to a local directory or "
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
                        logging.debug(
                            f"Attempted to download file {key}, "
                            f"but it is `None`. The file is being skipped..."
                        )
        self.path = directory_path
        return all(downloads)

    def validate(
        self, validate_onnxruntime: bool = True, minimal_validation: bool = False
    ) -> bool:
        """
        Validate the Model class object:
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

        if self.model_card is None:
            raise ValueError("Model card missing in the model!")

        return self.integration_validator.validate(minimal_validation)

    def analyze(self):
        raise NotImplementedError()

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.source.startswith(ZOO_STUB_PREFIX):
            return f"{self.__class__.__name__}(stub={self.source})"
        else:
            return f"{self.__class__.__name__}(directory={self.source})"

    def initialize_model_from_stub(
        self, stub: str
    ) -> Tuple[List[Dict[str, str]], str, str]:
        files, model_id, params = load_files_from_stub(
            stub, valid_params=list(PARAM_DICT.keys())
        )
        if params:
            self._validate_params(params)
            self._stub_params.update(params)
        path = os.path.join(CACHE_DIR, model_id)
        url = os.path.dirname(files[0]["url"])
        return files, path, url

    def initialize_model_from_directory(
        self, directory: str
    ) -> Tuple[List[Dict[str, str]], str, None]:
        files = load_files_from_directory(directory)
        path = directory
        url = None
        return files, path, url

    def data_loader(
        self, batch_size: int = 1, iter_steps: int = 0, batch_as_list: bool = True
    ) -> DataLoader:
        """
        Create a  data loader containing all of the available data for this model
        :param batch_size: the size of batches to create for the iterator
        :param iter_steps: the number of steps (batches) to create.
            Set to -1 for infinite, 0 for running through the loaded data once,
            or a positive integer for the desired number of steps
        :param batch_as_list: True to create the items from each dataset
            as a list, False for an ordereddict
        :return: The created data loader
        """
        datasets = []

        if self.sample_inputs is not None:
            datasets.append(self.sample_inputs.dataset())
        if self.sample_outputs is not None:
            datasets.append(next(val for val in self.sample_outputs.values()).dataset())

        if len(datasets) < 1:
            raise FileNotFoundError(
                "no datasets available for this model to create a loader from"
            )

        return DataLoader(
            *datasets,
            batch_size=batch_size,
            iter_steps=iter_steps,
            batch_as_list=batch_as_list,
        )

    def sample_batch(
        self, batch_index: int = 0, batch_size: int = 1, batch_as_list: bool = True
    ) -> Union[List[numpy.ndarray], Dict[str, numpy.ndarray]]:
        """
        Get a sample batch of data from the data loader
        :param batch_index: the index of the batch to get
        :param batch_size: the size of the batches to create the loader for
        :param batch_as_list: True to return multiple inputs/outputs/etc
            within the dataset as lists, False for an ordereddict
        :return: The sample batch for use with the model
        """
        loader = self.data_loader(batch_size=batch_size, batch_as_list=batch_as_list)

        return loader.get_batch(bath_index=batch_index)

    def _get_directory(
        self,
        file: Dict[str, Any],
        directory_class: Directory,
        display_name: Optional[str] = None,
        regex: Optional[bool] = False,
        **kwargs,
    ) -> Union[Directory, None]:
        # Takes a file dictionary and returns a Directory() object, if successful.
        # Optionally, can do:
        #   string matching (if `display_name` not None)
        #   regex (if `display_name` not None and `regex`)
        #   file validation (if file's _path is not None)

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
            logging.debug(
                "Could not find a directory with "
                f"display_name / regex_pattern: {display_name}"
            )
            return None

        name, path, url = file.get("display_name"), file.get("path"), file.get("url")

        # directory is a tar file
        if self._is_file_tar(file):
            directory = directory_class(
                files=[], name=name, path=path, url=url, parent_directory=self._path
            )
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
                files=files_in_directory,
                name=name,
                path=path,
                url=url,
                parent_directory=self._path,
                **kwargs,
            )
            return directory

    def _get_file(
        self,
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

                if display_name == "model.onnx" and "file_type" in file:
                    # handling a corner case when `files` only contains
                    # file dicts with `file_type="deployment"`
                    if file["file_type"] == "deployment":
                        match = False

        if not match:
            logging.debug(
                "Could not find a file with "
                f"display_name / regex_pattern: {display_name}"
            )
            return None
        else:
            file = File.from_dict(file, parent_directory=self._path)

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
        **kwargs,
    ) -> Union[Directory, None]:

        # Takes a list of file dictionaries and returns
        # a Directory() object, if successful,
        # otherwise None.
        if all([file_dict.get("file_type") for file_dict in files]):
            # if file_dict is retrieved from the API as `request_json`
            # first check if a directory can be created from the
            # "loose" files (alternative to parsing a .tar.gz file as
            # a directory)
            directory = self._get_directory_from_loose_api_files(
                files=files,
                directory_class=directory_class,
                display_name=display_name,
                parent_directory=self._path,
                **kwargs,
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
                    **kwargs,
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
                    [isinstance(file, _class) for _class in [Directory, NumpyDirectory]]
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
    def _get_directory_from_loose_api_files(
        files, directory_class, display_name, parent_directory, **kwargs
    ):
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
            files = [
                File.from_dict(
                    file, parent_directory=os.path.join(parent_directory, display_name)
                )
                for file in files
            ]
            directory = directory_class(
                files=files,
                name=display_name,
                path=None,
                url=None,
                parent_directory=parent_directory,
                **kwargs,
            )
            return directory

    @staticmethod
    def _is_file_tar(file):
        extension = file["display_name"].split(".")[-2:]
        return extension == ["tar", "gz"]

    @staticmethod
    def _validate_params(params: Dict[str, str]):
        # make sure that the extracted
        # params are correct
        if len(params) == 0:
            return
        for key, value in params.items():
            if key not in PARAM_DICT.keys():
                raise ValueError(
                    f"String argument {key} not found in the "
                    f"expected set of string arguments {list(PARAM_DICT.keys())}!"
                )
            if PARAM_DICT[key] and value not in PARAM_DICT[key]:
                raise ValueError(
                    f"String argument {key} has value {value}, "
                    "cannot be found in the "
                    f"expected set of values {list(PARAM_DICT[key])}!"
                )
