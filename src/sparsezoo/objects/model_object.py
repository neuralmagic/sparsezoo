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
from typing import Any, Dict, List, Optional, Union

from src.sparsezoo.objects.model_folder_objects import (
    Directory,
    File,
    FrameworkFiles,
    NumpyDirectory,
    SampleOriginals,
)


FRAMEWORKS = ["pytorch", "keras", "tensorflow"]


def file_dictionary(**kwargs):
    return kwargs


class ModelDirectory:
    """
    Object to represent SparseZoo Model Directory.

    The suggested way to create a class object is to use
    any of the two factory methods:
    - `from_zoo_api()` -> by using the output from the
        `src.sparsezoo.requests.download.download_model_get_request()` function.
    - `from_directory()` -> by using the directory to the model on your machine.

    :param files: list of files, where every file
        is represented by a dictionary (not File() object)
    """

    def __init__(self, files: List[Dict[str, Any]]):

        self.framework_files: FrameworkFiles = self._directory_from_files(
            files,
            directory_class=FrameworkFiles,
            display_name="framework-files",
            regex=True,
        )
        self.sample_originals: SampleOriginals = self._directory_from_files(
            files,
            directory_class=SampleOriginals,
            display_name="sample-originals",
            regex=True,
        )
        self.sample_inputs: NumpyDirectory = self._directory_from_files(
            files,
            directory_class=NumpyDirectory,
            display_name="sample-inputs",
            regex=True,
        )
        # TODO: Ignoring the directory part of the statement below
        self.sample_outputs: Dict[str, NumpyDirectory] = self._directory_from_files(
            files,
            directory_class=NumpyDirectory,
            display_name="sample-outputs",
            regex=True,
        )  # key by engine name.
        self.sample_labels: Directory = self._directory_from_files(
            files, directory_class=Directory, display_name="sample-labels", regex=True
        )
        self.onnx_model: File = self._file_from_files(
            files, display_name="model.onnx"
        )  # model.onnx
        self.onnx_models: List[File] = self._file_from_files(
            files, display_name="model{.*}.onnx", regex=True
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
            files, display_name="recipe{.*}.md", regex=True
        )  # recipe{_tag}.md

    @classmethod
    def from_zoo_api(cls, request_json: List[Dict]) -> "ModelDirectory":
        """
        Factory method for creating ModelDirectory class object
        from the output of the NeuralMagic API
        """
        files = request_json["files"]

        # group all the framework files under one, single file dictionary
        # this is to make sure that the file structure from both
        # `from_zoo_api` and `from_directory` factory methods
        # is fairly consistent
        framework_files = [x for x in files if x["file_type"] == "framework"]
        files.append(
            file_dictionary(display_name="framework-files", contains=framework_files)
        )
        # remove framework files from files
        [files.remove(file) for file in framework_files]

        return ModelDirectory(files=files)

    @classmethod
    def from_directory(cls, directory_path: str) -> "ModelDirectory":
        """
        Factory method for creating ModelDirectory class object
        from the local directory
        """
        files = []
        paths = glob.glob(os.path.join(directory_path, "*"))
        for path in paths:
            display_name = os.path.basename(path)
            # swap specific framework `display_name` (e.g. pytorch, keras) for
            # a generic one (`framework-files`)
            if display_name in FRAMEWORKS:
                display_name = "framework-files"
            files.append(file_dictionary(display_name=display_name, path=path))

        return ModelDirectory(files=files)

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
            # directory.validate() # TODO: Include directory validation later on
            return directory

        # TODO: Include nested directory parsing in the future
        # directory is folder
        else:
            # is directory using the 'contains' key.
            # this is a placeholder for the nested files that
            # the directory contains

            if file.get("contains"):
                files_within = file["contains"]

            # is directory locally on the machine
            else:
                paths_within = glob.glob(os.path.join(path, "*"))
                files_within = [
                    file_dictionary(display_name=os.path.basename(path), path=path)
                    for path in paths_within
                ]

            files = [self._get_file(file=file) for file in files_within]
            directory = directory_class(files=files, name=name, path=path, url=url)
            # directory.validate() # TODO: Include directory validation later on
            return directory

    @staticmethod
    def _is_file_tar(file):
        _, *extension = file["display_name"].split(".")
        return extension == ["tar", "gz"]

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
            file = File.from_file_dict(file)
            if file.path:
                file.validate()
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
        regex: Optional[bool] = False,
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
            raise ValueError()
        else:
            return directories_found[0]
