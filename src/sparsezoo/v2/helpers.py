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
Helper functions pertaining to the creation of ModelDirectory
"""

import copy
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Union

from sparsezoo.v2 import Directory, File, NumpyDirectory, SampleOriginals


def setup_model_directory(
    output_dir: str,
    training: Union[str, Directory, List[Union[str, Directory]]],
    deployment: Union[str, Directory, List[Union[str, Directory]]],
    onnx_model: Union[File, str],
    sample_inputs: Union[str, NumpyDirectory],
    sample_outputs: Union[
        List[Union[str, NumpyDirectory]], str, NumpyDirectory, None
    ] = None,
    sample_labels: Union[Directory, str, None] = None,
    sample_originals: Union[SampleOriginals, str, None] = None,
    logs: Union[Directory, str, None] = None,
    analysis: Union[File, str, None] = None,
    benchmarks: Union[File, str, None] = None,
    eval_results: Union[File, str, None] = None,
    model_card: Union[File, str, None] = None,
    recipes: Union[List[Union[str, File]], str, File, None] = None,
) -> None:
    """

    The function takes Files and Directories that are expected by the
    ModelDirectory (some Files/Directories are mandatory, some are optional),
    and then creates a new directory where the files are being copied to.
    The format of the new directory adheres to the structure expected by the
    `ModelDirectory` class factory methods.

    Note: Some of the   "loose" files/directories that would then be copied


    :params output_dir: path to the target directory
    :params training: pointer (path or File) to the training directory
        (can also pass an "unstructured" list containing a mix of paths/Files)
    :params deployment: pointer (path or File) to the deployment directory
        (can also pass an "unstructured" list containing a mix of paths/Files)
    :params onnx_model: pointer (path or File) to the model.onnx file
    :params sample_inputs: pointer (path or File) to the sample_inputs directory
    :params sample_outputs: pointer (path or File) to the sample_outputs directory
            (also supports list of paths or Files)
    :params sample_labels: pointer (path or File) to the sample_labels directory
    :params sample_originals: pointer (path or File) to the sample_originals directory
    :params logs: pointer (path or File) to the logs directory
        (can also pass an "unstructured" list containing a mix of paths/Files)
    :params analysis: pointer (path or File) to the analysis.yaml file
    :params benchmarks: pointer (path or File) to the benchmarks.yaml file
    :params eval_results: pointer (path or File) to the eval.yaml file
    :params model_card: pointer (path or File) to the model.md file
    :params recipes: pointer (path or File) to the recipe.yaml file
            (also supports list of paths or Files)
    """
    # create new directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # fetch the function arguments as a dictionary
    files_dict = copy.deepcopy(locals())
    del files_dict["output_dir"]

    # iterate over the arguments (files)
    for name, file in files_dict.items():
        if file is None:
            logging.debug(f"File {name} not provided. It will be omitted.")
        else:
            if isinstance(file, str):
                # if file is a string, convert it to
                # File/Directory class object
                file = _create_file_from_path(file)
            elif isinstance(file, list):
                # if file is a list, we need to call
                # _create_file_from_path on every object
                # that is a path (string)
                for idx, _file in enumerate(file):
                    if isinstance(_file, str):
                        file[idx] = _create_file_from_path(_file)
                    elif isinstance(_file, File):
                        continue
                    else:
                        raise ValueError(
                            "Expected `file` to be either a string (path) "
                            "or a File/Directory class object. "
                            f"However, it's type is {type(file)}."
                        )
            # otherwise, the file is File/Directory class object
            # and can be directly copied over

            _copy_file_contents(output_dir, file, name)


def _create_file_from_path(
    path: str,
) -> Union[File, Directory]:
    # create a File or Directory given a path
    file = File(name=os.path.basename(path), path=path)
    if os.path.isdir(path):
        directory = Directory.from_file(file=file)
        return directory
    else:
        return file


def _copy_file_contents(
    output_dir: str, file: Union[File, Directory], name: Optional[str] = None
) -> None:
    # optional argument `name` only used to make sure
    # that the names of the saved folders are consistent
    if name in ["training", "deployment", "logs"]:
        # for the `unstructured` directories (can contain
        # different files depending on the integration/circumstances
        if isinstance(file, list):
            # files passed as an unstructured list of files
            for _file in file:
                Path(os.path.join(output_dir, name)).mkdir(parents=True, exist_ok=True)
                copy_path = os.path.join(output_dir, name, _file.name)
                copy_func = (
                    shutil.copytree if isinstance(_file, Directory) else shutil.copyfile
                )
                _copy_and_overwrite(_file.path, copy_path, copy_func)
        else:
            # files passed as a Directory class instance
            copy_path = os.path.join(output_dir, name)
            _copy_and_overwrite(file.path, copy_path, shutil.copytree)
    else:
        # for the structured directories/files
        if isinstance(file, list):
            for _file in file:
                copy_path = os.path.join(output_dir, os.path.basename(_file.path))
                _copy_and_overwrite(_file.path, copy_path, shutil.copyfile)
        elif isinstance(file, Directory):
            copy_path = os.path.join(output_dir, os.path.basename(file.path))
            _copy_and_overwrite(file.path, copy_path, shutil.copytree)
        else:
            # if not Directory then File class object
            copy_path = os.path.join(output_dir, os.path.basename(file.path))
            _copy_and_overwrite(file.path, copy_path, shutil.copyfile)


def _copy_and_overwrite(from_path, to_path, func):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    func(from_path, to_path)
