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
    training: Union[str, Directory],
    deployment: Union[str, Directory],
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


    :params output_dir: path to the target directory
    :params training: pointer (path or File) to the training directory
    :params deployment: pointer (path or File) to the deployment directory
    :params onnx_model: pointer (path or File) to the model.onnx file
    :params sample_inputs: pointer (path or File) to the sample_inputs directory
    :params sample_outputs: pointer (path or File) to the sample_outputs directory
            (also supports list of paths or Files)
    :params sample_labels: pointer (path or File) to the sample_labels directory
    :params sample_originals: pointer (path or File) to the sample_originals directory
    :params logs: pointer (path or File) to the logs directory
    :params analysis: pointer (path or File) to the analysis.yaml file
    :params benchmarks: pointer (path or File) to the benchmarks.yaml file
    :params eval_results: pointer (path or File) to the eval.yaml file
    :params model_card: pointer (path or File) to the model.md file
    :params recipes: pointer (path or File) to the recipe.yaml file
            (also supports list of paths or Files)
    """
    # create new directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # fetch the function arguments as a dictionary
    files_dict = copy.deepcopy(locals())
    del files_dict["output_dir"]

    for name, file in files_dict.items():
        if file is None:
            logging.debug(f"File {name} not provided, skipping...")
        else:
            if isinstance(file, str):
                file = _create_file_from_path(file)
            elif isinstance(file, list) and isinstance(file[0], str):
                file = [_create_file_from_path(_file) for _file in file]

            _copy_file_contents(output_dir, file, name)


def _create_file_from_path(path: str) -> Union[File, Directory]:
    # create a File or Directory given a path
    file = File(name=os.path.basename(path), path=path)
    if os.path.isdir(path):
        directory = Directory.from_file(file=file)
        return directory
    else:
        return file


def _copy_file_contents(
    output_dir: str, file: Union[File, Directory], name: Optional[str] = None
)-> None:
    # optional argument `name` only used to make sure
    # that the names of the saved folders are consistent
    if isinstance(file, list):
        for _file in file:
            _copy_file_contents(output_dir, _file)

    elif isinstance(file, Directory):
        copy_path = (
            os.path.join(output_dir, name)
            if name in ["training", "deployment", "logs"]
            else os.path.join(output_dir, os.path.basename(file.path))
        )
        shutil.copytree(file.path, copy_path)
    else:
        copy_path = os.path.join(output_dir, os.path.basename(file.path))
        shutil.copyfile(file.path, copy_path)
