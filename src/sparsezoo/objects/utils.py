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
A set of helper functions that serve
as a temporary bridge between
sparsezoo v1 and v2
"""
from __future__ import annotations

import copy
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from sparsezoo.utils import save_numpy

from .directory import Directory
from .file import File
from .model_objects import NumpyDirectory


__all__ = [
    "restructure_request_json",
    "fetch_from_request_json",
    "save_outputs_to_tar",
    "setup_model",
]

ALLOWED_FILE_TYPES = {
    "originals",
    "recipe",
    "onnx",
    "labels",
    "card",
    "inputs",
    "training",
    "deployment",
    "benchmarking",
    "outputs",
}


def save_outputs_to_tar(
    sample_inputs: NumpyDirectory, iterator: Callable, engine_type: str
):
    """
    Save the output from the `engine_type` engine, conditioned on the sample_inputs.
    The output is fetched from the iterator.
    The outputs will be saved in the same directory as sample_inputs in the
    corresponding structure.
    The outputs by default will be saved as tar.gz file.
    :param sample_inputs: Sample inputs directory
    :param iterator: Iterator that contains sample outputs
    :param engine_type: Inference engine name e.g. `deepsparse` or `onnxruntime`
    """
    output_files = []

    path = os.path.join(
        os.path.dirname(sample_inputs.path),
        f"sample_outputs_{engine_type}",
    )
    if not os.path.exists(path):
        os.mkdir(path)

    for input_file, output in zip(sample_inputs.files, iterator()):
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


def restructure_request_json(
    request_json: List[Dict[str, Any]], allowed_file_types: Set = ALLOWED_FILE_TYPES
) -> Dict[str, Any]:
    """
    Takes the legacy API response and restructures it, so that the output is
    compatible with the structure of Model.

    :params request_json: data structure describing the
        files in the Model (output from NeuralMagic API).
    :params allowed_file_types: a set of `file_types`,
        that will not be filtered out during restructuring
    :return: restructured request_json
    """
    # create `training` folder
    training_dicts_list = fetch_from_request_json(
        request_json, "file_type", "framework"
    )
    for (idx, training_file_dict) in training_dicts_list:
        training_file_dict["file_type"] = "training"
        request_json[idx] = training_file_dict

    # create `deployment` folder
    onnx_model_dict_list = fetch_from_request_json(
        request_json, "display_name", "model.onnx"
    )
    assert len(onnx_model_dict_list) == 1
    _, onnx_model_file_dict = copy.copy(onnx_model_dict_list[0])
    onnx_model_file_dict["file_type"] = "deployment"
    request_json.append(onnx_model_file_dict)

    # if NLP model, add `config.json` and `tokenizer.json` to `deployment`
    training_file_names = [
        x[1]["display_name"]
        for x in fetch_from_request_json(request_json, "file_type", "training")
    ]
    nlp_folder = (
        True
        if (("config.json") in training_file_names)
        and (("tokenizer.json") in training_file_names)
        else False
    )
    if nlp_folder:
        for file_name in ["config.json", "tokenizer.json"]:
            file_dict_training = fetch_from_request_json(
                request_json, "display_name", file_name
            )
            assert len(file_dict_training) == 1
            file_dict_deployment = copy.copy(file_dict_training[0][1])
            file_dict_deployment["file_type"] = "deployment"
            request_json.append(file_dict_deployment)

    # create recipes
    recipe_dicts_list = fetch_from_request_json(request_json, "file_type", "recipe")
    for (idx, file_dict) in recipe_dicts_list:
        display_name = file_dict["display_name"]
        # make sure that recipe name has a
        # format `recipe_{...}`.
        prefix = "recipe_"
        if not display_name.startswith(prefix):
            display_name = prefix + display_name
            file_dict["display_name"] = display_name
            request_json[idx] = file_dict

    # restructure inputs/labels/originals/outputs directories
    # use `sample-inputs.tar.gz` to simulate non-existent directories

    files_to_create = [
        "sample_inputs.tar.gz",
        "sample_labels.tar.gz",
        "sample_originals.tar.gz",
        "sample_outputs.tar.gz",
    ]
    types = ["inputs", "labels", "originals", "outputs"]
    for file_name, type in zip(files_to_create, types):
        data = fetch_from_request_json(
            request_json, "display_name", file_name.replace("_", "-")
        )
        if len(data) == 1:
            # file present but needs
            # restructuring
            file_dict = data[0][1]
            file_dict["display_name"] = file_name
            file_dict["file_type"] = type
            idx = data[0][0]
            request_json[idx] = file_dict

    # remove all undesired or duplicate files
    request_json = [
        file_dict
        for file_dict in request_json
        if file_dict["file_type"] in allowed_file_types
    ]

    return request_json


def fetch_from_request_json(
    request_json: List[Dict[str, Any]], key: str, value: str
) -> List[Tuple[int, Dict[str, Any]]]:
    # searches through the `request_json` list to find a
    # dictionary, that contains the requested
    # key-value pair.
    # return a list of tuples
    # (every tuple is a file_dict, together
    # with the respective list index)
    return [
        (idx, copy.copy(file_dict))
        for (idx, file_dict) in enumerate(request_json)
        if file_dict[key] == value
    ]


def setup_model(
    output_dir: str,
    training: Union[str, Directory, List[Union[str, Directory]]],
    deployment: Union[str, Directory, List[Union[str, Directory]]],
    onnx_model: Union[File, str],
    sample_inputs: Union[str, NumpyDirectory],
    sample_outputs: Union[
        List[Union[str, NumpyDirectory]], str, NumpyDirectory, None
    ] = None,
    sample_labels: Union[Directory, str, None] = None,
    sample_originals: Union[Directory, str, None] = None,
    logs: Union[Directory, str, None] = None,
    analysis: Union[File, str, None] = None,
    benchmarks: Union[File, str, None] = None,
    eval_results: Union[File, str, None] = None,
    model_card: Union[File, str, None] = None,
    recipes: Union[List[Union[str, File]], str, File, None] = None,
) -> None:
    """

    The function takes Files and Directories that are expected by the
    Model (some Files/Directories are mandatory, some are optional),
    and then creates a new directory where the files are being copied to.
    The format of the new directory adheres to the structure expected by the
    `Model` class factory methods.

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
