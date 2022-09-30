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

import copy
import logging
import os
import shutil
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from sparsezoo.model.result_utils import (
    ModelResult,
    ThroughputResults,
    ValidationResult,
)
from sparsezoo.objects import Directory, File, NumpyDirectory
from sparsezoo.utils import download_get_request, save_numpy


__all__ = [
    "restructure_request_json",
    "fetch_from_request_json",
    "setup_model",
    "load_files_from_stub",
    "load_files_from_directory",
    "ZOO_STUB_PREFIX",
    "SAVE_DIR",
    "COMPRESSED_FILE_NAME",
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
    "onnx_gz",
}

_LOGGER = logging.getLogger(__name__)

ZOO_STUB_PREFIX = "zoo:"
CACHE_DIR = os.path.expanduser(os.path.join("~", ".cache", "sparsezoo"))
SAVE_DIR = os.getenv("SPARSEZOO_MODELS_PATH", CACHE_DIR)
COMPRESSED_FILE_NAME = "model.onnx.tar.gz"


def load_files_from_directory(directory_path: str) -> List[Dict[str, Any]]:
    """
    :param directory_path: a path to the directory,
        that contains model files in the expected structure
    :return list of file dictionaries
    """
    display_names = os.listdir(directory_path)
    if not display_names:
        raise ValueError(
            "The directory path is empty. "
            "Check whether the indicated directory exists."
        )
    files = [
        dict(display_name=display_name, path=os.path.join(directory_path, display_name))
        for display_name in display_names
    ]
    return files


def _get_compressed_size(files: List[Dict[str, Any]]) -> Optional[int]:
    """
    Utility method to return compressed file size in bytes, if the size cannot
    be inferred `None` is returned

    :param files: List of file dictionaries
    :return: `None` if file size cannot be determined, else an int representing
        compressed size of the model in bytes
    """
    for file in files:
        if file.get("display_name") == COMPRESSED_FILE_NAME:
            return file.get("file_size")

    _LOGGER.info("Compressed file-size not found!")


def load_files_from_stub(
    stub: str,
    valid_params: Optional[List[str]] = None,
    force_token_refresh: bool = False,
) -> Tuple[
    List[Dict[str, Any]], str, Dict[str, str], Dict[str, List[ModelResult]], int
]:
    """
    :param stub: the SparseZoo stub path to the model (optionally
        may include string arguments)
    :param valid_params: list of expected parameter names to be encoded in the
        stub. Will raise a warning if any unexpected param names are given. Leave
        as None to not raise any warnings. Default is None
    :param force_token_refresh: True to refresh the auth token, False otherwise
    :return: The tuple of
        - list of file dictionaries
        - model_id (from the server)
        - parsed param dictionary
        - validation results dictionary
        - compressed model size in bytes
    """
    params = None
    if isinstance(stub, str):
        stub, params = parse_zoo_stub(stub=stub, valid_params=valid_params)
    _LOGGER.debug(f"load_files_from_stub: loading files from {stub}")
    response = download_get_request(
        args=stub,
        force_token_refresh=force_token_refresh,
    )

    # piece of code required for backwards compatibility
    model_response = response.get("model", {})
    files = model_response.get("files", [])
    files = restructure_request_json(request_json=files)
    compressed_file_size = _get_compressed_size(files=files)
    model_id = model_response.get("model_id")
    if params is not None:
        files = filter_files(files=files, params=params)

    model_results = model_response.get("results")
    validation_results = _parse_validation_metrics(model_results_response=model_results)
    return files, model_id, params, validation_results, compressed_file_size


def filter_files(
    files: List[Dict[str, Any]], params: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Use the `params` to extract only the relevant files from `files`

    :param files: a list of file dictionaries
    :param params: a dictionary with filtering parameters
    :return a filtered `files` object
    """
    available_params = set(params.keys())
    files_filtered = []
    num_recipe_file_dicts = 0
    for file_dict in files:
        if "recipe" in available_params and file_dict["file_type"] == "recipe":
            expected_recipe_name = params["recipe"]
            if not file_dict["display_name"].startswith(
                "recipe_" + expected_recipe_name
            ):
                continue
            else:
                num_recipe_file_dicts += 1
        if "checkpoint" in available_params and file_dict["file_type"] == "training":
            pass

        if "deployment" in available_params and file_dict["file_type"] == "deployment":
            pass

        files_filtered.append(file_dict)

    if not files_filtered:
        raise ValueError("No files found - the list of files is empty!")

    if num_recipe_file_dicts >= 2:
        recipe_names = [
            file_dict["display_name"]
            for file_dict in files_filtered
            if file_dict["file_type"] == "recipe"
        ]
        raise ValueError(
            f"Found multiple recipes: {recipe_names}, "
            f"for the string argument {expected_recipe_name}"
        )
    else:
        return files_filtered


def parse_zoo_stub(
    stub: str, valid_params: Optional[List[str]] = None
) -> Tuple[str, Dict[str, str]]:
    """
    :param stub: A SparseZoo model stub. i.e. 'model/stub/path',
        'zoo:model/stub/path', 'zoo:model/stub/path?param1=value1&param2=value2'
    :param valid_params: list of expected parameter names to be encoded in the
        stub. Will raise a warning if any unexpected param names are given. Leave
        as None to not raise any warnings. Default is None
    :return: the parsed base stub and a dictionary of parameter names and their values
    """
    # strip optional zoo stub prefix
    if stub.startswith(ZOO_STUB_PREFIX):
        stub = stub[len(ZOO_STUB_PREFIX) :]

    if "?" not in stub:
        return stub, {}

    stub_parts = stub.split("?")
    if len(stub_parts) > 2:
        raise ValueError(
            "Invalid SparseZoo stub, query string must be preceded by only one '?'"
            f"given {stub}"
        )
    stub, params = stub_parts
    params = dict(param.split("=") for param in params.split("&"))

    if valid_params is not None and any(param not in valid_params for param in params):
        warnings.warn(
            f"Invalid query string for stub {stub} valid params include {valid_params},"
            f" given {list(params.keys())}"
        )

    return stub, params


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
    request_json: Union[Dict[str, Any], List[Dict[str, Any]]],
    allowed_file_types: Set = ALLOWED_FILE_TYPES,
) -> List[Dict[str, Any]]:
    """
    Takes the legacy API response and restructures it, so that the output is
    compatible with the structure of Model.

    :params files: data structure describing the
        files in the Model (output from NeuralMagic API).
    :params allowed_file_types: a set of `file_types`,
        that will not be filtered out during restructuring
    :return: restructured files
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

    training_file_names = [
        file_dict["display_name"]
        for idx, file_dict in fetch_from_request_json(
            request_json, "file_type", "training"
        )
    ]
    # if NLP model,
    # add `config.json`,`tokenizer.json`,`tokenizer_config.json` to `deployment`
    nlp_deployment_files = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    nlp_folder = nlp_deployment_files.issubset(set(training_file_names))

    if nlp_folder:
        for file_name in nlp_deployment_files:
            file_dict_training_list = fetch_from_request_json(
                request_json, "display_name", file_name
            )
            assert len(file_dict_training_list) == 1
            _, file_dict_training = file_dict_training_list[0]
            file_dict_deployment = copy.copy(file_dict_training)
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
            idx, file_dict = data[0]
            file_dict["display_name"] = file_name
            file_dict["file_type"] = type
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
    """
    Searches through the `request_json` list to find a
    dictionary, that contains the requested key-value pair.

    :param request_json: A list of file dictionaries
    :param key: lookup key for the file dictionary
    :param value: lookup value for the file dictionary
    :return a list of tuples
        (index - the found file dictionary's position in the `request_json`,
        the found file dictionary)
    """
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

    Note: Some of the "loose" files/directories that would then be copied


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
    output_dir: str,
    file: Union[File, Directory],
    name: Optional[str] = None,
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


def _parse_validation_metrics(
    model_results_response: List[Dict[str, Union[str, float, int]]]
) -> Dict[str, List[ModelResult]]:
    results: Dict[str, List[ModelResult]] = defaultdict(list)
    for result in model_results_response:
        recorded_units = result.get("recorded_units").lower()
        if recorded_units in ["items/seconds", "items/second"]:
            key = "throughput"
            current_result = ThroughputResults(
                result_type=result.get("result_type"),
                recorded_value=result.get("recorded_value"),
                recorded_units=result.get("recorded_units"),
                device_info=result.get("device_info"),
                num_cores=result.get("num_cores"),
                batch_size=result.get("batch_size"),
            )

        else:
            current_result = ValidationResult(
                result_type=result.get("result_type"),
                recorded_value=result.get("recorded_value"),
                recorded_units=recorded_units,
                dataset_name=result.get("dataset_name"),
                dataset_type=result.get("dataset_type"),
            )
            key = "validation"

        results[key].append(current_result)

    return results
