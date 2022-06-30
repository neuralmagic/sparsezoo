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
Helper functions for SparseZoo v2
"""

import copy
from typing import Any, Dict, List


def restructure_request_json(
    request_json: List[Dict[str, Any]],
    clone_sample_outputs: bool = True,
    tars: bool = True,
) -> Dict[str, Any]:
    """
    Takes the raw API response and restructures it, so that the output mocks the
    planned, future structure of the ModelDirectory.

    :params request_json: the output from NeuralMagic API
    :params clone_sample_outputs: whether we should convert `sample_outputs` folder
        to `sample_outputs_deepsparse` and `sample_outputs_onnxruntime`
    :params tars: whether to represent directories within ModelDirectory as
        archived folders or loose files
    :return: restructured request_json
    """

    if tars:
        request_json = _create_tar_directories(request_json)

    else:
        # create `training` and `deployment` files
        data = _fetch_from_request_json(request_json, "file_type", "framework")
        for (idx, file_dict) in data:
            training_file_dict, deployment_file_dict = file_dict, copy.copy(file_dict)
            training_file_dict["file_type"], deployment_file_dict["file_type"] = (
                "training",
                "deployment",
            )
            request_json[idx] = training_file_dict
            request_json.append(deployment_file_dict)

        # append `model.onnx` to `deployment` files
        data = _fetch_from_request_json(request_json, "display_name", "model.onnx")
        assert len(data) == 1
        _, file_dict = data[0]
        file_dict["file_type"] = "deployment"
        request_json.append(file_dict)

        # create files for onnx directory (use `model.onnx` to simulate them)
        files_to_create = [
            "model.11.onnx",
            "model.14.onnx",
        ]
        data = _fetch_from_request_json(request_json, "file_type", "onnx")
        assert len(data) == 1
        for file in files_to_create:
            file_dict = copy.copy(data[0][1])
            file_dict["display_name"] = file
            file_dict["operator_version"] = int(file.split(".")[-2])
            request_json.append(file_dict)

        # create logs directory (use `model.md` to simulate it)
        data = _fetch_from_request_json(request_json, "display_name", "model.md")
        assert len(data) == 1
        _, file_dict = data[0]
        file_dict["display_name"] = "logs.yaml"
        file_dict["file_type"] = "logs"
        request_json.append(file_dict)

    # create recipes
    data = _fetch_from_request_json(request_json, "file_type", "recipe")
    only_one_recipe = len(data) == 1
    for (idx, file_dict) in data:
        display_name = file_dict["display_name"]
        # make sure that recipe name has a
        # format `recipe_{...}`.
        if not display_name.startswith("recipe"):
            display_name = "recipe_" + display_name
            file_dict["display_name"] = display_name
            request_json[idx] = file_dict

    # add one more recipe to files
    # just to make it more complex
    # for testing
    if only_one_recipe:
        additional_file_dict = copy.copy(file_dict)
        additional_file_dict["display_name"] = "recipe_foo.md"
        request_json.append(additional_file_dict)

    # create yaml files (use `model.md` to simulate them)
    files_to_create = ["analysis.yaml", "benchmarks.yaml", "eval.yaml"]
    data = _fetch_from_request_json(request_json, "display_name", "model.md")
    assert len(data) == 1
    for file_name in files_to_create:
        file_dict = copy.copy(data[0][1])
        file_dict["display_name"] = file_name
        file_dict["file_type"] = "benchmarking"
        request_json.append(file_dict)

    # restructure inputs/labels/originals/outputs directories

    # use `sample-inputs.tar.gz` to simulate non-existent directories
    dummy_data = _fetch_from_request_json(
        request_json, "display_name", "sample-inputs.tar.gz"
    )
    files_to_create = [
        "sample_inputs.tar.gz",
        "sample_labels.tar.gz",
        "sample_originals.tar.gz",
        "sample_outputs.tar.gz",
    ]
    types = ["inputs", "labels", "originals", "outputs"]
    for file_name, type in zip(files_to_create, types):
        data = _fetch_from_request_json(
            request_json, "display_name", file_name.replace("_", "-")
        )
        if len(data) == 0:
            # file missing
            file_dict = copy.copy(dummy_data[0][1])
            file_dict["display_name"] = file_name
            file_dict["file_type"] = type
            request_json.append(file_dict)
        elif len(data) == 1:
            # file present but needs
            # restructuring
            file_dict = data[0][1]
            file_dict["display_name"] = file_name
            file_dict["file_type"] = type
            idx = data[0][0]
            request_json[idx] = file_dict
        else:
            raise ValueError("We should not end up here.")

    # generate engine-specific files
    # use `sample_outputs.tar.gz` to simulate
    if clone_sample_outputs:
        files_to_create = [
            "sample_outputs_deepsparse.tar.gz",
            "sample_outputs_onnxruntime.tar.gz",
        ]
        data = _fetch_from_request_json(
            request_json, "display_name", "sample_outputs.tar.gz"
        )
        assert len(data) == 1
        for file in files_to_create:
            idx, file_dict = copy.copy(data[0])
            file_dict["display_name"] = file
            request_json.append(copy.copy(file_dict))
        del request_json[idx]

    return request_json


def _create_tar_directories(request_json: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    # filter out the unused files
    request_json = [
        x
        for x in request_json
        if x["file_type"] not in ["framework", "tar_gz", "onnx_gz"]
    ]

    # create all archived directories (using `sample-inputs.tar.gz` to simulate)
    data = _fetch_from_request_json(
        request_json, "display_name", "sample-inputs.tar.gz"
    )
    assert len(data) == 1
    _, file_dict = data[0]
    for folder in [
        "training.tar.gz",
        "logs.tar.gz",
        "deployment.tar.gz",
        "onnx.tar.gz",
    ]:
        file_dict_ = copy.copy(file_dict)
        file_dict_["display_name"] = folder
        file_dict_["file_type"] = folder.replace(".tar.gz", "")
        request_json.append(file_dict_)

    return request_json


def _fetch_from_request_json(request_json: List[Dict[str, Any]], key: str, value: str):
    return [
        (idx, copy.copy(file_dict))
        for (idx, file_dict) in enumerate(request_json)
        if file_dict[key] == value
    ]
