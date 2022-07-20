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
import copy
from typing import Any, Dict, List, Set, Tuple


__all__ = ["restructure_request_json", "fetch_from_request_json"]

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


def restructure_request_json(
    request_json: List[Dict[str, Any]], allowed_file_types: Set = ALLOWED_FILE_TYPES
) -> Dict[str, Any]:
    """
    Takes the legacy API response and restructures it, so that the output is
    compatible with the structure of ModelDirectory.

    :params request_json: data structure describing the
        files in the ModelDirectory (output from NeuralMagic API).
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
    nlp_folder = {"config.json", "tokenizer.json"}.issubset(set(training_file_names))

    if nlp_folder:
        for file_name in ["config.json", "tokenizer.json"]:
            training_file_dict = fetch_from_request_json(
                request_json, "display_name", file_name
            )
            assert len(training_file_dict) == 1
            deployment_file_dict = copy.copy(training_file_dict[0][1])
            deployment_file_dict["file_type"] = "deployment"
            request_json.append(deployment_file_dict)

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
