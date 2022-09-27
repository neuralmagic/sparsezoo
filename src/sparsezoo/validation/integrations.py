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
Helper functions that provide the validation data for the
validation procedure described in IntegrationValidator
"""

__all__ = [
    "validate_nlp",
    "validate_cv_detection",
    "validate_cv_segmentation",
    "validate_cv_classification",
]


def validate_nlp():
    training_files = {
        "config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "trainer_state.json",
        "training_args.bin",
        "vocab.txt",
        "recipe.yaml",
    }
    optional_training_files = {}
    deployment_files = {
        "model.onnx",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    return training_files, optional_training_files, deployment_files


def validate_cv_classification():
    training_files = {
        "model.pth",
    }
    optional_training_files = {"recipe.yaml"}
    deployment_files = {"model.onnx"}
    return training_files, optional_training_files, deployment_files


def validate_cv_detection():
    training_files = {"model.pt"}
    optional_training_files = {"recipe.yaml", "model.ckpt.pt"}
    deployment_files = {"model.onnx"}
    return training_files, optional_training_files, deployment_files


def validate_cv_segmentation():
    training_files = {
        "model.pth",
    }
    optional_training_files = {"recipe.yaml"}
    deployment_files = {"model.onnx"}
    return training_files, optional_training_files, deployment_files
