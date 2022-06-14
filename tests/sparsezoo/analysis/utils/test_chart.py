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

import pytest

from sparsezoo import Zoo
from sparsezoo.analysis import ModelAnalysis
from sparsezoo.analysis.utils.chart import (
    draw_operation_chart,
    draw_parameter_chart,
    draw_parameter_operation_combined_chart,
    draw_sparsity_by_layer_chart,
)


@pytest.fixture(scope="session")
def get_model_analysis():
    model_stubs = {
        "yolact_none": "zoo:cv/segmentation/yolact-darknet53/"
        "pytorch/dbolya/coco/base-none",
        "mobilenet_v1_pruned_moderate": "zoo:cv/classification/mobilenet_v1-1.0/"
        "pytorch/sparseml/imagenet/pruned-moderate",
        "bert_pruned_quantized": "zoo:nlp/question_answering/bert-base/"
        "pytorch/huggingface/squad/"
        "12layer_pruned80_quant-none-vnni",
        "resnet50_pruned_quantized": "zoo:cv/classification/resnet_v1-50"
        "/pytorch/sparseml/imagenet/pruned85_quant-none-vnni",
    }

    model_analyses = {}
    for model_name, model_stub in model_stubs.items():
        model_stub = model_stubs[model_name]
        model = Zoo.load_model_from_stub(model_stub)
        model.onnx_file.download()
        onnx_path = model.onnx_file.downloaded_path()
        analysis = ModelAnalysis.from_onnx_model(onnx_path)
        model_analyses[model_name] = analysis

    def _get_model_analysis(model_name):
        return model_analyses[model_name]

    return _get_model_analysis


@pytest.mark.parametrize(
    "model_name",
    [
        ("yolact_none"),
        ("mobilenet_v1_pruned_moderate"),
        ("bert_pruned_quantized"),
        ("resnet50_pruned_quantized"),
    ],
)
def test_draw_sparsity_by_layer_chart(model_name, get_model_analysis):
    model_analysis = get_model_analysis(model_name)
    out_path = f"/Users/poketopa/Desktop/analysis_tool/draw_sparsity_by_layer_chart_{model_name}.png"
    draw_sparsity_by_layer_chart(
        model_analysis, out_path=out_path, model_name=model_name
    )


@pytest.mark.parametrize(
    "model_name",
    [
        ("yolact_none"),
        ("mobilenet_v1_pruned_moderate"),
        ("bert_pruned_quantized"),
        ("resnet50_pruned_quantized"),
    ],
)
def test_draw_operation_chart(model_name, get_model_analysis):
    model_analysis = get_model_analysis(model_name)
    out_path = (
        f"/Users/poketopa/Desktop/analysis_tool/draw_operation_chart_{model_name}.png"
    )
    draw_operation_chart(model_analysis, out_path=out_path, model_name=model_name)


@pytest.mark.parametrize(
    "model_name",
    [
        ("yolact_none"),
        ("mobilenet_v1_pruned_moderate"),
        ("bert_pruned_quantized"),
        ("resnet50_pruned_quantized"),
    ],
)
def test_draw_parameter_chart(model_name, get_model_analysis):
    model_analysis = get_model_analysis(model_name)
    out_path = (
        f"/Users/poketopa/Desktop/analysis_tool/draw_parameter_chart_{model_name}.png"
    )
    draw_parameter_chart(model_analysis, out_path=out_path, model_name=model_name)


@pytest.mark.parametrize(
    "model_name",
    [
        ("yolact_none"),
        ("mobilenet_v1_pruned_moderate"),
        ("bert_pruned_quantized"),
        ("resnet50_pruned_quantized"),
    ],
)
def test_draw_parameter_operation_combined_chart(model_name, get_model_analysis):
    model_analysis = get_model_analysis(model_name)
    out_path = f"/Users/poketopa/Desktop/analysis_tool/draw_parameter_operation_combined_chart_{model_name}.png"
    draw_parameter_operation_combined_chart(
        model_analysis, out_path=out_path, model_name=model_name
    )
