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
The analysis tests of Sparsezoo require ground truth
yaml files to compare to. This script updates those
files for when the models or implementation change


##########
Command help:
usage: update_analysis_tests.py

##########
usage: sparsezoo.py

Download a specific model from the repo.

##########
Example:
python3 scripts/update_analysis_tests.py
"""

from sparsezoo import Zoo
from sparsezoo.analysis import ModelAnalysis


# reference: tests.sparsezoo.analysis.helpers
_MODEL_PATHS = {
    "yolact_none": {
        "stub": "zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/" "base-none",
        "truth": "tests/sparsezoo/analysis/yolact_none.yaml",
    },
    "mobilenet_v1_pruned_moderate": {
        "stub": "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/"
        "imagenet/pruned-moderate",
        "truth": "tests/sparsezoo/analysis/mobilenet_v1_pruned_moderate.yaml",
    },
    "bert_pruned_quantized": {
        "stub": "zoo:nlp/question_answering/bert-base/pytorch/huggingface/"
        "squad/12layer_pruned80_quant-none-vnni",
        "truth": "tests/sparsezoo/analysis/bert_pruned_quantized.yaml",
    },
    "resnet50_pruned_quantized": {
        "stub": "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/"
        "imagenet/pruned85_quant-none-vnni",
        "truth": "tests/sparsezoo/analysis/resnet50_pruned_quantized.yaml",
    },
}


def update_analysis_tests(model_paths):
    for model_path in model_paths.values():
        model = Zoo.load_model_from_stub(model_path["stub"])
        model.onnx_file.download()
        onnx_path = model.onnx_file.downloaded_path()

        model_analysis = ModelAnalysis.from_onnx(onnx_path)
        model_analysis.yaml(model_path["truth"])


if __name__ == "__main__":
    update_analysis_tests(_MODEL_PATHS)
