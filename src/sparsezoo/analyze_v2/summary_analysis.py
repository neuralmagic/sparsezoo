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

from typing import Any, Dict

import yaml

from sparsezoo.analyze_v2.memory_access_analysis import MemoryAccessAnalysis
from sparsezoo.analyze_v2.node_analysis import NodeAnalysis
from sparsezoo.analyze_v2.operation_analysis import OperationAnalysis
from sparsezoo.analyze_v2.parameter_analysis import ParameterAnalysis
from sparsezoo.analyze_v2.schemas import SummaryAnalysisSchema


class SummaryAnalysis:
    def __init__(
        self,
    ):
        self.parameter_analysis = {}
        self.operation_analysis = {}
        self.memory_access_analysis = {}

    def aggregate_analysis_metrics_from_node_analysis(
        self, node_analysis: NodeAnalysis
    ) -> None:
        """
        Given node_analysis, sum up its metrics into
        self.operation_analysis and self.parameter_analysis
        """
        parameter_analysis: ParameterAnalysis = node_analysis.parameter_analysis
        operation_analysis: OperationAnalysis = node_analysis.operation_analysis
        memory_access_analysis: MemoryAccessAnalysis = (
            node_analysis.memory_access_analysis
        )

        if parameter_analysis is not None:
            self.aggregate_metrics_from_parameter_analysis(parameter_analysis)
            self.aggregate_metrics_from_operation_analysis(operation_analysis)
            self.aggregate_metrics_from_memory_access_analysis(memory_access_analysis)

    def aggregate_metrics_from_parameter_analysis(self, analysis: ParameterAnalysis):
        sparsity_analysis_model = analysis.sparsity_analysis_model
        quantization_analysis_model = analysis.quantization_analysis_model

        if sparsity_analysis_model:
            model_dct = self.parameter_analysis
            if "sparsity" not in model_dct:
                model_dct["sparsity"] = {}

            sparsity_dct = model_dct["sparsity"]
            for model in sparsity_analysis_model:
                grouping = model.grouping
                if grouping not in sparsity_dct:
                    sparsity_dct[grouping] = model
                    continue
                sparsity_dct[grouping] += model

        if quantization_analysis_model is not None:
            if "quantization" not in model_dct:
                model_dct["quantization"] = {}
            quantization_dct = model_dct["quantization"]

            for model in quantization_analysis_model:
                grouping = model.grouping
                if grouping not in quantization_dct:
                    quantization_dct[grouping] = model
                    continue
                quantization_dct[grouping] += model

    def aggregate_metrics_from_operation_analysis(self, analysis: OperationAnalysis):
        sparsity_analysis_model = analysis.sparsity_analysis_model
        quantization_analysis_model = analysis.quantization_analysis_model

        if sparsity_analysis_model is not None:
            model_dct = self.operation_analysis
            if "sparsity" not in model_dct:
                model_dct["sparsity"] = {}

            sparsity_dct = model_dct["sparsity"]
            for model in sparsity_analysis_model:
                grouping = model.grouping
                if grouping not in sparsity_dct:
                    sparsity_dct[grouping] = model
                    continue
                sparsity_dct[grouping] += model

        if quantization_analysis_model is not None:
            if "quantization" not in model_dct:
                model_dct["quantization"] = {}
            quantization_dct = model_dct["quantization"]

            for model in quantization_analysis_model:
                grouping = model.grouping
                if grouping not in quantization_dct:
                    quantization_dct[grouping] = model
                    continue
                quantization_dct[grouping] += model

    def aggregate_metrics_from_memory_access_analysis(
        self, analysis: MemoryAccessAnalysis
    ):
        sparsity_analysis_model = analysis.sparsity_analysis_model
        quantization_analysis_model = analysis.quantization_analysis_model

        if sparsity_analysis_model:
            model_dct = self.memory_access_analysis
            if "sparsity" not in model_dct:
                model_dct["sparsity"] = {}

            sparsity_dct = model_dct["sparsity"]
            for model in sparsity_analysis_model:
                grouping = model.grouping
                if grouping not in sparsity_dct:
                    sparsity_dct[grouping] = model
                    continue
                sparsity_dct[grouping] += model

        if quantization_analysis_model is not None:
            if "quantization" not in model_dct:
                model_dct["quantization"] = {}
            quantization_dct = model_dct["quantization"]

            for model in quantization_analysis_model:
                grouping = model.grouping
                if grouping not in quantization_dct:
                    quantization_dct[grouping] = model
                    continue
                quantization_dct[grouping] += model

    def to_dict(self) -> Dict[str, Any]:
        if self.parameter_analysis:
            return SummaryAnalysisSchema(
                params=self.parameter_analysis,
                ops=self.operation_analysis,
                mem_access=self.memory_access_analysis,
            ).dict()

    def to_yaml(self) -> str:
        if self.parameter_analysis:
            return yaml.dump(self.to_dict())
