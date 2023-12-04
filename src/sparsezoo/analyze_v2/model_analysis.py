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


from typing import Dict

import yaml

from sparsezoo.analyze_v2.model_validator import ModelAnalysisModel
from sparsezoo.analyze_v2.node_analysis import NodeAnalysis
from sparsezoo.analyze_v2.summary_analysis import SummaryAnalysis


class ModelAnalysis:
    def __init__(
        self, summary_analysis: SummaryAnalysis, node_analyses: Dict[str, NodeAnalysis]
    ):
        self.summary_analysis = summary_analysis
        self.node_analyses = node_analyses

    def to_dict(self):
        return ModelAnalysisModel(
            summaries=self.summary_analysis.to_dict(),
            nodes={id: node.to_dict() for id, node in self.node_analyses.items()},
        ).dict()

    def to_yaml(self):
        return yaml.dump(self.to_dict())
