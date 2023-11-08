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

from typing import Dict, List

from pydantic import BaseModel


class DistributionAnalysis(BaseModel):
    count: int
    mean: float
    median: float
    modes: List[float]
    sum_val: float
    min_val: float
    max_val: float
    percentiles: Dict[float, float]  # 0.1, 0.25, 0.5, 0.75, 0.9
    std_dev: float
    skewness: float
    kurtosis: float
    entropy: float
    bin_width: float  # (max_val - min_val) / num_bins
    num_bins: int  # 25
    bins: List[float]  # counts in each bin
    bin_centers: List[float]  # center value for each bin
