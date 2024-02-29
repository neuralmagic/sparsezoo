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


import numpy
import yaml


__all__ = [
    "numpy_array_representer",
]


def numpy_array_representer(dumper: yaml.Dumper, data: numpy.ndarray):
    """
    A representer for numpy arrays to be used with pyyaml
    """
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data.tolist())
