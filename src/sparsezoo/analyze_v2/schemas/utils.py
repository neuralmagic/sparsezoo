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

from typing import Any

import numpy


def type_validator(value) -> Any:
    """
    Pydantic validator, mostly used to convert numpy to
     non-numpy types
    """
    if isinstance(value, numpy.generic):
        if isinstance(value, float):
            return float(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, list):
            return value.tolist()
    return value
