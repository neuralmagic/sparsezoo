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

import os
import tempfile
from typing import List

import pytest


def _get_file_count(directory: str) -> List[str]:
    list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            list.append(os.path.join(os.path.abspath(root), file))
    return list


@pytest.fixture(scope="session", autouse=True)
def check_for_created_files():
    start_file_count_root = _get_file_count(directory=r".")
    start_file_count_temp = _get_file_count(directory=tempfile.gettempdir())
    yield

    end_file_count_root = _get_file_count(directory=r".")
    end_file_count_temp = _get_file_count(directory=tempfile.gettempdir())

    assert len(start_file_count_root) >= len(end_file_count_root), (
        f"{len(end_file_count_root) - len(start_file_count_root)} "
        f"files created in current working "
        f"directory during pytest run."
    )
    max_allowed_temp_files = 5
    print(set(end_file_count_temp) - set(start_file_count_temp))
    assert len(start_file_count_temp) + max_allowed_temp_files >= len(
        end_file_count_temp
    ), (
        f"{len(end_file_count_temp) - len(start_file_count_temp)} "
        f"files created in /tmp "
        f"directory during pytest run. Diff: "
        f"{set(end_file_count_temp) - set(start_file_count_temp)}"
    )
