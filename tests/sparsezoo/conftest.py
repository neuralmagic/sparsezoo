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


def _get_files(directory: str) -> List[str]:
    list_filepaths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            list_filepaths.append(os.path.join(os.path.abspath(root), file))
    return list_filepaths


@pytest.fixture(scope="session", autouse=True)
def check_for_created_files():
    start_files_root = _get_files(directory=r".")
    start_files_temp = _get_files(directory=tempfile.gettempdir())
    yield
    end_files_root = _get_files(directory=r".")
    end_files_temp = _get_files(directory=tempfile.gettempdir())

    assert len(start_files_root) >= len(end_files_root), (
        f"{len(end_files_root) - len(start_files_root)} "
        f"files created in current working "
        f"directory during pytest run. "
        f"Created files: {set(end_files_root) - set(start_files_root)}"
    )
    max_allowed_sized_temp_files_megabytes = 1
    size_of_temp_files_bytes = sum(
        os.path.getsize(path) for path in set(end_files_temp) - set(start_files_temp)
    )
    size_of_temp_files_megabytes = size_of_temp_files_bytes / 1024 / 1024

    assert max_allowed_sized_temp_files_megabytes >= size_of_temp_files_megabytes, (
        f"{size_of_temp_files_megabytes} "
        f"megabytes of temp files created in temp directory during pytest run. "
        f"Created files: {set(end_files_temp) - set(start_files_temp)}"
    )
