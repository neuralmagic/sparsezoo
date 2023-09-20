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

import logging

import requests


LOGGER = logging.getLogger(__name__)


def test_sparsezoo_website_health():
    try:
        response = requests.get("https://sparsezoo.neuralmagic.com", timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        LOGGER.error(f"Error when accessing SparseZoo website: {e}")
        raise AssertionError(f"Failed to access SparseZoo website: {e}") from e
