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

from copy import deepcopy

from sparsezoo.analytics import sparsezoo_analytics


def test_send_event_success():
    test_analytics = deepcopy(sparsezoo_analytics)
    test_analytics._disabled = False

    test_analytics.send_event(
        "tests__analytics__test_send_event_success",
        raise_errors=True,
        _await_response=True,
    )
