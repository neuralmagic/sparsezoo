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

import numpy as np
import pytest as pytest

from sparsezoo.utils import Dataset


@pytest.mark.parametrize(
    "_data",
    [
        Dataset(data=np.empty(shape=(100, 100, 10, 10)), name="4d"),
    ],
)
def test_has_iter_batches(_data):
    assert hasattr(_data, "iter_batches")


@pytest.mark.parametrize(
    "_data",
    [
        Dataset(data=np.empty(shape=(100, 100, 10, 10)), name="4d"),
        Dataset(data=np.empty(shape=(100, 100, 10)), name="3d"),
        Dataset(data=np.empty(shape=(100, 10)), name="2d"),
        Dataset(data=np.empty(shape=(100,)), name="1d"),
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        10,
        100,
    ],
)
@pytest.mark.parametrize(
    "iterations",
    [
        1,
        10,
        100,
    ],
)
def test_batched_iteration(_data, batch_size, iterations):
    data_loader = _data.iter_batches(batch_size=batch_size, iterations=iterations)
    data_shape = _data.data.shape

    # fix 1-d numpy array shape
    if len(data_shape) == 1:
        data_shape = (data_shape[0], 1)

    for iteration, batch in enumerate(data_loader):
        batch_shape = batch[0].shape

        assert isinstance(batch, list)
        assert len(batch) == data_shape[1]
        assert len(batch_shape) == len(data_shape) - 1
        assert all([a == b for a, b in zip(batch_shape[1:], data_shape[2:])])

    assert iteration + 1 == iterations
