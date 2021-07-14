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


@pytest.fixture
def dummy_dataset():
    _dummy_array_1 = np.random.rand(2, 3)
    _dummy_array_2 = np.random.rand(34, 3)
    return Dataset(data=[_dummy_array_1, _dummy_array_2], name="dummy")


def test_has_iter_batches(dummy_dataset):
    assert hasattr(dummy_dataset, "iter_batches")


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        2,
        3,
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
def test_batched_iteration(dummy_dataset, batch_size, iterations):
    data_loader = dummy_dataset.iter_batches(
        batch_size=batch_size, iterations=iterations
    )
    data_shape = dummy_dataset.data[0].shape

    for iteration, batch in enumerate(data_loader):
        batch_element_shape = batch[0].shape

        assert isinstance(batch, list)
        assert batch_element_shape[0] == batch_size
        assert all(
            (
                batch_dimension == data_dimension
                for batch_dimension, data_dimension in zip(
                    batch_element_shape[1:], data_shape[1:]
                )
            )
        )

    assert iteration + 1 == iterations
