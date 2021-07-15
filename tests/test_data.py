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
    "dataset",
    [Dataset(data=(np.random.rand(2, 45), np.random.rand(2, 45)), name="tuple")],
)
def test_has_iter_batches(dataset):
    assert hasattr(dataset, "iter_batches")


@pytest.mark.parametrize(
    "dataset",
    [
        Dataset(data=[np.random.rand(2, 45), np.random.rand(2, 45)], name="list"),
        Dataset(data=(np.random.rand(2, 45), np.random.rand(2, 45)), name="tuple"),
    ],
)
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
def test_batched_iteration(dataset, batch_size, iterations):
    data_loader = dataset.iter_batches(batch_size=batch_size, iterations=iterations)
    data_shape = dataset.data[0].shape

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
