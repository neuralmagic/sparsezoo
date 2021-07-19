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
from typing import Iterable

import numpy
import numpy as np
import pytest

from sparsezoo.utils import Dataset


@pytest.fixture
def dummy_dataset():
    return Dataset(data=[np.random.rand(100, 10)], name="dummy-dataset")


@pytest.fixture
def single_input_dataset():
    data = [np.random.rand(3, 2), np.random.rand(3, 2), np.random.rand(3, 2)]
    return Dataset(data=data, name="single-input-test-dataset")


@pytest.fixture
def multi_input_dataset():
    data = [
        [np.random.rand(1, 2), np.random.rand(1, 3)],
        [np.random.rand(1, 2), np.random.rand(1, 3)],
    ]
    return Dataset(data=data, name="multi-input-test-dataset")


@pytest.fixture
def both_datasets(single_input_dataset, multi_input_dataset):
    return [single_input_dataset, multi_input_dataset]


def test_has_iter_batches(dummy_dataset):
    assert hasattr(dummy_dataset, "iter_batches")


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        2,
        10,
    ],
)
@pytest.mark.parametrize(
    "iterations",
    [
        1,
        2,
        4,
        10,
    ],
)
def test_iter_batches_returns_iterable(both_datasets, batch_size, iterations):
    for dataset in both_datasets:
        loader = dataset.iter_batches(batch_size=batch_size, iterations=iterations)
        assert isinstance(loader, Iterable)


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        2,
        10,
    ],
)
@pytest.mark.parametrize(
    "iterations",
    [
        1,
        2,
        4,
        10,
    ],
)
def test_batch_is_in_list(multi_input_dataset, batch_size, iterations):
    loader = multi_input_dataset.iter_batches(
        batch_size=batch_size, iterations=iterations
    )
    for batch in loader:
        assert isinstance(batch, list)


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        2,
        10,
    ],
)
@pytest.mark.parametrize(
    "iterations",
    [
        1,
        2,
        4,
        10,
    ],
)
def test_batch_not_in_list_for_single_input(
    single_input_dataset, batch_size, iterations
):
    loader = single_input_dataset.iter_batches(
        batch_size=batch_size, iterations=iterations
    )
    for batch in loader:
        assert not isinstance(batch, list) and isinstance(batch, numpy.ndarray)


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        2,
        10,
    ],
)
@pytest.mark.parametrize(
    "iterations",
    [
        1,
        2,
        4,
        10,
    ],
)
def test_iter_batches_single_input_batch_shape(
    single_input_dataset, batch_size, iterations
):
    loader = single_input_dataset.iter_batches(
        batch_size=batch_size, iterations=iterations
    )

    _data_dimensions = single_input_dataset.data[0].shape
    print("data dimensions", _data_dimensions)
    expected_batch_shape = (batch_size, *_data_dimensions)
    print(expected_batch_shape)
    for batch in loader:
        assert batch.shape == expected_batch_shape


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        2,
        10,
    ],
)
@pytest.mark.parametrize(
    "iterations",
    [
        1,
        2,
        4,
        10,
    ],
)
def test_iter_batches_number_of_iterations(both_datasets, batch_size, iterations):
    for dataset in both_datasets:
        loader = dataset.iter_batches(batch_size=batch_size, iterations=iterations)
        for iteration, batch in enumerate(loader):
            pass
        assert iteration + 1 == iterations


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        2,
        3,
    ],
)
@pytest.mark.parametrize(
    "iterations",
    [
        1,
        2,
        3,
    ],
)
def test_iter_batches_multi_input_batch_shape(
    multi_input_dataset, batch_size, iterations
):
    expected_batch_dimensions = [
        (batch_size, *multi_input.shape) for multi_input in multi_input_dataset.data[0]
    ]
    loader = multi_input_dataset.iter_batches(
        batch_size=batch_size, iterations=iterations
    )

    for batch in loader:
        assert all(
            expected_batch_dimensions[idx] == multi_input.shape
            for idx, multi_input in enumerate(batch)
        )
