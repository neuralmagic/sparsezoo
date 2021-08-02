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


from collections import OrderedDict
from typing import Tuple

import pytest

from sparsezoo.utils.data import DataLoader, Dataset, RandomDataset


# TESTS for DataLoader


@pytest.fixture
def dataset() -> Dataset:
    """
    A Dataset containing 4 samples of 3 channel images
    """
    name, typed_shapes, num_samples = "rd1", {"inp": ([3, 224, 224], None)}, 4
    data = RandomDataset(name=name, typed_shapes=typed_shapes, num_samples=num_samples)
    yield data
    del data


@pytest.fixture
def dataset_b() -> Dataset:
    """
    A Dataset containing 1 sample of a single channel image
    """
    name, typed_shapes, num_samples = "rd2", {"inp": ([224, 224], None)}, 1
    data = RandomDataset(name=name, typed_shapes=typed_shapes, num_samples=num_samples)
    yield data
    del data


@pytest.fixture
def empty_dataset() -> Dataset:
    """
    A Dataset with no samples
    """
    name, typed_shapes, num_samples = "rd3", {"inp": ([224, 224], None)}, 0
    data = RandomDataset(name=name, typed_shapes=typed_shapes, num_samples=num_samples)
    yield data
    del data


@pytest.fixture
def dataset_and_shape(dataset) -> Tuple[Dataset, Tuple[int, ...]]:
    """
    Fixture to get the 3 channel image dataset along with image shape
    """
    yield dataset, (3, 224, 224)


@pytest.fixture
def loader_and_shape(dataset_and_shape) -> Tuple[DataLoader, Tuple[int, ...]]:
    """
    Fixture to get a DataLoader instance for the 3 channel image dataset along
    with expected batch shape
    """
    _dataset, _shape = dataset_and_shape
    batch_size = 2
    expected_batch_shape = (batch_size, *_shape)
    loader = DataLoader(
        _dataset, batch_size=batch_size, iter_steps=3, batch_as_list=False
    )
    yield loader, expected_batch_shape


class TestDataLoader:
    def test_no_dataset(self):
        with pytest.raises(ValueError):
            DataLoader(batch_size=1, iter_steps=1, batch_as_list=True)

    def test_empty_dataset(self, empty_dataset):
        with pytest.raises(ValueError):
            DataLoader(empty_dataset, batch_size=1, iter_steps=1, batch_as_list=True)

    def test_zero_batch_size(self, dataset):
        with pytest.raises(ValueError):
            DataLoader(dataset, batch_size=0, iter_steps=1, batch_as_list=True)

    def test_negative_iter_steps(self, dataset):
        with pytest.raises(ValueError):
            DataLoader(dataset, batch_size=1, iter_steps=-2, batch_as_list=True)

    def test_datasets_with_different_size(self, dataset, dataset_b):
        with pytest.raises(ValueError):
            DataLoader(
                dataset,
                dataset_b,
                batch_size=1,
                iter_steps=-2,
                batch_as_list=True,
            )

    def test_batch_is_list(self, dataset):
        loader = DataLoader(dataset, batch_size=1, iter_steps=1, batch_as_list=True)
        assert isinstance(next(loader), list)

    def test_batch_is_not_list(self, dataset):
        loader = DataLoader(dataset, batch_size=1, iter_steps=1, batch_as_list=False)
        assert isinstance(next(loader), OrderedDict)

    def test_batch_shape_and_number(self, loader_and_shape):
        _loader, _expected_batch_shape = loader_and_shape
        _expected_batches = _loader.iter_steps
        for index, batch in enumerate(_loader):
            assert batch["inp"].shape == _expected_batch_shape
        assert index + 1 == _expected_batches

    def test_infinite_iteration(self, dataset):
        ITERATION_LIMIT = 1000
        loader = DataLoader(dataset, batch_size=1, iter_steps=-1, batch_as_list=False)

        for index, batch in enumerate(loader):
            if index == ITERATION_LIMIT:
                break
        else:
            assert False

    def test_batch_index_greater_than_batches(self, dataset_b):
        with pytest.raises(IndexError):
            DataLoader(
                dataset_b,
                batch_size=1,
                iter_steps=1,
                batch_as_list=True,
            ).get_batch(10)
