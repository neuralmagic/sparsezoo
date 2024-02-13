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


import pytest

from sparsezoo.utils.download import Downloader


@pytest.fixture
def mock_downloader(monkeypatch):
    def mock_head(*args, **kwargs):
        class MockResponse:
            def __init__(self, headers):
                self.headers = headers

        return MockResponse({"accept-ranges": "bytes", "Content-Length": "1500000000"})

    monkeypatch.setattr("requests.head", mock_head)

    downloader = Downloader(
        "http://foo.com/get/model.onnx.tar.gz", "/path/to/download/file.zip"
    )
    return downloader


def test_is_range_header_supported(mock_downloader):
    assert mock_downloader.is_range_header_supported() is True


def test_get_file_size(mock_downloader):
    assert mock_downloader.get_file_size() == 1500000000


def test_queue_chunk_download_jobs(mock_downloader):

    downloader = mock_downloader
    assert downloader.job_queues.qsize() == 0

    downloader.queue_chunk_download_jobs()

    # one download, one chunk combining job
    assert downloader.job_queues.qsize() == 2

    download_queue, combine_queue = (
        downloader.job_queues.get(),
        downloader.job_queues.get(),
    )

    # There should be 3 download jobs (for 1.5 GB with 500 MB chunks)
    assert download_queue.qsize() == 3

    expected_ranges = [
        "bytes=0-500000000",
        "bytes=500000001-1000000000",
        "bytes=1000000001-1500000000",
    ]
    for _ in range(download_queue.qsize()):
        job = download_queue.get()
        assert job.func_kwargs["headers"]["Range"] in expected_ranges

    # There should be 1 combine job
    assert combine_queue.qsize() == 1
