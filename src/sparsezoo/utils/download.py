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

import concurrent.futures
import logging
import os
import re
import shutil
import threading
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Callable, Dict, Optional

import requests
from tqdm import tqdm

from .helpers import create_parent_dirs


__all__ = ["download_file"]

_LOGGER = logging.getLogger(__name__)


def download_file(
    url_path: str,
    dest_path: str,
    num_retries: int = 3,
    **kwargs,
):
    """
    Download a file from the given url to the desired local path
    :param url_path: the source url to download the file from
    :param dest_path: the local file path to save the downloaded file to
    :param overwrite: True to overwrite any previous files if they exist,
        False to not overwrite and raise an error if a file exists
    :param num_retries: number of times to retry the download if it fails
    :param show_progress: True to show a progress bar for the download,
        False otherwise
    :param progress_title: The title to show with the progress bar
    :raise PreviouslyDownloadedError: raised if file already exists at dest_path
        nad overwrite is False
    """
    downloader = Downloader(
        url=url_path, download_path=dest_path, max_retries=num_retries
    )
    downloader.download()


@dataclass
class Job:
    id: int
    func: Callable
    max_retries: int = 3
    retries: int = 0

    description: str = ""
    func_kwargs: Dict[str, Any] = field(default_factory=dict)


class JobQueue(Queue):
    def __init__(self, maxsize=0, description: str = ""):
        super().__init__(maxsize)
        self.description = description


class Downloader:
    def __init__(
        self,
        url: str,
        download_path: str,
        max_retries: int = 3,
        chunk_bytes: int = 500_000_000,
    ):

        self.url = url
        self.download_path = download_path
        self.max_retries = max_retries
        self.file_size = self.get_file_size()
        self.chunk_bytes = chunk_bytes
        self.job_queues = Queue()
        self._lock = threading.Lock()

    def is_range_header_supported(self):
        response = requests.head(
            self.url, headers={"Accept-Encoding": "identity"}, allow_redirects=True
        )
        if response.headers.get("accept-ranges") is not None:
            return True
        return False

    def get_file_size(self):
        response = requests.head(
            self.url, headers={"Accept-Encoding": "identity"}, allow_redirects=True
        )
        file_size = int(response.headers.get("Content-Length", -1))
        if file_size > 0:
            return file_size
        raise ValueError(f"Invalid download URL: {self.url}")

    def queue_chunk_download_jobs(self):

        download_jobs: Queue[Job] = JobQueue(description="Downloading Chunks")
        num_download_jobs = self.file_size // self.chunk_bytes + int(
            self.chunk_bytes % (self.file_size - 1) != 0
            if self.file_size > self.chunk_bytes
            else 1
        )

        for job_id in range(num_download_jobs):
            start_byte = job_id * self.chunk_bytes
            end_byte = (
                start_byte + self.chunk_bytes - 1
                if self.chunk_bytes * (job_id + 1) < self.file_size
                else self.file_size
            )
            bytes_range = f"bytes={start_byte}-{end_byte}"

            func_kwargs = {
                "download_path": self.get_chunk_file_path(
                    f"{job_id:05d}_{bytes_range}"
                ),
                "headers": {
                    "Range": bytes_range,
                },
            }

            download_jobs.put(
                Job(
                    id=job_id,
                    func=self.download_file,
                    func_kwargs=func_kwargs,
                )
            )

        # add chunk combine job at the end
        chunk_combine_job = JobQueue(description="Combining Chunks")
        chunk_combine_job.put(
            Job(
                id=download_jobs.qsize() + 1,
                func=self.combine_chunks_and_delete,
                func_kwargs={
                    "download_path": self.download_path,
                },
            )
        )
        self.job_queues.put(download_jobs)
        self.job_queues.put(chunk_combine_job)

    def download(self, *args, **kwargs):
        self.queue_jobs(*args, **kwargs)
        self.run()

    def queue_jobs(self):
        # queue of queue. Run each job in job_queue sequentially
        # run each job in job_queue in parallel

        # Chunk stream download
        if self.is_range_header_supported():
            self.queue_chunk_download_jobs()
            return

        # regular stream download
        job_queue = JobQueue(maxsize=0)
        func_kwargs = {
            "download_path": self.download_path,
        }
        job_queue.put(
            Job(
                id=1,
                func=self.download_file,
                func_kwargs=func_kwargs,
            )
        )
        self.job_queues.put(job_queue)

    def run(self, num_threads: int = 10):
        is_prev_job_queue_success = True
        while not self.job_queues.empty() and is_prev_job_queue_success:
            job_queue = self.job_queues.get()
            is_prev_job_queue_success = False
            with tqdm(
                total=self.file_size,
                unit="B",
                unit_scale=True,
                desc=job_queue.description,
                leave=True,
            ) as progress_bar:

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_threads
                ) as executor:
                    futures = []
                    while not job_queue.empty():
                        for i in range(min(num_threads, job_queue.qsize())):
                            future = executor.submit(
                                self.execute_job_from_queue,
                                job_queue,
                                progress_bar=progress_bar,
                            )
                            futures.append(future)

                        for future in futures:
                            future.result()
                is_prev_job_queue_success = True

    def execute_job_from_queue(self, job_queue: Queue[Job], **kwargs):
        with self._lock:
            job: Job = job_queue.get()
        success = False
        while not success and job.retries < job.max_retries:
            try:
                job.func(**job.func_kwargs, **kwargs)
                success = True
            except Exception as _err:
                _LOGGER.debug(
                    f"{job.retries/self.max_retries}: "
                    "Failed running {self.func} with kwargs {job.func_kwargs}"
                )
                _LOGGER.debug(_err)
                job.retries += 1
                if job.retries < job.max_retries:
                    job_queue.put(job)

        if not success:
            _LOGGER.debug(f"Chunk download failed after {self.max_retries} retries.")
            raise ValueError

    def download_file(
        self,
        download_path: str,
        progress_bar: tqdm,
        headers: Dict[str, Any] = {},
        write_chunk_size: Optional[int] = None,
    ):
        write_chunk_size = min(500_000_000, self.file_size)
        create_parent_dirs(download_path)
        response = requests.get(
            self.url, headers=headers, stream=True, allow_redirects=True
        )
        with open(download_path, "wb") as f:
            for chunk in response.iter_content(write_chunk_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

    def combine_chunks_and_delete(self, download_path: str, progress_bar: tqdm):
        parent_directory = os.path.dirname(download_path)
        chunk_directory = os.path.join(parent_directory, "chunks")

        pattern = re.compile(r"\d+_bytes=")
        files = os.listdir(chunk_directory)

        chunk_files = [chunk_file for chunk_file in files if pattern.match(chunk_file)]

        sorted_chunk_files = list(sorted(chunk_files))

        create_parent_dirs(self.download_path)

        with open(self.download_path, "ab") as combined_file:
            for file_path in sorted_chunk_files:
                chunk_path = os.path.join(chunk_directory, file_path)
                with open(chunk_path, "rb") as infile:
                    data = infile.read()
                    combined_file.write(data)
                    progress_bar.update(len(data))

        shutil.rmtree(chunk_directory)

    def get_chunk_file_path(self, file_range: str):
        parent_directory = os.path.dirname(self.download_path)
        return os.path.join(parent_directory, "chunks", file_range)
