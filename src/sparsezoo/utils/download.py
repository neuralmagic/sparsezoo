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
import math
import multiprocessing
import os
import re
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

import requests
from tqdm import tqdm

from .helpers import create_parent_dirs


__all__ = ["download_file", "Downloader"]

_LOGGER = logging.getLogger(__name__)

CHUNK_BYTES = 500_000_000  # 500 MB


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
    :param num_retries: number of times to retry the download if it fails

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
    """Add description to display the status in tqdm progress bar"""

    def __init__(self, maxsize=0, description: str = ""):
        super().__init__(maxsize)
        self.description = description


class Downloader:
    def __init__(
        self,
        url: str,
        download_path: str,
        max_retries: int = 3,
        chunk_bytes: int = CHUNK_BYTES,
    ):

        self.url = url
        self.download_path = download_path
        self.max_retries = max_retries
        self.file_size: int = self.get_file_size()
        self.chunk_bytes = chunk_bytes
        self.job_queues = Queue()
        self._lock = threading.Lock()
        self.chunk_download_path = self.get_chunk_download_path(download_path)

    def get_chunk_download_path(self, path: str) -> str:
        """Get the path where chunks will be downloaded"""

        # make the folder name from the model name and file to be downloaded
        stub = path.split(os.path.sep)[-3]
        path = "_".join(path.split(os.path.sep)[-2:])
        file_name_as_folder = path.replace(".", "_")
        file_id = str(uuid4())[:4]

        # Note: parallel download may cause multiple processes to download
        # the same file
        # save the chunks on a different folder than the root model folder
        # ~/.cache/sparsezoo/neuralmagic/chunks/stub/file_id/tokenizer_json/{chunk1, ...} # noqa
        return os.path.join(
            str(Path.home()),
            ".cache",
            "sparsezoo",
            "neuralmagic",
            "chunks",
            stub,
            file_id,
            file_name_as_folder,
        )

    def is_range_header_supported(self) -> bool:
        """Check if chunck download is supported"""
        response = requests.head(
            self.url, headers={"Accept-Encoding": "identity"}, allow_redirects=True
        )
        if response.headers.get("accept-ranges") is not None:
            return True
        return False

    def get_file_size(self) -> int:
        """Get file size of the requested file"""
        response = requests.head(
            self.url, headers={"Accept-Encoding": "identity"}, allow_redirects=True
        )
        file_size = int(response.headers.get("Content-Length", -1))
        if file_size > 0:
            return file_size
        raise ValueError(f"Invalid download URL: {self.url}")

    def queue_chunk_download_jobs(self) -> None:
        """
        Queues two jobs: file chunk download and combining the chunks.

        This method calculates the number of download jobs needed based on the size of
         the file to be downloaded and the size of each chunk. It creates download jobs
         for each chunk, where each job is responsible for downloading a specific byte
         range of the file. After queuing all the chunk download jobs, it queues an
         additional job to combine these chunks into the final file.

        Each download job is added to a JobQueue with a description (shown in the
         progress bar), and the jobs are configured with the necessary parameters,
         including the path where the chunk will be saved and the 'Range' header to
         download the specified byte range. The method ensures that the byte ranges
         are calculated correctly to cover the entire file, even if the file size is
         not a multiple of the chunk size.

        After all chunk download jobs are queued, if at least one chunk download job
         exists, a final job is queued to combine all downloaded chunks into the final
         file and delete the individual chunk files. This combining job is added to a
         separate JobQueue with its own descriptive label.

        Side effects:
            - Queues jobs in the instance's `job_queues` attribute, which should be a
               Queue of JobQueues.
            - Utilizes the `download_file` method to download each chunk.
            - Utilizes the `get_chunk_file_path` method to determine the path for each
               chunk's temporary file.
            - Utilizes the `combine_chunks_and_delete` method as the final step to
               combine all chunks and clean up.

        Note:
            This method does not start the download jobs; it only queues them.
            The jobs need to be executed by a worker or scheduler that processes the
             queued JobQueues.
        """
        file_name = self.download_path.split(os.path.sep)[-1]
        download_jobs: Queue = JobQueue(
            description=f"Downloading Chunks for {file_name}"
        )
        num_download_jobs = math.ceil(self.file_size / self.chunk_bytes)
        for job_id in range(num_download_jobs):
            start_byte = 0 if job_id == 0 else job_id * (self.chunk_bytes) + 1
            end_byte = (
                max(0, start_byte - 1) + self.chunk_bytes
                if self.chunk_bytes * (job_id + 1) < self.file_size
                else self.file_size
            )
            bytes_range = f"bytes={start_byte}-{end_byte}"

            func_kwargs = {
                "download_path": (
                    os.path.join(
                        self.chunk_download_path, f"{job_id:05d}_{bytes_range}"
                    )
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
        if download_jobs.qsize() > 0:
            chunk_combine_job = JobQueue(description="Combining Chunks")
            chunk_combine_job.put(
                Job(
                    id=download_jobs.qsize(),
                    func=self.combine_chunks_and_delete,
                    func_kwargs={
                        "download_path": self.download_path,
                    },
                )
            )
            self.job_queues.put(download_jobs)
            self.job_queues.put(chunk_combine_job)

    def download(self, *args, **kwargs) -> None:
        self.queue_jobs(*args, **kwargs)
        self.run()

    def queue_jobs(self) -> None:
        """
        Determines the download strategy and queues the appropriate jobs.

        This method decides whether to use chunked download or a regular download based
         on the server's support forrange requests. If the server supports range
         requests,it queues chunk download jobs for different portions of the file,
         allowing paralleldownloads of file chunks. Otherwise, it falls back to a
         regulardownload,queuing a single job to download the entire file at once.

        For chunked downloads, it calls `queue_chunk_download_jobs` to calculate
         the number of chunks and queue separate download jobs for each chunk. It
         then queues a finaljob to combine these chunks into the final file.

        For a regular download, it queues a single job with the `download_file` method
         as thetarget, passing the download path as an argument. This job is added to
         the job queue to be executed by the downloader.

        """
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

    def run(self, num_threads: int = 1) -> None:
        """
        Executes queued download jobs in parallel using multiple threads.

        This method starts the download process by executing the jobs queued in
        `self.job_queues`. It processes each job queue sequentially, but runs the
         jobs within each queue in parallel,up to the specified number of threads.
         The next job_queue is executed if the previousjob_queue executes successfully

        :param num_threads: The maximum number of worker threads to use for downloading
         file chunks in parallel. Defaults to 10.

        """
        available_threads = multiprocessing.cpu_count() - threading.active_count()
        num_threads = max(available_threads // 2, num_threads)

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

    def execute_job_from_queue(self, job_queue: Queue, **kwargs) -> None:
        """
        Executes a single job from the given job queue, managing retries and
         synchronization.

        Retrieves and executes the next job from the specified job queue, applying
         any additional keyword arguments provided. Lock used to be thread safe
         accessing next job from the queue. If the job execution fails, it will
         retry the job based on  the job's retry policy until it succeeds or
         exceeds the maximum number of retries.

        :param job_queue: The queue from which to retrieve and execute a job.

        """
        with self._lock:
            job: Job = job_queue.get()
        success = False
        err = ""
        while not success and job.retries < job.max_retries:
            try:
                job.func(**job.func_kwargs, **kwargs)
                success = True
            except Exception as _err:
                err = _err
                _LOGGER.debug(
                    f"{job.retries/self.max_retries}: "
                    "Failed running {self.func} with kwargs {job.func_kwargs}"
                )
                _LOGGER.error(_err)
                job.retries += 1
                if job.retries < job.max_retries:
                    job_queue.put(job)

        if not success:
            _LOGGER.debug(f"Chunk download failed after {self.max_retries} retries.")
            raise ValueError(err)

    def download_file(
        self,
        download_path: str,
        progress_bar: tqdm,
        headers: Dict[str, Any] = {},
        write_chunk_size: Optional[int] = None,
    ) -> None:
        """
        Downloads a file or a chunk of a file to the specified path, updating a progress
         bar as it downloads.

        Downloads the file (or a chunk of it, if specified by the 'Range' header in
         `headers`) from the URL associated with the downloader instance. It writes the
         downloaded content to the file specified by `download_path`. The method updates
         the given `progress_bar` object to reflect the progress of the download.

        :param download_path: The path to which the file (or chunk) will be downloaded.
        :param progress_bar: A tqdm progress bar instance to be updated as the file
         downloads.
        :param headers: Additional headers to include in the download request, such as
            'Range' for chunked downloads. Defaults to an empty dictionary.
        :param write_chunk_size: The size of chunks to write to the file at a time.
         If not specified, a default size of 500MB is used

        """
        write_chunk_size = min(CHUNK_BYTES, self.file_size)
        _LOGGER.debug("creating ", download_path)

        create_parent_dirs(download_path)

        response = requests.get(
            self.url, headers=headers, stream=True, allow_redirects=True
        )
        with open(download_path, "wb") as f:
            for chunk in response.iter_content(write_chunk_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

    def combine_chunks_and_delete(self, download_path: str, progress_bar: tqdm) -> None:
        """
        Combines the chunks and deletes the folder containing chunk files

        :param download_path: path to combine the chunks, specified in object
         instantiation
        :param progress_bar: tqdm object showing the progress of combining chunks

        """
        _LOGGER.debug("Combing and deleting ", self.chunk_download_path)

        pattern = re.compile(r"\d+_bytes=")
        files = os.listdir(self.chunk_download_path)

        chunk_files = [chunk_file for chunk_file in files if pattern.match(chunk_file)]

        sorted_chunk_files = list(sorted(chunk_files))

        create_parent_dirs(self.download_path)
        with open(self.download_path, "wb") as combined_file:
            for file_path in sorted_chunk_files:
                chunk_path = os.path.join(self.chunk_download_path, file_path)
                with open(chunk_path, "rb") as infile:
                    data = infile.read()
                    combined_file.write(data)
                    progress_bar.update(len(data))

        shutil.rmtree(os.path.dirname(self.chunk_download_path))

    def get_chunk_file_path(self, file_range: str) -> str:
        """
        Generate chunk file path based on the download path specified in
         object instantiation

        :param file_range: additional string to extend to the file displaying the
         range of chunk of the file

        """
        parent_directory = os.path.dirname(self.download_path)
        return os.path.join(parent_directory, "chunks", file_range)
