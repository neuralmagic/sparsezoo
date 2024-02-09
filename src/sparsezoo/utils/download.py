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
import os
import threading
from queue import Queue
from typing import Iterator, NamedTuple, Union

import requests
from tqdm import auto, tqdm, tqdm_notebook

from .helpers import clean_path, create_parent_dirs


__all__ = ["download_file", "download_file_iter"]

_LOGGER = logging.getLogger(__name__)


def create_tqdm_auto_constructor() -> Union[tqdm, tqdm_notebook]:
    """
    :return: the tqdm instance to use for progress.
        If ipywidgets is installed then will return auto.tqdm,
        if not will return tqdm so that notebooks will not break
    """
    try:
        import ipywidgets as widgets  # noqa: F401

        return auto.tqdm
    except Exception:
        pass

    return tqdm


tqdm_auto = create_tqdm_auto_constructor()


DownloadProgress = NamedTuple(
    "DownloadProgress",
    [
        ("chunk_size", int),
        ("downloaded", int),
        ("content_length", Union[None, int]),
        ("path", str),
    ],
)


class PreviouslyDownloadedError(Exception):
    """
    Error raised when a file has already been downloaded and overwrite is False
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def download_part(
    url, start_byte, end_byte, part_num, file, chunk_size, lock, completed
):
    _LOGGER.debug(f"bytes={start_byte}-{end_byte}")
    print(f"bytes={start_byte}-{end_byte}")
    # breakpoint()
    headers = {"Range": f"bytes={start_byte}-{end_byte}"}

    response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
    # breakpoint()
    response.raise_for_status()
    # if response.status_code != 206:
    #     print(headers)
    #     raise

    total_bytes_obtained = 0
    for chunk in response.iter_content(chunk_size=chunk_size):
        if chunk:
            total_bytes_obtained += len(chunk)
            # print("writing...", total_bytes_obtained)
            file.seek(start_byte)  # Seek to the correct position
            file.write(chunk)

    # print(headers)
    completed.add(part_num)

    print(total_bytes_obtained, part_num, completed, len(completed))


def download_range(url, ranges, part_num, file, chunk_size, lock, completed):

    # headers = {"Range": f"bytes={start_byte}-{end_byte}"}
    headers = {"Range": ", ".join(f"bytes={start}-{end}" for start, end in ranges)}
    print(headers)
    # breakpoint()
    response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
    response.raise_for_status()
    # breakpoint()

    # content = response.content
    completed.add(part_num)

    # print(len(content), len(completed), part_num )

    total_bytes_obtained = 0
    for chunk in response.iter_content(chunk_size=chunk_size):
        if chunk:
            total_bytes_obtained += len(chunk)
            # print("writing...", total_bytes_obtained)
            # file.seek(start_byte)  # Seek to the correct position
            file.write(chunk)


def _download_iter(
    url_path: str,
    dest_path: str,
    chunk_size=1e8,
    num_threads: int = 5,
) -> Iterator[DownloadProgress]:

    print(url_path)

    # 100 MB
    # breakpoint()
    num_threads = 100000
    # chunk_size = 5e7

    chunk_size = int(chunk_size)
    _LOGGER.debug(f"downloading file from {url_path} to {dest_path}")

    response = requests.head(url_path, stream=True, allow_redirects=True)
    file_size = int(response.headers["Content-Length"])
    if file_size < chunk_size:
        chunk_size = file_size

    lock = threading.Lock()
    completed = set()

    try:
        with open(dest_path, "wb") as f:
            f.truncate(file_size)

        chunk_iter = (file_size // int(chunk_size)) + int(file_size % chunk_size != 0)
        yield DownloadProgress(0, 0, file_size, dest_path)

        with open(dest_path, "r+b") as file:
            downloaded_chunk = 0
            threads = []
            chunk_iter_group = max(chunk_iter // num_threads, 1)
            ranges = []
            for chunk_group in range(chunk_iter_group):
                if chunk_group == chunk_iter_group - 1:
                    num_threads = chunk_iter % num_threads
                for chunk_num in range(num_threads):
                    start_byte = (
                        chunk_num * chunk_size + chunk_group * chunk_size * num_threads
                    )
                    end_byte = (
                        start_byte + chunk_size - 1
                        if chunk_num < chunk_iter - 1
                        else file_size
                    )
                    ranges.append((start_byte, end_byte))
                    thread = threading.Thread(
                        target=download_part,
                        args=(
                            url_path,
                            start_byte,
                            end_byte,
                            chunk_num,
                            file,
                            chunk_size,
                            lock,
                            completed,
                        ),
                    )
                    # breakpoint()
                    thread.start()
                    # thread.join()

                    threads.append(thread)

                    # thread.start()
                    # thread.join()

                # download_range(url_path, ranges, chunk_num, file, chunk_size, lock, completed),
                for thread in threads:
                    thread.join()
                    downloaded_chunk += file_size / chunk_iter
                    yield DownloadProgress(
                        chunk_size, downloaded_chunk, file_size, dest_path
                    )

        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)
        print(0)

    except Exception as err:
        _LOGGER.error(f"error downloading file from {url_path} to {dest_path}: {err}")

        try:
            os.remove(dest_path)
        except Exception:
            pass
        raise err


def download_file_iter(
    url_path: str,
    dest_path: str,
    overwrite: bool,
    num_retries: int = 3,
) -> Iterator[DownloadProgress]:
    """
    Download a file from the given url to the desired local path
    :param url_path: the source url to download the file from
    :param dest_path: the local file path to save the downloaded file to
    :param overwrite: True to overwrite any previous files if they exist,
        False to not overwrite and raise an error if a file exists
    :param num_retries: number of times to retry the download if it fails
    :return: an iterator representing the progress for the file download
    :raise PreviouslyDownloadedError: raised if file already exists at dest_path
        nad overwrite is False
    """
    dest_path = clean_path(dest_path)

    create_parent_dirs(dest_path)

    if not overwrite and os.path.exists(dest_path):
        raise PreviouslyDownloadedError()

    if os.path.exists(dest_path):
        _LOGGER.debug(f"removing previously downloaded file at {dest_path}")

        try:
            os.remove(dest_path)
        except OSError as err:
            _LOGGER.warning(
                "error encountered when removing older "
                f"cache_file at {dest_path}: {err}"
            )

    retry_err = None

    for retry in range(num_retries + 1):
        _LOGGER.debug(
            f"downloading attempt {retry} for file from {url_path} to {dest_path}"
        )

        try:
            for progress in _download_iter(url_path, dest_path):
                yield progress
            break
        except PreviouslyDownloadedError as err:
            raise err
        except Exception as err:
            _LOGGER.error(
                f"error while downloading file from {url_path} to {dest_path}"
            )
            retry_err = err

    if retry_err is not None:
        raise retry_err

def download_file(
    url_path: str,
    dest_path: str,
    overwrite: bool,
    num_retries: int = 3,
    show_progress: bool = True,
    progress_title: str = None,
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
    # dest_path = "/home/george/sparsezoo/tmp"

    downloader = Downloader(
        url=url_path, download_path=dest_path, max_retries=num_retries
    )
    downloader.download()


from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import re
from dataclasses import dataclass


@dataclass
class Job:
    id: int
    chunk_range: str
    is_completed: bool


class Downloader:
    def __init__(
        self,
        url: str,
        download_path: str,
        max_retries: int = 3,
    ):
        self.url = url
        self.download_path = download_path
        self.max_retries = max_retries
        self.file_size = self.get_file_size()
        self.queue = Queue(maxsize=0)

    def is_chunk_downloable(self):
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

    def download(self, *args, **kwargs):
        if self.is_chunk_downloable():
            self.chunk_download(*args, **kwargs)
            self.combine_chunks(*args, **kwargs)
            return
        
        self.download_file(download_path=self.download_path, **kwargs)
        
    def combine_chunks(self):
        parent_directory = os.path.dirname(self.download_path)
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
                    combined_file.write(infile.read())
        
        
    
    def get_chunk_file_path(self, file_range: str):
        parent_directory = os.path.dirname(self.download_path)
        return os.path.join(parent_directory, "chunks", file_range)
    
    
    def populate_job_queue(self, chunk_size: int):
        iter =  self.file_size //chunk_size + int(chunk_size % (self.file_size -1) != 0)
        for job_id in range(iter):
            start_byte = job_id * chunk_size
            end_byte = start_byte + chunk_size - 1 if chunk_size * (job_id + 1) < self.file_size else self.file_size
            self.queue.put(
               (job_id,  f"bytes={start_byte}-{end_byte}")
            )


    def chunk_download(self, num_threads: int = 40, chunk_size: int =500_000_000):
        
        self.populate_job_queue(chunk_size)

        with tqdm(total=self.file_size, unit='B', unit_scale=True, desc='Downloading', leave=True) as progress_bar:
            
            while self.queue:
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = []
                    for _ in range(num_threads):
                        job = self.queue.get()
                        bytes_range = job[1]
                        headers = {"Range": bytes_range}

                        download_path = self.get_chunk_file_path(f"{i:05d}_{bytes_range}")
                        kwargs = dict(
                            headers=headers, 
                            download_path=download_path,
                            progress_bar = progress_bar,
                        )
                        future = executor.submit(self.run, **kwargs)
                        futures.append(future)

                    for future in futures:
                        future.result()  # This will propagate any exceptions if they occurred in the threads

    

    def download_file(
        self,
        download_path: str,
        **kwargs,
    ):
        with tqdm(
            total=self.file_size,
            unit="B",
            unit_scale=True,
            desc="Downloading",
            leave=True,
        ) as progress_bar:
            self.run(
                download_path=download_path,
                progress_bar=progress_bar,
                **kwargs
            )

    def run(self, download_path: str, progress_bar, headers: Dict[str, Any] = {}, write_chunk_size: Optional[int] = None):
        
        retries = 0
        success = False
        create_parent_dirs(download_path)
        while not success and retries < self.max_retries:
            try:
                response = requests.get(
                    self.url, headers=headers, stream=True, allow_redirects=True
                )
                with open(download_path, "wb") as f:
                    for chunk in response.iter_content(write_chunk_size):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))
                success = True
            except Exception as e:
                print(
                    f"Chunk download failed. Retrying... ({retries+1}/{self.max_retries})"
                )
                retries += 1
        if not success:
            print(f"Chunk download failed after {self.max_retries} retries.")
