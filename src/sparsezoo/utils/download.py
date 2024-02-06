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

def download_part(url, start_byte, end_byte, part_num, file, chunk_size, lock, completed):
    _LOGGER.debug(f"bytes={start_byte}-{end_byte}")
    print(f"bytes={start_byte}-{end_byte}")
    
    headers = {"Range": f"bytes={start_byte}-{end_byte}"}
    response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
    response.raise_for_status()
    
    total_written = 0
    for chunk in response.iter_content(chunk_size=chunk_size):
        if chunk:
            file.seek(start_byte + total_written)  # Seek to the correct position
            file.write(chunk)
            total_written += len(chunk)  # Update the total bytes written
            
    completed.add(part_num)
    print(completed)

    
def _download_iter(
    url_path: str, dest_path: str, chunk_div=1
) -> Iterator[DownloadProgress]:

    _LOGGER.debug(f"downloading file from {url_path} to {dest_path}")

    response = requests.head(url_path, stream=True, allow_redirects=True)
    file_size = int(response.headers["Content-Length"])
    
    lock = threading.Lock()
    completed = set()

    try:
        with open(dest_path, "wb") as f:
            f.truncate(file_size)

        chunk_size = file_size // chunk_div
        
        with open(dest_path, "r+b") as file:
            threads = []
            for chunk_num in range(chunk_div):
                start_byte = chunk_num * chunk_size
                end_byte = (
                    start_byte + chunk_size - 1 if chunk_num < chunk_div - 1 else file_size
                )
                thread = threading.Thread(
                    target=download_part,
                    args=(url_path, start_byte, end_byte, chunk_num, file, chunk_size, lock, completed),
                )
                thread.start()
                threads.append(thread)

            yield DownloadProgress(0, 0, file_size, dest_path)
            
            downloaded_chunk=0
            for thread in threads:
                thread.join()
                downloaded_chunk += file_size / chunk_div
                yield DownloadProgress(chunk_size, downloaded_chunk, file_size, dest_path)
                

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
    bar = None
    for progress in download_file_iter(url_path, dest_path, overwrite, num_retries):
        if (
            bar is None
            and show_progress
            and progress.content_length
            and progress.content_length > 0
        ):
            bar = tqdm_auto(
                total=progress.content_length,  # the total iteration
                desc=progress_title
                if progress_title
                else f"Downloading (…){dest_path[-20:]}: ",
                unit="B",  # unit string to be displayed
                unit_scale=True,  # let tqdm to determine the scale in kilo, mega, etc.
                unit_divisor=1024,  # is used when unit_scale is true
            )

        if bar:
            bar.update(n=progress.chunk_size)

    if bar:
        bar.close()
