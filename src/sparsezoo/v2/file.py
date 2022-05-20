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

import functools
import json
import logging
import os
import time
import traceback
from typing import Any, Dict, Optional

import onnx
import yaml

from PIL import Image
from sparsezoo.utils.downloader import download_file
from sparsezoo.utils.numpy import load_numpy_list


__all__ = ["File"]


def retry(retries: int, retry_sleep_sec: int):
    """
    Retry Decorator
    Retries the wrapped function/method
        - `retry_num` times if the exceptions listed
        -  with `retry_sleep_sec` interval between each attempt
    
    :param retries: The number of times to repeat the wrapped function/method
    :type retry_sleep_sec: How long to wait between attempts
    """

    def decorator(func):
        """decorator"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """wrapper"""
            for attempt in range(retries):
                try:
                    return func(
                        *args, **kwargs
                    )  # should return the raw function's return value
                except Exception as err:
                    logging.error(err)
                    logging.error(traceback.format_exc())
                    time.sleep(retry_sleep_sec)
                logging.error("Trying attempt %s of %s.", attempt + 1, retries)
            logging.error("Func %s retry failed", func)
            raise Exception("Exceed max retry num: {} failed".format(retries))

        return wrapper

    return decorator


class File:
    """
    Object to wrap around common files. Currently, supporting:
    - numpy files
    - onnx files
    - markdown files
    - json files
    - csv files
    - image files

    :param name: name of the File
    :param path: path of the File
    :param url: url of the File
    """

    def __init__(
        self, name: str, path: Optional[str] = None, url: Optional[str] = None
    ):
        self.name = name
        self.url = url
        self.path = path

        # self.path can have any extension, including no extension.
        # However, the File object also contains information
        # About its loadable extensions.
        # Loadable files can be read into the memory.

        self.loadable_extensions = {
            ".npz": self._validate_numpy,
            ".onnx": self._validate_onnx,
            ".md": self._validate_markdown,
            ".json": self._validate_json,
            ".csv": self._validate_csv,
            ".jpg": self._validate_img,
            ".png": self._validate_img,
            ".jpeg": self._validate_img,
            ".yaml": self._validate_yaml,
        }

    @classmethod
    def from_dict(cls, file: Dict[str, Any]) -> "File":
        """
        Factory method for creating a File class object
        from a file dictionary
        (useful when working with the `request_json` from NeuralMagic).

        :param file: a dictionary which contains an information about
            the file (as returned by NeuralMagic API)
        :return: File class object
        """
        name = file.get("display_name")
        path = file.get("path")
        url = file.get("url")

        return cls(
            name=name,
            path=path,
            url=url,
        )

    @retry(retries=3, retry_sleep_sec=5)
    def download(self, destination_path: str, overwrite: bool = True):
        """
        Download the contents of the file from the url.

        :param destination_path: the local file path to save the downloaded file to
        :param overwrite: True to overwrite any previous files if they exist,
            False to not overwrite and raise an error if a file exists
        """
        new_file_path = os.path.join(destination_path, self.name)

        if self.url is None:
            raise ValueError(
                "The File object requires a valid attribute `url` to download "
                "the file contents from. However, `url` is None."
            )

        if self.path is not None:
            logging.warning(
                f"Overwriting the current location of the File: {self.path} "
                f"with the new location: {new_file_path}."
            )

        download_file(
            url_path=self.url,
            dest_path=new_file_path,
            overwrite=overwrite,
        )

        self.path = new_file_path

    # TODO: Add support for various integrations
    def validate(
        self, strict_mode: bool = True, integration: Optional[str] = None
    ) -> bool:
        """
        Validate whether the File object is loadable or not.

        :param strict_mode: specifies the behavior of private `_validate_{}` methods:
            - if strict_mode: method will raise ValueError on error
            - if not strict_mode: method will raise warning on
                error
        :param integration: name of the specific integration
            (e.g. transformers, YOLOv5 etc.)
        :return: boolean flag; True if File instance is loadable, otherwise False
        """
        if not self.name or (not self.path and not self.url):
            logging.warning(
                "Failed to validate a file. A valid file needs to "
                "have a valid `name` AND a valid `path` or `url`."
            )
            return False

        else:
            _, extension = os.path.splitext(self.name)

            if extension in self.loadable_extensions.keys():
                validation_function = self.loadable_extensions[extension]
                validation_function(strict_mode=strict_mode)
                return True
            else:
                return False

    def _validate_numpy(self, strict_mode):
        if not load_numpy_list(self.path):
            self._throw_error(
                error_msg="Numpy file could not been loaded properly",
                strict_mode=strict_mode,
            )

    def _validate_onnx(self, strict_mode):
        if not onnx.load(self.path):
            self._throw_error(
                error_msg="Onnx file could not been loaded properly",
                strict_mode=strict_mode,
            )

    def _validate_markdown(self, strict_mode):
        try:
            with open(self.path) as file:
                file.readlines()
        except Exception as error:  # noqa: F841
            self._throw_error(
                error_msg="Markdown file could not been loaded properly",
                strict_mode=strict_mode,
            )

    def _validate_json(self, strict_mode):
        try:
            with open(self.path) as file:
                json.load(file)
        except Exception as error:  # noqa: F841
            self._throw_error(
                error_msg="Json file could not been loaded properly",
                strict_mode=strict_mode,
            )

    def _validate_csv(self, strict_mode):
        try:
            with open(self.path) as file:
                file.readlines()
        except Exception as error:  # noqa: F841
            self._throw_error(
                error_msg="Csv file could not been loaded properly",
                strict_mode=strict_mode,
            )

    def _validate_img(self, strict_mode):
        if not Image.open(self.path):
            self._throw_error(
                error_msg="Image file could not been loaded properly",
                strict_mode=strict_mode,
            )

    def _validate_yaml(self, strict_mode):
        try:
            with open(self.path) as file:
                yaml.load(file, Loader=yaml.FullLoader)
        except Exception as error:  # noqa: F841
            self._throw_error(
                error_msg="Yaml file could not been loaded properly",
                strict_mode=strict_mode,
            )

    def _throw_error(self, error_msg, strict_mode):
        if strict_mode:
            raise ValueError(error_msg)
        else:
            logging.warning(error_msg)
