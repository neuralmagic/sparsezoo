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

import json
import logging
import os
import re
import time
import traceback
from typing import Any, Dict, Optional

import onnx
import yaml

from PIL import Image
from sparsezoo.utils.downloader import download_file
from sparsezoo.utils.numpy import load_numpy_list


__all__ = ["File"]


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

    def download(
        self,
        destination_path: str,
        overwrite: bool = True,
        retries: int = 1,
        retry_sleep_sec: int = 5,
    ):
        """
        Download the contents of the file from the url.

        :param destination_path: the local file path to save the downloaded file to
        :param overwrite: True to overwrite any previous files if they exist,
            False to not overwrite and raise an error if a file exists
        :param retries: The maximum number of times to ping the API for the response
        :type retry_sleep_sec: How long to wait between `retry` attempts
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
            for attempt in range(retries):
                try:
                    download_file(
                        url_path=self.url,
                        dest_path=new_file_path,
                        overwrite=overwrite,
                    )

                    self.path = new_file_path
                    return

                except Exception as err:
                    logging.error(err)
                    logging.error(traceback.format_exc())
                    time.sleep(retry_sleep_sec)
                logging.error(
                    f"Trying attempt {attempt + 1} of {retries}.", attempt + 1, retries
                )
            logging.error("Download retry failed...")
            raise Exception("Exceed max retry attempts: {} failed".format(retries))

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

    def _validate_recipe(self, strict_mode):
        # only validate whether a file is a recipe if sparseml is installed.
        # this is optional, since we do not want to have an explicit dependency
        # on sparseml in sparsezoo.
        from sparseml.pytorch.optim import ScheduledModifierManager

        try:
            manager = ScheduledModifierManager.from_yaml(self.path)  # noqa  F841
        except Exception as error:  # noqa: F841
            self._throw_error(
                error_msg="Markdown file could not been loaded properly",
                strict_mode=strict_mode,
            )

    def _validate_model_card(self):
        try:
            with open(self.path, "r") as yaml_file:
                yaml_str = yaml_file.read()

            # extract YAML front matter from markdown recipe card
            # adapted from
            # https://github.com/jonbeebe/frontmatter/blob/master/frontmatter
            yaml_delim = r"(?:---|\+\+\+)"
            _yaml = r"(.*?)"
            re_pattern = r"^\s*" + yaml_delim + _yaml + yaml_delim
            regex = re.compile(re_pattern, re.S | re.M)
            result = regex.search(yaml_str)
            yaml_str = result.group(1)
            yaml_dict = yaml.safe_load(yaml_str)
            # returns a string "{domain}-{sub_domain}, if valid
            # this makes the method reusable to fetch the integration
            # name for the integration validation
            return yaml_dict

        except Exception as error:  # noqa: F841
            logging.error(error)

    def _validate_markdown(self, strict_mode):
        # test if .md file is a model_card
        is_valid_model_card = self._validate_model_card()
        # if not, attempt to check if it is a recipe file
        if not is_valid_model_card:
            try:
                from sparseml.pytorch.optim import (  # noqa  F401
                    ScheduledModifierManager,
                )
            except Exception as error:  # noqa  F841
                # if not model card and unable to check if recipe,
                # optimistically assume the .md file is valid.
                return
            self._validate_recipe(strict_mode=strict_mode)

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