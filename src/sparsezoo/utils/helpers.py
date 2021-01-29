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

"""
Code related to helper functions for model zoo
"""

import errno
import os
from typing import Union

from tqdm import auto, tqdm, tqdm_notebook


__all__ = [
    "CACHE_DIR",
    "clean_path",
    "create_dirs",
    "create_parent_dirs",
    "create_tqdm_auto_constructor",
    "tqdm_auto",
]

CACHE_DIR = os.path.expanduser(os.path.join("~", ".cache", "sparsezoo"))


def clean_path(path: str) -> str:
    """
    :param path: the directory or file path to clean
    :return: a cleaned version that expands the user path and creates an absolute path
    """
    return os.path.abspath(os.path.expanduser(path))


def create_dirs(path: str):
    """
    :param path: the directory path to try and create
    """
    path = clean_path(path)

    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            # Unexpected OSError, re-raise.
            raise


def create_parent_dirs(path: str):
    """
    :param path: the file path to try to create the parent directories for
    """
    parent = os.path.dirname(path)
    create_dirs(parent)


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
