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

import glob
import os
from typing import Any


__all__ = [
    "create_dirs",
    "create_parent_dirs",
    "clean_path",
    "remove_tar_duplicates",
    "convert_to_bool",
]


def convert_to_bool(val: Any):
    """
    :param val: a value
    :return: False if value is a Falsy value e.g. 0, f, false, None, otherwise True.
    """
    return (
        bool(val)
        if not isinstance(val, str)
        else bool(val) and "f" not in val.lower() and "0" not in val.lower()
    )


def remove_tar_duplicates(directory: str):
    """
    If a directory contains similar the same data, both one
    as a directory and a .tar file, remove the .tar file.
    Example:

    Before:
        [directory_A, directory_B, directory_A.tar.gz,
        directory_B.tar.gz, directory_C.tar.gz]
    After:
        [directory_A, directory_B, directory_C.tar.gz]

    :param directory: A directory where the removal of tar duplicates is to happen
    """
    extension_to_remove = ".tar.gz"
    files = glob.glob(os.path.join(directory, "*"))
    possible_duplicates = [
        file.replace(extension_to_remove, "")
        for file in files
        if file.endswith(extension_to_remove)
    ]
    remaining_files = [file for file in files if not file.endswith(extension_to_remove)]
    duplicates = [file for file in remaining_files if file in possible_duplicates]
    [os.remove(duplicate + extension_to_remove) for duplicate in duplicates]


def create_dirs(path: str):
    """
    :param path: the directory path to try and create
    """
    path = clean_path(path)

    os.makedirs(path, exist_ok=True)


def create_parent_dirs(path: str):
    """
    :param path: the file path to try to create the parent directories for
    """
    parent = os.path.dirname(path)
    create_dirs(parent)


def clean_path(path: str) -> str:
    """
    :param path: the directory or file path to clean
    :return: a cleaned version that expands the user path and creates an absolute path
    """
    return os.path.abspath(os.path.expanduser(path))
