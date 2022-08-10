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


import os
import re
from typing import Dict

import requests
import yaml

from .constants import GIT_SERVER_REGEX, RAW_FILE_CONTENT_URL


__all__ = [
    "extract_git_server_metadata",
    "get_model_file",
    "local_load",
    "web_load",
]


def extract_git_server_metadata(git_server_path: str) -> Dict[str, str]:
    """
    Extract name and namespace from git_server_path. Accepts:
        model url           : https://git.neuralmagic.com/neuralmagic/cary
        git clone ssh url   : git@git.neuralmagic.com:neuralmagic/cary.git
        git clone http url  : https://git.neuralmagic.com/neuralmagic/cary.git

    :param git_server_path: path of the git server
    :return: dict containing 'name' and 'namespace'
    """

    matches = re.match(GIT_SERVER_REGEX, git_server_path)
    return {
        "name": matches.group("name"),
        "namespace": matches.group("namespace"),
    }


def get_model_file(
    path: str, file_name: str, platform: str = "web", branch: str = "main"
) -> Dict:
    """
    Get the file info from local or git-server

    :param path: path containing the desired file
    :param file_name: name of the desired file
    :param platform: either 'web' or 'platform'
    :param branch: git server branch
    :return: Dict of the desired file's raw metadata
    """
    if platform == "web":
        return web_load(path, file_name=file_name, branch=branch)
    elif platform == "local":
        return local_load(path, file_name=file_name)
    raise ValueError(
        "[get_model_file]: input arg 'platform' must be 'web' or 'local'"
        f"for path: {path}, file_name {file_name}"
    )


def local_load(path: str, file_name: str) -> Dict:
    """
    Given a file_path, return a dict with its contents as metadata

    :param path: path containing the desired file
    :param file_name: name of the desired file
    :return: raw metadata of file_name
    """

    file_path = os.path.join(path, file_name)
    with open(file_path, "r") as yaml_file:
        raw_data = next(yaml.safe_load_all(yaml_file.read()))
    return raw_data


def web_load(git_server_url: str, file_name: str, branch: str = "main") -> Dict:
    """
    Given a gitserver url (url of the gitserver model, or git clone ssh/http),
    and the file name, get the contents of the file using http request

    :param git_server_url: url /git clone of the git server model
    :param file_name: name of the file to be loaded
    :param branch: git server branch

    :return: a text/str of the response
    """
    git_server_metadata = extract_git_server_metadata(git_server_url)
    response = requests.get(
        RAW_FILE_CONTENT_URL.format(
            **git_server_metadata, **{"file_name": file_name, "branch": branch}
        )
    )
    response.raise_for_status()
    return next(yaml.safe_load_all(response.text))
