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


import re
from email.contentmanager import raw_data_manager

import requests
import yaml

from .constants import GIT_SERVER_REGEX, RAW_FILE_CONTENT_URL


def extract_git_server_metadata(git_server_path: str):
    """
    Extract name and namespace from git_server_path. Accepts:
        model url           : https://git.neuralmagic.com/neuralmagic/cary
        git clone ssh url   : git@git.neuralmagic.com:neuralmagic/cary.git
        git clone http url  : https://git.neuralmagic.com/neuralmagic/cary.git
    """

    matches = re.match(GIT_SERVER_REGEX, git_server_path)
    return {
        "name": matches.group("name"),
        "namespace": matches.group("namespace"),
    }


def local_load(file_path: str):
    """Given a file_path, return a dict with its contents as metadata"""

    with open(file_path, "r") as yaml_file:
        raw_data = next(yaml.safe_load_all(yaml_file.read()))
    return raw_data


def web_load(git_server_url: str, filename: str, branch: str = "main"):
    """
    Given a gitserver url (url of the gitserver model, or git clone ssh/http),
    and the file name, get the contents of the file using http request

    returns a text/str of the response
    """
    git_server_metadata = extract_git_server_metadata(git_server_url)
    response = requests.get(
        RAW_FILE_CONTENT_URL.format(
            **git_server_metadata, **{"filename": filename, "branch": branch}
        )
    )
    response.raise_for_status()
    return response.text

    pass
