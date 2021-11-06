# flake8: noqa

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
- Functionality for accessing models, recipes, and supporting files in the SparseZoo
- Notify the user the last pypi package version
"""

# flake8: noqa
# isort: skip_file

from .version import *
from .main import *
from .models.zoo import *
from .objects import *

from .requests.base import LATEST_PACKAGE_VERSION_URL

import logging
import requests
import threading
import os

_LOGGER = logging.getLogger(__name__)

def _version_check(package_name, package_version, package_integration):
    url = f"{LATEST_PACKAGE_VERSION_URL}?"\
        f"packages={package_name}"\
        f"&versions={package_version}"\
        f"&integrations={package_integration}"
    try:
        response = requests.get(url)  # no token-headers required
        response.raise_for_status()

        response_json = response.json()
        for checkedPackage in response_json["checkedPackages"]:
            if not checkedPackage["isLatest"]:
                _LOGGER.warning(
                    "WARNING: "\
                    f"You are using {checkedPackage['packageName']} "\
                    f"version {checkedPackage['userVersion']} "\
                    f"however version {checkedPackage['latestVersion']} "\
                    "is available.\n"\
                    "You should consider upgrading via executing the "\
                    f"'pip install {checkedPackage['packageName']}'"\
                    "command"
                )
                _LOGGER.warning(
                    "WARNING: "\
                    "To turn off version check, set an environmental variable " \
                    "NM_VERSION_CHECK=false"\
                )
    except:
        pass

package_integration = "HuggingFace"  # temp holder

# Run thread in the background
if os.getenv("NM_VERSION_CHECK") == None or os.getenv("NM_VERSION_CHECK").lower().strip() != 'false':
    threading.Thread(
        target=_version_check,
        kwargs={
            "package_name": __name__,
            "package_version": version,
            "package_integration": package_integration,
        },
    ).start()
else:
    print('skipping version check')


