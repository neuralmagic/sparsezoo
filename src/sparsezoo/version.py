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
Functionality for storing and setting the version info for SparseZoo
"""


import logging
import threading
from datetime import date

import requests

from sparsezoo.requests.base import LATEST_PACKAGE_VERSION_URL


_LOGGER = logging.getLogger(__name__)

version_base = "0.9.0"
is_release = False  # change to True to set the generated version as a release version


def _generate_version():
    return (
        version_base
        if is_release
        else f"{version_base}.{date.today().strftime('%Y%m%d')}"
    )


def _version_check(package_name, package_version, package_integration):

    url = f"{LATEST_PACKAGE_VERSION_URL}/\
        packages={package_name}\
        &versions={package_version}\
        &integrations={package_integration}"
    response = requests.get(url)  # no token-headers required
    response.raise_for_status()

    response_json = response.json()
    for checkedPackage in response_json["checkedPackages"]:
        if not checkedPackage["isLatest"]:
            _LOGGER.warning(
                f"Latest version {checkedPackage.isLatestVersion} \
                available for {checkedPackage.packageName}. \
                Current version {checkedPackage.userVersion}"
            )


__all__ = [
    "__version__",
    "version_base",
    "is_release",
    "version",
    "version_major",
    "version_minor",
    "version_bug",
    "version_major_minor",
]
__version__ = _generate_version()

version = __version__
version_major, version_minor, version_bug, version_build = version.split(".") + (
    [None] if len(version.split(".")) < 4 else []
)  # handle conditional for version being 3 parts or 4 (4 containing build date)
version_major_minor = f"{version_major}.{version_minor}"


package_integration = ""  # temp holder

threading.Thread(
    target=_version_check,
    kwargs={
        "package_name": __name__,
        "package_version": version,
        "package_integration": package_integration,
    },
).start()
