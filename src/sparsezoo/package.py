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
from typing import Optional

import requests

from sparsezoo.utils import LATEST_PACKAGE_VERSION_URL


LOGGER = logging.getLogger(__name__)


def package_version_check_request(
    package_name: str, package_version: str, package_integration: Optional[str]
):
    """
    Make an api call to api-neuralmagic.com, retrieve payload and check if the
    user is on the latest package version. Lambda: nm-get-latest-version

    :param package_name: package name of the client
    :param package_version: package version of the client
    :param package_integration: package integration of the client
    """
    url = (
        f"{LATEST_PACKAGE_VERSION_URL}?"
        f"packages={package_name}"
        f"&integrations={package_integration}"
        f"&versions={package_version}"
    )
    try:
        response = requests.post(url)  # no token-headers required
        response.raise_for_status()
        response_json = response.json()

        for checked_package in response_json["checked_packages"]:
            if not checked_package["is_latest"]:
                LOGGER.warning(
                    "WARNING: "
                    f"You are using {checked_package['package_name']} "
                    f"version {checked_package['user_package_version']} "
                    f"however version {checked_package['latest_package_version']} "
                    "is available.\n"
                    "Consider upgrading via executing the "
                    f"'pip install --upgrade' command.\n"
                    "To turn off set an environmental variable "
                    "NM_VERSION_CHECK=false"
                )
    except Exception as err:
        raise RuntimeError(
            f"Exception occured in the Neural Magic's internal version-api check\n{err}"
        )


def version_check_execution_condition(
    package_name: str, package_version: str, package_integration: Optional[str]
):
    """
    Check if conditions are met to run the version-check api

    :param package_name: package name of the client
    :param package_version: package version of the client
    :param package_integration: package integration of the client
    """
    if (
        os.getenv("NM_VERSION_CHECK") is not None
        and os.getenv("NM_VERSION_CHECK").lower().strip() == "false"
    ):
        LOGGER.info("Skipping Neural Magic's latest package version check")
        return

    try:
        package_version_check_request(
            package_name=package_name,
            package_integration=package_integration,
            package_version=package_version,
        )
    except Exception as err:
        LOGGER.warning(
            "Neural Magic's latest package version check raised an exception. "
            "To turn off set the following in the environment NM_VERSION_CHECK=false "
            f"Exception: {err}"
        )


def check_package_version(
    package_name: str, package_version: str, package_integration: Optional[str] = None
):
    """
    Run a background thread to run version-check api

    :param package_name: package name of the client
    :param package_version: package version of the client
    :param package_integration: package integration of the client
    """
    threading.Thread(
        target=version_check_execution_condition,
        kwargs={
            "package_name": package_name,
            "package_version": package_version,
            "package_integration": package_integration,
        },
    ).start()
