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
import os
import threading
import uuid
from functools import wraps
from typing import Dict, Optional

import machineid
import requests

from sparsezoo.utils.gdpr import is_gdpr_country
from sparsezoo.utils.helpers import disable_request_logs
from sparsezoo.version import version as sparsezoo_version


__all__ = ["GoogleAnalytics", "analytics_disabled", "sparsezoo_analytics"]


_DEBUG = os.getenv("NM_DEBUG_ANALYTICS")


def analytics_disabled():
    """
    :return: True if analytics should be disabled, False otherwise
    """
    env_disabled = os.getenv("NM_DISABLE_ANALYTICS")

    return env_disabled or is_gdpr_country()


class GoogleAnalytics:
    """
    Google Analytics client for sending events for the given package and version

    :param package: the name of the package to send events for
    :param version: the version of the package to send events for,
        if the version string has a nightly datestamp, the datestamp portion will
        be replaced by 'nightly'
    :param package_params: optional dictionary of parameters to send with each event
    """

    @staticmethod
    def get_client_id():
        try:
            return str(uuid.UUID(machineid.id()))
        except Exception:
            return str(uuid.uuid4())

    def __init__(
        self,
        package: str,
        version: str,
        package_params: Optional[Dict[str, str]] = None,
    ):
        self._disabled = analytics_disabled()
        self._url = (
            "https://www.google-analytics.com/mp/collect"
            if not _DEBUG
            else "https://www.google-analytics.com/debug/mp/collect"
        )
        self._url += "?api_secret=YVDgNAMrTV6JScn7N3ZPBQ&measurement_id=G-8DK54R8077"
        self._client_id = (
            GoogleAnalytics.get_client_id() if not self._disabled else None
        )
        self._package = package
        self._version = _maybe_replace_nightly_date(version)
        self._package_params = package_params if package_params is not None else {}

    def send_event_decorator(
        self,
        event_name: str,
        event_params: Optional[Dict[str, str]] = None,
    ):
        """
        Send an event when the decorated function is called

        :param event_name: the name of the event to send
        :param event_params: optional dictionary of parameters to send with the event
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.send_event(event_name, event_params)
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def send_event(
        self,
        event_name: str,
        event_params: Optional[Dict[str, str]] = None,
        raise_errors: bool = False,
        _await_response: bool = False,
    ):
        """
        Send an event

        :param event_name: the name of the event to send
        :param event_params: optional dictionary of parameters to send with the event
        :param raise_errors: True to raise any errors that occur, False otherwise
        :param _await_response: (testing only) True to wait for analytics request
            thread to complete before returning. Default False
        """
        if self._disabled:
            if _await_response:
                raise ValueError(
                    "_await_response cannot be set to True when analytics is disabled"
                )
            return

        if not event_params:
            event_params = {}

        def _send_request():
            event_params.update(self._package_params)
            event_params["package"] = self._package
            event_params["version"] = self._version
            payload = {
                "client_id": self._client_id,
                "events": [{"name": event_name, "params": event_params}],
            }
            headers = {
                "Content-Type": "application/json",
            }
            data = json.dumps(payload)
            with disable_request_logs():
                try:
                    response = requests.post(self._url, headers=headers, data=data)
                    response.raise_for_status()
                    body = response.content
                    if _DEBUG:
                        print(body)
                except Exception as exception:
                    if _DEBUG:
                        print(exception)

                    if raise_errors:
                        raise exception

        thread = threading.Thread(target=_send_request)
        thread.start()

        if _await_response:
            thread.join()


def _maybe_replace_nightly_date(version: str) -> str:
    version_parts = version.split(".")
    if len(version_parts) == 4 and len(version_parts[3]) == 8:
        # nightly would be the 4th part of version string
        # and would be 8 character YYYYMMDD, assume this check is
        # sufficient, replace datestamp with 'nightly'
        version_parts[3] = "nightly"
        version = ".".join(version_parts)

    return version


# analytics client for sparsezoo, to disable set NM_DISABLE_ANALYTICS=1
sparsezoo_analytics = GoogleAnalytics("sparsezoo", sparsezoo_version)
