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

import contextlib
from typing import Optional

import geocoder
import requests

from sparsezoo.utils.helpers import disable_request_logs


__all__ = ["get_external_ip", "get_country_code", "is_gdpr_country"]


_GDPR_COUNTRY_CODES = [
    "AT",
    "BE",
    "BG",
    "HR",
    "CY",
    "CZ",
    "DK",
    "EE",
    "FI",
    "FR",
    "DE",
    "GR",
    "HU",
    "IE",
    "IT",
    "LV",
    "LT",
    "LU",
    "MT",
    "NL",
    "PL",
    "PT",
    "RO",
    "SK",
    "SI",
    "ES",
    "SE",
]


def get_external_ip() -> Optional[str]:
    """
    :return: the external ip of the machine, None if unable to get
    """
    try:
        with disable_request_logs():
            response = requests.get("https://ident.me")
        external_ip = response.text.strip()

        return external_ip
    except Exception:
        return None


def get_country_code() -> Optional[str]:
    """
    :return: the country code of the machine, None if unable to get
    """
    try:
        ip = get_external_ip()
        geo = geocoder.ip(ip)

        return geo.country
    except Exception:
        return None


def is_gdpr_country() -> bool:
    """
    :return: True if the country code of the machine is in the GDPR list,
             False otherwise
    """
    with contextlib.redirect_stderr(None):
        # suppress geocoder error logging
        country_code = get_country_code()

    return country_code is None or country_code in _GDPR_COUNTRY_CODES
